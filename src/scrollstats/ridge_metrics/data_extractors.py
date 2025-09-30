"""
This module contains the classes and functions used to extract data along the transects.

Data is extracted at three scales: bend, transect, and ridge.
Packet-scale metrics are just the ridge scale metrics aggregated within the packet boundaries. No unique extraction is done for the packet-scale.

Rules:
- null attributes are None, df values are NaN

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.io import DatasetReader as RasterioDatasetReader
from scipy import ndimage
from shapely.geometry import LineString, Point
from tqdm import tqdm

from .ridge_amplitude import calc_ridge_amps


def calc_dist(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """p1 and p2 are both (n,2) arrays of coordinates"""
    return np.sqrt((p1[:, 0] - p2[:, 0]) ** 2 + (p1[:, 1] - p2[:, 1]) ** 2)


def remove_coincident_points(coord_array: np.ndarray) -> np.ndarray:
    """Removes coincident points that happen to be adjacent"""

    # Calc distances between each point
    dist = calc_dist(coord_array[:-1], coord_array[1:])

    # Find where these distances are not near zero
    unique_idx = (
        np.nonzero(dist > 0.001)[0] + 1
    )  # Add one to the index b/c first point is guaranteed ok

    return coord_array[unique_idx]


def explode(line: LineString) -> list[LineString]:
    """Return a list of all 2 coordinate line segments in a given LineString"""
    return list(map(LineString, zip(line.coords[:-1], line.coords[1:], strict=False)))


def densify_segment(line: LineString) -> LineString:
    """Densify a line segment between two points to 1pt/m. Assumes line coordinates are in meters"""
    x, y = line.xy

    num_points = int(round(line.length))  # noqa: RUF046

    xs = np.linspace(*x, num_points + 1)
    ys = np.linspace(*y, num_points + 1)

    return LineString(zip(xs, ys, strict=False))


def densify_line(line: LineString) -> LineString:
    """Return the given LineString densified coordinates. Density = 1pt/unit length"""

    all_points = []

    # Break each line into its straight segments
    for seg in explode(line):
        # Densify segment and get coords
        coords = densify_segment(seg).coords

        # Append all coords to all_points bucket
        for coord_pair in coords:
            all_points.append(coord_pair)

    # Remove all coincident AND adjacent points
    unique_points = remove_coincident_points(np.array(all_points))

    return LineString(unique_points)


def transform_coords(
    coord_array: np.ndarray, bin_raster: RasterioDatasetReader
) -> np.ndarray:
    """Transform the coordinates from geo to image for indexing a in_bin_raster"""

    transform = bin_raster.transform
    t_coords = [~transform * coord for coord in coord_array]

    return np.round(t_coords).astype(int)  # round coords for indexing


class RidgeDataExtractor:
    """
    Responsible for calculating ridge metrics at each intersection of a ridge and transect.
    The geometry for this class is a 3-vertex LineString
    """

    def __init__(
        self,
        geometry: LineString,
        position: int,
        ridges: gpd.GeoDataFrame,
        dem_signal: np.ndarray[float] | None = None,
        bin_signal: np.ndarray[float] | None = None,
    ) -> None:
        # Inputs
        self.id = None
        self.geometry = geometry
        self.position = position
        self.ridges = ridges
        self.dem_signal = dem_signal
        self.dem_signal_selection = self.dem_signal
        self.bin_signal = bin_signal
        self.bool_mask = self.boolify_mask()
        self.signal_length = self.determine_signal_length()

        # Create GeoDataFrame
        self.data_columns = {
            "p_id": str,
            "ridge_id": str,
            "bend_id": str,
            "mig_dist": float,
            "mig_time": float,
            "mig_rate": float,
            "deposit_year": float,
            "ridge_width": float,
            "ridge_amp": float,
            "geometry": gpd.array.GeometryDtype(),
        }

        # Assess Geometry
        self.gdf = self.create_point_gdf()
        self.gdf = self.add_point_geometries(self.gdf, self.geometry)
        self.gdf = self.join_ridge_info(self.gdf, self.ridges)
        self.gdf = self.calc_values_from_ridge_info(self.gdf)
        self.gdf = self.calc_relative_vertex_distance(self.gdf, self.geometry)
        self.gdf = self.calc_vertex_indices(self.gdf, self.signal_length)

        # Process Binary Signal
        self.metric_confidence = self.determine_metric_confidence()
        self.swale_dq_adjustment = 0
        self.bool_mask = self.dq_first_swale()
        self.ridge_com = self.calc_ridge_coms()
        self.single_ridge_num = -9999  # Set by self.find_closest_ridge()
        self.single_ridge_bin_signal = self.find_closest_ridge()

        # Ridge Metrics
        self.ridge_width_px = self.calc_ridge_width_px()
        self.ridge_amp_series = self.calc_every_ridge_amp()
        self.ridge_amp = self.determine_ridge_amp()

        self.gdf = self.coerce_dtypes(self.gdf)

    def determine_signal_length(self) -> float:
        """Return length of dem/bin signal if provided"""
        if isinstance(self.bin_signal, np.ndarray) and isinstance(
            self.dem_signal, np.ndarray
        ):
            return self.dem_signal.size  # type: ignore[no-any-return]
        return np.nan  # type: ignore[no-any-return]

    def create_point_gdf(self) -> gpd.GeoDataFrame:
        """Create a 3 point GeoDataFrame to contain all relevant info for other methods."""
        gdf = gpd.GeoDataFrame(
            columns=self.data_columns, geometry="geometry", crs=self.ridges.crs
        )
        gdf = gdf.set_index("p_id")
        return gdf

    def add_point_geometries(
        self, gdf: gpd.GeoDataFrame, line: LineString
    ) -> gpd.GeoDataFrame:
        """Add the vertices from the 3vertex line as point geometries"""
        # Add geometry info
        id_list = []
        p_list = []

        for i, p in enumerate(line.coords):
            id_list.append(f"p{i}")
            p_list.append(Point(p))

        gdf = gdf.reset_index()
        gdf["p_id"] = id_list
        gdf["geometry"] = p_list
        gdf = gdf.set_index("p_id")

        return gdf

    def join_ridge_info(
        self, gdf: gpd.GeoDataFrame, ridges: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Get ridge ids, time, distance, and migration rates via spatial join from the ridge features"""

        # Use slight buffer to ensure intersection on spatial join
        gdf["geometry_buff"] = gdf.buffer(1e-5)
        gdf.set_geometry("geometry_buff", inplace=True)
        join_gdf = gdf.sjoin(ridges, how="left")

        gdf["ridge_id"] = join_gdf["ridge_id_right"]
        gdf["bend_id"] = join_gdf["bend_id_right"]
        gdf["deposit_year"] = join_gdf["deposit_year_right"].astype(float)

        # Reset geometry to points
        gdf.set_geometry("geometry", inplace=True)

        return gdf

    def calc_values_from_ridge_info(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Calculates the migration time, distance, and rate both before and after the center ridge.
        If the ridge does not have values for the deposit year, then mig_rate will be NaN.
        """

        # Calculate values from joined info
        gdf["mig_time"] = gdf["deposit_year"].diff().abs()
        gdf["mig_dist"] = gdf.distance(gdf.loc[["p0", "p0", "p1"]], align=False)
        gdf["mig_rate"] = gdf["mig_dist"] / gdf["mig_time"]

        return gdf

    def calc_relative_vertex_distance(
        self, gdf: gpd.GeoDataFrame, line: LineString
    ) -> gpd.GeoDataFrame:
        """Calculate the relative distance of each vertex along the transect."""

        gdf["relative_distance"] = np.cumsum(gdf["mig_dist"]) / line.length

        return gdf

    def calc_vertex_indices(
        self, gdf: gpd.GeoDataFrame, signal_length: float
    ) -> gpd.GeoDataFrame:
        """
        Calculate the array index of all vertices.
        If `self.signal_length` is nan, then return array of nans
        """

        gdf["vertex_indices"] = np.round(gdf["relative_distance"] * signal_length)

        # If any are nans, then you cannot cast to int
        if not gdf["vertex_indices"].isna().any():
            gdf["vertex_indices"] = gdf["vertex_indices"].astype(int)

        return gdf

    def boolify_mask(self) -> np.ndarray[bool] | None:
        """Simplifies the bin_sig (which may contain nans) to a pure boolean array"""

        if self.bin_signal is None:
            return None

        return np.where(np.isnan(self.bin_signal), 0, self.bin_signal).astype(bool)

    def determine_metric_confidence(self) -> int:
        """
        Assign a metric confidence score based on the boolean mask.
        """
        if self.bin_signal is None:
            return 0

        _labels, ridge_count = ndimage.label(self.bool_mask)
        _labels, swale_count = ndimage.label(~self.bool_mask)  # type: ignore[operator]

        # All swale
        if ridge_count == 0 and swale_count == 1:
            metric_confidence = 0
        # All ridge
        elif ridge_count == 1 and swale_count == 0:
            metric_confidence = 1
        # S-shape: 1 unbounded ridge and swale
        elif ridge_count == 1 and swale_count == 1:
            metric_confidence = 2
        # One bounded ridge, unbounded swales
        elif ridge_count < 3 and swale_count >= 1:
            metric_confidence = 3
        # One bounded ridge, two bounded swales
        elif ridge_count >= 3 and swale_count >= 2:
            metric_confidence = 4
        else:
            err_msg = f"Unexpected ridge-swale configuration in self.bool_mask. \
                            \n{self.bool_mask=} \
                            \n{ridge_count=} \
                            \n{swale_count=}"
            raise ValueError(err_msg)

        return metric_confidence

    def dq_first_swale(self) -> np.ndarray[bool] | None:
        """If the ridge position of the signal is 0, then remove the first chunk of false values"""
        if self.bin_signal is None:
            return None

        if self.metric_confidence == 0:
            return self.bool_mask

        if self.position == 0:
            first_positive = np.flatnonzero(self.bool_mask)[0]
            self.bool_mask = self.bool_mask[first_positive:]  # type: ignore[index]
            self.dem_signal_selection = self.dem_signal_selection[first_positive:]  # type: ignore[index]
            self.metric_confidence = self.determine_metric_confidence()
            self.swale_dq_adjustment = len(self.bin_signal) - len(self.bool_mask)
        return self.bool_mask

    def calc_ridge_coms(self) -> np.ndarray[bool] | None:
        """Find the center of mass for each ridge in the input binary signal."""

        if self.bin_signal is None:
            return None

        # Create a copy to not modify the original
        sig = self.bool_mask.copy()  # type: ignore[union-attr]

        # Find individual ridge areas
        labels, numfeats = ndimage.label(sig)

        # Find the centerpoint of each ridge along the transect
        coms = ndimage.center_of_mass(sig, labels, np.arange(numfeats) + 1)
        coms = np.round(coms).astype(int).reshape(numfeats)

        # Create boolean array where True indicates com of a ridge
        coms_signal = np.zeros(sig.shape).astype(bool)
        coms_signal[coms] = True

        return coms_signal

    def find_closest_ridge(self) -> np.ndarray[float]:
        """
        The bin_signal may have more than two ridges present.
        This method identifies which ridge is closest to the transect-ridge intersection point.
        """
        if self.bin_signal is None:
            return None

        if not self.ridge_com.any():  # type: ignore[union-attr]
            return np.zeros(self.bin_signal.shape)

        # Get index of center vertex
        poi_idx = self.gdf.loc["p1", "vertex_indices"]

        # Find indices of ridge centers of mass
        ## If the first swale area was disqualified (if self.position==0), then self.bool_mask
        ## was cropped and the ridge_com indices need to be adjusted by the distance of that dq'd swale.
        ## If first swale wasn't disqualified, then the adjustment is 0
        b = self.bool_mask
        ridge_midpoints = np.flatnonzero(self.ridge_com) + self.swale_dq_adjustment

        # Find the closest ridge
        dist_from_poi = np.absolute(ridge_midpoints - poi_idx)
        closest_ridge_num = np.flatnonzero(dist_from_poi == dist_from_poi.min())[0]
        self.single_ridge_num = closest_ridge_num

        # Erase all ridges that are not closest
        label, _num_feats = ndimage.label(b)
        single_ridge = (label == closest_ridge_num + 1).astype(float)

        return single_ridge

    def calc_ridge_width_px(self) -> float | None:
        """
        Calculate the width of the single ridge in pixels
        """
        if self.bin_signal is None:
            return None

        if self.metric_confidence < 2:
            return float(np.nan)

        return float(np.nansum(self.single_ridge_bin_signal))

    def calc_every_ridge_amp(self) -> np.ndarray[float] | list[None]:
        """
        Calculates the average amplitude of each observed ridges in the units of the DEM.
        """
        if self.bin_signal is None:
            return []

        return calc_ridge_amps(self.dem_signal_selection, self.bool_mask)

    def determine_ridge_amp(self) -> float | None:
        """
        Select the correct ridge amplitude calculated by `calc_every_ridge_amp()` based on the number of ridges present
        """
        if self.bin_signal is None:
            return None

        if len(self.ridge_amp_series) == 0:
            amp = np.nan
        elif len(self.ridge_amp_series) == 1:
            amp = self.ridge_amp_series[0]
        elif self.single_ridge_num >= 0:
            amp = self.ridge_amp_series[int(self.single_ridge_num)]
        else:
            err_msg = "Multiple ridges identified in signal, but the current ridge position was not set"
            raise ValueError(err_msg)

        return float(amp)

    def coerce_dtypes(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Coerce the the 'object' dtypes into their proper numeric types"""

        gdf = gdf.reset_index().astype(self.data_columns)
        gdf = gdf.set_index("p_id")
        return gdf

    def dump_data(self) -> dict[str, Any]:
        """Dump all the relevant info for the middle point."""
        d = {}

        d["ridge_id"] = self.gdf.loc["p1", "ridge_id"]
        d["transect_position"] = self.position
        d["metric_confidence"] = self.metric_confidence
        d["pre_mig_dist"] = self.gdf.loc["p2", "mig_dist"]
        d["post_mig_dist"] = self.gdf.loc["p1", "mig_dist"]
        d["pre_mig_time"] = self.gdf.loc["p2", "mig_time"]
        d["post_mig_time"] = self.gdf.loc["p1", "mig_time"]
        d["pre_mig_rate"] = self.gdf.loc["p2", "mig_rate"]
        d["post_mig_rate"] = self.gdf.loc["p1", "mig_rate"]
        d["deposit_year"] = self.gdf.loc["p1", "deposit_year"]
        d["bend_id"] = self.gdf.loc["p1", "bend_id"]
        d["geometry"] = self.gdf.loc["p1", "geometry"]

        d["bool_mask"] = self.bool_mask
        d["swale_dq_adjustment"] = self.swale_dq_adjustment
        d["dem_signal_selection"] = self.dem_signal_selection
        d["ridge_width"] = self.ridge_width_px
        d["ridge_amp"] = self.ridge_amp

        return d


class TransectDataExtractor:
    """
    Responsible for extracting ridge metrics along a transect.

    TransectDataExtractor will ultimately return a GeoDataFrame where each row is an eligible intersection between the transect and the ridge
    An eligible intersection is one that has a vertex before and after it so that the raster underneath can be sampled along the full width of the ridge.
    If a transect contains no eligible intersections, the gdf will be empty.
    """

    def __init__(
        self,
        transect_id: str,
        geometry: LineString,
        dem_signal: np.ndarray[float] | None = None,
        bin_signal: np.ndarray[float] | None = None,
        ridges: gpd.GeoDataFrame | None = None,
    ) -> None:
        # Inputs
        self.transect_id = transect_id
        self.geometry = geometry
        self.raw_dem_signal = dem_signal
        self.raw_bin_signal = bin_signal
        self.ridges = ridges

        # Create GeoDataFrame

        self.data_columns = {
            "ridge_id": str,
            "transect_id": str,
            "bend_id": str,
            "start_distances": float,
            "transect_position": int,
            "metric_confidence": int,
            "relative_vertex_distances": None,
            "vertex_indices": None,
            "dem_signal": None,
            "dem_signal_selection": None,
            "bin_signal": None,
            "bool_mask": None,
            "pre_mig_dist": float,
            "post_mig_dist": float,
            "pre_mig_time": float,
            "post_mig_time": float,
            "pre_mig_rate": float,
            "post_mig_rate": float,
            "ridge_width": float,
            "ridge_amp": float,
            "deposit_year": float,
            "substring_geometry": gpd.array.GeometryDtype(),
            "geometry": gpd.array.GeometryDtype(),
        }

        # Add Geometries
        self.itx_gdf = self.create_itx_gdf()
        self.itx_gdf = self.add_substring_geometry(self.itx_gdf)
        self.itx_gdf = self.add_point_geometry(self.itx_gdf)

        # Add transect_id
        self.itx_gdf = self.add_transect_id(self.itx_gdf)

        # Process binary and DEM signals
        self.itx_gdf = self.determine_substring_starts(self.itx_gdf)
        self.itx_gdf = self.add_relative_vertex_distances(self.itx_gdf)
        self.itx_gdf = self.calc_vertex_indices(self.itx_gdf)
        self.itx_gdf = self.slice_bin_signal(self.itx_gdf)
        self.itx_gdf = self.slice_dem_signal(self.itx_gdf)

    def create_itx_gdf(self) -> gpd.GeoDataFrame:
        """Create the gdf that will contain all the ridge data for each intersection."""
        gdf = gpd.GeoDataFrame(columns=self.data_columns, geometry="geometry").set_crs(
            self.ridges.crs  # type: ignore[union-attr]
        )

        return gdf

    def determine_eligible_coords(self, ls: LineString) -> list[tuple[float, float]]:
        """
        Determine coordinates in the transect linestring that are eligible to be a start of a substring.
        Because the substrings are all 3 vertices long, the last two are not eligible.
        These eligible coords are defined because multiple functions need to use these coordinates.
        """

        return ls.coords[:-2]  # type: ignore[no-any-return]

    def create_substrings(self, ls: LineString) -> list[LineString]:
        """Create substrings starting from the eligible coordinates of the given linestring"""

        eligible_coords = self.determine_eligible_coords(ls)
        substrings = [
            LineString(ls.coords[i : i + 3]) for i, v in enumerate(eligible_coords)
        ]

        return substrings

    def add_substring_geometry(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Adds the 3 vertex substring that corresponds to each itx."""

        gdf["substring_geometry"] = self.create_substrings(self.geometry)

        return gdf

    def add_point_geometry(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add the intersection (middle) point of the 3 vertex substring as its own point"""

        gdf["geometry"] = gdf["substring_geometry"].apply(lambda x: Point(x.coords[1]))

        return gdf

    def add_transect_id(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add the transect id as a column"""

        gdf["transect_id"] = self.transect_id

        return gdf

    def calc_cumulative_dist(
        self, coords: Sequence[tuple[float, float]]
    ) -> np.ndarray[float]:
        """Calculate the cumulative distances along a coordinate series"""

        coords = np.asarray(coords)
        dists = np.insert(calc_dist(coords[:-1], coords[1:]), 0, 0)
        cumdists = np.cumsum(dists)

        return cumdists

    def determine_substring_starts(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Determine the along-transect distance of the points of each substring"""

        eligible_coords = self.determine_eligible_coords(self.geometry)

        if eligible_coords:
            start_dists = self.calc_cumulative_dist(eligible_coords)
        else:
            start_dists = eligible_coords

        gdf["start_distances"] = start_dists

        return gdf

    def calc_relative_vertex_distances(
        self, ls: LineString, start_dist: float
    ) -> np.ndarray[float]:
        """Calculate the relative distance of each vertex along the transect."""

        sub_dists = self.calc_cumulative_dist(ls.coords)
        dists_from_start = sub_dists + start_dist
        rel_dists = dists_from_start / self.geometry.length

        return rel_dists

    def add_relative_vertex_distances(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Calculate the distance between the substring coordinates relative to the length of the whole line."""
        bucket = []

        for _i, row in gdf[["substring_geometry", "start_distances"]].iterrows():
            geom, dist = row
            relative_distances = self.calc_relative_vertex_distances(geom, dist)
            bucket.append(relative_distances)

        gdf["relative_vertex_distances"] = bucket
        return gdf

    def calc_vertex_indices(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Calculates the corresponding signal index of each of the substring vertices"""

        if self.raw_bin_signal is not None:
            gdf["vertex_indices"] = gdf["relative_vertex_distances"].apply(
                lambda x: np.round(x * self.raw_bin_signal.size).astype(int)
            )
        return gdf

    def slice_dem_signal(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Slice the DEM between the two end vertices of the substrings"""
        if self.raw_bin_signal is not None:
            gdf["dem_signal"] = gdf["vertex_indices"].apply(
                lambda x: self.raw_dem_signal[x[0] : x[2]]  # type: ignore[index]
            )

        return gdf

    def slice_bin_signal(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Slice the binary signal between the two end vertices of the substrings"""
        if self.raw_bin_signal is not None:
            gdf["bin_signal"] = gdf["vertex_indices"].apply(
                lambda x: self.raw_bin_signal[x[0] : x[2]]
            )

        return gdf

    def calc_ridge_metrics(self) -> gpd.GeoDataFrame:
        """
        Calculate ridge width and amplitude at every transect-ridge intersection.
        Return a GeoDataFrame with Point geometries.
        """

        for i, row in self.itx_gdf.iterrows():
            row[row.isna()] = None
            rde = RidgeDataExtractor(
                row["substring_geometry"],
                i,
                self.ridges,
                row["dem_signal"],
                row["bin_signal"],
            )

            ridge_metrics = rde.dump_data()

            self.itx_gdf.loc[i, list(ridge_metrics.keys())] = ridge_metrics

        self.itx_gdf = self.itx_gdf.astype(self.data_columns)

        return self.itx_gdf


class BendDataExtractor:
    """Responsible for extraction of ridge metrics across an entire bend."""

    def __init__(
        self,
        transects: gpd.GeoDataFrame,
        bin_raster: RasterioDatasetReader | None = None,
        dem: RasterioDatasetReader | None = None,
        ridges: gpd.GeoDataFrame | None = None,
        packets: gpd.GeoDataFrame | None = None,
    ) -> None:
        self.transects = transects
        self.bin_raster = bin_raster
        self.dem = dem
        self.ridges = ridges
        self.packets = packets

        # Calculate Metrics at the smaller scales
        self.rich_transects = self.calc_transect_metrics()
        self.itx_metrics = self.calc_itx_metrics()

    def disqualify_coords(
        self, coord_array: np.ndarray[float], raster: RasterioDatasetReader
    ) -> np.ndarray[bool]:
        """
        Some coordinates may be out of the in_bin_raster.
        This function disqualifies these coordinates and returns a boolean array showing the location of all disqualified coordinates.

        Coordinates are checked to see if they are 1) negative, 2) too large in x, or 3) too large in y
        """
        # Find location of all negative coords
        too_small = np.any(coord_array < 0, axis=1)

        # Find location of too large x
        x_max = raster.profile["width"]
        too_large_x = coord_array[:, 0] >= x_max

        # Find location of too large y
        y_max = raster.profile["height"]
        too_large_y = coord_array[:, 1] >= y_max

        # return true for a coord if that coord failed any test
        return np.any(np.vstack((too_small, too_large_x, too_large_y)), axis=0)

    def sample_array(
        self, coord_array: np.ndarray[float], raster: RasterioDatasetReader
    ) -> np.ndarray[float]:
        """
        Takes in an array of image coordinates, samples the image, and returns the sampled values.
        Assumes that the coord array and in_bin_raster dataset share the same crs.
        """
        # Prep the coords
        ## Some coordinates may be out of bounds for the in_bin_raster
        ## So we need to only sample with valid image coords, then pad the array with nans for all out of bounds areas
        disq = self.disqualify_coords(coord_array, raster)  # boolean array
        in_bounds = coord_array[~disq]

        # Sample the array with valid coords
        arr = raster.read(1)
        signal = arr[
            in_bounds[:, 1], in_bounds[:, 0]
        ].flatten()  # remember image coords are (y,x)

        # Pad either side of the signal with nans for disqualified points
        out_signal = np.zeros(disq.shape) * np.nan
        out_signal[~disq] = signal

        return out_signal

    def count_ridges(self, signal: np.ndarray[float]) -> int:
        """Counts ridges in binary waves."""

        mask = np.isnan(signal)
        _labs, numfeats = ndimage.label(signal[~mask])

        return int(numfeats)

    def dense_sample(
        self, line: LineString, raster: RasterioDatasetReader
    ) -> np.ndarray[float]:
        """Sample an underlying in_bin_raster along a given LineString at a frequency of ~1m"""

        # Densify points
        d_line = densify_line(line)

        # Extract coordinates
        d_coords = np.asarray(d_line.coords[:])

        # Apply inverse geotranseform (geo -> image coords)
        t_coords = transform_coords(d_coords, raster)

        # Sample (index) underlying in_bin_raster at each coord
        ## coords out of in_bin_raster bounds will be returned with np.nan
        return self.sample_array(t_coords, raster)

    def trans_fft(
        self, signal: np.ndarray[float]
    ) -> tuple[np.ndarray[float], np.ndarray[float]]:
        """
        Calculates the fast fourier transform for a 1D signal.

        If you wish to see the power spectra, plot the sampled frequencies (x) vs their measured amplitude (y)
        The dominant wavelength within a given signal can be found with the function `dominant_wavelength()` below
        """

        # Define Scroll Signature
        scroll_sig = np.array(signal)

        ## Set Variables
        # We want to sample the signal 1 time every meter
        # Therefore the interval between samples is equal to 1/1 meters
        sampling_freq = 1

        ## Fourier Transform
        # Normalize the amplitude to the number of samples
        # The second half of the fft array seems to be the mirror image of the first half. So we only need the first half
        amps = np.fft.fft(scroll_sig) / scroll_sig.size  # normalize amplitude
        amps = abs(amps[range(int(scroll_sig.size / 2))])  # exclude sampling

        ## Variables
        # make a new range of sampling points
        # define the "time period" (length) of the signal
        values = np.arange(int(scroll_sig.size / 2))
        time_period = scroll_sig.size / sampling_freq
        freqs = values / time_period

        return (freqs, amps)

    def dominant_wavelength(self, ridge_count: int, signal: np.ndarray) -> float:
        """
        Identifies the dominant wavelength from an input binary signal
        """

        # Test to see if signal has at least 2 ridges
        if ridge_count < 2:
            return float(np.nan)

        # Remove nans and zero values from signal
        real_signal = signal[~np.isnan(signal)] - 0.5

        # Calculate the fft
        ## To see the power spectra, plot freq x amps as a line
        freq, amp = self.trans_fft(real_signal)

        # Find the max value in amps that is not the first value
        # Use np.nonzero().min() here because some transects have the max amp in multiple locations
        max_amp_loc = np.nonzero(amp == amp[1:].max())[0].min()

        # Convert freq to wavelength
        dom_wav = round(1 / freq[max_amp_loc])

        return float(dom_wav)

    def calc_transect_metrics(self) -> gpd.GeoDataFrame:
        rich_transects = self.transects.copy()

        if self.dem is not None:
            rich_transects["dem_signal"] = rich_transects["geometry"].apply(
                lambda x: self.dense_sample(x, self.dem)
            )
            rich_transects["dem_signal"] = rich_transects["dem_signal"].apply(
                lambda x: np.where(x <= 0, np.nan, x)
            )

        if self.bin_raster is not None:
            rich_transects["bin_signal"] = rich_transects["geometry"].apply(
                lambda x: self.dense_sample(x, self.bin_raster)
            )

            rich_transects["ridge_count_raster"] = rich_transects["bin_signal"].apply(
                lambda x: self.count_ridges(x)  # pylint: disable=W0108
            )

            rich_transects["fft_spacing"] = rich_transects[
                ["ridge_count_raster", "bin_signal"]
            ].apply(lambda x: self.dominant_wavelength(*x), axis=1)

        return rich_transects.sort_index()

    def calc_itx_metrics(self) -> gpd.GeoDataFrame:
        """For each transect found in transects, calculate the itx metrics."""

        # Determine input columns
        input_columns = ["transect_id", "geometry"]

        if (
            self.ridges is not None
            and self.dem is not None
            and self.bin_raster is not None
        ):
            input_columns.append("dem_signal")
            input_columns.append("bin_signal")

        # Use TransectDataExtractor for every transect to create the itx dataframe
        tde_list = []
        pbar = tqdm(total=len(self.rich_transects), desc="Ridge Metrics", ascii=True)
        for _i, row in self.rich_transects[input_columns].iterrows():
            tde = TransectDataExtractor(**row, ridges=self.ridges).calc_ridge_metrics()
            tde_list.append(tde)
            pbar.update()
        pbar.close()

        itx = pd.concat(tde_list).set_index(["bend_id", "transect_id", "ridge_id"])

        return itx.sort_index()
