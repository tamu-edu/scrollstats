# Testing for ScrollStats
from __future__ import annotations

import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.io import DatasetReader
from shapely.geometry import LineString, Point

from scrollstats import (
    BendDataExtractor,
    LineSmoother,
    RidgeDataExtractor,
    TransectDataExtractor,
    create_transects,
)
from scrollstats.delineation import (
    create_ridge_area_raster,
    create_ridge_area_raster_fs,
)
from scrollstats.delineation.raster_classifiers import (
    quadratic_profile_curvature,
    residual_topography,
)
from scrollstats.delineation.ridge_area_raster import clip_raster

# Data Paths
DEM_PATH = Path("example_data/input/LBR_025_dem.tif")
BEND_PATH = Path("example_data/input/LBR_025_bend.geojson")
PACKET_PATH = Path("example_data/input/LBR_025_packets.geojson")
CENTERLINE_PATH = Path("example_data/input/LBR_025_cl.geojson")
MANUAL_RIDGE_PATH = Path("example_data/input/LBR_025_ridges_manual.geojson")


class MockRidgeData:
    def __init__(
        self,
        wavelength: int = 30,
        amp: int = 1,
        vert_adj: int = 5,
        reps: int = 5,
        crs: str = "EPSG:32139",
    ) -> None:
        self.wavelength = wavelength
        self.amp = amp
        self.vert_adj = vert_adj
        self.reps = reps
        self.crs = crs

    def generate_waves(self) -> np.ndarray:
        """
        Generate a 2D cosine wave with known properties for testing
        """

        p = 2 * np.pi / self.wavelength
        signal_length = self.wavelength * self.reps
        x = np.arange(signal_length)

        dem_1d = self.amp * np.cos(p * x) + self.vert_adj

        dem_2d = np.multiply(np.ones((signal_length, signal_length)), dem_1d)

        return dem_2d

    def generate_ridges(self) -> gpd.GeoDataFrame:
        """Generate mock ridge lines with known spacing for testing"""

        r = {
            "ridge_id": [f"r_{i:03d}" for i in range(self.reps - 1)],
            "bend_id": ["LBR_999" for i in range(self.reps - 1)],
            "deposit_year": [np.nan for i in range(self.reps - 1)],
            "geometry": [
                LineString([[i, 0], [i, self.wavelength * self.reps]])
                for i in range(
                    self.wavelength, self.wavelength * self.reps, self.wavelength
                )
            ],
        }
        return gpd.GeoDataFrame(data=r, geometry="geometry", crs=self.crs)

    def generate_transects(self) -> gpd.GeoDataFrame:
        """Generate mock transects with known spacing for testing"""

        t = {
            "transect_id": [f"t_{i:03d}" for i in range(self.reps - 1)],
            "bend_id": ["LBR_999" for i in range(self.reps - 1)],
            "geometry": [
                LineString(
                    [
                        [j, i]
                        for j in range(0, self.wavelength * self.reps, self.wavelength)
                    ]
                )
                for i in range(
                    self.wavelength, self.wavelength * self.reps, self.wavelength
                )
            ],
        }
        return gpd.GeoDataFrame(data=t, geometry="geometry", crs=self.crs)

    def generate_bend_area(self) -> gpd.GeoDataFrame:
        """
        Generate a mock bend area polygon of a known area in the middle of the wave array.
        Return the bend area polygon as a GeoDataFrame
        """

        # Find the midpoint of the array
        dem_shape = np.ones(2) * (self.wavelength * self.reps)
        point = Point(dem_shape // 2)

        # Buffer out this point to create a circle slightly smaller than the wave array
        circle = point.buffer((self.wavelength / 2) * (self.reps - 1))

        b = {"bend_id": ["LBR_999"], "geometry": [circle]}
        return gpd.GeoDataFrame(data=b, geometry="geometry", crs=self.crs)

    def generate_raster(self, array: np.ndarray, no_data=None) -> DatasetReader:
        """Create a rasterio.DatasetReader object from a 2D array"""

        # Write DEM to disk
        with tempfile.NamedTemporaryFile(suffix=".tif") as fp:
            with rasterio.open(
                fp.name,
                "w",
                driver="GTiff",
                width=array.shape[1],
                height=array.shape[0],
                count=1,
                dtype=array.dtype,
                crs=self.crs,
                nodata=no_data,
            ) as dst:
                dst.write(array, 1)

            out_ds = rasterio.open(fp.name)

        return out_ds


def test_line_smoother_density() -> None:
    """Ensure that LineSmoother generates LineStrings with a sufficient point density"""

    manual_ridges = gpd.read_file(MANUAL_RIDGE_PATH)

    spacing = 1
    window = 5
    ls = LineSmoother(manual_ridges, spacing=spacing, window=window)
    smooth_ridges = ls.execute()

    tolerance = 0.01
    deviance = smooth_ridges.geometry.apply(
        lambda x: abs((len(x.coords) / x.length) - spacing)
    )
    assert all(deviance < tolerance)


def test_quadratic_profile_curvature() -> None:
    """Check that the profile curvature transformation accurately identifies ridge areas"""
    # Generate 2D cosine waves with known properties
    data = MockRidgeData()
    dem = data.generate_waves()

    # Apply profile curvature to assess landscape convexity
    profc = quadratic_profile_curvature(dem, window=int(data.wavelength / 2))

    # Select areas in the generated waves known to be ridges
    known_ridge_area = dem > data.vert_adj
    known_swale_area = dem < data.vert_adj

    # Assess if the known ridge/swale areas are +/-
    accurate_ridges = profc[known_ridge_area] > 0
    accurate_swales = profc[known_swale_area] < 0

    assert np.sum(accurate_ridges) > 0.9 * accurate_ridges.size
    assert np.sum(accurate_swales) > 0.9 * accurate_swales.size


def test_residual_topography() -> None:
    """Check that the residual topography transformation accurately identifies ridge areas"""
    # Generate 2D cosine waves with known properties
    data = MockRidgeData()
    dem = data.generate_waves()

    # Apply residual topography to assess landscape prominence
    rt = residual_topography(dem, w=int(data.wavelength / 2))
    rt_not_nan = ~np.isnan(rt)

    # Select areas in the generated waves known to be ridges
    known_ridge_area = dem > data.vert_adj
    known_swale_area = dem < data.vert_adj

    # Assess if the known ridge/swale areas are +/- (excluding nan areas)
    accurate_ridges = rt[known_ridge_area & rt_not_nan] > 0
    accurate_swales = rt[known_swale_area & rt_not_nan] < 0

    assert np.sum(accurate_ridges) > 0.9 * accurate_ridges.size
    assert np.sum(accurate_swales) > 0.9 * accurate_swales.size


def test_clip_raster() -> None:
    """Test if the array window shrunk as a result of the clip and if all values outside to the geometry are cast to np.nan"""
    dem_ds = rasterio.open(DEM_PATH)
    gdf = gpd.read_file(BEND_PATH)
    geom = gdf.loc[0, "geometry"]

    array_clip, clipped_mask, clipped_meta = clip_raster(dem_ds, geom, no_data=np.nan)

    assert clipped_meta["width"] * clipped_meta["height"] < dem_ds.width * dem_ds.height
    assert np.all(np.isnan(array_clip[clipped_mask]))


def test_create_ridge_area_raster() -> None:
    """
    Test the following aspects of the create_ridge_area_raster function:
        1. no data is set correctly
        2. the output binary raster accurately identifies ridge areas
        3. the output dem raster window has shrunk as a result of the clip and if all values outside to the geometry are cast to the no data value
    """

    # Mock Data
    data = MockRidgeData()
    dem = data.generate_waves()
    no_data_value = np.nan
    dem_ds = data.generate_raster(dem, no_data=no_data_value)
    circle = data.generate_bend_area().loc[0, "geometry"]  # get Polygon from gdf

    # Identify ridge areas within the mocked dem
    binary_clip, dem_clip, _binary_meta = create_ridge_area_raster(
        dem_ds=dem_ds,
        geometry=circle,
        no_data_value=no_data_value,
        window=int(data.wavelength / 2),
        dx=1,
        small_feats_size=1,
    )

    known_ridge_areas = dem_clip > data.vert_adj
    no_data_area = (binary_clip != 0) & (binary_clip != 1)

    # Do all classified ridge pixels correspond to known ridge pixels?
    assert known_ridge_areas[binary_clip == 1].all()

    # Were nearly all of the known ridge pixels classified as ridge pixels?
    assert ((binary_clip == 1).sum() / known_ridge_areas.sum()) > 0.9

    # Was the no data value set correctly?
    if np.isnan(no_data_value):
        assert np.isnan(binary_clip[no_data_area]).all()
    else:
        assert (binary_clip[no_data_area] == no_data_value).all()


def test_create_ridge_area_raster_fs() -> None:
    """
    Test that the file system interface for the create_ridge_area_raster function can matches the output from create_ridge_area_raster
    """
    # Generate a bend dataset known properties
    data = MockRidgeData()
    dem = data.generate_waves()
    bend_area = data.generate_bend_area()
    no_data_value = np.nan
    dem_ds = data.generate_raster(dem, no_data=no_data_value)

    # Create ridge area raster and clipped dem with file system interface
    with tempfile.NamedTemporaryFile(suffix=".tif") as dem_path:
        with rasterio.open(
            dem_path.name,
            "w",
            driver="GTiff",
            width=dem.shape[1],
            height=dem.shape[0],
            count=1,
            dtype=dem.dtype,
            crs=data.crs,
            nodata=no_data_value,
        ) as dst:
            dst.write(dem, 1)

        with tempfile.NamedTemporaryFile(suffix=".geojson") as bend_path:
            bend_area.to_file(bend_path.name, driver="GeoJSON", index=False)

            out_dir = tempfile.gettempdir()
            binary_out_path, dem_out_path = create_ridge_area_raster_fs(
                dem_path=Path(dem_path.name),
                geometry_path=Path(bend_path.name),
                out_dir=Path(out_dir),
                no_data_value=no_data_value,
                window=int(data.wavelength / 2),
                dx=1,
                small_feats_size=1,
            )

    binary_from_disk = rasterio.open(binary_out_path).read(1)
    dem_from_disk = rasterio.open(dem_out_path).read(1)

    # Create ridge area raster and clipped dem directly from objects in memory
    binary_from_mem, dem_from_mem, _binary_meta = create_ridge_area_raster(
        dem_ds=dem_ds,
        geometry=bend_area.loc[0, "geometry"],
        no_data_value=no_data_value,
        window=int(data.wavelength / 2),
        dx=1,
        small_feats_size=1,
    )

    assert np.array_equal(binary_from_disk, binary_from_mem, equal_nan=True)
    assert np.array_equal(dem_from_disk, dem_from_mem, equal_nan=True)


def test_create_transects() -> None:
    """Test that the transects intersect all of the ridges"""

    centerline = gpd.read_file(CENTERLINE_PATH)
    ridges_manual = gpd.read_file(MANUAL_RIDGE_PATH)

    ls = LineSmoother(ridges_manual, 1, window=5)
    ridges_smooth = ls.execute()

    transects = create_transects(centerline, ridges_smooth, 100, 300, 200, 5)

    ridge_ids = ridges_smooth["ridge_id"]
    intersections = transects.overlay(ridges_smooth, keep_geom_type=False)
    ridge_ids_from_itx = intersections["ridge_id"].unique()

    assert [i for i in ridge_ids if i not in ridge_ids_from_itx] == []


def test_ridge_data_extractor() -> None:
    """Test that the RidgeDataExtractor calculates expected ridge width, amplitude, and spacing"""

    # Generate a 1D cosine wave with known properties
    data = MockRidgeData()
    dem = data.generate_waves()[0]
    transects = data.generate_transects()
    ridges = data.generate_ridges()

    # RidgeDataExtractor calculates ridge metrics at the intersection of a single transect_substring and the ridge it intersects
    ## Take the first 3 vertices of the first transect to form the transect_substring
    transect_substring = LineString(transects.loc[0, "geometry"].coords[:3])
    position = 1

    ## Take the first ridge as the intersecting ridge
    ridge = ridges.loc[[0]]

    # Create a binary wave from the DEM where swales are 0 and ridges are 1
    bw = dem.copy()
    bw[bw <= data.vert_adj] = 0
    bw[bw > data.vert_adj] = 1

    # Calculate the three core metrics (ridge amplitude, width, and spacing) at the transect-ridge intersection
    rde_data = RidgeDataExtractor(
        transect_substring, position, ridge, dem, bw
    ).dump_data()

    assert rde_data["ridge_amp"] == data.amp * 2
    assert rde_data["ridge_width"] == data.wavelength / 2
    assert rde_data["pre_mig_dist"] == data.wavelength


def test_transect_data_extractor() -> None:
    """
    Test that the TransectDataExtractor calculates expected ridge width, amplitude, and spacing for an entire transect

    Mock DEM is a simple cosine wave with a given wavelength.
    Mock ridges and transects are perpendicular to each other on a grid with a regular spacing equal to the mock DEM's wavelength.
    """

    data = MockRidgeData()
    dem = data.generate_waves()
    transects = data.generate_transects()
    ridges = data.generate_ridges()

    # Calculate ridge metrics for the first transect
    tde = TransectDataExtractor(
        **transects.loc[0, ["transect_id", "geometry"]],
        dem_signal=dem[0],
        bin_signal=(dem[0] > data.vert_adj).astype(int),
        ridges=ridges,
    )

    transect_metrics = tde.calc_ridge_metrics()

    assert all(transect_metrics["ridge_amp"] == data.amp * 2)
    assert all(transect_metrics["ridge_width"] == data.wavelength / 2)
    assert all(transect_metrics["pre_mig_dist"] == data.wavelength)


def test_bend_data_extractor() -> None:
    """
    Test that the BendDataExtractor calculates expected ridge width, amplitude, and spacing for an entire bend.

    Mock DEM is a simple cosine wave with a given wavelength.
    Mock ridges and transects are perpendicular to each other on a grid with a regular spacing equal to the mock DEM's wavelength.
    Mock Binary Raster is a binary classification of DEM where 1s are values greater than a threshold and 0s are less than.
    Both DEM and Binary Raster are converted into temporary rasterio Dataset objects as required by BendDataExtractor
    """

    # Generate mock data
    data = MockRidgeData()
    dem = data.generate_waves()
    transects = data.generate_transects()
    ridges = data.generate_ridges()

    bin_arr = (dem > data.vert_adj).astype(int)

    # Create rasterio.DatasetReader objects from arrays
    dem_ras = data.generate_raster(dem, no_data=np.nan)
    bin_ras = data.generate_raster(bin_arr)

    # Create BendDataExtractor to calculate metrics
    bde = BendDataExtractor(transects, bin_ras, dem_ras, ridges)

    assert all(bde.itx_metrics["ridge_amp"] == data.amp * 2)
    assert all(bde.itx_metrics["ridge_width"] == data.wavelength / 2)
    assert all(bde.itx_metrics["pre_mig_dist"] == data.wavelength)
