"""
This module contains the classes and functions used to extract data along the transects.

Data is extracted at three scales bend, transect, and ridge.
Packet-scale metrics are just the ridge scale metrics aggregated within the packet boundaries. No unique extraction is done for the packet-scale.
"""

from typing import List

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import ndimage
from shapely.geometry import Point, LineString

from .ridgeAmplitudes import calc_ridge_amps
from ..utils import calc_dist, transform_coords, densify_line


class SignalScrubber:
    """Responsible for cleaning a given binary signal"""
    def __init__(self, signal, th=3) -> None:
        self.signal = np.array(signal).astype(float)
        self.th = th
        self.scrubbed_signal = self.clean_signal()

        pass

    def remove_leading_ones(self, sig):
        '''
        This function replaces all 1s that preceed a 0 in the input signal with a NaN.

        Run this function across the signal in both directions like so
        `clean_sig = remove_ones(remove_ones(sig)[::-1])[::-1]`
        '''

        # Create a watch variable;
        make_nan = 1

        for i, v in enumerate(sig):
            if make_nan == 1:
                if v != 0:
                    # Define the new value
                    sig[i] = np.nan

                # Once a zero is encountered, just return the same value and turn off the watch variable
                if v == 0:
                    sig[i] = v
                    make_nan = 0
            # Once make_nan==0, then stop altering values
            else:
                pass
        return sig
    
    def flip_bin(self):
        '''
        flips binary values
        '''

        # Get locs
        one_loc = (self.signal==1)
        z_loc = (self.signal==0)

        # Flip array values
        self.signal[one_loc] = 0
        self.signal[z_loc] = 1

    
    def remove_small_feats(self):
        '''
        Removes features smaller than the threshold `th` in the given signal array.
        '''

        # Label all unique features
        labels, numfeats = ndimage.label(self.signal)

        # Find their counts (widths)
        val, count = np.unique(labels, return_counts=True)

        # Find all labels corresponding to small features
        small_ridge_vals = val[count <= self.th]

        # Redefine small features to 0s
        for i in small_ridge_vals:
            self.signal[labels==i]=0


    def clean_signal(self):
        '''
        Apply several functions to the raw transect signal to remove small or incomplete
        features from the signal.
        '''

        # Remove partial ridges
        self.signal = self.remove_leading_ones(self.signal)
        self.signal = self.remove_leading_ones(self.signal[::-1])[::-1]

        # Remove small ridges
        self.remove_small_feats()

        # Flip values and repeat to eliminate small swales
        self.flip_bin()
        self.remove_small_feats()

        # Flip values back
        self.flip_bin()

        return self.signal


class RidgeDataExtractor:
    """
    Responsible for calcualting ridge metrics at each intersection of a ridge and transect.
    The geometry for this class is a 3-vertex LineString
    """

    def __init__(self, geometry, ridges, dem_signal=None, bin_signal=None) -> None:
        # Inputs
        self.id = None
        self.geometry = geometry
        self.ridges = ridges
        self.dem_signal = dem_signal
        self.bin_signal = bin_signal
        self.signal_length = self.determine_signal_length()
        print("Started RDE")

        # Create GeoDataFrame
        self.data_columns = ["p_id", "ridge_id", "bend_id", 
                             "mig_dist", "mig_time", "mig_rate", "deposit_year", 
                             "ridge_width", "ridge_amp",
                             "geometry"]
        
        # Assess Geometry
        self.gdf = self.create_point_gdf()
        self.gdf = self.add_point_geometries(self.gdf, self.geometry)
        self.gdf = self.join_ridge_info(self.gdf, self.ridges)
        self.gdf = self.calc_values_from_ridge_info(self.gdf)
        self.gdf = self.calc_relative_vertex_distance(self.gdf, self.geometry)
        self.gdf = self.calc_vertex_indices(self.gdf, self.signal_length)
        
        # Process Binary Signal 
        self.ridge_com = self.calc_ridge_coms()
        self.single_ridge_num = None  # Set by self.find_closest_ridge()
        self.single_ridge_bin_signal = self.find_closest_ridge()

        # Ridge Metrics
        self.ridge_width_px = self.calc_ridge_width_px()
        self.ridge_amp_series = self.calc_every_ridge_amp()
        self.ridge_amp = self.determine_ridge_amp()

        self.gdf = self.coerce_dtypes(self.gdf)
        pass

    def determine_signal_length(self):
        """Return length of dem/bin signal if provided"""
        if isinstance(self.bin_signal, np.ndarray):
            return self.dem_signal.size
        else:
            return np.nan

    def create_point_gdf(self):
        """Create a 3 point GeoDataFrame to contain all relevant info for other methods."""
        gdf = gpd.GeoDataFrame(columns = self.data_columns,
                               geometry = "geometry",
                               crs = self.ridges.crs)
        gdf = gdf.set_index("p_id")
        return gdf

    def add_point_geometries(self, gdf, line):
        """Add the vertices from the 3vertex line as point geometries """
        # Add geometry info
        id_list = []
        p_list = []

        for i, p in enumerate(line.coords):
            id_list.append(f"p{i}" )
            p_list.append(Point(p))

        gdf = gdf.reset_index()
        gdf["p_id"] = id_list
        gdf["geometry"] = p_list
        gdf = gdf.set_index("p_id")

        return gdf

    def join_ridge_info(self, gdf, ridges):
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

    def calc_values_from_ridge_info(self, gdf):
        """
        Calculates the migration time, distance, and rate both before and after the center ridge.
        If the ridge does not have values for the deposit year, then mig_rate will be NaN.
        """

        # Calculate values from joined info
        gdf["mig_time"] = gdf["deposit_year"].diff().abs()
        gdf["mig_dist"] = gdf.distance(gdf.loc[["p0", "p0", "p1"]], align=False)
        gdf["mig_rate"] = gdf["mig_dist"] / gdf["mig_time"]

        return gdf
    
    def calc_relative_vertex_distance(self, gdf, line):
        """Calculate the relative distance of each vertex along the transect."""

        # coords = np.asarray(self.geometry.coords)
        # dists = np.insert(calc_dist(coords[:-1], coords[1:]), 0, 0)

        gdf["relative_distance"] = np.cumsum(gdf["mig_dist"]) / line.length

        return gdf
    
    def calc_vertex_indices(self, gdf, signal_length):
        """
        Calcualte the array index of all vertices.
        If `self.signal_length` is nan, then return array of nans
        """

        gdf["vertex_indices"] = (gdf["relative_distance"] * signal_length).round()

        # If any are nans, then you cannot cast to int
        if not gdf["vertex_indices"].isna().any():
             gdf["vertex_indices"] =  gdf["vertex_indices"].astype(int)

        return gdf

    def calc_ridge_coms(self):
        """Find the center of mass for each ridge in the input binary signal."""

        if self.bin_signal is None:
            return None

        # Create a copy to not modify the original
        sig = self.bin_signal.copy()
        sig[np.isnan(sig)] = 0

        # Find individual ridge areas
        labels, numfeats = ndimage.label(sig)

        # Find the centerpoint of each ridge along the transect
        coms = ndimage.center_of_mass(sig, labels, np.arange(numfeats)+1)
        coms = np.round(coms).astype(int).reshape(numfeats)

        # Create boolean array where True indicates com of a ridge
        coms_signal = np.zeros(sig.shape).astype(bool)
        coms_signal[coms] = True

        return coms_signal

    def find_closest_ridge(self):
        """
        The bin_signal may have more than two ridges present. 
        This method identifies which ridge is closest to the transect-ridge intersection point.
        """
        if self.bin_signal is None:
            return None

        # Get index of center vertex
        poi_idx = self.gdf.loc["p1", "vertex_indices"]

        # Find indices of ridge centers of mass
        bin = self.bin_signal
        ridge_midpoints = np.flatnonzero(self.ridge_com)
        
        # Find the closest ridge
        dist_from_poi = np.absolute(ridge_midpoints - poi_idx)
        closest_ridge_num = np.flatnonzero(dist_from_poi == dist_from_poi.min())[0]
        self.single_ridge_num = closest_ridge_num
        
        # Erase all ridges that are not closest
        label, num_feats = ndimage.label(bin==1)
        single_ridge = (label == closest_ridge_num+1).astype(float)
        single_ridge[np.isnan(bin)] = np.nan
        
        return single_ridge
    
    def calc_ridge_width_px(self)->int:
        """
        Calculate the width of the single ridge in pixels
        """
        if self.bin_signal is None:
            return None
        return np.nansum(self.single_ridge_bin_signal)
    
    def calc_every_ridge_amp(self)->int:
        """
        Calculates the average amplitude of each observed ridges in the units of the DEM.
        """
        if self.bin_signal is None:
            return None
        return calc_ridge_amps(self.dem_signal, self.bin_signal)

    def determine_ridge_amp(self):
        if self.bin_signal is None:
            return None
        return self.ridge_amp_series[self.single_ridge_num]
    
    def coerce_dtypes(self, gdf):
        """Coerce the the 'object' dtypes into their proper numeric types"""

        dtypes = {"p_id":str, "ridge_id":str, "bend_id":str, 
                    "mig_dist":float, "mig_time":float, "mig_rate":float, "deposit_year":float, 
                    "ridge_width":float, "ridge_amp":float}

        gdf = gdf.reset_index().astype(dtypes)
        gdf = gdf.set_index("p_id")
        return gdf
    
    def dump_data(self):
        """Dump all the relevant info for the middle point."""
        d = {}

        d["ridge_id"] = self.gdf.loc["p1", "ridge_id"]
        d["pre_mig_dist"] = self.gdf.loc["p2", "mig_dist"]
        d["post_mig_dist"] = self.gdf.loc["p1", "mig_dist"]
        d["pre_mig_time"] = self.gdf.loc["p2", "mig_time"]
        d["post_mig_time"] = self.gdf.loc["p1", "mig_time"]
        d["pre_mig_rate"] = self.gdf.loc["p2", "mig_rate"]
        d["post_mig_rate"] = self.gdf.loc["p1", "mig_rate"]
        d["deposit_year"] = self.gdf.loc["p1", "deposit_year"]
        d["bend_id"] = self.gdf.loc["p1", "bend_id"]
        d["geometry"] = self.gdf.loc["p1", "geometry"]

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

    def __init__(self, transect_id, geometry, dem_signal=None, bin_signal=None, ridges=None) -> None:

        # Inputs
        self.transect_id = transect_id
        self.geometry = geometry
        self.raw_dem_signal = dem_signal
        self.raw_bin_signal = bin_signal
        self.ridges = ridges
        print(f"Started TDE for {self.transect_id}")

        # Create GeoDataFrame
        self.data_columns = ["ridge_id", "transect_id", "bend_id",
                             "relative_vertex_distances", "vertex_indices",
                             "dem_signal", "bin_signal", 
                             "deposit_year",
                             "pre_mig_dist", "post_mig_dist", 
                             "pre_mig_time", "post_mig_time", 
                             "pre_mig_rate", "post_mig_rate", 
                             "ridge_width", "ridge_amp", 
                             "substring_geometry", "geometry"]
        
        # Add Geometries
        self.itx_gdf = self.create_itx_gdf()
        self.itx_gdf = self.add_substring_geometry(self.itx_gdf)
        self.itx_gdf = self.add_point_geometry(self.itx_gdf)
        
        # Add transect_id
        self.itx_gdf = self.add_transect_id(self.itx_gdf)


        # Process binary and DEM signals
        self.clean_bin_signal = self.scrub_bin_signal()
        self.itx_gdf = self.add_relative_vertex_distances(self.itx_gdf)    
        self.itx_gdf = self.calc_vertex_indices(self.itx_gdf)
        self.itx_gdf = self.slice_bin_signal(self.itx_gdf)
        self.itx_gdf = self.slice_dem_signal(self.itx_gdf)
        

        
    def create_itx_gdf(self):
        """Create the gdf that will contain all the ridge data for each intersection."""
        gdf = gpd.GeoDataFrame(columns=self.data_columns, geometry="geometry").set_crs(self.ridges.crs)

        return gdf
    
    def create_substrings(self, ls:LineString)-> List[LineString]:
        """
        Break up a LineString into many overlaping 'substrings' 
        Each substring constructed from three consecutive vertices of the input LineString.
        """
        
        # Create a list of lists where each sublist corresponds to a vertex position
        # eg. verts = [[back_verts], [center_verts], [forward_verts]]
        verts = [ls.coords[i:len(ls.coords)-(3-(i+1))] for i in range(3)]
        
        # Return a list of LineStrings
        return list(map(LineString, zip(*verts)))
    
    def add_substring_geometry(self, gdf):
        """Adds the 3 vertex substring that corresponds to each itx."""

        gdf["substring_geometry"] = self.create_substrings(self.geometry)

        return gdf

    def add_point_geometry(self, gdf):
        """Add the intersection (middle) point of the 3 vertex substring as its own point"""

        gdf["geometry"] = gdf["substring_geometry"].apply(lambda x: Point(x.coords[1]))

        return gdf
    
    def add_transect_id(self, gdf):
        """Add the transect id as a column"""

        gdf["transect_id"] = self.transect_id

        return gdf
    
    def scrub_bin_signal(self):
        """
        Clean errant noise extracted along the binary signal. 
        Examples of noise removed:
            - positive areas smaller than a given threshold
            - incomplete ridges (signal starts or ends with ones)
        """
        if self.raw_bin_signal is not None:
            return SignalScrubber(self.raw_bin_signal).scrubbed_signal

    def calc_relative_vertex_distances(self, ls):
        """Calculate the relative distance of each vertex along the transect."""

        coords = np.asarray(ls.coords)
        dists = np.insert(calc_dist(coords[:-1], coords[1:]), 0, 0)
        return np.cumsum(dists) / ls.length
    
    def add_relative_vertex_distances(self, gdf):
        """Calculate the distance between the substring coordinates relative to the length of the whole line."""

        gdf["relative_vertex_distances"] = gdf["substring_geometry"].apply(self.calc_relative_vertex_distances)
        return gdf
    
    def calc_vertex_indices(self, gdf):
        """Calculates the corresponding signal index of each of the substring vertices"""

        if self.raw_bin_signal is not None:
            gdf["vertex_indices"] = gdf["relative_vertex_distances"].apply(lambda x: np.round(x * self.raw_bin_signal.size).astype(int))
        return gdf

    def slice_dem_signal(self, gdf):
        """Slice the DEM between the two end vertices of the substrings"""
        if self.raw_bin_signal is not None:
            gdf["dem_signal"] = gdf["vertex_indices"].apply(lambda x: self.raw_dem_signal[x[0]:x[2]])
        
        return gdf
    
    def slice_bin_signal(self, gdf):
        """Slice the binary signal between the two end vertices of the substrings"""
        if self.raw_bin_signal is not None:
            gdf["bin_signal"] = gdf["vertex_indices"].apply(lambda x: self.clean_bin_signal[x[0]:x[2]])

        return gdf
       
    def calc_ridge_metrics(self):
        """
        Calculate ridge width and amplitude at every transect-ridge intersection.
        Return a GeoDataFrame with Point geometries.
        """

        for i, row in self.itx_gdf.iterrows():
            row[row.isna()] = None
            rde = RidgeDataExtractor(row["substring_geometry"], self.ridges, row["dem_signal"], row["bin_signal"])

            ridge_metrics = rde.dump_data()

            self.itx_gdf.loc[i, list(ridge_metrics.keys())] = ridge_metrics

        self.itx_gdf = self.coerce_dtypes(self.itx_gdf)

        return self.itx_gdf

    def coerce_dtypes(self, gdf):
        """Coerce the the 'object' dtypes into their proper numeric types"""

        dtypes = {"ridge_id":str,"transect_id":str,"bend_id":str,
                    "pre_mig_dist":float,"post_mig_dist":float,
                    "pre_mig_time":float,"post_mig_time":float,
                    "pre_mig_rate":float,"post_mig_rate":float,
                    "ridge_width":float,"ridge_amp":float,
                    "deposit_year":float}
        
        gdf = gdf.astype(dtypes)
        return gdf

class BendDataExtractor:
    """Responsible for extraction of ridge metrics across an entire bend."""
    def __init__(self, transects, bin_raster=None, dem=None, ridges=None, packets=None) -> None:
        self.transects = transects
        self.bin_raster = bin_raster
        self.dem = dem
        self.ridges = ridges
        self.packets = packets
        print("Started BDE")

        # Calculate Metrics at the smaller scales
        self.rich_transects = self.calc_transect_metrics()
        self.itx_metrics = self.calc_itx_metrics()

    def disqualify_coords(self, coord_array, raster):
        """ 
        Some coordinates may be out of the in_bin_raster. 
        This function disqualifies these coordinates and returns a boolean arry showing the location of all disqualified coordinates.
        
        Coordinates are checked to see if they are 1) negative, 2) too large in x, or 3) too large in y
        """
        # Find location of all negative coords
        too_small = np.any(coord_array < 0, axis=1)

        # Find location of too large x
        x_max = raster.profile["width"]
        too_large_x = coord_array[:,0] >= x_max

        # Find location of too large y
        y_max = raster.profile["height"]
        too_large_y = coord_array[:,1] >= y_max

        # return true for a coord if that coord failed any test 
        return np.any(np.vstack((too_small, too_large_x, too_large_y)), axis=0)


    def sample_array(self, coord_array, raster):
        """
        Takes in an array of image coordinats, samples the image, and returns the sampled values.
        Assumes that the coord array and in_bin_raster dataset share the same crs.
        """
        # Prep the coords 
        ## Some coordinates may be out of bounds for the in_bin_raster
        ## So we need to only sample with valid image coords, then pad the array with nans for all out of bounds areas
        disq = self.disqualify_coords(coord_array, raster) # boolean array
        in_bounds = coord_array[~disq]
        
        # Sample the array with valid coords
        arr = raster.read(1)
        signal = arr[in_bounds[:,1], in_bounds[:,0]].flatten() # remember image coords are (y,x)
        
        # Pad either side of the signal with nans for disqualified points
        out_signal = np.zeros(disq.shape)*np.nan
        out_signal[~disq] = signal
        
        return out_signal

    def count_ridges(self, signal):
        """Counts ridges in binary waves."""

        mask = np.isnan(signal)
        labs, numfeats = ndimage.label(signal[~mask])

        return numfeats

    def dense_sample(self, line, raster):
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

    def trans_fft(self, signal):
        '''
        Calculates the fast fourier transform for a 1D signal.

        If you wish to see the power spectra, plot the sampled frequencies (x) vs their measured amplitude (y)
        The dominant wavelength within a given signal can be found with the function `dominant_wavelength()` below
        '''

        # Define Scroll Signature
        scroll_sig = np.array(signal)

        ## Set Variables
        # We want to sample the signal 1 time every meter
        # Therefore the interval between samples is equal to 1/1 meters
        sampling_freq = 1
        sampling_intv = 1/sampling_freq


        ## Fourier Transform
        # Some reason we have to normalize the amplitude to the number of samples
        # The second half of the fft array seems to be the mirror image of the first half. So we only need the first half
        amps = np.fft.fft(scroll_sig) / scroll_sig.size   # normalize amplitude
        amps = abs(amps[range(int(scroll_sig.size/2))])  # exclude sampling


        ## Variables
        # make a new range of sampling points; 0 - 499
        # define the "time period" (length) of the signal; 500m
        # All available frequencies the fft can identfy - or maybe just x values

        values = np.arange(int(scroll_sig.size/2))
        time_period  = scroll_sig.size/sampling_freq
        freqs = values/time_period

        return (freqs, amps)
    
    def dominant_wavelength(self, ridge_count, signal):
        '''
        Identifies the dominant wavelength from an input binary signal
        '''

        # Test to see if signal has at least 2 ridges
        if ridge_count < 2:
            return np.nan

        # Remove nans and zero values from signal
        real_signal = signal[~np.isnan(signal)] - 0.5

        # Calcualte the fft
        ## To see the power spectra, plot freq x amps as a line
        freq, amp = self.trans_fft(real_signal)

        # Find the max value in amps that is not the first value
        # Use np.nonzero().min() here because some transects have the max amp in multiple locations
        max_amp_loc = np.nonzero(amp == amp[1:].max())[0].min()

        # Convert freq to wavelength
        dom_wav = round(1/freq[max_amp_loc])

        return dom_wav

    def create_amp_signal(self, bin_signal, dem_signal):
        """Create a transect signal where the positive areas in clean_bin_sig are replaced with amplitude."""

        # Boolify bin_signal
        bin_sig = bin_signal.copy()
        bin_sig[np.isnan(bin_sig)] = 0

        # Create labels for each ridge area
        labels, numfeats = ndimage.label(bin_sig)
        float_labels = labels.astype(float)  # cast to float, otherwise precision is not stored when redefining

        # Calculate amplitude for each positive area in bin_signal
        ridge_amps = calc_ridge_amps(dem_signal, bin_sig)

        # Redefine the positive areas in bin_sig to amps
        for i, amp in enumerate(ridge_amps):
            float_labels[labels==i+1] = amp

        return float_labels
    

    def calc_transect_metrics(self):

        rich_transects = self.transects.copy()

        if self.dem is not None:
            rich_transects["dem_signal"] = rich_transects["geometry"].apply(lambda x: self.dense_sample(x, self.dem))
            rich_transects["dem_signal"] = rich_transects["dem_signal"].apply(lambda x: np.where(x<=0, np.nan, x))

        if self.bin_raster is not None:
            rich_transects["bin_signal"] = rich_transects["geometry"].apply(lambda x: self.dense_sample(x, self.bin_raster))
            rich_transects["clean_bin_signal"] = rich_transects["bin_signal"].apply(lambda x: SignalScrubber(x).scrubbed_signal)
            rich_transects["ridge_count_raster"] = rich_transects["clean_bin_signal"].apply(lambda x: self.count_ridges(x))
            rich_transects["fft_spacing"] = rich_transects[["ridge_count_raster", "clean_bin_signal"]].apply(lambda x: self.dominant_wavelength(*x), axis=1)

        if self.dem is not None and self.bin_raster is not None:
            rich_transects["amp_signal"] = rich_transects[["clean_bin_signal", "dem_signal"]].apply(lambda x: self.create_amp_signal(*x), axis=1)
            rich_transects["fft_amps"] = rich_transects[["ridge_count_raster", "amp_signal"]].apply(lambda x: self.dominant_wavelength(*x), axis=1)
                
        return rich_transects.sort_index()
    
    def calc_itx_metrics(self):
        """For each transect found in transects, calculate the itx metrics."""

        if self.ridges is not None and self.dem is not None and self.bin_raster is not None:

            tde_list = []
            for i, row in self.rich_transects[["transect_id", "geometry", "dem_signal", "clean_bin_signal"]].iterrows():
                tde = TransectDataExtractor(*row, ridges = self.ridges).calc_ridge_metrics()
                tde_list.append(tde)

            itx = pd.concat(tde_list).set_index(["bend_id", "transect_id", "ridge_id"])    


        else:
            tde_list = []
            for i, row in self.rich_transects[["transect_id", "geometry"]].iterrows():
                tde = TransectDataExtractor(*row, ridges = self.ridges).calc_ridge_metrics()
                tde_list.append(tde)

            itx = pd.concat(tde_list).set_index(["bend_id", "transect_id", "ridge_id"])

        return itx.sort_index()
    