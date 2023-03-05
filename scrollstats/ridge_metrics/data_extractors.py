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

    def __init__(self, geometry, dem_signal, bin_signal) -> None:
        # Inputs
        self.id = None
        self.geometry = geometry
        self.dem_signal = dem_signal
        self.bin_signal = bin_signal


        # Assess Geometry
        self.itx_point = Point(self.geometry.coords[1])
        self.itx_idx = None  # Set by `self.find_closest_ridge()`
        self.relative_vertex_distances = self.calc_relative_vertex_distance()
        self.vertex_indices = np.round(self.relative_vertex_distances * self.bin_signal.size).astype(int)

        # Process Binary Signal 
        self.ridge_com = self.calc_ridge_coms()
        self.single_ridge_num = None  # Set by self.find_closest_ridge()
        self.single_ridge_bin_signal = self.find_closest_ridge()

        # Ridge Metrics
        self.ridge_width_px = self.calc_ridge_width_px()
        self.ridge_amp_series = self.calc_every_ridge_amp()
        self.ridge_amp = self.ridge_amp_series[self.single_ridge_num]
        self.ridge_migration = self.calc_migration()
        
        pass

    def calc_relative_vertex_distance(self):
        """Calculate the relative distance of each vertex along the transect."""

        coords = np.asarray(self.geometry.coords)
        dists = np.insert(calc_dist(coords[:-1], coords[1:]), 0, 0)

        return np.cumsum(dists) / self.geometry.length
    

    def calc_ridge_coms(self):
        """Find the center of mass for each ridge in the input binary signal."""

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
        """The bin_signal may have more than two ridges present. 
        This method identifies which ridge is closest to the transect-ridge intersection point. """
    
        # Find relative distance of the center vertex
        poi_idx = self.vertex_indices[1]
        self.itx_idx = poi_idx

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
        """Calculate the width of the single ridge in pixels"""
        return np.nansum(self.single_ridge_bin_signal)
    
    def calc_every_ridge_amp(self)->int:
        """
        Calculates the average amplitude of each observed ridges in the units of the DEM.
        """
        return calc_ridge_amps(self.dem_signal, self.bin_signal)
    
    def calc_migration(self) -> float:
        """Calculates the distance between the current ridge and the ridge deposited before it."""

        _p1, p2, p3 = (Point(i) for i in self.geometry.coords)
        return p2.distance(p3)

class TransectDataExtractor:
    """Responsible for extracting ridge metrics along a transect"""
    def __init__(self, transect_id, geometry, dem_signal, bin_signal, crs, ridges) -> None:

        # Inputs
        self.transect_id = transect_id
        self.geometry = geometry
        self.raw_dem_signal = dem_signal
        self.raw_bin_signal = bin_signal
        self.crs = crs
        self.ridges = ridges

        # Assess binary signal
        self.clean_bin_signal = SignalScrubber(self.raw_bin_signal).scrubbed_signal
        self.nan_mask = np.isnan(self.clean_bin_signal)
        self.has_observations = not(all(self.nan_mask))
 
        # Assess geometry and its relative position along the 1D signal 
        self.relative_vertex_distances = self.calc_relative_vertex_distance()
        self.vertex_indices = np.round(self.relative_vertex_distances * self.raw_bin_signal.size).astype(int)
        self.substrings = self.create_substrings(self.geometry, 3)
        self.substring_indices = self.get_substring_indices(3)
        pass

    def calc_relative_vertex_distance(self):
        """Calculate the relative distance of each vertex along the transect."""

        coords = np.asarray(self.geometry.coords)
        dists = np.insert(calc_dist(coords[:-1], coords[1:]), 0, 0)

        return np.cumsum(dists) / self.geometry.length
    

    def create_substrings(self, ls:LineString, n:int)-> List[LineString]:
        """
        Break up a linestring into many overlaping linestrings constructed from the vertices of input LineString.
        Length of the resulting linestrings (in vertices) is determined by `n`
        """
        
        # Create a list of lists where each sublist corresponds to a vertex position
        # eg. for n=3, verts = [[back_verts], [center_verts], [forward_verts]]
        verts = [ls.coords[i:len(ls.coords)-(n-(i+1))] for i in range(n)]
        
        # Return a list of LineStrings
        return list(map(LineString, zip(*verts)))


    def get_substring_indices(self, n:int):
        """Get the array indices that correspond to the start and end vertices of each transect."""
        starts = self.vertex_indices[:-(n-1)]
        ends = self.vertex_indices[n-1:]

        return list(zip(starts, ends))

    def generate_rde(self, i:int):
        """Generate a RidgeDataExtractor at a given vertex index"""
            
        geom = self.substrings[i]
        idx = self.substring_indices[i]
        dem = self.raw_dem_signal[idx[0]:idx[1]]
        bin = self.clean_bin_signal[idx[0]:idx[1]]
        rde = RidgeDataExtractor(geom, dem, bin)

        return rde
    
    def generate_rde_list(self):
        """Generate a list of RidgeDataExtractors for each available vertex in the transect"""
        return [self.generate_rde(i) for i, sub in enumerate(self.substrings)]

    def calc_ridge_metrics(self):
        """
        Calculate ridge width and amplitude at every transect-ridge intersection.
        Return a GeoDataFrame with Point geometries.
        """
        gdf_list = []

        for rde in self.generate_rde_list():
            t_id = self.transect_id
            width = rde.ridge_width_px
            amp = rde.ridge_amp
            mig = rde.ridge_migration
            point = rde.itx_point

            gdf_list.append((t_id, width, amp, mig, point))

        itx_columns = ["transect_id","width", "amplitude", "migration","geometry"]
        ridge_metrics = gpd.GeoDataFrame(columns=itx_columns, data=gdf_list, geometry="geometry", crs=self.crs)

        # Apply buffer for spatial join
        ridge_metrics.geometry = ridge_metrics.buffer(1e-5)

        ridge_columns = ["ridge_id", "bend_id"]
        ridge_metrics = ridge_metrics.sjoin(self.ridges, how="left")[itx_columns + ridge_columns]
        ridge_metrics.geometry = ridge_metrics.centroid

        # Cast dtypes
        dtypes = {"width":float, "amplitude":float, "migration":float}
        ridge_metrics = ridge_metrics.astype(dtypes)

        return ridge_metrics

class BendDataExtractor:
    """Responsible for extraction of ridge metrics across an entire bend """
    def __init__(self, transects, bin_raster, dem, ridges, packets=None) -> None:
        self.transects = transects
        self.bin_raster = bin_raster
        self.dem = dem
        self.ridges = ridges
        self.packets = packets

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

        rich_transects["dem_signal"] =rich_transects["geometry"].apply(lambda x: self.dense_sample(x, self.dem))
        rich_transects["dem_signal"] =rich_transects["dem_signal"].apply(lambda x: np.where(x<=0, np.nan, x))
        rich_transects["bin_signal"] =rich_transects["geometry"].apply(lambda x: self.dense_sample(x, self.bin_raster))
        rich_transects["clean_bin_signal"] =rich_transects["bin_signal"].apply(lambda x: SignalScrubber(x).scrubbed_signal)
        rich_transects["ridge_count_raster"] =rich_transects["clean_bin_signal"].apply(lambda x: self.count_ridges(x))
        rich_transects["fft_spacing"] =rich_transects[["ridge_count_raster", "clean_bin_signal"]].apply(lambda x: self.dominant_wavelength(*x), axis=1)
        rich_transects["amp_signal"] =rich_transects[["clean_bin_signal", "dem_signal"]].apply(lambda x: self.create_amp_signal(*x), axis=1)
        rich_transects["fft_amps"] =rich_transects[["ridge_count_raster", "amp_signal"]].apply(lambda x: self.dominant_wavelength(*x), axis=1)
            
        return rich_transects.sort_index()
    
    def calc_itx_metrics(self):
        """For each transect found in transects, calculate the itx metrics."""

        itx = pd.concat(
            [TransectDataExtractor(*row, self.rich_transects.crs, self.ridges).calc_ridge_metrics() \
                for i, row in self.rich_transects[["transect_id", "geometry", "dem_signal", "clean_bin_signal"]].iterrows()]
        ).set_index(["bend_id", "transect_id", "ridge_id"])

        return itx.sort_index()