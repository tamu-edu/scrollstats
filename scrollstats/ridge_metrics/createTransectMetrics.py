# This script is intended to contain all relevant fucntions to create a
# geodataframe of bend transects and their associated scroll metrics

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.io import DatasetReader
from shapely.geometry import *
from scipy import ndimage

from .migration_rates import calc_rel_mig_rates
from .ridgeAmplitudes import calc_ridge_amps, map_amp_values


def calc_dist(p1: np.array, p2: np.array) -> np.array:
    """p1 and p2 are both (n,2) arrays of coordinates"""
    return np.sqrt((p1[:, 0] - p2[:, 0]) ** 2 + (p1[:, 1] - p2[:, 1]) ** 2)


def remove_coincident_points(coord_array: np.array) -> np.array:
    """Removes coincident points that happen to be adjacent"""
    
    # Calc distances between each point
    dist = calc_dist(coord_array[:-1], coord_array[1:])
    
    # Find where these distances are not near zero
    unique_idx = np.nonzero(dist>0.001)[0] + 1     # Add one to the index b/c first point is guaranteed ok
    
    return coord_array[unique_idx]


def explode(line):
    """Return a list of all 2 coordinate line segments in a given LineString"""
    return list(map(LineString, zip(line.coords[:-1], line.coords[1:])))


def densify_segment(line):
    """Densify a line segment between two points to 1pt/m. Assumes line coordinates are in meters """
    x, y = line.xy
    
    num_points = int(round(line.length))

    xs = np.linspace(*x, num_points+1)
    ys = np.linspace(*y, num_points+1)
    
    return LineString(zip(xs,ys))


def densify_line(line):
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


def transform_coords(coord_array, bin_raster):
    """Transform the coordinates from geo to image for indexing a in_bin_raster"""
    
    transform = bin_raster.transform
    t_coords = [~transform * coord for coord in coord_array]
    
    return np.round(t_coords).astype(int)  # round coords for indexing


def disqualify_coords(coord_array, bin_raster):
    """ 
    Some coordinates may be out of the in_bin_raster. 
    This function disqualifies these coordinates and returns a boolean arry showing the location of all disqualified coordinates.
    
    Coordinates are checked to see if they are 1) negative, 2) too large in x, or 3) too large in y
    """
    # Find location of all negative coords
    too_small = np.any(coord_array < 0, axis=1)

    # Find location of too large x
    x_max = bin_raster.profile["width"]
    too_large_x = coord_array[:,0] >= x_max

    # Find location of too large y
    y_max = bin_raster.profile["height"]
    too_large_y = coord_array[:,1] >= y_max

    # return true for a coord if that coord failed any test 
    return np.any(np.vstack((too_small, too_large_x, too_large_y)), axis=0)


def sample_array(coord_array, bin_raster:DatasetReader):
    """
    Takes in an array of image coordinats, samples the image, and returns the sampled values.
    Assumes that the coord array and in_bin_raster dataset share the same crs.
    """
    # Prep the coords 
    ## Some coordinates may be out of bounds for the in_bin_raster
    ## So we need to only sample with valid image coords, then pad the array with nans for all out of bounds areas
    disq = disqualify_coords(coord_array, bin_raster) # boolean array
    in_bounds = coord_array[~disq]
    
    # Sample the array with valid coords
    arr = bin_raster.read(1)
    signal = arr[in_bounds[:,1], in_bounds[:,0]].flatten() # remember image coords are (y,x)
    
    # Pad either side of the signal with nans for disqualified points
    out_signal = np.zeros(disq.shape)*np.nan
    out_signal[~disq] = signal
    
    return out_signal


def dense_sample(line, bin_raster):
    """Sample an underlying in_bin_raster along a given LineString at a frequency of ~1m"""
    
    # Densify points 
    d_line = densify_line(line)
    
    # Extract coordinates
    d_coords = np.asarray(d_line.coords[:])
    
    # Apply inverse geotranseform (geo -> image coords)
    t_coords = transform_coords(d_coords, bin_raster)
    
    # Sample (index) underlying in_bin_raster at each coord
    ## coords out of in_bin_raster bounds will be returned with np.nan
    return sample_array(t_coords, bin_raster)


def plot_signal(signal):
    '''
    Simple plot of a single signal.
    '''
    fig, ax = plt.subplots(1, figsize=(8,2))
    ax.plot(signal)
    ax.set_xlabel('Distance from Channel')
    return ax


def remove_ones(signal):
    '''
    This function replaces all 1s that preceed a 0 in the input signal with a NaN.

    Run this function across the signal in both directions like so

    `clean_sig = remove_ones(remove_ones(sig)[::-1])[::-1]`
    '''

    # Create copy of the array
    sigc = signal.copy()

    # Create a watch variable;
    make_nan = 1
    for i, v in enumerate(signal):

        if make_nan == 1:
            if v != 0:
                # Define the new value
                sigc[i] = np.nan

            # Once a zero is encountered, just return the same value and turn off the watch variable
            if v == 0:
                sigc[i] = v
                make_nan = 0
        # Once the watch variable =0, then stop altering values
        else:
            pass

    return sigc


def flip_bin(signal):
    '''
    flips binary values
    '''
    signal_c = signal.copy()

    # Get locs
    one_loc = (signal==1)
    z_loc = (signal==0)

    # Flip array values
    signal_c[one_loc] = 0
    signal_c[z_loc] = 1

    return signal_c


def remove_small_feats(signal, th=3):
    '''
    removes small features within an array smaller than the threshold `th`
    '''

    signal_c = signal.copy()

    # Label all unique features
    labels, numfeats = ndimage.label(signal_c)

    # Find their counts (widths)
    val, count = np.unique(labels, return_counts=True)

    # Find all labels corresponding to small features
    small_ridge_vals = val[count <= th]

    # Redefine small features to 0s
    for i in small_ridge_vals:
        signal_c[labels==i]=0

    return signal_c


def clean_signal(signal):
    '''
    Apply several functions to the raw transect signal to remove small or incomplete
    features from the signal.
    '''

    signal_c = signal.copy()

    # Remove partial ridges
    signal_c = remove_ones(remove_ones(signal_c)[::-1])[::-1]

    # Remove small ridges
    signal_c = remove_small_feats(signal_c)

    # Flip values and repeat to eliminate small swales
    signal_c = flip_bin(signal_c)
    signal_c = remove_small_feats(signal_c)

    # Flip values back
    signal_c = flip_bin(signal_c)

    return signal_c


def count_ridges(signal):
    """Counts ridges in binary waves."""
    mask = np.isnan(signal)
    labs, numfeats = ndimage.label(signal[~mask])
    return numfeats


def trans_fft(raster_val):
    '''
    Calcualtes the fast fourier transform for a 1D signal.

    If you wish to see the power spectra, plot the sampled frequencies (x) vs their measured amplitude (y)
    The dominant wavelength within a given signal can be found with the function `dominant_wavelength()` below
    '''

    # Define Scroll Signature
    scroll_sig = np.array(raster_val)

    ## Set Variables
    # We want to sample the signal 1 time every meter
    # Therefore the interval between samples is equal to 1/1 meters
    samplingFreq = 1
    samplingInt = 1/samplingFreq


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
    timePeriod  = scroll_sig.size/samplingFreq
    freqs = values/timePeriod

    return(freqs, amps)


def dominant_wavelength(signal):
    '''
    Identifies the dominant wavelength from an input binary signal
    '''

    # Test to see if signal has at least 2 ridges
    if count_ridges(signal) > 1:

        # Remove nans and zero values from signal
        real_signal = signal[~np.isnan(signal)] - 0.5

        # Calcualte the fft
        ## To see the power spectra, plot freq x amps as a line
        freq, amp = trans_fft(real_signal)

        # Find the max value in amps that is not the first value
        # Use np.nonzero().min() here because some transects have the max amp in multiple locations
        max_amp_loc = np.nonzero(amp == amp[1:].max())[0].min()

        # Convert freq to wavelength
        dom_wav = round(1/freq[max_amp_loc])

    else:
        dom_wav = np.nan

    return dom_wav


def ridge_width_series(signal):
    """
    Calcualte the width of each ridge area in the 1D signal.
    Widths are calculated as the length of each unique string of 1s in the input binary signal.
    Returns an array of np.nan with the ridge width at the midpoint location.
    
    Ex) np.array([nan, nan, nan, 3, nan, nan, nan, 2, nan, nan])
    """
    
    # Check for presense of nans in signal
    mask = np.isnan(signal)
    if all(mask):
        return signal
    
    # Remove potential nans from beginning and end of signal
    clean_signal = signal[~mask]
    
    # Find individual ridge areas
    labels, numfeats = ndimage.label(clean_signal)
    
    # Find the width of each ridge by counting each group in labels
    vals, counts = np.unique(labels, return_counts=True)
    
    # Find the centerpoint of each ridge along the transect
    center_points = ndimage.center_of_mass(clean_signal, labels, np.arange(numfeats)+1)
    center_points = np.round(center_points).astype(int).reshape(numfeats)
    
    # Create nan array and fill with ridge widths (counts) at centerpoints
    rw = clean_signal*np.nan
    rw[center_points] = counts[1:]
    
    # Pad either side of rw with nans from original signal
    out_rw = signal*np.nan
    out_rw[~mask] = rw
    
    return out_rw

    
def calc_avg_width(width_series):
    return np.nanmean(width_series)

def calc_avg_amp(amp_series):
    return np.nanmean(amp_series)


def calc_curvature(line):
    
    # Check for empty line
    if line.is_empty:
        return 0
    
    x, y = line.xy
    
    dist = np.sqrt((x[0]-x[-1])**2 + (y[0]-y[-1])**2)
    return line.length / dist




def create_dataframe(transects, bin_raster, dem):
    '''
    Calculates all scroll metrics that require raster sampling along the generated transects.

    Input:
    ------
    in_transects: GeoDataFrame of automatically generated transects
    in_bin_raster: rasterio dataset to binary bin_raster to be sampled by the transects found in `in_transects`
    in_dem: rasterio dataset for the DEM of the bend

    Output:
    -------
    transects: geodataframe with extra rows containing ridge signals and metrics for each transect
    '''

    # Add columns to geodataframe
    transects["dem_signal"] = transects.geometry.apply(lambda x: dense_sample(x, dem))
    transects["dem_signal"] = transects.dem_signal.apply(lambda x: np.where(x==0, np.nan, x))
    transects["bin_signal"] = transects.geometry.apply(lambda x: dense_sample(x, bin_raster))
    transects["clean_bin_signal"] = transects.bin_signal.apply(clean_signal)
    transects["ridge_count"] = transects.clean_bin_signal.apply(count_ridges)
    transects["dom_wav"] = transects.clean_bin_signal.apply(dominant_wavelength)
    transects["width_series"] = transects.clean_bin_signal.apply(ridge_width_series)
    transects["avg_width"] = transects.width_series.apply(calc_avg_width)
    transects["amp_series"] = transects[["dem_signal", "clean_bin_signal"]].apply(lambda x: calc_ridge_amps(*x), axis=1)
    transects["amp_series"] = transects[["amp_series", "width_series"]].apply(lambda x: map_amp_values(*x), axis=1)
    transects["avg_amp"] = transects.amp_series.apply(calc_avg_amp)
    transects["avg_curv"] = transects.geometry.apply(calc_curvature)

    return transects


def calculate_transect_metrics(in_transects, in_bin_raster, in_dem, in_ridges=None, in_packets=None):
    """
    Master funtion to calculate scroll metrics.

    If in_ridges is specified, relative migration rates will be calculated. Relative migration rate is the distance along a transect from a particular ridge to the curren centerline.
    If in_packets is specified, then all metrics (except relative migration rates) will be calcualted for the transect fragement within each packet.

    All arguments can be provided as a file path or in-memory object (vector: GeoDataFrame, raster: rasterio dataset)
    """


    # Check if args are paths or objects in memory
    if isinstance(in_transects, gpd.GeoDataFrame):
        transects = in_transects.copy()
    else:
        transects = gpd.read_file(in_transects)
    
    if isinstance(in_bin_raster, rasterio.io.DatasetReader):
        bin_raster = in_bin_raster
    else:
        bin_raster = rasterio.open(in_bin_raster)
    
    if isinstance(in_dem, rasterio.io.DatasetReader):
        dem = in_dem
    else:
        dem = rasterio.open(in_dem)

    if isinstance(in_ridges, gpd.GeoDataFrame) or in_ridges is None:
        ridges = in_ridges
    else:
        ridges = gpd.read_file(in_ridges)
    
    if isinstance(in_packets, gpd.GeoDataFrame) or in_packets is None:
        packets = in_packets
    else:
        packets = gpd.read_file(in_packets)
    

    # If ridges are provided, calculate relative migration rates
    if isinstance(ridges, gpd.GeoDataFrame):
        transects = calc_rel_mig_rates(transects, ridges)
        
    # If packets are provided, create intersection with packets and return MultiIndex DataFrame
    if isinstance(packets, gpd.GeoDataFrame):
        transects = transects.overlay(packets, how="intersection").set_index(["packet_id", "transect_id"])
    
    # Calculate sampled ridge metrics
    transects = create_dataframe(transects, bin_raster, dem).sort_index()
    
    return transects
