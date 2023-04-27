from typing import List

import numpy as np
from scipy import ndimage
import geopandas as gpd


def calc_ridge_maxes(dem_sig, mask):
    """Calculate the max value of the dem signal within each of the ridge areas in the mask"""

    # Find each unique ridge
    labels, numfeats = ndimage.label(mask)
    
    # Index dem_sig with each unique labeled area
    dem_maxes = [dem_sig[labels==i].max() for i in np.arange(numfeats)+1]
    
    # Return list as array
    return np.array(dem_maxes)


def calc_swale_mins(dem_sig, mask):
    """Calcualte the minimun value within each of the swale areas of the mask"""
    
    # Find each unique swale
    labels, numfeats = ndimage.label(~mask)
    
    # Calculate minimum value for each swale (excluding nans)
    dem_mins = [np.nanmin(dem_sig[labels==i]) for i in np.arange(numfeats)+1]
    
    # Return as array
    return np.array(dem_mins)


def s00_amps(maxes, mins):
    """
    Amplitude calcualtion if binary signal is capped with 0s on each end. 
          _   _   _
    ex. _| |_| |_| |_
    
    Amplitude for each ridge is calcualted as the average of the differences between ridge max and the swale mins on each side.
    However, the amplitude of the ridge at the beginning is calculated only by the following swale because of channel effects on the DEM signal.
    """
    
    # Calcualte diffs between ridge maxes and the preceeding swale mins
    d1 = maxes[1:] - mins[1:-1]
    
    # Calcualte diffs between ridge maxes and following swale mins
    d2 = maxes - mins[1:]
    
    # Insert the first value from d2 into d1 to make arrays the same size
    d1 = np.insert(d1, 0, d2[0])
    
    # Return the average of the differences
    return np.vstack([d1, d2]).mean(axis=0)


def s11_amps(maxes, mins):
    """
    Amplitude calcualtion if binary signal is capped with 1s on each end. 
        _   _   _
    ex.  |_| |_| 
    
    Amplitude for each ridge is calcualted as the average of the differences between ridge max and the swale mins on each side.
    However, because there is a ridge at both ends, the first and last ridge only gets one amplitude measurement

    """
    
    # Calcualte diffs between ridge maxes and the preceeding swale mins
    d1 = maxes[1:] - mins
    
    # Calcualte diffs between ridge maxes and following swale mins
    d2 = maxes[:-1] - mins

    # Insert the first value from d2 because d1 is missing a measurement for the first ridge max
    d1 = np.insert(d1, 0, d2[0])    
    
    # Append the last value from d1 because d2 is missing a measurement for the last ridge max
    d2 = np.append(d2, d1[-1] )

    # Return the average of the differences
    return np.vstack([d1, d2]).mean(axis=0)


def s01_amps(maxes, mins):
    """
    Amplitude calcualtion if binary signal starts with a swale and ends with a ridge.
          _   _   _
    ex. _| |_| |_|
    
    Amplitude for each ridge is calcualted as the average of the differences between ridge max and the swale mins on each side.
    However, because there is no swale at the end, the last ridge only gets one amplitude measurement
    Additionally, the amplitude of the ridge at the beginning is calculated only by the following swale because of channel effects on the DEM signal.
    """
    
    # Calcualte diffs between ridge maxes and the preceeding swale mins
    d1 = maxes[1:] - mins[1:]
    
    # Calcualte diffs between ridge maxes and following swale mins
    d2 = maxes[:-1] - mins[1:]
    
    # Append the last measurement from d1 to d2 to make arrays the same size
    d2 = np.append(d2, d1[-1])
    
    # Insert the first measurement from d2 into d1 to make arrays the same size
    d1 = np.insert(d1, 0, d2[0])
    
    # Return the mean of the two arrays
    return np.vstack([d1, d2]).mean(axis=0)
    
    
def s10_amps(maxes, mins):
    """
    Amplitude calcualtion if binary signal starts with a ridge and ends with a swale.
        _   _   _
    ex.  |_| |_| |_
    
    Amplitude for each ridge is calcualted as the average of the differences between ridge max and the swale mins on each side.
    However, because there is no swale at the beginning, the first ridge only gets one amplitude measurement
    """
    
    # Calcualte diffs between ridge maxes and the preceeding swale mins
    d1 = maxes[1:] - mins[:-1]
    
    # Calcualte diffs between ridge maxes and following swale mins
    d2 = maxes - mins
    
    # Insert the first measurement from d2 into d1 to make arrays the same size
    d1 = np.insert(d1, 0, d2[0])
    
    # Return the mean of the two arrays
    return np.vstack([d1, d2]).mean(axis=0)


def determine_complex_strategy(bool_mask:List[bool]):
    """
    This function determines which multi ridge/swale amplitude calculation strategy to use.

    Generally, the ridge amplitude is calcualted as the average differences between the ridge max and the two swale mins.
    However, exceptions must be made if the boolean_mask begins or ends with a ridge (this beginning/ending ridge will not have a swale on one of the sides)

    Importantly, this function only deals with boolean arrays with more than one ridges and or swales. 
    This means that this function does not have to deal with edge cases of flat or s-shaped signals.
    """

    if not bool_mask[0] and not bool_mask[-1]:
        strategy = s00_amps
    elif bool_mask[0] and not bool_mask[-1]:
        strategy = s10_amps
    elif not bool_mask[0] and bool_mask[-1]:
        strategy = s01_amps
    elif bool_mask[0] and bool_mask[-1]:
        strategy = s11_amps
    else:
        raise Exception(f"bool_mask is of unexpected type {type(bool_mask)} or contains unexpexted values\n{bool_mask=}")
    
    return strategy


def calc_ridge_amps(dem_sig, bin_sig):
    """
    Calculate the ridge amplitudes from a DEM profile using the boolean mask signal.
    
    Different strategies are used to calculate the ridge amplitude based on the ridge and swale count 
    found within the boolean mask signal.
    """
        
    # Create a boolean mask from the binay signal
    mask = np.where(np.isnan(bin_sig), 0, bin_sig).astype(bool)

    # Calculate ridge maxes
    maxes = calc_ridge_maxes(dem_sig, mask)
    ridge_count = len(maxes)
    
    # Calculate swale mins
    mins = calc_swale_mins(dem_sig, mask)
    swale_count = len(mins)

    if (ridge_count==1 and swale_count==0) or (ridge_count==0 and swale_count==1):
        amps = np.array([np.nanmax(dem_sig) - np.nanmin(dem_sig)])

    elif ridge_count==1 and swale_count==1:
        amps = maxes - mins
    
    elif ridge_count>=1 or swale_count>=1:
        strategy = determine_complex_strategy(mask)
        amps = strategy(maxes, mins)

    else:
        raise Exception(f"Unexpected configuration/count of ridge and swales. \
                        \n{bin_sig=} \
                        \n{mask=} \
                        \n{dem_sig=} \
                        \n{maxes=} \
                        \n{mins=}")

    return amps


def map_amp_values(amp_series, width_series):
    """
    Map the ridge amplidute values to their assumed location along the transect.
    Assumed location is the approximate midpoint of the ridge.
    """
    
    # Create a new array that is the length of the transect
    new_series = width_series*np.nan
    
    # Map amp values onto 
    new_series[~np.isnan(width_series)] = amp_series
    
    return new_series
    

    