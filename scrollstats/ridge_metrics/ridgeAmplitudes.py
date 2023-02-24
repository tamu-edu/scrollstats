import numpy as np
from scipy import ndimage
import geopandas as gpd


def calc_ridge_maxes(dem_sig, mask):
    """Calculate the max value of the dem sinal within each of the ridge areas in the mask"""

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


def calc_ridge_amps(dem_sig, bin_sig):
    """
    Calculate the amplitude of each ridge in the DEM signal.
    
    Amplitude is calcuated as the mean of the max ridge elevation minus the swale minimums of both sides of the ridge.
    
    mean( [(max(elev_ri) - min(elev_si)), (max(elev_ri) - min(elev_si+1))])
    """
    
    # Create a boolean mask from the binay signal
    mask = np.where(np.isnan(bin_sig), 0, bin_sig).astype(bool)
    
    # Calculate ridge maxes
    maxes = calc_ridge_maxes(dem_sig, mask)
    
    # Calcualte swale mins
    mins = calc_swale_mins(dem_sig, mask)

    # Catch scenarios where bin_sig is either all ridge or all swale
    if (mins.size==0 or maxes.size==0):
        amps = np.array([])
    
    # Test for different ridge-swale scenarios:
    elif mins.size == maxes.size:
        
        # Only 1 ridge-swale pair
        if mins.size==1:
            amps = maxes - mins
        
        # Starts with ridge, ends with swale
        elif mask[0] and not mask[-1]:
            amps = s10_amps(maxes, mins)
        
        # Starts with swale and ends with ridge
        elif not mask[0] and mask[-1]:
            amps = s01_amps(maxes, mins)
        
        else:
            raise Exception("Equal ridge and swale count, but in an unexpected pattern.")
    
    elif mins.size - maxes.size == 1:
        
        # Starts and ends with ridge
        if mask[0] and mask[-1]:
            amps = s11_amps(maxes, mins)
        
        # Starts and ends with swale
        elif not mask[0] and not mask[-1]:
            amps = s00_amps(maxes, mins)
    
        else:
            raise Exception("Ridge and swale count are off by one, but in an unexpected pattern.")
    
    else:
        raise Exception("Ridge and swale count are not equal and differ by more than one.")
    
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
    

    