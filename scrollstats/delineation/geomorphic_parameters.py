# Contains the functions used to calculate geomorphic parameters on DEMs
import numpy as np
from numpy import array
from scipy.signal import convolve2d

def profile_curvature():
    """Instruct the user on how to use the `r.param.scale` tool in QGIS"""

    instructions = ""

    return instructions


def residual_topography(dem:array, w:int) -> array:
    """
    Calculate the residual topography for a 2D array.
    """
    # Create weighted window with which to convolve the DEM
    win = np.ones((w, w)) / w**2

    # Convolve the image to reassign a given pixel value to the average of its neighborhood
    avg = convolve2d(dem, win, mode='same', fillvalue=np.nan)

    # Subtract avg from dem to see which features stand out from the landscape
    rt = dem - avg

    return rt


