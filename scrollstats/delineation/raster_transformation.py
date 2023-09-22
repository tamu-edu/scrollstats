import numpy as np
from numpy import array
import rasterio
from scipy.signal import convolve2d


def profile_curvature_instructions():
    """Instructions on how to calculate profile curvature in QGIS3"""

    qgis_string = """"""

    return qgis_string


class CalcResidualTopography:
    def __init__(self, dem_path, window_size, out_dir) -> None:
        self.dem_path = dem_path
        self.window_size = window_size
        self.out_dir = out_dir

        self.suffix = "rt"
        self.out_name = f"{self.dem_path.stem}_{self.suffix}{self.window_size}px.tif"
        self.out_path = self.out_dir / self.out_name

    def residual_topography(self, dem:array, w:int) -> array:
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
    
    def execute(self):
        """
        Execute residual topography
        """

        # Open DEM 
        dem_raster = rasterio.open(self.dem_path)
        profile = dem_raster.profile
        dem = dem_raster.read(1)

        # Mask out no-data pixels with nans
        dem[dem<-1e6] = np.nan
        
        # Apply residual topography transformation to array
        rt = self.residual_topography(dem, self.window_size)

        # Save array to disk 
        with rasterio.open(self.out_path, "w", **profile) as dst:
            dst.write(rt, 1)

        return self.out_path