import sys
import os
import subprocess
from pathlib import Path

import numpy as np
from numpy import array
import rasterio
from scipy.signal import convolve2d
from parameters import GRASS_DIR, GRASS_BASE, GRASS_BIN


class CalcProfileCurvature:
    def __init__(self, dem_path, window_size, out_dir, out_name=None) -> None:
        self.dem_path = dem_path
        self.window_size = window_size
        self.out_dir = out_dir
        self.grass_dir = GRASS_DIR

        # File name attributes
        self.suffix = "profc"
        if out_name:
            self.out_name = out_name
        else:
            self.out_name = f"{self.dem_path.stem}_{self.suffix}{self.window_size}px.tif"
        self.out_path = self.out_dir / self.out_name

        # Create location specific directory for grass outputs
        crs = rasterio.open(self.dem_path).crs.to_string()
        if not crs == '':
            self.location_path = self.create_grass_project(crs)
        else:
            raise ValueError(f"Detected CRS of the tif is not valid.\n Detected CRS: {crs}")

    def create_grass_project(self, crs:str) -> Path:
        """
        Creates the GRASS GIS Database location (project) folder structure as defined below
        
        grass_dir: the base directory that will contain all GRASS GIS data
        location: a subdirectory of `grass_dir` that contains all data for a project. All data must share the same CRS

        See link below for more detail on GRASS GIS Database structure.
        https://grass.osgeo.org/grass84/manuals/grass_database.html
        """

        # Create CRS specific location; see doc string
        location_path = self.grass_dir / f"CRS_{crs.replace(':', '_')}"

        # Initialize the GRASS location
        # This generates all required data/directories for the location
        startcmd = f'"{str(GRASS_BIN)}" -c {crs} -e "{location_path}"'

        if not location_path.exists():
            p = subprocess.Popen(startcmd, shell=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = p.communicate()
            if p.returncode != 0:
                print(f"ERROR: {err}")
                sys.exit(-1)
            else:
                print(f"Created location {location_path}")
        
        return location_path
    
    def execute(self) -> Path:
        """Launch a headless GRASS GIS session to calculate profile curvature"""

        # `grass` library added to PYTHON_PATH in parameters.py
        import grass.script as gs
        from grass.script import array as garray
        from grass.script import setup as gsetup

        # Intialize GRASS modules
        gsetup.init(str(GRASS_BASE), self.grass_dir, self.location_path.stem, "PERMANENT")

        # Open DEM 
        dem = rasterio.open(self.dem_path)
        dem_array = dem.read(1)
        west, south, east, north = dem.bounds
        nrows, ncols = np.shape(dem_array)

        # Define a temporary region for the tif
        gs.run_command('g.region',
                       n=north, s=south, e=east, w=west,
                       rows=nrows, cols=ncols, 
                       flags="p", save="my_tmp", overwrite=True)

        # Import DEM as a GRASS raster into the region created above
        elev_arr = garray.array()
        elev_arr[:] = dem_array.copy()
        elev_arr.write(mapname='elev', title='elev', null=dem.nodata, overwrite=True)

        # Run r.param.scale tool to generate the profile curvature raster
        # default values are explicitly written below
        profc_result = "profc_result" # string variable to hold name of raster result
        gs.run_command('r.param.scale', 
                       input = 'elev',
                       output = profc_result,
                       slope_tolerance=1.0,             # default value
                       curvature_tolerance = 0.0001,    # default value
                       size = self.window_size,
                       method = 'profc',
                       exponent = 0.0,                  # default value
                       zscale = 1.0,                    # default value
                       overwrite=True)

        # # Uncomment below to print test info for output raster
        # gs.run_command('r.info', map_=profc_result)

        # Convert GRASS raster to numpy array
        profc_arr = garray.array()
        profc_arr.read(profc_result)
        profc_arr_np = np.asarray(profc_arr, dtype = np.float32)

        # Save array to disk 
        with rasterio.open(self.out_path, "w", **dem.profile) as dst:
            dst.write(profc_arr_np, 1)

        return self.out_path


class CalcResidualTopography:
    def __init__(self, dem_path, window_size, out_dir, out_name=None) -> None:
        self.dem_path = dem_path
        self.window_size = window_size
        self.out_dir = out_dir

        # File name attributes
        self.suffix = "rt"
        if out_name:
            self.out_name = out_name
        else:
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
    
