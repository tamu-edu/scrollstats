import sys
import pathlib
import numpy as np
import rasterio
from scipy.signal import convolve2d
from scipy.ndimage import generic_filter

# Numba stuff
from numba import jit

# This script is designed to calculate the residual topography for one or many dems

# INPUTS
inpath = sys.argv[1] # path to directory containing all the rasters to be classified
inPath = pathlib.Path(inpath)

if len(sys.argv) > 2:
    win = int(sys.argv[2])   # window size for the rt operation
else:
    win = 45

if len(sys.argv) > 3:
    outdir = pathlib.Path(sys.argv[3])
else:
    outdir = pathlib.Path('./output') # set deafult output location to 'output' dir



# Creates a sorted list of file paths for a given dir - one for each bend of the Lower Brazos
def listFilePaths(in_dir, ext='.tif', win=45):

    in_dir = pathlib.Path(in_dir)
    # Make a list of all relevant files
    paths = [i for i in in_dir.iterdir() if i.suffix == ext]

    # Check if `paths` variable acutally contains any file paths
    if len(paths) == 0:
        print(f"ERROR: No paths found with given extension ({ext})")
        return []
    elif len(paths) != 101:
        print(f"WARNING: Less than 101 files ({len(paths)}) were found in `{in_dir}` with extension `{ext}`")


    # Filter paths for the desired window
    # paths = [i for i in paths if i.stem.split('_')[4] == f"{win}px"]

    # Sort the output for niceness
    # spaths = sorted(paths, key = lambda x: x.name.split('_')[2])

    # Return sorted paths
    return paths


def res_topo(dem, w):
    # Create weighted window with which to convolve the DEM
    win = np.ones((w, w)) / w**2

    # Convolve the image to reassign a given pixel value to the average of its neighborhood
    avg = convolve2d(dem, win, mode='same', fillvalue=np.nan)

    # Subtract avg from dem to see which features stand out from the landscape
    rt = dem - avg

    return rt

# Check for single or batch process intended
if inPath.is_file():
    target = [inPath]
else:
    target = listFilePaths(inPath)

for ras_path in target:

    ras = rasterio.open(ras_path)
    profile = ras.profile
    dem = ras.read(1)

    dem[dem<0]=np.nan

    rt = res_topo(dem, win)


    # Write rt to disk
    outname = f"{ras_path.stem}_rt_{win}px.tif"
    outpath = outdir / outname

    # Create the out dir if it does not already exist
    if not outpath.parent.exists():
        outpath.parent.mkdir(parents=True)

    with rasterio.open(outpath, 'w', **profile) as dst:
        dst.write(rt, 1)

    print('Wrote res. topo raster to disk: ', outpath)
