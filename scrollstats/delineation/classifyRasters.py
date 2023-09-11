# This script is designed to apply a simple binary classification to a set of rasters
# This threshold value can be set by the user, but the deafult value is 0

import sys
import pathlib
import numpy as np
import rasterio


# INPUTS
inpath = sys.argv[1] # path to directory containing all the rasters to be classified
inPath = pathlib.Path(inpath)

if len(sys.argv) > 2:
    th = sys.argv[2]   # threshold value used to decide the binary classification
else:
    th = 0

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


    # # Filter paths for the desired window
    # paths = [i for i in paths if i.stem.split('_')[4] == f"{win}px"]

    # Sort the output for niceness
    spaths = sorted(paths, key = lambda x: x.name.split('_')[2])

    # Return sorted paths
    return spaths


def binclass(a, th):

    # Create mask for nans
    mask = np.isnan(a)

    # Classify raster values
    a[a <= th] = 0
    a[a > th] = 1

    # Redefine nan locations
    a[mask] = np.nan

    return a


# PROCESSING

# Check for single or batch process intended
if inPath.is_file():
    target = [inPath]
else:
    target = listFilePaths(inPath)

for i, path in enumerate(target):

    # Open dataset
    ds = rasterio.open(path)
    profile = ds.profile

    # Read Array
    arr = ds.read(1)

    # Classify Array
    out_arr = binclass(arr, th=th)

    # Create outpath
    outname = f"{path.stem}_binclass.tif"
    # outdir = path.parents[1] / f"{path.parts[-2]}-binclass"
    outpath = outdir / outname

    # Create the out dir if it does not already exist
    if not outpath.parent.exists():
        outpath.parent.mkdir(parents=True)

    # Write to disk
    with rasterio.open(outpath, 'w', **profile) as dst:
        dst.write(out_arr, 1)

    print('')
    print(f"Classified raster: {i+1} of {len(target)}", end='\r')

print('')
print(f"Classification Complete! Check `{outdir.resolve()}` for classified rasters")
