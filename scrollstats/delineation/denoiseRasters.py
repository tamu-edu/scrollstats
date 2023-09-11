# This script contains all the functions to denoise an image.

import sys
import pathlib
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage, spatial

# INPUTS
rpath = sys.argv[1] # path to directory containing all the rasters to be processed
rPath = pathlib.Path(rpath)

if len(sys.argv) > 2:
    outdir = pathlib.Path(sys.argv[2])
else:
    outdir = pathlib.Path('./output')        # Set outdir to current directory as default

## Params
### should be a tuple given in the following order
### (buff, small, et)
try:
    dn_params = sys.argv[3]
except IndexError:
    dn_params = 'd'

if dn_params == 'd':
    buff = 0
    small = 500
    et = 0.8
    msg = 'Denoise params not specified. Using default values:'
else:
    buff, small, et = tuple(float(num) for num in dn_params.split(','))
    msg = 'Denoise params given:'

print('')
print(f"{msg}\n{'-'*len(msg)}")
print(f"Buffer               : {buff}m")
print(f"Min feat size        : {small}m")
print(f"Elongation Threshold : {int(et*100)}%")
print('')


# DEFINE FUNCTIONS

#############################################################################################
#############################################################################################
#############################################################################################
# From `delin_funcs.py`

# Remove feats
def clean_small_feats(img, size):

    '''
    Removes any patch/feature in a binary image that is below a certian pixel count

    Parameters
    ----------
    img : binary ndarray
    size : (int), minimum patch size needed to be kept in the image

    Returns
    -------
    out : binary ndarray

    '''
    # Label all unique features in binary image
    label, numfeats = ndimage.label(img)

    # Get list of unique feat ids as well as pixel counts
    u, cnt = np.unique(label, return_counts=True)

    # list of feat ids that are too small
    ids = u[cnt < size]

    # Wipe out patches with id that is in `ids` list
    for id in ids:
        label[label==id] = 0

    # Get a list of remaining unique IDs as a check
    u2 = np.unique(label)

    # Convert all labels to 1
    label[label!=0] = 1

    # Feedback
    msg = f'Removing Small Features (<{size}px):'
    print('\n')
    print(f"{msg}\n{'-'*len(msg)}")
    print('Features in: ', len(u)-1) # -1 for 0 (background)
    print('Features out: ', len(u2)-1) # -1 for 0 (background)
    print('Features removed: ', len(ids))

    return label

## Calcualte morphological charcteristics for each patch in the image
## Return df with values, patches, and patch locations to reconstruct image
def classify_feats(img):

    '''
    Calcualtes morphological charcteristics for each patch in the image.

    Returns a df with values, patch area, and patch locations to reconstruct original
    binary image after the target patches are removed

    Input:
    ------
    img : binary ndarray

    Returns:
    --------
    label : (2D array) array where all indv patches have a unique ID (for reference with df)
    df: (dataframe) Contains: PatchID, raw patch area (px), filled patch area (px),
        diameter of circumscribing circle, area of circle, elongation index,
        isolated patch as 2D array, location of each patch within the image

    '''

    # Label individual image features
    label, numfeats = ndimage.label(img)

    # Get list of unique features as well as pixel counts
    ids = np.unique(label)[1:]

    # Find location of every object in label
    locs = ndimage.find_objects(label)

    # Index labeled image for every feature location
    patches = [label[loc] for loc in locs]

    # Isolate just the patch in question for every slice in patches, then convert
    # it to a value of 1, all else to 0
    for i, patch in enumerate(patches):
        patches[i] = np.where(patch==(i+1), 1, 0)

    # Calc raw area for each patch
    p_area = np.array([patch.sum() for patch in patches]) # used later for feedback

    # Calc the circular area for every patch to calc the elongation index
    #  Step 1 - Fill holes in all patches
    #  Step 2 - Erode filled patch, then subtract erosion from filled patch to get boundary
    #  Step 3 - Get image coords for each boundary pixel
    #  Step 4 - Calc distance between each coord, take max value as diameter
    #  Step 5 - Calc circular area for each patch
    #  Step 6 - Divide filled patch area by circle area for index value


    # Step 1 - Fill holes in all patches
    filled = [ndimage.binary_fill_holes(patch).astype(int) for patch in patches]

    ## Calc filled area for each patch
    f_area = np.array([fill.sum() for fill in filled])

    # Step 2 - Erode filled patch, then subtract erosion from filled patch to get boundary
    ero = [ndimage.binary_erosion(fill) for fill in filled]
    bounds = [i - j for i, j in zip(filled, ero)]

    ## Calc boundary count for feedback
    px_count2 = sorted([bound.sum() for bound in bounds[0:5]], reverse=True)

    # Step 3 - Get image coords for each boundary pixel
    coords = [np.transpose(np.nonzero(bound)) for bound in bounds]

    # Step 4 - Calc distance between each coord, take max value as diameter
    di = np.array([spatial.distance.pdist(coord).max() for coord in coords])

    # Step 5 - Calc circular area for each patch
    pr2 = np.pi*(di/2)**2

    #  Step 6 - Divide filled patch area by circle area for index value
    ## I used filled features here because features with holes will be counted
    ## as more elongated without consideration of thier outer boundaries
    ## consider how an empty circle would score
    elong = f_area / pr2

    # Create data frame of all relevant data
    df = pd.DataFrame({'ID': ids,
                       'PatchArea':p_area,
                       'FillArea':f_area,
                       'Diameter': di,
                       'PiR2': pr2,
                       'ElongIndex': elong,
                       'Patch': patches,
                       'PatchLoc': locs
                       }).set_index('ID')

    # Feedback
    msg = f'Classifying Remaining Features (n={len(df.index)})'
    print('\n')
    print(f"{msg}\n{'-'*len(msg)}")
    print('Number of pixels in 5 largest patches: ', np.sort(p_area)[-1:-6:-1])
    print('Number of pixels in 5 largest patches after erosion: ', px_count2)

    return (label, df)

## Rebuild image from df output of `classify_feats()`
def build_img(img, df, th):

    '''
    Rebuild image from df output of `classify_feats()`

    Input:
    ------
    img : (2D array) used solely for image dimensions
    df : dataframe from `classify_feats()`

    Returns:
    --------
    new_img : (2D array) array built from patches that satisfy morph criteria

    '''
    # Create blank image to receive features
    new_img = np.zeros(img.shape)

    # Query df for records with satisfactory elongation index
    new_df = df[df.ElongIndex < th]

    # Add all satisfactory patches to blank image
    for patch, loc in zip(new_df.Patch, new_df.PatchLoc):
        new_img[loc] += patch

    # Feedback
    msg = f'Filtering image for circular patches (ElongIndex > {th})'
    print('\n')
    print(f"{msg}\n{'-'*len(msg)}")
    print('Features in: ', len(df.index))
    print('Features out: ', len(new_df.index))
    print('Features removed: ', len(df.index) - len(new_df.index))


    return new_img

def master_denoise(arr, small_feats_size, elongation_threshold):
    # Remove single pixe//very small objects from image
    close_arr = ndimage.binary_closing(arr).astype(int)
    open_arr = ndimage.binary_opening(close_arr).astype(int)

    ## Remove features that are small enough to only be considered noise
    clean = clean_small_feats(open_arr, small_feats_size)

    ## Calculate elongation criteria for all features, return info in df
    img, df = classify_feats(clean)

    ## Build new image from features with satisfactory elongation values
    new_img = build_img(img, df, elongation_threshold)

    return new_img

#############################################################################################
#############################################################################################
#############################################################################################


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
    paths = [i for i in paths if f"{win}px" in i.stem.split('_')]

    # Sort the output for niceness
    spaths = sorted(paths, key = lambda x: x.name.split('_')[2])

    # Return sorted paths
    return spaths


def flip_array(a):

    ones = a == 1
    zeros = a == 0

    a[ones] = 0
    a[zeros] = 1

    return a

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# DATA PROCESSING

# Check for single or batch process intended
if rPath.is_file():
    target = [rPath]
else:
    target = listFilePaths(rPath)

for path in target:

    # Open raster file
    ds = rasterio.open(path)
    profile = ds.profile
    a = ds.read(1)

    # Create mask for later
    corner_mask = np.isnan(a)

    # Remove errant features from the swale areas
    print('--- REMOVE ERRANT FEATURES FROM SWALE AREAS ---')

    ## Remove errant objects from swales
    a = master_denoise(a, small, et)

    # Remove errant features from the ridge areas
    print('--- REMOVE ERRANT FEATURES FROM RIDGE AREAS ---')

    # Flip ones and zeros; leaves NaNs alone
    a = flip_array(a)

    # a[cmask] = 0

    # Remove artifacts originally classified as non-ridge
    a = master_denoise(a, small, et)

    # Flip array back
    a = flip_array(a)

    # Mask out the channel and cliped regions of the raster
    a[corner_mask] = np.nan
    # a[ch_mask] = np.nan

    # Write denoised image to disk
    outname = '_'.join([path.stem, 'dn', f'Buff{int(buff)}m', f'SmFt{int(small)}m', f'ET{int(100*et)}p'])
    # outpath = path.parents[1] / f"{path.parts[-2]}-denoise-nobuff" / f"{outname}.tif"
    outpath = outdir / f"{outname}.tif"

    if not outpath.parent.exists():
        outpath.parent.mkdir(parents=True)

    with rasterio.open(outpath, 'w', **profile) as dst:
        dst.write(a, 1)

    print('Wrote binclass to disk: ', outpath)

print('')
print(f"Denoising Complete! Check `{outpath.parent}` for denoised rasters")
