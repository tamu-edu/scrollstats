# This script is responsible for calculating an agreement raster (a binary raster
# which represernts the agreement between two input binary rasters) for a single
# bend and writes it to disk.

# Info on this result is then aggregated by the "batch" script.

import sys, os, pathlib
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt


###############################################################################
# Define Functions

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
    paths = [i for i in paths if i.stem.split('_')[4] == f"{win}px"]

    # Sort the output for niceness
    spaths = sorted(paths, key = lambda x: x.name.split('_')[2])

    # Return sorted paths
    return spaths


# Determine adjustment for the given dimension
# Here adjustment is defined as the value added to the beginning and end of the given dimension
def calc_adjust(a1, a2, dim):

    # used to translate dimension to an axis to aggregate accross
    dim_dict = {'row':1, 'col':0}


    # Determine where each array has solid nans along the given dimension
    a1_nans = np.isnan(a1).all(axis=dim_dict.get(dim))
    a2_nans = np.isnan(a2).all(axis=dim_dict.get(dim))

    sa, la = sorted([a1_nans,a2_nans], key = lambda x: x.shape)

    if la[0] == sa[0]: # Both arrays match at the beginning, so buffering occurs at the end of the array
        adj = 0
    elif la[0]: # NaN buffer at the beginning of la only, so sa needs to be moved one over
        adj = 1
    else:         # This would imply that la has a 2x Nan buffer [... True, True] - unlikely
        print((sa, sa.shape), (la, la.shape))
        print('Something went wrong')

    return adj


def adjust_rasters_v3(a1, a2):

    # Determine the smallest array that can hold both arrays
    mr = np.max([a1.shape[0], a2.shape[0]])
    mc = np.max([a1.shape[1], a2.shape[1]])
    na = np.zeros((mr, mc))*np.nan

    # Calculate the adjustments that need to be made for the array that is smaller in a given dimension
    ## Row adjustment
    adj_r = calc_adjust(a1, a2, 'row')

    ## Col adjustment
    adj_c = calc_adjust(a1, a2, 'col')

#     sa_r, la_r = sorted([a1, a2], key=lambda x: x.shape[0])
    # If an array is smaller in the first dimension:
    #  It needs the row adjustment

    # for both of the input arrays, determine which array needs to be adjusted in which direction before adding it to the canvas na array
    # If an array is smaller in a given dimension, then it is assumed that it needs the adjustment
    new_arrays = []
    for g in [a1, a2]:

        # Empty array
        canvas = na.copy()

        # Is this array both smaller in the first and second dimensions?
        if (g.shape[0]<mr) and (g.shape[1]<mc):
            canvas[0+adj_r : g.shape[0]+adj_r, 0+adj_c : g.shape[1]+adj_c] = g
            new_arrays.append(canvas.copy())

        # Is this array only smaller in the first dimension?
        elif g.shape[0]<mr:
            canvas[0+adj_r : g.shape[0]+adj_r, 0: g.shape[1]] = g
            new_arrays.append(canvas.copy())

        # Is this array only smaller in the second dimension?
        elif g.shape[1]<mc:
            canvas[0 : g.shape[0], 0+adj_c : g.shape[1]+adj_c] = g
            new_arrays.append(canvas.copy())

        # This array must be larger in both dimensions, and therefore does not need adjustment
        else:
            new_arrays.append(g)


    return new_arrays


def writeToDisk(a, r_type, path, outdir):
    '''
    a      : input array that needs to be written to disk
    r_type : type of array that will be written, changes the output path and filename
    path   : path of the raster whose profile will be emulated to write the new array to disk
    '''

    # Get default profile and edit to fit dimensions of `a`
    _ds = rasterio.open(path)
    _profile = _ds.profile
    _profile['width'] = a.shape[1]
    _profile['height'] = a.shape[0]

    # Create outname
    ## Break up name elements of the profc_name
    elem = path.name.split('_')
    ## Redefine the geomorphic parameter element to r_type ('composite' or 'agreement')
    elem[3] = r_type
    ## rejoin elements
    outname = '_'.join(elem)

    # Append outname to the rest of outpath
    outpath = outdir / outname

    # Create the out dir if it does not already exist
    if not outpath.parent.exists():
        outpath.parent.mkdir(parents=True)

    # Write to disk with rasterio
    with rasterio.open(outpath, 'w', **_profile) as dst:
        dst.write(a, 1)

    return outpath

def calcStats(cdir):

    df_rows = []
    # Make a list of all file paths within comp_dir
    path_list = listFilePaths(cdir)
    path_list = [i for i in path_list if 'composite' in i.name]
    print(f"Path list is now filtered to {len(path_list)} records")

    for path in path_list:
        fn = pathlib.Path(path).name
        elem = fn.split('_')
        bendID = '_'.join(elem[0:3])

        # open raster
        ds = rasterio.open(path)
        a = ds.read(1)

        # Get unique values within `a` as well as their counts
        vals, counts = np.unique(a, return_counts=True)

        # Get proportion of real pixels
        props = counts / (a.size - counts[-1]) # counts[-1] is the count of NaNs

        # Get simple labels of values
        if len(vals)==5:
            labels = ['BothNo', 'Profc', 'ResTopo', 'BothYes']
        else:
            print(f"Strange raster values in comp/agreement raster\n{vals}")


        row = {label:prop for label, prop in zip(labels, props)}
        row['BendID'] = bendID

        df_rows.append(row)

    df = pd.DataFrame(df_rows)
    df['TotalAgree'] = df['BothNo'] + df['BothYes']
    df['TotalDisagree'] = df['Profc'] + df['ResTopo']
    df = df.set_index('BendID')

    return df


################################################################################
# Begin Raster Processing

# Read in rasters
pcpath = sys.argv[1] # /Volumes/GoogleDrive/My Drive/FLUD/BrazosScrolls/data/raster/binclass-clip/sb_1_006_profc_45px_binclass_Buff100m_SmFt100m_ET80p_clip.tif
rtpath = sys.argv[2] # /Volumes/GoogleDrive/My Drive/FLUD/BrazosScrolls/data/raster/rtbinclass-clip/sb_1_006_rt_45px_binclass_Buff100m_SmFt100m_ET80p_clip.tif


if len(sys.argv) > 3:
    outdir = pathlib.Path(sys.argv[3])
else:
    outdir = pathlib.Path('./output')

pcPath = pathlib.Path(pcpath)
rtPath = pathlib.Path(rtpath)

if pcPath.is_file() and rtPath.is_file():
    target = [(pcPath, rtPath)]
elif pcPath.is_dir() and rtPath.is_dir():
    target = list(zip(listFilePaths(pcPath), listFilePaths(rtPath)))
else:
    sys.exit(f"Inputs are not complimentary. Both need to be either dirs or indv files. \nGiven: {pcpath} and {rtpath}")
# # Dir for output arrays
# comp_output = sys.argv[3] # /Volumes/GoogleDrive/My Drive/FLUD/BrazosScrolls/data/raster/composite
# agree_output = sys.argv[4] # /Volumes/GoogleDrive/My Drive/FLUD/BrazosScrolls/data/raster/agreement

for pc_path, rt_path in target:
    msg = f"Assessing agreement for bend {pc_path.name.split('_')[2]}"
    print(f"{msg}\n{'-'*len(msg)}")
    # Read in rasters with rasterio and get arrays
    pc = rasterio.open(pc_path)
    rt = rasterio.open(rt_path)

    pca = pc.read(1)
    rta = rt.read(1)

    # The incoming clipped rasters do not always have the same size - likely due to the clipping process.
    # Therefore each set of rasters need to hvae their lengths slightly adjusted in
    # one dimension or another to that they may be easily compared. This is done by
    # finding the smallest possible array footprint that can hold both of the incoming rasters
    # then by adjusting the shape of either or both rasters by adding a row/column of nans
    # to the image.
    if pca.size != rta.size:
        pca, rta = adjust_rasters_v3(pca, rta)

    # Redefine RT foreground values as 10 for comparison later
    rta[rta==1] = 10
    print(f"PROFC Values: {np.unique(pca)}")
    print(f"RT Values: {np.unique(rta)}")

    # Add arrays together to form the composite array
    # Because the RT foreground values were redefined to a value of 10, when individual
    # pixels are added together the values of each digit of a cell comunicate how the
    # different methods agreed (or not), as shown int the confusion matrix below

    #                   | PC Ridge (01)| Not PC Ridge (00)|
    # ------------------|--------------|------------------|
    # RT Ridge     (10) |      11      |        10        |
    # ------------------|--------------|------------------|
    # Not RT Ridge (00) |      01      |        00        |
    # ------------------|--------------|------------------|


    # Create composite array & write to disk
    comp = pca+rta
    print(f"COMP Values: {np.unique(comp)}")
    comp_path = writeToDisk(comp, 'composite', rt_path, outdir)

    # Create agreement array & write to disk
    agr = comp.copy()
    agr[agr==1] = 0 # redefine all disagreement as 0
    agr[agr==10] = 0 # redefine all disagreement as 0
    agr[agr==11] = 1 # Keep all positve agreement as 1
    print(f"AGR Values: {np.unique(agr)}\n")
    agr_path = writeToDisk(agr, 'agreement', rt_path, outdir)


# Calc agreement stats for composite rasters
# save table in both composite and agreement dirs
comp_stats = calcStats(comp_path.parent)

comp_stats.to_csv(comp_path.parent / 'agreementStats.csv')
comp_stats.to_csv(agr_path.parent / 'agreementStats.csv')

# print_stats = 
print(f"Comparison Stats:\n {pd.DataFrame({'Mean': comp_stats.mean(), 'StDev': comp_stats.std()})}")
# print(f"Stand. Deviation: {comp_stats.std()}")
