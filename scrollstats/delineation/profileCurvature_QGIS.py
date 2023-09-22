# For help:
# processing.algorithmHelp('grass7:r.param.scale')

import processing
from pathlib import Path


# Define input/output locations
# dem_dir = Path("path/to/data/input/dem")
# output_dir = Path("path/to/data/output/profc")

dem_dir = Path("/Users/avan/FLUD/scrollstats/data/input/dem")
output_dir = Path("/Users/avan/FLUD/scrollstats/data/output/profc")

# Define window_size (set in ______.txt)
window_size = 45

## regex-like pattern matching to filter dems; 
## leave as "*" to match all files in `dem_dir`
regex_str = "sb_5*.tif"

## function used to sort the dems within `dem_dir`
## leave as is to use default filename sorting
def sort_func(x):
    return x

# Create a list of dem paths 
dem_paths = sorted(dem_dir.glob(regex_str), key=sort_func)


######################################

for dem in dem_paths:

    # Set output names for r.param.scale
    temp_output_name = f"TEMP_{dem.stem}.tif"
    temp_output_path = output_dir / temp_output_name

    # Calculate profile curvature
    profc = processing.run('grass7:r.param.scale',{
    'input': str(dem),
    'size': window_size,
    'method': 3,
    'output': str(temp_output_path)
    })

    # Set output names for gdal.warpreproject   
    output_name = f'{dem.stem}_profc{window_size}px.tif'
    output_path = output_dir / output_name

    # Reproject the unprojected output of r.param.scale
    warp = processing.run('gdal:warpreproject',{
    'INPUT': profc['output'],
    'SOURCE_CRS':str(dem),
    'TARGET_CRS':str(dem),
    'OUTPUT': str(output_path)
    })


# Delete temp files from r.param.scale
temp_files = output_dir.glob("TEMP_*")
for file in temp_files:
    name = file.name
    file.unlink()
    print(f"Auto Deleted: {name}")

