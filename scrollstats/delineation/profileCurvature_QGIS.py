# For help:
# processing.algorithmHelp('grass7:r.param.scale')

import processing
from pathlib import Path


# Define input/output locations
dem_path = Path("path/to/example_data/input/LBR_025_dem.tif")
output_dir = Path("path/to/example_data/output/")

# Define window_size (set in parameters.py)
window_size = 45


######################################

# Set output names for r.param.scale
temp_output_name = f"TEMP_{dem_path.stem}.tif"
temp_output_path = output_dir / temp_output_name

# Calculate profile curvature
profc = processing.run('grass7:r.param.scale',{
'input': str(dem_path),
'size': window_size,
'method': 3,
'output': str(temp_output_path)
})

# Set output names for gdal.warpreproject   
output_name = f'{dem_path.stem}_profc{window_size}px.tif'
output_path = output_dir / output_name

# Reproject the unprojected output of r.param.scale
warp = processing.run('gdal:warpreproject',{
'INPUT': profc['output'],
'SOURCE_CRS':str(dem_path),
'TARGET_CRS':str(dem_path),
'OUTPUT': str(output_path)
})


# Delete temp files from r.param.scale
temp_files = output_dir.glob("TEMP_*")
for file in temp_files:
    name = file.name
    file.unlink()
    print(f"Auto Deleted: {name}")

