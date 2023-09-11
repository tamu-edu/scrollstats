# This script is designed to clip a provided raster to the provided bounds.
# All areas outside of the polgon bounds are given an output value of `np.nan`



import sys
import pathlib
import numpy as np
import rasterio
import rasterio.mask
import geopandas as gpd


# INPUTS
raster_path = pathlib.Path(sys.argv[1])
vec_path = pathlib.Path(sys.argv[2])

if len(sys.argv) > 3:
    out_dir = pathlib.Path(sys.argv[3])
else:
    out_dir = pathlib.Path('./output')        # Set out_dir to current directory as default


# PROCESSING
gdf = gpd.read_file(vec_path)
poly = gdf.geometry

with rasterio.open(raster_path) as src:
    out_img, out_transform = rasterio.mask.mask(src, poly, nodata=np.nan, crop=True)
    out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                     "height": out_img.shape[1],
                     "width": out_img.shape[2],
                     "transform": out_transform,
                     "nodata": np.nan})

out_name = f"{raster_path.stem}_clip.tif"
out_path = out_dir / out_name

# Create the out_dir if it does not already exist
if not out_path.parent.exists():
    out_path.parent.mkdir(parents=True)

# Write to disk with rasterio
with rasterio.open(out_path, 'w', **out_meta) as dst:
    dst.write(out_img)

# Feedback
print(f"Clipping raster complete! Check {out_dir.resolve()} for clipped raster.")

