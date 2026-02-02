from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from scrollstats import create_ridge_area_raster

# User provided parameters
RASTER_WINDOW_SIZE = 45  # kernel width for image processing; measured in px
SMALL_FEATS_SIZE = 500  # minimum feature size for image classification; all features smaller will be removed in denoising process; measured in px^2


# Input Dataset Paths
dem_path = Path("example_data/input/LBR_025_dem.tif")
bend_path = Path("example_data/input/LBR_025_bend.geojson")

# Output directory
output_dir = Path("example_data/output")
if not output_dir.is_dir():
    output_dir.mkdir()

# Read in the DEM
dem_ds = rasterio.open(dem_path)

# Read in the bend area as a GeoDataFrame; then get the Polygon geometry
bend = gpd.read_file(bend_path)
bend_geom = bend.loc[0, "geometry"]

# Delineate ridge areas
## This function applies two independent classification functions to the DEM to identify ridge areas: profile curvature and residual topography
## Both of these functions return 2D float arrays where values greater 0 than indicate the presense of a ridge. A threshold of 0 is applied to both of these float arrays to create binary arrays.
## The union of these two binary arrays is then subject to a denoising process to create the ridge area raster.
## `create_ridge_area_raster` returns the ridge area raster and the clipped DEM as numpy arrays along with the required metadata to write them to disk as tifs with rasterio
ridge_area_raster, dem_clip, clip_meta = create_ridge_area_raster(
    dem_ds=dem_ds,  # input DEM; rasterio.DatasetReader not np array
    geometry=bend_geom,  # bend polygon containing ridge & swale topography
    # Set kwargs for raster delineation functions
    no_data=np.nan,  ## Set no data value for the clipped raster
    window=RASTER_WINDOW_SIZE,  ## Set kernel size for localized operations (profile curvature & residual topography)
    dx=1,  ## Set grid spacing of raster
    small_feats_size=SMALL_FEATS_SIZE,  ## Set min size of image objects to be preserved in image denoising
)

# Write arrays to disk
binary_out_path = output_dir / f"{dem_path.stem}_ridge_area_raster.tif"
dem_out_path = output_dir / f"{dem_path.stem}_clip.tif"

with rasterio.open(binary_out_path, "w", **clip_meta) as dst:
    dst.write(ridge_area_raster, 1)
    print(f"Wrote ridge area raster to disk: {binary_out_path}")

with rasterio.open(dem_out_path, "w", **clip_meta) as dst:
    dst.write(dem_clip, 1)
    print(f"Wrote clipped DEM to disk: {dem_out_path}")


# Plot the delineated areas

## Locate clipped ridge area raster (currently 2D array w/o crs) within DEM
### Get geo coordinates of ridge area raster upperleft corner
affine = clip_meta.get("transform")
ul_clip_geox = affine.xoff
ul_clip_geoy = affine.yoff

### Get img coordinates of DEM array corresponding to ridge area raster upperleft corner
ul_clip_imgx, ul_clip_imgy = ~dem_ds.transform * (ul_clip_geox, ul_clip_geoy)

## Prepare DEM and ridge area raster for the plot
### Clip DEM to ridge area raster envelope
dem_vis = dem_ds.read(1)[
    int(ul_clip_imgy) : int(ul_clip_imgy) + ridge_area_raster.shape[0],
    int(ul_clip_imgx) : int(ul_clip_imgx) + ridge_area_raster.shape[1],
]

### Make all ridge area raster values transparent except for ridge areas
ridge_area_raster_vis = ridge_area_raster.copy()
ridge_area_raster_vis[ridge_area_raster_vis == 0] = np.nan

## Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.imshow(dem_vis, cmap="Greys_r")
ax.imshow(ridge_area_raster_vis, cmap="viridis_r")

ax.set_title("Delineated Ridge Areas")
ax.set_axis_off()
plt.show()
