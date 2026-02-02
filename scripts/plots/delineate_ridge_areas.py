# Generates plots for docs

from functools import partial
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy import ndimage

from scrollstats.delineation.raster_classifiers import (
    quadratic_profile_curvature,
    residual_topography,
)
from scrollstats.delineation.raster_denoisers import remove_small_feats_w_flip
from scrollstats.delineation.ridge_area_raster import clip_raster

# Input Paths
dem_path = Path("example_data/input/LBR_025_dem.tif")
bend_path = Path("example_data/input/LBR_025_bend.geojson")

# Output Directory
output_dir = Path("example_data/output")

# Image variables
img_dir = Path("img")
dpi = 100


# User provided parameters
RASTER_WINDOW_SIZE = 45  # kernel size for image processing; measured in px
SMALL_FEATS_SIZE = 500   # all features smaller will be removed in denoising process; measured in px^2

# Open the DEM dataset
dem_handle = rasterio.open(dem_path)
dem = dem_handle.read(1)

# Find no data pixels
if np.isnan(dem_handle.nodata):
    no_data_mask = np.isnan(dem_handle.read(1))
else:
    no_data_mask = dem == dem_handle.nodata

# Set no data values to absurd integer to avoid errors from the classifier functions with infinite values
dem[no_data_mask] = -99999

# Calculate the transformation rasters
profc = quadratic_profile_curvature(dem, RASTER_WINDOW_SIZE, dx=1)
rt = residual_topography(dem, RASTER_WINDOW_SIZE)

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

mapper = ax1.imshow(profc, vmin=-0.05, vmax=0.05, cmap="seismic")
fig.colorbar(mapper, ax=ax1)
ax1.set_axis_off()
ax1.set_title("Profile Curvature")

mapper = ax2.imshow(dem, vmin=60)
fig.colorbar(mapper, ax=ax2)
ax2.set_axis_off()
ax2.set_title("Original DEM")

mapper = ax3.imshow(rt, vmin=-2, vmax=2, cmap="seismic")
fig.colorbar(mapper, ax=ax3)
ax3.set_axis_off()
ax3.set_title("Residual Topography")

plt.tight_layout()
plt.savefig(img_dir/"pc_v_dem_v_rt.png", dpi=dpi)



# classify each transformed raster from step 1
profc_bc = profc > 0
rt_bc = rt > 0

# Plot the classification results
## 00 - no ridge
## 01 - profile curvature
## 10 - residual topography
## 11 - both
profc_vis = np.zeros(profc.shape)
profc_vis[profc_bc] = 1

rt_vis = np.zeros(rt_bc.shape)
rt_vis[rt_bc] = 10

comp_vis = profc_vis + rt_vis

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
cmap = ListedColormap(["k", "dodgerblue", "yellow", "limegreen"])
bounds = np.array([0, 1, 10, 11, 12])
norm = BoundaryNorm(bounds, cmap.N)
img = ax.imshow(comp_vis, cmap=cmap, norm=norm)
ax.set_axis_off()
ax.set_title("Binary Classification Results")

cbar = plt.colorbar(img)
cbar.set_ticks([0.5, 5.5, 10.5, 11.5], minor=False)
cbar.ax.set_yticklabels(
    ["No ridge", "Profile\nCurvature", "Residual\nTopography", "Agreement"]
)

plt.tight_layout()
plt.savefig(img_dir/"binary_classification.png", dpi=dpi)


# Find the union of the two binary rasters
agr = profc_bc & rt_bc

# Plot
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
cmap = ListedColormap(["k", "limegreen"])
bounds = np.array([0, 1, 2])
norm = BoundaryNorm(bounds, cmap.N)
img = ax.imshow(agr, cmap=cmap, norm=norm)
ax.set_axis_off()
ax.set_title("Binary Classification Agreement")

cbar = plt.colorbar(img)
cbar.set_ticks([0.5, 1.5], minor=False)
cbar.ax.set_yticklabels(["No ridge", "Ridge\nAgreement"])

plt.tight_layout()
plt.savefig(img_dir/"agreement_raster.png", dpi=dpi)


bend = gpd.read_file(bend_path)
bend_geom = bend.loc[0, "geometry"]

# Clip the DEM and agreement raster by assigning all pixels outside the bend geometry to np.nan
dem_clip, dem_mask, dem_meta = clip_raster(dem_handle, bend_geom, no_data=np.nan)
agr_clip, agr_mask, agr_meta = clip_raster(
    dem_handle, bend_geom, array=agr, no_data=np.nan
)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.5))

mapper = ax1.imshow(dem_clip)
ax1.set_axis_off()
ax1.set_title("Clipped DEM")

agr_clip_vis = agr_clip.astype(float)
agr_clip_vis[agr_mask] = np.nan

mapper = ax2.imshow(agr_clip_vis)
ax2.set_axis_off()
ax2.set_title("Clipped Agreement Raster")

plt.tight_layout()
plt.savefig(img_dir/"dem_v_agreement.png", dpi=dpi)



denoiser_funcs = [
    ndimage.binary_closing,
    ndimage.binary_opening,
    partial(remove_small_feats_w_flip, small_feats_size=SMALL_FEATS_SIZE),
]

# Denoise the binary array
ridge_area_array = agr_clip.copy()
for func in denoiser_funcs:
    ridge_area_array = func(ridge_area_array)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.5))

mapper = ax1.imshow(agr_clip_vis)
ax1.set_axis_off()
ax1.set_title("Agreement")

ridge_area_array_vis = ridge_area_array.astype(float)
ridge_area_array_vis[agr_mask] = np.nan
mapper = ax2.imshow(ridge_area_array_vis)
ax2.set_axis_off()
ax2.set_title("Denoised")

plt.tight_layout()
plt.savefig(img_dir/"agreement_v_denoised.png", dpi=dpi)
