from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio

from scrollstats import calculate_ridge_metrics

########################################

# Set input/output paths
# Vector Datasets
ridge_path = Path("example_data/output/LBR_025_ridges_manual_smoothed.geojson")
transect_path = Path("example_data/output/LBR_025_transects.geojson")
packet_path = Path("example_data/input/LBR_025_packets.geojson")
centerline_path = Path("example_data/input/LBR_025_cl.geojson")

# Raster Datasets
bin_clip_path = Path("example_data/output/LBR_025_dem_ridge_area_raster.tif")
dem_clip_path = Path("example_data/output/LBR_025_dem_clip.tif")

# Output
output_dir = Path("example_data/output")

# Image variables
img_dir = Path("img")
dpi = 100

########################################

# Read in datasets

# Vector Data
ridges = gpd.read_file(ridge_path)
transects = gpd.read_file(transect_path)
packets = gpd.read_file(packet_path)
cl = gpd.read_file(centerline_path)

# Raster Data
bin_raster = rasterio.open(bin_clip_path)
dem = rasterio.open(dem_clip_path)

########################################

# Calculate ridge metrics
rich_transects, itx = calculate_ridge_metrics(transects, ridges, bin_raster, dem)
itx = itx.loc["LBR_025"]

# Join packet info
itx_w_packets = itx.sjoin(packets.drop("bend_id", axis=1))
itx_w_packets = itx_w_packets.reset_index().set_index(
    ["transect_id", "ridge_id", "packet_id"]
)
ridge_metrics_w_packets = itx_w_packets[["ridge_amp", "ridge_width", "pre_mig_dist"]]
ridge_metrics_w_packets.columns = ridge_metrics_w_packets.columns.rename("metrics")

########################################

# Plot ridge amplitudes at intersections
# Plot itx
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

ridges.plot(ax=ax, color="k", ls="--", lw=0.5, zorder=0)
transects.plot(ax=ax, color="k", lw=1, zorder=1)
cl.plot(ax=ax, color="tab:blue", lw=5, zorder=2)

itx_w_packets.plot(
    column="ridge_amp",
    ax=ax,
    zorder=2,
    legend=True,
    legend_kwds={"label": "Ridge Amplitude [m]"},
)

ax.set_title("Ridge amplitude at each intersection")

plt.tight_layout()
plt.savefig(img_dir/"ridge_amplitude.png", dpi=dpi)