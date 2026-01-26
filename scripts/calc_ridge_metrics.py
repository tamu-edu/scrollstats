from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio

from scrollstats import calculate_ridge_metrics

# Set input/output paths
## Vector Datasets
ridge_path = Path("example_data/output/LBR_025_ridges_manual_smoothed.geojson")
transect_path = Path("example_data/output/LBR_025_transects.geojson")
packet_path = Path("example_data/input/LBR_025_packets.geojson")
centerline_path = Path("example_data/input/LBR_025_cl.geojson")

## Raster Datasets
bin_clip_path = Path("example_data/output/LBR_025_dem_ridge_area_raster.tif")
dem_clip_path = Path("example_data/output/LBR_025_dem_clip.tif")

## Output
output_dir = Path("example_data/output")
if not output_dir.is_dir():
    output_dir.mkdir()


# Read in datasets
## Vector Data
ridges = gpd.read_file(ridge_path)
transects = gpd.read_file(transect_path)
packets = gpd.read_file(packet_path)
cl = gpd.read_file(centerline_path)

## Raster Data
bin_raster = rasterio.open(bin_clip_path)
dem = rasterio.open(dem_clip_path)


# Calculate ridge metrics
## This function calculates the ridge metrics (amplitude, width, and spacing) at the intersection of every transect and ridge.
## Two GeoDataFrames are returned: 
### `rich_transects` contains the transect geometry (LineString) as well as the elevation and binary arrays sampled along the transects plus other values calculated for the entire transect
### `itx` contains all ridge-transect intersection Points as well as their ridge metrics plus other intermediate values 
rich_transects, itx = calculate_ridge_metrics(transects, ridges, bin_raster, dem)

## Index the bend_id to effectively remove it from the dataframe for future convenience
itx = itx.loc["LBR_025"] 

## Spatial join packet info (only packet_id for example data) to itx points for potential inter/intra-packet analysis 
## packet_id, like ridge_id, is incremental and increases with distance from the channel
itx_w_packets = itx.sjoin(packets.drop("bend_id", axis=1))
itx_w_packets = itx_w_packets.reset_index().set_index(
    ["transect_id", "ridge_id", "packet_id"]
)


# Plot ridge amplitudes at intersections
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

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
ax.set_axis_off()

plt.show()