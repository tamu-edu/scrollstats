# Complete ScrollStats Workflow

This page contains the entire process for creating intersection-scale ridge
measurements (ridge amplitude, width, and spacing) from the required input data
(dem, manual ridge lines, packets, bend area, centerline). This process is the
same process that has been split up across the other 3 pages, but without
context and explanation.

```python
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.mask

from scrollstats import (
    LineSmoother,
    calculate_ridge_metrics,
    create_ridge_area_raster_fs,
    create_transects,
)

# User defined parameters
# Ridge Area Raster
RASTER_WINDOW_SIZE = 45  # kernel size for image processing; measured in px
SMALL_FEATS_SIZE = (
    500  # all features smaller will be removed in denoising process; measured in px^2
)

# LineSmoother
SMOOTHING_WINDOW_SIZE = 5  # Measured in vertices
VERTEX_SPACING = 1  # Distance between densified vertices; Measured in linear unit of dataset (meters for example datasets)

# Migration Pathway
SHOOT_DISTANCE = 300  # Distance that the N1 coordinate will shoot out from point P1; measured in linear unit of dataset
SEARCH_DISTANCE = 200  # Buffer radius used to search for an N2 coordinate on R2; measured in linear unit of dataset
DEV_FROM_90 = 5  # Max angular deviation from 90° allowed when searching for an N2 coordinate on R2; measured in degrees

# Bend ID
bend_id = "LBR_025"

# Raster Paths
dem_path = Path(f"example_data/input/{bend_id}_dem.tif")

# Vector Paths
bend_path = Path(f"example_data/input/{bend_id}_bend.geojson")
packet_path = Path(f"example_data/input/{bend_id}_packets.geojson")
centerline_path = Path(f"example_data/input/{bend_id}_cl.geojson")
manual_ridge_path = Path(f"example_data/input/{bend_id}_ridges_manual.geojson")

# Output Directory
output_dir = Path("example_data/output")

##########################################################################################
# 1. Delineate Ridge Areas

binary_path_out, dem_path_out = create_ridge_area_raster_fs(
    dem_path=dem_path,
    geometry_path=bend_path,
    out_dir=output_dir,
    no_data=np.nan,
    window=RASTER_WINDOW_SIZE,
    dx=1,
    small_feats_size=SMALL_FEATS_SIZE,
)

########################################################################################
# 2. Create Vector Datasets

manual_ridges = gpd.read_file(manual_ridge_path)
cl = gpd.read_file(centerline_path)
packets = gpd.read_file(packet_path).set_index("packet_id")

# Smooth and densify the lines
ls = LineSmoother(manual_ridges, VERTEX_SPACING, SMOOTHING_WINDOW_SIZE)
smooth_ridges = ls.execute()

# Save smooth ridges to disk
output_dir = Path("example_data/output")
smooth_ridge_name = manual_ridge_path.with_stem(
    manual_ridge_path.stem + "_smoothed"
).name
smooth_ridge_path = output_dir / smooth_ridge_name

smooth_ridges.to_file(smooth_ridge_path, driver="GeoJSON", index=False)

# define the distance between transects
step = 100

# With a vertex spacing of ~1m, take every `step`th vertex along the centerline
starts = np.asarray(cl.geometry[0].xy).T[::step]

# Generate Transects
transects = create_transects(
    cl, smooth_ridges, step, SHOOT_DISTANCE, SEARCH_DISTANCE, DEV_FROM_90
)

# Save transects to disk
transect_path = output_dir / f"{bend_id}_transects.geojson"
transects.to_file(transect_path, driver="GeoJSON", index=True)

########################################################################################
# 3. Calculate Ridge Metrics

# Vector Data
ridges = gpd.read_file(smooth_ridge_path)
transects = gpd.read_file(transect_path)
packets = gpd.read_file(packet_path)
cl = gpd.read_file(centerline_path)

# Raster Data
bin_raster = rasterio.open(binary_path_out)
dem = rasterio.open(dem_path_out)
rich_transects, itx = calculate_ridge_metrics(transects, ridges, bin_raster, dem)
itx = itx.loc["LBR_025"]

# Add packets
itx_w_packets = itx.sjoin(packets.drop("bend_id", axis=1))
itx_w_packets = itx_w_packets.reset_index().set_index(
    ["transect_id", "ridge_id", "packet_id"]
)
ridge_metrics_w_packets = itx_w_packets[
    ["ridge_amp", "ridge_width", "pre_mig_dist", "geometry"]
]
ridge_metrics_w_packets.columns = ridge_metrics_w_packets.columns.rename("metrics")

# Save to disk
itx_path = output_dir / f"{bend_id}_intersections.geojson"
ridge_metrics_w_packets.to_file(itx_path, driver="GeoJSON", index=True)
```
