from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import MultiPoint

from scrollstats import LineSmoother, create_transects

# User defined parameters
# LineSmoother
SMOOTHING_WINDOW_SIZE = 5  # Measured in vertices
VERTEX_SPACING = 1  # Distance between densified vertices; Measured in linear unit of dataset (meters for example datasets)

# Migration Pathway
SHOOT_DISTANCE = 300  # Distance that the N1 coordinate will shoot out from point P1; measured in linear unit of dataset
SEARCH_DISTANCE = 200  # Buffer radius used to search for an N2 coordinate on R2; measured in linear unit of dataset
DEV_FROM_90 = 5  # Max angular deviation from 90° allowed when searching for an N2 coordinate on R2; measured in degrees

output_dir = Path("example_data/output")
if not output_dir.exists():
    output_dir.mkdir(parents=True)

# Image variables
img_dir = Path("img")
dpi = 100

########################################
bend_area = gpd.read_file("example_data/input/LBR_025_bend.geojson").set_index(
    "bend_id"
)
packets = gpd.read_file("example_data/input/LBR_025_packets.geojson").set_index(
    "packet_id"
)

fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

packets.boundary.plot(color="grey", ax=ax, label="Packet Boundaries")
packets.plot(color="grey", alpha=0.5, ax=ax)
bend_area.boundary.plot(color="k", lw=3, ax=ax, label="Bend Boundary")

ax.legend(loc="upper left")
ax.set_axis_off()

plt.tight_layout()
plt.savefig(img_dir / "bend_geometry.png", dpi=100)

########################################
ridge_path = Path("example_data/input/LBR_025_ridges_manual.geojson")
manual_ridges = gpd.read_file(ridge_path)

# Centerline is already smoothed and densified
cl_path = Path("example_data/input/LBR_025_cl.geojson")
cl = gpd.read_file(cl_path)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

manual_ridges.plot(color="k", ls=":", ax=ax, label="Ridges (Manual)")
cl.plot(color="k", lw=2, ax=ax, label="Centerline")

ax.legend(loc="upper left")
ax.set_axis_off()

plt.tight_layout()
plt.savefig(img_dir / "ridges_and_centerline.png", dpi=100)

########################################
# Smooth and densify the lines
ls = LineSmoother(manual_ridges, VERTEX_SPACING, SMOOTHING_WINDOW_SIZE)
smooth_ridges = ls.execute()

# Save smooth ridges to disk
smooth_ridge_name = ridge_path.with_stem(ridge_path.stem + "_smoothed").name
smooth_ridge_path = output_dir / smooth_ridge_name

smooth_ridges.to_file(smooth_ridge_path, driver="GeoJSON", index=False)

# Plot manual and smoothed ridges
fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
ax.set_aspect("equal")

manual_ridges_points = manual_ridges.geometry.apply(lambda x: MultiPoint(x.coords))
smooth_ridges_points = smooth_ridges.geometry.apply(lambda x: MultiPoint(x.coords))

manual_ridges.plot(ax=ax, color="grey", lw=3, zorder=1, label="Manual Ridges")
manual_ridges_points.plot(ax=ax, color="black", markersize=40, zorder=2)
smooth_ridges.plot(
    ax=ax, color="red", lw=1.5, alpha=0.6, zorder=3, label="Smoothed Ridges"
)
smooth_ridges_points.plot(ax=ax, color="red", markersize=15, alpha=0.6, zorder=4)

# scalebar
x_sb = 1067860
y_sb = 3111455
len_sb = 10
ax.plot((x_sb, x_sb + len_sb), (y_sb, y_sb), lw=3, color="black")
ax.text(x=x_sb + len_sb / 2, y=y_sb - 2, s=f"{len_sb}m", horizontalalignment="center")

ax.set_ylim(3111450, 3111490)
ax.set_xlim(1067820, 1067880)
ax.legend(loc="upper left")
ax.set_axis_off()

plt.tight_layout()
plt.savefig(img_dir / "smoothed_ridges.png", dpi=100)

########################################
# define the distance between transects
step = 100

# With a vertex spacing of ~1m, take every `step`th vertex along the centerline
starts = np.asarray(cl.geometry[0].xy).T[::step]

# Generate transects
transects = create_transects(
    cl, smooth_ridges, step, SHOOT_DISTANCE, SEARCH_DISTANCE, DEV_FROM_90
)

# Save transects to disk
transect_path = output_dir / "LBR_025_transects.geojson"
transects.to_file(transect_path, driver="GeoJSON", index=True)

# Plot migration pathways alongside ridges and centerlines
fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

cl.plot(color="k", lw=2, ax=ax, zorder=0, label="Centerline")
smooth_ridges.plot(color="k", ls=":", ax=ax, zorder=1, label="Ridges")
transects.plot(color="r", ax=ax, zorder=2, label="Migration Pathways")
plt.scatter(starts[:, ::2], starts[:, 1::2], color="r", zorder=3)

ax.legend()
ax.set_axis_off()

plt.tight_layout()
plt.savefig(img_dir / "migration_pathways.png", dpi=100)

########################################
