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
VERTEX_SPACING = 1         # Distance between densified vertices; Measured in linear unit of dataset (meters for example datasets)

# Migration Pathway
SHOOT_DISTANCE = 300   # Distance that the N1 coordinate will shoot out from point P1; measured in linear unit of dataset
SEARCH_DISTANCE = 200  # Buffer radius used to search for an N2 coordinate on R2; measured in linear unit of dataset
DEV_FROM_90 = 5        # Max angular deviation from 90° allowed when searching for an N2 coordinate on R2; measured in degrees

# Set file paths
bend_area_path = Path("example_data/input/LBR_025_bend.geojson")       # Polygon containing the ridge and swale topography
packet_path = Path("example_data/input/LBR_025_packets.geojson")       # Polygons that divide the bend area into depositional packets
ridge_path = Path("example_data/input/LBR_025_ridges_manual.geojson")  # Polylines marking the location of ridges
centerline_path = Path("example_data/input/LBR_025_cl.geojson")        # Polyline marking the center of the channel 
output_dir = Path("example_data/output")
if not output_dir.is_dir():
    output_dir.mkdir()

##############################
# Plot manually digitized data

bend_area = gpd.read_file(bend_area_path).set_index("bend_id")
packets = gpd.read_file(packet_path).set_index("packet_id")
manual_ridges = gpd.read_file(ridge_path)
cl = gpd.read_file(centerline_path) # Centerline is already smoothed and densified

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

packets.boundary.plot(color="grey", ax=ax, label="Packet Boundaries")
packets.plot(color="grey", alpha=0.2, ax=ax)
bend_area.boundary.plot(color="k", lw=3, ax=ax, label="Bend Boundary")
manual_ridges.plot(color="k", ls=":", ax=ax, label="Ridges")
cl.plot(color="k", ls="--", lw=2, ax=ax, label="Centerline")

ax.legend(loc="upper left")
ax.set_axis_off()
ax.set_title("Manually Generated Datasets")


#################
# Ridge Smoothing

# Smooth and densify manual ridge lines for transect creation
ls = LineSmoother(manual_ridges, VERTEX_SPACING, SMOOTHING_WINDOW_SIZE)
smooth_ridges = ls.execute()

# Save smooth ridges to disk
smooth_ridge_name = ridge_path.with_stem(ridge_path.stem + "_smoothed").name
smooth_ridge_path = output_dir / smooth_ridge_name
smooth_ridges.to_file(smooth_ridge_path, driver="GeoJSON", index=False)

# Plot manual and smoothed ridges
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.set_aspect("equal")

manual_ridges_points = manual_ridges.geometry.apply(lambda x: MultiPoint(x.coords))
smooth_ridges_points = smooth_ridges.geometry.apply(lambda x: MultiPoint(x.coords))

manual_ridges.plot(ax=ax, color="grey", lw=3, zorder=1, label="Manual Ridges")
manual_ridges_points.plot(ax=ax, color="black", markersize=30, zorder=2)

smooth_ridges.plot(
    ax=ax, color="red", lw=1.5, alpha=0.6, zorder=3, label="Smoothed Ridges"
)
smooth_ridges_points.plot(ax=ax, color="red", markersize=20, alpha=0.5, zorder=4)

# Create scalebar
x_sb = 1067860
y_sb = 3111455
len_sb = 10
ax.plot((x_sb, x_sb + len_sb), (y_sb, y_sb), lw=3, color="black")
ax.text(x=x_sb + len_sb / 2, y=y_sb - 2, s=f"{len_sb}m", horizontalalignment="center")

ax.set_ylim(3111450, 3111490)
ax.set_xlim(1067820, 1067880)
ax.legend(loc="upper left")
ax.set_axis_off()
ax.set_title("Smoothed vs Manual Ridges")


###########################
# Create Migration Pathways

# Define the desired distance between each migration pathway along the centerline; measured in centerline vertices
# Centerline from example data has a vertex spaceing of 1m
step = 100

# Migration Pathway Algorithm 
## 1. Let the channel centerline be called R1
## 2. Select a starting point on R1, let this point be called P1
## 3. From P1, shoot a given distance perpendicular from R1 in the direction of the convex bank.
## 4. If the line intersects a ridge, call this ridge R2 and proceed to step 5. If not, move on to the next location on R1 and repeat step 2.
## 5. Where this new line intersects R2, make a new point and call it N1
## 6. Buffer N1 by a given radius and search all vertices of R2 that intersect this buffer for a point from which a line may be drawn back to P1 that is perpendicular to R2. Call this new point on R2, N2.
## 7. Treating the lines P1->N1 and P1->N2 as vectors, calculate their vertical resultant and place a new point at the end of this vertical resultant. Let this point be called VR.
## 8. Where the line P1->VR intersects R2, place a point and let this point be called P2. Line P1->P2 is the migration pathway for this location on the floodplain.
## 9. Redefine P2 as P1
## 10. Repeat steps 3-9 until the perpendicular shot from P1 fails to intersect any ridges.

# This function iteratively creates a series of migration pathways using the algorithm above
transects = create_transects(
    centerline=cl,                      # centerline (GeoDataFrame) from which to start transects
    ridges=smooth_ridges,               # ridge features (GeoDataFrame) that the transects will navigate through
    step=step,                          # distance between transect start locations
    shoot_distance=SHOOT_DISTANCE,      # distance that the migration pathway will shoot from the start ridge (R1) to intersect the next ridge (R2)
    search_distance=SEARCH_DISTANCE,    # buffer distance used to search vertices of R2 for point N2
    dev_from_90=DEV_FROM_90             # allowed deviance from a 90 degree angle for point N2
)

# Save transects to disk
transect_path = output_dir / "LBR_025_transects.geojson"
transects.to_file(transect_path, driver="GeoJSON", index=True)

# Plot migration pathways alongside ridges and centerlines
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

cl.plot(color="k", lw=2, ax=ax, zorder=0, label="Centerline")
smooth_ridges.plot(color="k", ls=":", ax=ax, zorder=1, label="Ridges")
transects.plot(color="r", ax=ax, zorder=2, label="Migration Pathways")

# Plot migration pathway start points 
starts = np.asarray(cl.geometry[0].xy).T[::step]
ax.scatter(starts[:, ::2], starts[:, 1::2], color="r", zorder=3)

ax.legend()
ax.set_axis_off()
ax.set_title("Migration Pathways Generated")

plt.show()