
"""
General purpose functions used to manipulate geometries or calcualte geometric properties.

Geometries can either be numpy arrays of coordinates or Shapely objects such as Point, LineString, Polygon, etc.
"""

import numpy as np
from shapely.geometry import LineString




def calc_dist(p1: np.array, p2: np.array) -> np.array:
    """p1 and p2 are both (n,2) arrays of coordinates"""
    return np.sqrt((p1[:, 0] - p2[:, 0]) ** 2 + (p1[:, 1] - p2[:, 1]) ** 2)


def remove_coincident_points(coord_array: np.array) -> np.array:
    """Removes coincident points that happen to be adjacent"""
    
    # Calc distances between each point
    dist = calc_dist(coord_array[:-1], coord_array[1:])
    
    # Find where these distances are not near zero
    unique_idx = np.nonzero(dist>0.001)[0] + 1     # Add one to the index b/c first point is guaranteed ok
    
    return coord_array[unique_idx]


def explode(line):
    """Return a list of all 2 coordinate line segments in a given LineString"""
    return list(map(LineString, zip(line.coords[:-1], line.coords[1:])))


def densify_segment(line):
    """Densify a line segment between two points to 1pt/m. Assumes line coordinates are in meters """
    x, y = line.xy
    
    num_points = int(round(line.length))

    xs = np.linspace(*x, num_points+1)
    ys = np.linspace(*y, num_points+1)
    
    return LineString(zip(xs,ys))


def densify_line(line):
    """Return the given LineString densified coordinates. Density = 1pt/unit length"""

    all_points = []
    
    # Break each line into its straight segments
    for seg in explode(line):
        
        # Densify segment and get coords
        coords = densify_segment(seg).coords
        
        # Append all coords to all_points bucket
        for coord_pair in coords:
            all_points.append(coord_pair)

    # Remove all coincident AND adjacent points
    unique_points = remove_coincident_points(np.array(all_points))
    
    return LineString(unique_points)


def transform_coords(coord_array, bin_raster):
    """Transform the coordinates from geo to image for indexing a in_bin_raster"""
    
    transform = bin_raster.transform
    t_coords = [~transform * coord for coord in coord_array]
    
    return np.round(t_coords).astype(int)  # round coords for indexing
