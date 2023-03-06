
"""
General purpose functions used to manipulate geometries or calcualte geometric properties.

Geometries can either be numpy arrays of coordinates or Shapely objects such as Point, LineString, Polygon, etc.
"""

import numpy as np
from shapely.geometry import LineString
from scipy.interpolate import CubicSpline




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


def meanfilt(line: LineString, w:int) -> LineString:
    '''
    Mean filter to smooth the xy points of the line with a window size of w.

    Appends the first and last coord to the new line to account for erosion via convolution
    '''

    mode = 'valid'
    x, y= line.xy

    xm = np.convolve(x, np.ones(w)*1/w, mode=mode)
    xm = np.insert(xm, 0, x[0])
    xm = np.append(xm, x[-1])

    ym = np.convolve(y, np.ones(w)*1/w, mode=mode)
    ym = np.insert(ym, 0, y[0])
    ym = np.append(ym, y[-1])

    return LineString([(x, y) for x, y in zip(xm, ym)])


def GetS(x,y):
    """ Calc distance along the line """

    xdiff = np.ediff1d(x)
    ydiff = np.ediff1d(y)

    return \
        np.cumsum(
            np.insert(
                np.sqrt(
                    np.add(
                        np.power(xdiff,2),
                        np.power(ydiff,2)
                        )
                    ),
                    0,
                    0
                )
            ).tolist()


def calc_cubic_spline(line, spacing):
    """
    Fits a function to explain the change in distance over x and y independently.

    Spacing determines the distance between points
    """
    # Get x,y values from LineString
    x, y = line.xy

    # Calc distance along the line
    s = GetS(x,y)

    # Get the total length of the line
    l = s[-1]

    # Total number of output points
    n = int(l//spacing)

    # Interpolated distances for each output point
    interp_dist = np.linspace(0, l, n+1, endpoint=True)

    # Create spline function of x and y
    cx_func = CubicSpline(s, x)
    cy_func = CubicSpline(s, y)

    # Apply Spline
    cx = cx_func(interp_dist)
    cy = cy_func(interp_dist)

    return LineString(zip(cx, cy))
