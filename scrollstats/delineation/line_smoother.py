# This script takes rough hand-drawn ridges as input and applies a mean filter
# cublic spline to the coordinates to both smooth and densify these lines.

import sys
from pathlib import Path
import numpy as np
from shapely.geometry import LineString
import geopandas as gpd
from scipy.interpolate import CubicSpline

# from vector_funcs import meanfilt, calc_cubic_spline

# Read in Data
in_path = Path(sys.argv[1])


# Specify window size (in number of coordinates) for mean filter
if len(sys.argv) > 2:
    window = float(sys.argv[2])
else:
    window = 5

# Specify spacing between points for the cubic spline
if len(sys.argv) > 3:
    spacing = float(sys.argv[3])
else:
    spacing = 1

# Define output path
if len(sys.argv) > 4:
    out_dir = Path(sys.argv[4])
else:
    out_dir = Path('./output')



#########################
# From vector_funcs.py
def meanfilt(line, w):
    '''
    use a mean filter to smooth the xy points of the line
    this particular method appends the first and last coord to the new line to account for erosion via convolution
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

# Calc distance along the line
def GetS(x,y):
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

## Spline function
# Basically fitting a function to explain the change in distance over x and y independently

def calc_cubic_spline(line, spacing):
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



# Create gdf of ridge file
ridges = gpd.read_file(in_path)

# Apply mean filter to lines
ridges.geometry = ridges.geometry.apply(lambda x: meanfilt(x, window))

# Apply Cubic Spline to densify and further smooth ridgelines
ridges.geometry = ridges.geometry.apply(calc_cubic_spline, spacing=spacing)


# Write new lines to disk
out_name = '_'.join([in_path.stem, f"smoothed_mf{window}_cs{str(spacing).replace('.', 'p')}"])
out_path = out_dir / f"{out_name}.json"

# Create the out dir if it does not already exist
if not out_path.parent.exists():
    out_path.parent.mkdir(parents=True)

ridges.to_file(out_path, driver='GeoJSON')

print(f"Completed smoothing lines. Check `{out_dir.resolve()}` for outputs.")
