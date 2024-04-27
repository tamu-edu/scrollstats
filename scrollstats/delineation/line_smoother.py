import numpy as np
from shapely.geometry import LineString
from geopandas import GeoDataFrame
from scipy.interpolate import CubicSpline


class LineSmoother:
    """
    Smooth and densify rough, manually drawn LineStrings.

    Smoothing is accomplished with the use of a mean filter and densifying is accomplished with the use of a piecewise cubic spline.
    The GeoDataFrame provided must only contain LineStrings. MuliLineStrings or other geometries are not supported.
    The vertex count of any ridge cannot be lower than the window size for the mean filter

    Values used for the Lower Brazos Ridges were:
        window = 5 (vertices)
        spacing = 1 (meters)
    """

    def __init__(self, lines: GeoDataFrame, spacing: int, window: int) -> None:
        self.lines = lines
        self.spacing = spacing
        self.window = window

        # Preform checks on inputs
        self.check_geometry_type()
        self.check_vertex_count()

    def check_geometry_type(self):
        """Check that all geomerties are of type LineString"""
        if any(self.lines.geom_type != "LineString"):
            raise ValueError(
                "Not all geometries are of type LineString. Remove the non-LineString geometry from `lines`"
            )

    def check_vertex_count(self):
        """Check that all ridges have at least as many vertices as the smoothing window is long"""
        if any(self.lines.geometry.apply(lambda x: len(x.coords)) < self.window):
            raise ValueError(
                f"One or more ridges have fewer vertices than the smoothing window is long (window={self.window}). Remove these ridges or add more vertices to them."
            )

    def meanfilt(self, line: LineString, w: int) -> LineString:
        """
        Use a mean filter to smooth the xy points of the line.
        This is done by passing a moving window with size `w` over the x and y coordinates separately and replacing the central value of the window with the mean value of the window.
        This particular method appends the first and last coord to the new line to account for erosion via convolution
        """

        mode = "valid"
        x, y = line.xy

        xm = np.convolve(x, np.ones(w) * 1 / w, mode=mode)
        xm = np.insert(xm, 0, x[0])
        xm = np.append(xm, x[-1])

        ym = np.convolve(y, np.ones(w) * 1 / w, mode=mode)
        ym = np.insert(ym, 0, y[0])
        ym = np.append(ym, y[-1])

        return LineString([(x, y) for x, y in zip(xm, ym)])

    def calc_dist(self, x, y):
        """Calc distance along the line"""
        xdiff = np.ediff1d(x)
        ydiff = np.ediff1d(y)

        return np.cumsum(
            np.insert(np.sqrt(np.add(np.power(xdiff, 2), np.power(ydiff, 2))), 0, 0)
        ).tolist()

    def calc_cubic_spline(self, line: LineString, spacing: int) -> LineString:
        """
        Fit a cubic spline function to a LineString then sample that function at the given `spacing`
        """

        # Get x,y values from LineString
        x, y = line.xy

        # Calc distance along the line
        s = self.calc_dist(x, y)

        # Get the total length of the line
        l = s[-1]

        # Total number of output points
        n = int(l // spacing)

        # Interpolated distances for each output point
        interp_dist = np.linspace(0, l, n + 1, endpoint=True)

        # Create spline function of x and y
        cx_func = CubicSpline(s, x)
        cy_func = CubicSpline(s, y)

        # Apply Spline
        cx = cx_func(interp_dist)
        cy = cy_func(interp_dist)

        return LineString(zip(cx, cy))

    def execute(self) -> GeoDataFrame:
        """
        Apply the mean filter and cubic spline to each line in the geodataframe.
        Return a new geodataframe with the smooth lines
        """
        # Create a copy of the lines GeoDataFrame
        out_lines = self.lines.copy()

        # Smooth points with mean filter
        out_lines.geometry = out_lines.geometry.apply(
            lambda x: self.meanfilt(x, self.window)
        )

        # Densify points with cubic spline
        out_lines.geometry = out_lines.geometry.apply(
            lambda x: self.calc_cubic_spline(x, self.spacing)
        )

        return out_lines
