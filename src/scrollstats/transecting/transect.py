from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from tqdm import tqdm


## Simple Geometric functions: accepts and returns arrays of coordinates
def curvature(line: LineString) -> tuple[np.ndarray[float], np.ndarray[float]]:
    x, y = line.xy

    # Get dx, dy
    dx = np.ediff1d(x)
    dy = np.ediff1d(y)

    # Calculate angle between the origin and the point (alpha)
    a = np.arctan2(dy, dx)

    # Calc change in alpha
    da = np.ediff1d(a)
    da[da > np.pi] = (
        da[da > np.pi] - 2 * np.pi
    )  # Is this to change the sign of the angle?
    da[da < -1.0 * np.pi] = da[da < -1.0 * np.pi] + 2 * np.pi

    # Calc arc length
    xydist = np.power(np.add(np.power(dx, 2), np.power(dy, 2)), 0.5)
    arc = np.convolve(xydist, np.ones(2), mode="valid")

    # Calc curvature
    curv = np.divide(da, arc)
    curv = np.append(np.insert(curv, 0, 0), 0)

    # Pad arrays to make them the same shape
    a = np.insert(a, 0, 0)
    da = np.append(np.insert(da, 0, 0), 0)
    xydist = np.insert(xydist, 0, 0)

    return curv, a


def calc_dist_along_line(line: LineString) -> np.ndarray[float]:
    """Calculate an array of distance along a given line"""

    x, y = line.xy

    dx = np.ediff1d(x)
    dy = np.ediff1d(y)

    arc_length = np.insert(np.sqrt(dx**2 + dy**2), 0, 0)

    return np.cumsum(arc_length)


def calc_dist(p1: np.ndarray[float], p2: np.ndarray[float]) -> np.ndarray[float]:
    # p1 and p2 are both (n,2) arrays of coordinates
    return np.sqrt((p1[:, 0] - p2[:, 0]) ** 2 + (p1[:, 1] - p2[:, 1]) ** 2)


def law_of_cos(
    a: np.ndarray[float], b: np.ndarray[float], c: np.ndarray[float]
) -> np.ndarray[float]:
    """
    This function uses law of cosines to calc the interior angle formed by two connected lines
    Traveling down the line, vertices are a, b, c (a=beg, b=mid, c=end)
    Distances between vertices are represented with '_' ex. a_b is the distance between a and b
    """

    # Calculate distances between points
    a_b = calc_dist(a, b)
    b_c = calc_dist(b, c)
    c_a = calc_dist(c, a)

    # beta will be the interior angle at point b
    beta = np.arccos((a_b**2 + b_c**2 - c_a**2) / (2 * a_b * b_c))

    return beta


def calc_dxdy(angle: np.ndarray[float], length: float = 100) -> np.ndarray[float]:
    """Calculates the delta x and y given an angle and a length"""

    dx = np.cos(angle) * length
    dy = np.sin(angle) * length

    return np.vstack([dx, dy])


def add_sub_90(
    a: Point, b: Point, alpha: float, length: float = 100
) -> np.ndarray[float]:
    """
    This function determines which direction to shoot a new transect (+ or - 90°)
    The proper line should have the largest of the interior angles formed by the incoming transect and new shoot being formed
    This function returns the coordinate representing the end point of the proper line

    Args:
     - a: coord before itx
     - b: coord of itx
     - alpha: alpha of the ridge at itx (float)
    """

    # Plus 90
    angle = alpha + np.pi / 2
    dxdy = calc_dxdy(angle, length)
    pos_coord = (b.xy + dxdy).T

    # Minus 90
    angle = alpha - np.pi / 2
    dxdy = calc_dxdy(angle, length)
    neg_coord = (b.xy + dxdy).T

    coords = np.vstack((pos_coord, neg_coord))

    # Get xy coordinates of a and b
    a_xy = np.asarray(a.xy).T
    b_xy = np.asarray(b.xy).T

    # Calculate interior angle of both plus and minus 90
    int_angle = law_of_cos(a_xy, b_xy, coords)

    # Find idx of the larger (max) value
    max_idx = np.flatnonzero(int_angle == int_angle.max())[0]

    # Return coordinate pair with max interior angle
    return coords[max_idx]


def vert_res(p0: Point, p1: Point, p2: Point) -> np.ndarray[float]:
    """Calculates the vertical resultant of the two vectors (LineStrings) p0->p1 and p0->p2"""
    return np.asarray(p1.xy) + np.asarray(p2.xy) - np.asarray(p0.xy)


def find_closest_idx(point: Point, line: LineString) -> int:
    """Loops through the coordinates of `line` and returns the idx of the coord closest to `point`"""

    # List the indices and coordinates of the LineString
    tups = [(i, Point(v)) for i, v in enumerate(line.coords)]

    # Sort the list of tuples by distance to point
    s_tups = sorted(tups, key=lambda tup: point.distance(tup[1]))

    # Return idx of the closest point
    return s_tups[0][0]


def direction_alpha_at_point(point: Point, line: LineString) -> tuple[int, float]:
    """Calculates both the shot direction and alpha value at the coordinate of `line` closest to `point`"""
    curv, alpha = curvature(line)
    idx = find_closest_idx(point, line)

    # Ends of curv and alpha arrays are padded with zeros, take next closest idx instead
    if idx == 0:
        idx += 1
    elif idx == len(line.coords):
        idx -= 1

    direction = int(abs(curv[idx]) / curv[idx])

    return direction, alpha[idx]


########################################################################################################################
########################################################################################################################
########################################################################################################################


class H74Transect:
    """
    Instances of this class are used to retain information (point ID, intermediate values, vertex coordinates, etc.) about a transect being created following the geometric methods described in Hickin 1974.
    Instances of this class are modified by the `H74TransectConstructor` class
    """

    def __init__(self, origin: Point, point_id: str | None = None) -> None:
        # Initial information for the transect
        self.origin = Point(origin)
        self.point_id = point_id

        # Coordinate lists as the transect walks up the floodplain
        self.coord_list: list[Point] = []
        self.coord_list.append(self.origin)
        self.n1_shoot_list: list[Point] = []
        self.n1_coord_list: list[Point] = []
        self.n2_coord_list: list[Point] = []
        self.vr_shoot_list: list[Point] = []
        self.p2_coord_list: list[Point] = []
        self.linestring = LineString()

        # # Other geometric information
        self.search_area_list: list[Polygon] = []
        self.ridge_clip_list: list[LineString] = []

        self.termination_point = None
        self.termination_reason: str | None = None
        self.distance_along_cl = None


class H74TransectConstructor:
    """
    Takes a transect instance and builds the transect out with a given set of ridges and geometric parameters.

    Each hickin function in the class will return a shapely object and avoid changing state variables within the class
    """

    def __init__(
        self,
        origin: Point,
        transect_id: str,
        centerline: LineString,
        ridges: MultiLineString,
        shoot_distance: float,
        search_distance: float,
        dev_from_90: float,
        user_direction: int | None = None,
        verbose: int = 1,
    ) -> None:
        # Transect values
        self.transect = H74Transect(origin, transect_id)
        self.origin = Point(origin)
        self.centerline = centerline
        self.user_direction = user_direction
        self.initial_direction = None
        self.initial_alpha = None

        # Auxiliary geometry
        self.ridges = ridges
        self.ridges_centroid = self.ridges.centroid
        self.p1: Point = self.transect.coord_list[-1]
        self.p2 = Point()
        self.r1: LineString = self.centerline
        self.r2 = LineString()
        self.n1 = Point()
        self.n2 = Point()

        # State Variables
        self.walk_state = True
        self.iteration = 0
        self.verbose = verbose

        # Transect parameters
        self.shoot_distance = shoot_distance
        self.search_distance = search_distance
        self.dev_from_90 = dev_from_90
        self.max_iterations = 100

        # Update values after all attributes are initialized
        self.initial_alpha = direction_alpha_at_point(self.origin, self.r1)[1]
        self.initial_direction = self.calc_initial_direction(
            self.origin, self.initial_alpha, self.ridges
        )
        self.eval_user_direction()
        self.calc_dist_along_cl()

    def eval_user_direction(self) -> None:
        """Overrides the calculated shot direction value if direction is specified by the user."""

        if self.user_direction is not None:
            d_list = ("undetermined", "left", "right")
            if self.verbose == 2:
                print(
                    f"Overriding calculated direction value with user specified `{self.user_direction}` ({d_list[self.user_direction]})"
                )
            self.initial_direction = int(self.user_direction)

    def calc_initial_direction(
        self, origin: Point, alpha: float, ridges: MultiLineString
    ) -> int:
        """
        Determines the direction of the initial shot from the centerline.

        This is done by evaluating which shot direction (+ or - 90°) intersects the ridges.
        """

        output_values = np.array([1, -1])

        # +/- 90 to alpha
        alphas = np.ones(2) * alpha
        angles = alphas + (np.pi / 2 * output_values)

        # Calc dxdy to origin and add
        dxdy = calc_dxdy(angles, self.shoot_distance)
        new_points = (origin.xy + dxdy).T

        # Create lines for each shot
        lines = [LineString([origin, point]) for point in new_points]

        # Evaluate intersection of the new lines and ridges
        itx_result = [ridges.intersects(line) for line in lines]

        # Return the output_value corresponding to the shortest distance
        if any(itx_result):
            return output_values[itx_result][0]  # type: ignore[no-any-return]
        # If no intersection, then return 0 (tangent line)
        return 0

    def calc_dist_along_cl(self) -> None:
        # Get distance along centerline at transect origin
        dist = calc_dist_along_line(self.centerline)
        idx = find_closest_idx(self.origin, self.centerline)
        self.transect.distance_along_cl = dist[idx]

    def find_closest_ridge(
        self, line: LineString, ridges: MultiLineString
    ) -> LineString:
        """
        Takes a line and ridges as input and returns a snippet of a single ridge that the line intersects.
        Depending on the size of the search radius, line may still intersect ridge snippet at more than one point
        Initial intersection can have many geometry types. This function accounts for: empty geometry, Point, MultiPoint, GeometryCollection
        """

        # Start will actually be self.coords[-1]
        p1 = Point(line.coords[0])

        # Intersect line with ridges
        itx = line.intersection(ridges)

        # Remove start point from itx
        itx = itx.difference(p1.buffer(1e-3))

        # Return None if itx is empty
        if not itx.is_empty:
            # Feature maybe a single point, Multipoint, or geometry collection
            try:
                itx = MultiPoint(itx.coords)
            except NotImplementedError:
                # Break feature into points if intersection contains a polyline
                _points = []
                for i in itx.geoms:
                    for j in i.coords:
                        _points.append(j)
                itx = MultiPoint(_points)

            # Find point nearest the start point
            itx = sorted(list(itx.geoms), key=p1.distance)[0]

            # Isolate ridge closest to itx
            iso_ridge = sorted(list(ridges.geoms), key=itx.distance)[0]

            # Intersect ridge with itx buffered by the search distance
            search_area = itx.buffer(self.search_distance)
            self.transect.search_area_list.append(search_area)

            ridge_clip = iso_ridge.intersection(search_area)
            self.transect.ridge_clip_list.append(ridge_clip)

            # Isolate ridge piece that is closest to itx - assumes non-LineSting geoms are geometry collections
            if ridge_clip.geom_type != "LineString":
                ridge_clip = sorted(list(ridge_clip.geoms), key=itx.distance)[0]

            return ridge_clip

        return LineString()

    def shoot_point(self, p1: Point, r1: LineString, dist: float) -> Point:
        """
        Calculates the point perpendicular to `r1` at point `p1`, `dist` away from `p1`.
        Used for the first shot from the centerline into the ridge field.
        """

        # Determine angle for the next shot point
        direction = self.initial_direction
        alpha = direction_alpha_at_point(p1, r1)[1]

        # Override direction if user specified a value
        if self.user_direction:
            direction = self.user_direction

        # Heading (alpha) +/- 90° in radians
        angle = alpha + (direction * np.pi / 2)

        # Calc dxdy
        dxdy = calc_dxdy(angle, dist)

        # Return Point with dxdy
        return Point(p1.xy + dxdy)

    def shoot_point_rg(
        self, p0: Point, p1: Point, r1: LineString, dist: float
    ) -> Point:
        """
        Calculates the point perpendicular to `r1` at point `p1`, `dist` away from `p1`.
        Used when transect is moving from ridge to ridge.
        add_sub_90() is used to determine whether to add or subtract 90° from heading (alpha) value.
        """

        # Determine alpha value of r1 nearest p1
        _direction, alpha = direction_alpha_at_point(p1, r1)

        # Calculate the next shot point
        shoot = add_sub_90(p0, p1, alpha, dist)

        return Point(shoot)

    def src90(self, shot: LineString, ridge: LineString) -> Point:
        """Calculates the intersection between p1->shot_point and ridge r1 to define point p1."""

        itx = shot.intersection(ridge)

        if itx.geom_type == "Point":
            return itx
        if itx.geom_type == "MultiPoint":
            # Shot may intersect ridge piece more than once within the search buffer
            return sorted(
                [Point(i) for i in itx.geoms], key=lambda x: self.p1.distance(x)
            )[0]
        if self.verbose == 2:
            print(f"Failure at Src90: itx = {itx.wkt}")
            print(f"Shot: {shot.wkt}")
            print(f"Ridge: {ridge.wkt}")
            print("")
        return Point()

    def dest90(self, p1: Point, r2: LineString, dev_from_90: float) -> Point:
        """
        Calculates a point on `r2` from which a line that is perpendicular to `r2` AND intersects `p1` may be drawn.

        This is done by creating an angle between 3 points: (p1, ridge point, a point along the line tangent to the ridge at the ridge point)
        These angles are collected for a series of ridge points within a certain distance to the src90 point
        The ridge point with the angle closest to 90° is returned as point `p2`
        """

        # Calc alpha for every midpoint
        _curv, alpha = curvature(r2)
        dxdy = calc_dxdy(alpha)
        endpoints = r2.xy + dxdy

        # Prep coord arrays
        a = np.ones((endpoints.shape[1], 2)) * np.asarray(p1.coords)  # p1 array
        b = np.asarray(r2.coords)  # midpoint array
        c = endpoints.T  # endpoint array

        # Calc interior angles
        beta = np.abs(law_of_cos(a, b, c))

        # Calc deviation from 90°
        dev = np.abs(beta - np.pi / 2) * (180 / np.pi)

        # Find idx of lowest deviation
        min_idx = np.flatnonzero(dev == dev.min())[0]

        # Get midpoint coords where int_angle is smallest
        mid = Point(b[min_idx])

        # Print n2 info if verbose output
        if self.verbose == 2:
            print(
                f"n2 candidate:\t ANGLE: {beta[min_idx] * (180 / np.pi):.2f}° (dev={dev[min_idx]:.2f}°)\t COORDS: {tuple(b[min_idx])}"
            )

        # Determine if deviance from 90° is acceptable
        if dev[min_idx] < dev_from_90:
            if self.verbose == 2:
                print("Successfully created n2 point")
            return mid
        if self.verbose == 2:
            print(
                f"N2 point creation failed. Angle between ridge at n2 and line n2->p1 is not close enough to 90 (dev={dev[min_idx]:.2f}, tol={dev_from_90}); returned empty Point"
            )
        return Point()

    def result_coord(
        self, p1: Point, src90: Point, dest90: Point, r2: LineString
    ) -> Point:
        """Function takes coordinates of two points and their shared p1 to calculate the resultant vector"""

        # Calc vertical resultant coord of src90 and dest90
        vr = Point(vert_res(p1, src90, dest90))
        self.transect.vr_shoot_list.append(vr)

        # Create LineString of vertical resultant
        line = LineString([p1, vr])

        # Calc intersection between vertical resultant and ridge
        itx = r2.intersection(line)

        if itx.geom_type == "Point":
            return itx
        if itx.geom_type == "MultiPoint":
            # Shot may intersect ridge piece more than once within the search buffer
            return sorted(
                [Point(i) for i in itx.geoms], key=lambda x: self.p1.distance(x)
            )[0]
        if self.verbose == 2:
            print(f"Result_coord: {itx.wkt}")
        return Point()

    def walk_transect(self) -> H74Transect:
        """
        Iteratively walk the transect up the ridge field. Objects `self.r1` and `self.p1` are set in __init__ as
        the centerline and point on the centerline, respectively.
        """
        if self.verbose == 2:
            print(f"\n--- Walking Transect {self.transect.point_id} ---")
        # Walk state controls the iteration - when false, stop iterating the transect
        while self.walk_state:
            # TODO: develop more robust error handling
            # The following try except block will shut down the generation of a transect if an error is encountered
            # This behavior preserves and returns the data that was generated up until the point of the error
            try:
                # Calculate initial shot from r1 - different shoot method when shooting from centerline vs ridge
                if self.iteration == 0:
                    # Calculate the initial shot, see if it intersects a ridge
                    shot_point = self.shoot_point(self.p1, self.r1, self.shoot_distance)
                    shot = LineString([self.p1, shot_point])
                    self.transect.n1_shoot_list.append(shot_point)
                    self.r2 = self.find_closest_ridge(shot, self.ridges)

                else:
                    # p0 is p1 of the previous iteration, cannot exist on the first iteration (self.iteration = 0)
                    p0 = self.transect.coord_list[-2]

                    # Calc shot and ridge
                    shot_point = self.shoot_point_rg(
                        p0, self.p1, self.r1, self.shoot_distance
                    )
                    shot = LineString([self.p1, shot_point])
                    self.transect.n1_shoot_list.append(shot_point)
                    self.r2 = self.find_closest_ridge(shot, self.ridges)

                # Test if the shot actually intersects a ridge. If not, terminate the transect iteration
                if self.r2.is_empty:
                    self.walk_state = False
                    self.transect.termination_reason = "Failed ridge itx"
                    if self.verbose == 2:
                        print(
                            f"TRANSECT TERMINATED (iter={self.iteration}): n1 shot failed to intersect any more ridges."
                        )

                else:
                    # We do have r2, so we can calculate n1
                    n1 = self.src90(shot, self.r2)
                    self.transect.n1_coord_list.append(n1)

                    # Calculate n2 coord - point on r2 from which a line perpendicular to the tangent intersects p1
                    n2 = self.dest90(self.p1, self.r2, self.dev_from_90)

                    if n2.is_empty:
                        self.walk_state = False
                        self.transect.termination_reason = "Failed n2 creation"
                        if self.verbose == 2:
                            print(
                                f"TRANSECT TERMINATED (iter={self.iteration}): Failed to create n2 within a deviance of {self.dev_from_90:.1f}°"
                            )
                    else:
                        # Append n2 coord to list now that we know it's valid
                        self.transect.n2_coord_list.append(n2)

                        # We do have n2, so we can calculate p2

                        # Calculate p2 coord - the intersection of r2 and the vertical resultant of p1->n1 and p1->n2
                        p2 = self.result_coord(self.p1, n1, n2, self.r2)
                        self.transect.p2_coord_list.append(p2)

                        # Print iteration results
                        if self.verbose == 2:
                            print(
                                f"Iteration {self.iteration:02} result: [n1: {n1}, n2: {n2}, p2: {p2}]"
                            )

                        # Append coordinates, reset ridges and points
                        self.transect.coord_list.append(p2)
                        self.r1 = self.r2
                        self.p1 = p2
                        self.r2 = self.n1 = self.n2 = None

                        # Update the iteration count for this transect
                        self.iteration += 1

                # Add a break for run-away iterations
                if self.iteration > self.max_iterations:
                    self.walk_state = False
                    self.transect.termination_reason = "Iteration limit"
                    if self.verbose == 2:
                        print(
                            f"TRANSECT TERMINATED: Iteration counter reached iteration cap (max_iter={self.max_iterations})"
                        )

            except Exception as error:
                self.walk_state = False
                self.transect.termination_reason = "Unknown"
                if self.verbose == 2:
                    print(
                        f"TRANSECT TERMINATED: The following error occurred: `{type(error).__name__}: {error}`"
                    )

        # Replace linestring coordinates if transect does leave the centerline
        if self.iteration > 0:
            self.transect.linestring = LineString(self.transect.coord_list)

        # Return transect as output
        return self.transect


class MultiTransect:
    """
    Creates multiple instances of `H74Transect` from a given centerline, ridge dataset, and other parameters.

    The `create_transects` method is used to generate a GeoDataframe of transects.
    The `return_all_geometries` method returns the transects from `create_transects` as well as other intermediate geometries used in the creation of the transects. Useful for deubgging and plotting.

    This class is used in the `create_transects` convenience function in the public API.
    """

    def __init__(
        self,
        coord_list: list[Point],
        centerline: gpd.GeoDataFrame,
        ridges: gpd.GeoDataFrame,
        shoot_distance: float,
        search_distance: float,
        dev_from_90: float,
        user_direction: int | None = None,
        verbose: int = 1,
    ) -> None:
        self.coord_list = coord_list
        self.centerline = centerline
        self.ridges = ridges
        self.shoot_distance = shoot_distance
        self.search_distance = search_distance
        self.dev_from_90 = dev_from_90
        self.user_direction = user_direction
        self.verbose = verbose
        self.crs = self.centerline.crs

        # List to contain generated transects
        self.transect_list = self.create_transect_list()

        # GeoDataFrames for transect outputs
        self.transect_df = self.create_transect_df()
        self.point_df = self.create_point_df()
        self.search_area_df = self.create_search_area_df()
        self.ridge_clip_df = self.create_ridge_clip_df()

    def create_transect_list(self) -> list[H74Transect]:
        """Creates a set of transects and aux geometries for a bend."""

        # Reduce GeoDataFrames to their shapely representations
        centerline_ls = self.centerline.geometry[0]
        if len(self.ridges) > 1:
            ridges_mls = MultiLineString(self.ridges.geometry.values)
        else:
            ridges_mls = self.ridges.geometry[0]

        # Create a transect for each coord in `self.coord_list`
        transect_list = []
        for i, coord in enumerate(
            tqdm(
                self.coord_list,
                desc="Generate Transects",
                ascii=True,
                disable=(self.verbose != 1),
            )
        ):
            const = H74TransectConstructor(
                coord,
                f"t_{i:03}",
                centerline_ls,
                ridges_mls,
                self.shoot_distance,
                self.search_distance,
                self.dev_from_90,
                self.user_direction,
                self.verbose,
            )

            # Walk transect up the ridges
            t = const.walk_transect()
            transect_list.append(t)

        return transect_list

    def create_transect_df(self) -> gpd.GeoDataFrame:
        # Loop through all the transects which left the centerline to create the row
        ls_list = []
        for transect in self.transect_list:
            if transect.linestring:
                row = (
                    transect.point_id,
                    transect.distance_along_cl,
                    transect.linestring.length,
                    len(transect.linestring.coords),
                    self.shoot_distance,
                    self.search_distance,
                    self.dev_from_90,
                    transect.linestring,
                )
                ls_list.append(row)

        # Assemble GeoDataFrame from transect contents
        col_names = [
            "transect_id",
            "cl_distance",
            "length",
            "num_coords",
            "shoot_distance",
            "search_distance",
            "dev_from_90",
            "geometry",
        ]
        df = gpd.GeoDataFrame(
            data=ls_list, columns=col_names, geometry="geometry", crs=self.crs
        )

        return df.set_index("transect_id")

    def create_point_df(self) -> gpd.GeoDataFrame:
        # Loop through all the transects which left the centerline to create the row
        row_list = []

        for transect in self.transect_list:
            geom_dict = {
                "n1_shots": MultiPoint(transect.n1_shoot_list),
                "n1_coords": MultiPoint(transect.n1_coord_list),
                "n2_coords": MultiPoint(transect.n2_coord_list),
                "vr_shots": MultiPoint(transect.vr_shoot_list),
                "p2_coords": MultiPoint(transect.p2_coord_list),
            }

            for geom_type, geom in geom_dict.items():
                row = (
                    transect.point_id,
                    geom_type,
                    self.shoot_distance,
                    self.search_distance,
                    self.dev_from_90,
                    geom,
                )
                row_list.append(row)

        # Assemble GeoDataFrame from transect contents
        col_names = [
            "transect_id",
            "coord_type",
            "shoot_distance",
            "search_distance",
            "dev_from_90",
            "geometry",
        ]
        df = gpd.GeoDataFrame(
            data=row_list, columns=col_names, geometry="geometry", crs=self.crs
        )

        return df.set_index("transect_id")

    def create_search_area_df(self) -> gpd.GeoDataFrame:
        row_list = []

        for transect in self.transect_list:
            polys = MultiPolygon(transect.search_area_list)
            row = (
                transect.point_id,
                "n2_search_area",
                self.shoot_distance,
                self.search_distance,
                self.dev_from_90,
                polys,
            )
            row_list.append(row)

        col_names = [
            "transect_id",
            "poly_type",
            "shoot_distance",
            "search_distance",
            "dev_from_90",
            "geometry",
        ]
        df = gpd.GeoDataFrame(
            data=row_list, columns=col_names, geometry="geometry", crs=self.crs
        )

        return df.set_index("transect_id")

    def create_ridge_clip_df(self) -> gpd.GeoDataFrame:
        row_list = []

        for transect in self.transect_list:
            lines = MultiLineString(transect.ridge_clip_list)
            row = (
                transect.point_id,
                "ridge_clip",
                self.shoot_distance,
                self.search_distance,
                self.dev_from_90,
                lines,
            )
            row_list.append(row)

        col_names = [
            "transect_id",
            "line_type",
            "shoot_distance",
            "search_distance",
            "dev_from_90",
            "geometry",
        ]
        df = gpd.GeoDataFrame(
            data=row_list, columns=col_names, geometry="geometry", crs=self.crs
        )

        return df.set_index("transect_id")

    def return_all_geometries(
        self,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        return self.transect_df, self.point_df, self.search_area_df, self.ridge_clip_df


def create_transects(
    centerline: gpd.GeoDataFrame,
    ridges: gpd.GeoDataFrame,
    step: int,
    shoot_distance: float,
    search_distance: float,
    dev_from_90: float,
) -> gpd.GeoDataFrame:
    """
    Convenience function to create a series of transects from a given centerline, set of ridges, and the necessary parameters.

    Transects are created at the `step` provided by the user (ex. every nth vertex along the centerline).
    Centerline is assumed to have a vertex spacing of ~1m.
    """

    # Establish starting points for each transect
    starts = np.asarray(centerline.geometry[0].xy).T[::step]

    transects = MultiTransect(
        starts, centerline, ridges, shoot_distance, search_distance, dev_from_90
    )

    # Create all output geometries created during transect creation
    transect_df, _point_df, _search_area_df, _ridge_clip_df = (
        transects.return_all_geometries()
    )

    # Return just the transects
    return transect_df
