import numpy as np
import matplotlib.pyplot as plt
import scipy
from numba import jit

########
##
##   Gradient-based curvature metrics
##
########


def gradient_gaussian_curvature(z, dx=1):
    """Gaussian curvature in 2D.
    Implemented as described by Gauss, 1902.

    Gradient based calculation, only valid for window 3x3.

    Parameters
    ----------
    z : np.ndarray
        2d surface array, probably elevation
    dx : float
        Grid spacing in meters, so that output curvature is in units of 1/m.

    .. note:: As implemented, requires that dx == dy. I.e., array z is equal
    spacing in x and y directions. Also only computes for local 9-member
    neighborhood.

    """
    dx = float(dx)
    dzdy, dzdx = np.gradient(z)
    dzdxdy, dzdxdx = np.gradient(dzdx)
    dzdydy, dzdydx = np.gradient(dzdy)
    k = (dzdxdx * dzdydy - (dzdxdy**2)) / (1 + (dzdx**2) + (dzdy**2)) ** 2
    k = k * (1 / (dx**2)) ** 2
    return k


def gradient_plan_curvature(z, dx=1):
    """
    Implemented per following link:
    https://surferhelp.goldensoftware.com/gridops/plan_curvature.htm?tocpath=Gridding%7CGrid%20Operations%7CGrid%20Calculus%7C_____9

    Could implement other curvature there as well
    """
    dx = float(dx)
    dzdy, dzdx = np.gradient(z)
    dzdxdy, dzdxdx = np.gradient(dzdx)
    dzdydy, dzdydx = np.gradient(dzdy)
    p = dzdx**2 + dzdy**2
    p[p == 0] = 1e-16  # not necessary on real DEM probably
    k = (
        (dzdxdx * (dzdy**2)) - (2 * dzdxdy * dzdx * dzdy) + (dzdydy * (dzdx**2))
    ) / p ** (3 / 2)
    k = k * (1 / (dx**2)) ** 2
    return k


########
##
##   Quadratic surface-based curvature metrics
##
########


def _quadratic_coefficients_least_squares(
    elevation, window, weighting_exponent, constrained, dx
):
    """
    Find the quadratic surface that fits the observations at all points in the domain.

    This is the real workhorse of the code, and it is separated out so that
    the coefficients can be resued for different curvature metric
    calculations, if desired.
    """

    weight = _find_weight(window, exponent=weighting_exponent)
    normal = _find_normal(weight, window, dx)

    # set up padded elevations array
    pad_width = int((window - 1) / 2)  # half window to pad arrays
    idx_offset = 2 * pad_width + 1  # offset to add to ij to index correct etas
    cidx = int((window - 1) / 2)
    pad_eta = np.pad(elevation, pad_width, mode="edge")

    # perform the factorization that only needs to happen once
    if constrained:
        # perform LU decomp only using the first 5 coefficients
        lu_A = scipy.linalg.lu_factor(normal[:-1, :-1])
    else:
        # perform LU decomp using all 6 coefficients
        lu_A = scipy.linalg.lu_factor(normal)

    # preallocate the output coefficients array
    ncoeff = 5 if constrained else 6
    coefficients_matrix = np.zeros((elevation.shape[0], elevation.shape[1], ncoeff))

    # loop through the domain
    for i in np.arange(elevation.shape[0]):
        for j in np.arange(elevation.shape[1]):
            # get ij observations
            ij_elev = pad_eta[i : i + idx_offset, j : j + idx_offset]
            ij_elev = ij_elev - ij_elev[cidx, cidx]

            # get obs as vector
            ij_obs = _find_obs(ij_elev, weight, window, dx, constrained)

            # solve system for quadratic
            if constrained:
                ij_obs = ij_obs[:-1]
            ij_coeff = scipy.linalg.lu_solve(lu_A, ij_obs)

            coefficients_matrix[i, j, :] = ij_coeff

    if constrained:
        # add a zero coefficient to the whole matrix
        coefficients_matrix = np.dstack((coefficients_matrix, np.zeros_like(elevation)))

    return coefficients_matrix


def _unpack_coeff(_coeff):
    """
    Helper for handling constrained and unconstrained least squares coefficients.
    """
    if len(_coeff) == 6:  # unconstrained
        a, b, c, d, e, f = _coeff
    else:  # constrained
        a, b, c, d, e = _coeff
        f = 0
    return a, b, c, d, e, f


def __realize_quadratic_surface(_elev, _coeff, wsize):
    """Make a plot.

    Helpful for testing.
    """
    a, b, c, d, e, f = _unpack_coeff(_coeff)

    edge = int((window - 1) / 2)
    xmesh, ymesh = np.meshgrid(
        np.linspace(edge * -dx, edge * dx, window),
        np.linspace(edge * -dx, edge * dx, window),
    )
    xmesh_dist = xmesh - xmesh[edge, edge]
    ymesh_dist = ymesh - ymesh[edge, edge]

    pred = (
        a * xmesh_dist * xmesh_dist
        + b * ymesh_dist * ymesh_dist
        + c * xmesh_dist * ymesh_dist
        + d * xmesh_dist
        + e * ymesh_dist
        + f
    )

    fig, ax = plt.subplots(1, 2)
    im0 = ax[0].imshow(_elev)
    fig.colorbar(im0, shrink=0.4)
    im1 = ax[1].imshow(pred)
    fig.colorbar(im1, shrink=0.4)
    plt.show()


def parameter_quadratic_planform_curvature(_coeff):
    """
    Planform curvature from quadratic coefficients.

    Parameters
    ----------
    _coeff : np.ndarray
        Set of coefficients to quadratic surface equation.

    Returns
    -------
    Single parameter for local planform curvature.
    """
    a, b, c, d, e, f = _unpack_coeff(_coeff)

    if (d == 0) and (e == 0):
        _curv_param = 0.0
    else:
        _curv_param = (2.00 * (b * d * d + a * e * e - c * d * e)) / (
            (e * e + d * d) ** (1.5)
        )

    return _curv_param


def parameter_quadratic_profile_curvature(_coeff):
    """
    Profile curvature from quadratic coefficients.

    Parameters
    ----------
    _coeff : np.ndarray
        Set of coefficients to quadratic surface equation.

    Returns
    -------
    Single parameter for local profile curvature.
    """
    a, b, c, d, e, f = _unpack_coeff(_coeff)

    if (d == 0) and (e == 0):
        _curv_param = 0.0
    else:
        _curv_param = (-2.00 * (a * d * d + b * e * e + c * d * e)) / (
            (e * e + d * d) * (1 + e * e + d * d) ** (1.5)
        )

    return _curv_param


def quadratic_planform_curvature(
    elevation,
    window=3,
    weighting_exponent=1,
    constrained=True,
    dx=1,
    coefficients_matrix=None,
    return_coefficients_matrix=False,
):
    """
    Calculate the planform curvature for a digital elevation model.

    Parameters
    ----------
    elevation : np.ndarray, xr.DataArray
        Elevation data.

    window : float
        Window size. Default is 3.

    weighting_expoenent : float
        Inverse-distance weighting for quadratic
        surface least squares. 0 = no weighting, 1 = linear decay,
        2 = exponential decay. Default 1.

    constrained : bool
        Whether quadratic surface is constrained through center cell. Default `True`.

    dx : float
        Grid spacing. Default 1.

    coefficients_matrix : np.ndarray
        An array of same dimensions as elevation M x N x 6 (coefficients), which can
        be reused for the various metric caclulations, rather than needting to
        recalculate the coefficients each time.

    return_coefficients_matrix : bool

    """

    # sanitize inputs (could be put into private subfunction)
    assert elevation.ndim == 2
    assert window <= 499  # limit from GRASS
    assert isinstance(constrained, bool)

    # note, if the elevation is an xarray, we can get dx from the data. something like below
    # if isinstance(elevation, xr.DataArray):
    #    dx = ...

    if coefficients_matrix is not None:
        # reuse the matrix for the calculations
        pass
    else:
        coefficients_matrix = _quadratic_coefficients_least_squares(
            elevation, window, weighting_exponent, constrained, dx
        )

    # loop through the domain to make the parameter calculation
    # NOTE: This can readily be vectorized...
    domain_plan_curv = np.zeros_like(elevation)
    for i in np.arange(elevation.shape[0]):
        for j in np.arange(elevation.shape[1]):
            ij_coeff = coefficients_matrix[i, j, :].flatten()
            domain_plan_curv[i, j] = parameter_quadratic_planform_curvature(ij_coeff)

    if return_coefficients_matrix:
        return domain_plan_curv, coefficients_matrix
    else:
        return domain_plan_curv


def quadratic_profile_curvature(
    elevation,
    window=3,
    weighting_exponent=1,
    constrained=True,
    dx=1,
    coefficients_matrix=None,
    return_coefficients_matrix=False,
):
    """
    Calculate the profile curvature for a digital elevation model.

    Parameters
    ----------
    elevation : np.ndarray, xr.DataArray
        Elevation data.

    window : float
        Window size. Default is 3.

    weighting_expoenent : float
        Inverse-distance weighting for quadratic
        surface least squares. 0 = no weighting, 1 = linear decay,
        2 = exponential decay. Default 1.

    constrained : bool
        Whether quadratic surface is constrained through center cell. Default `True`.

    dx : float
        Grid spacing. Default 1.

    coefficients_matrix : np.ndarray
        An array of same dimensions as elevation M x N x 6 (coefficients), which can
        be reused for the various metric caclulations, rather than needting to
        recalculate the coefficients each time.

    return_coefficients_matrix : bool

    """

    # sanitize inputs (could be put into private subfunction)
    assert elevation.ndim == 2
    assert window <= 499  # limit from GRASS
    assert isinstance(constrained, bool)

    # note, if the elevation is an xarray, we can get dx from the data. something like below
    # if isinstance(elevation, xr.DataArray):
    #    dx = ...

    if coefficients_matrix is not None:
        # reuse the matrix for the calculations
        pass
    else:
        coefficients_matrix = _quadratic_coefficients_least_squares(
            elevation, window, weighting_exponent, constrained, dx
        )

    # loop through the domain to make the parameter calculation
    # NOTE: This can readily be vectorized...
    domain_prof_curv = np.zeros_like(elevation)
    for i in np.arange(elevation.shape[0]):
        for j in np.arange(elevation.shape[1]):
            ij_coeff = coefficients_matrix[i, j, :].flatten()
            domain_prof_curv[i, j] = parameter_quadratic_profile_curvature(ij_coeff)

    if return_coefficients_matrix:
        return domain_prof_curv, coefficients_matrix
    else:
        return domain_prof_curv


def _find_weight(wsize, exponent):
    """
    Function to find the weightings matrix for the */
    /*               observed cell values.                          */
    /*               Uses an inverse distance function that can be  */
    /*               calibrated with an exponent (0= no decay,      */
    /*               1=linear decay, 2=squared distance decay etc.) */
    /*               V.1.1, Jo Wood, 11th May, 1995.                */
    """
    edge = (wsize - 1) / 2
    _weight = np.zeros((wsize, wsize))
    for row in np.arange(wsize):
        for col in np.arange(wsize):
            dij = np.sqrt((edge - col) * (edge - col) + (edge - row) * (edge - row))
            dist = 1.0 / np.power(dij + 1.0, exponent)
            _weight[row, col] = dist
    return _weight


def _find_normal(weights, wsize, resoln):
    """
    /****************************************************************/
    /* find_normal() - Function to find the set of normal equations */
    /*                 that allow a quadratic trend surface to be   */
    /*                 fitted through N points using least squares */
    /*                 V.1.0, Jo Wood, 27th November, 1994.         */

    /****************************************************************/
    Code is adapted from the GRASS implementation, from Jo  Wood thesis.
    Normal equations defined for quadratic surface are from Unwin, 1975
    """
    edge = int((wsize - 1) / 2)

    # coefficients of cross products, preallocate
    x1, y1 = 0, 0
    x2, y2 = 0, 0
    x3, y3 = 0, 0
    x4, y4 = 0, 0
    xy2, x2y = 0, 0
    xy3, x3y = 0, 0
    x2y2 = 0
    xy = 0
    N = 0

    # Initialise sums-of-squares and cross products matrix
    normal = np.zeros((6, 6))

    # Calculate matrix of sums of squares and cross products
    cnt = 0
    for row in np.arange(wsize):
        for col in np.arange(wsize):
            w = weights[row, col]

            x = resoln * (col - edge)
            y = resoln * (row - edge)

            x4 += (x * x * x * x) * w
            x2y2 += (x * x * y * y) * w
            x3y += (x * x * x * y) * w
            x3 += (x * x * x) * w
            x2y += (x * x * y) * w
            x2 += (x * x) * w

            y4 += (y * y * y * y) * w
            xy3 += (x * y * y * y) * w
            xy2 += (x * y * y) * w
            y3 += (y * y * y) * w
            y2 += (y * y) * w

            xy += (x * y) * w

            x1 += (x) * w
            y1 += (y) * w

            N += cnt * w
            cnt += 1

    # /* --- Store cross-product matrix elements. --- */
    normal[0][0] = x4
    normal[0][1] = normal[1][0] = x2y2
    normal[0][2] = normal[2][0] = x3y
    normal[0][3] = normal[3][0] = x3
    normal[0][4] = normal[4][0] = x2y
    normal[0][5] = normal[5][0] = x2

    normal[1][1] = y4
    normal[1][2] = normal[2][1] = xy3
    normal[1][3] = normal[3][1] = xy2
    normal[1][4] = normal[4][1] = y3
    normal[1][5] = normal[5][1] = y2

    normal[2][2] = x2y2
    normal[2][3] = normal[3][2] = x2y
    normal[2][4] = normal[4][2] = xy2
    normal[2][5] = normal[5][2] = xy

    normal[3][3] = x2
    normal[3][4] = normal[4][3] = xy
    normal[3][5] = normal[5][3] = x1

    normal[4][4] = y2
    normal[4][5] = normal[5][4] = y1

    normal[5][5] = N

    return normal

@jit
def _find_obs(elevations, weights, wsize, resoln, constrained):
    """
    /****************************************************************/
    /* find_obs() - Function to find the observed vector as part of */
    /*              the set of normal equations for least squares.  */
    /*              V.1.0, Jo Wood, 11th December, 1994.            */

    /****************************************************************/

    return obs, the column vector in matrix form for LU decomp

    .. important:: this might be a good target for jitting!
    """
    edge = (wsize - 1) / 2
    # offset      /* Array offset for weights & z */

    obs = np.zeros((6,))  # column vector

    for row in np.arange(wsize):
        for col in np.arange(wsize):
            # /* Local window coordinates */
            x = resoln * (col - edge)
            y = resoln * (row - edge)

            w = weights[row, col]
            z = elevations[row, col]

            obs[0] += w * z * (x * x)
            obs[1] += w * z * (y * y)
            obs[2] += w * z * x * y
            obs[3] += w * z * x
            obs[4] += w * z * y

            if not constrained:  # /* If constrained, should remain 0.0 */
                obs[5] += w * z
    return obs
