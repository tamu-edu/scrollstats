from __future__ import annotations

from collections.abc import Callable
from functools import partial
from inspect import signature
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
from shapely.geometry import Polygon

from .array_types import (
    Array2D,
    BinaryArray2D,
    BinaryClassifierFn,
    BinaryDenoiserFn,
    ElevationArray2D,
)
from .raster_classifiers import DEFAULT_CLASSIFIERS
from .raster_denoisers import DEFAULT_DENOISERS


def clip_raster(
    ds: rasterio.DatasetReader,
    geometry: Polygon,
    array: np.ndarray | None = None,
    no_data: Any | None = None,
) -> tuple[ElevationArray2D, BinaryArray2D, dict[Any, Any]]:
    # Replace optional values
    if isinstance(array, np.ndarray):  # noqa: SIM108
        array_copy = array.copy()
    else:
        array_copy = ds.read(1)

    if not no_data:
        no_data = ds.nodata

    # For cropped_mask, True is area outside of geometry
    clipped_mask, transform, window = rasterio.mask.raster_geometry_mask(
        dataset=ds, shapes=[geometry], crop=True
    )

    # Update size, transform, and nodata value for output raster
    clipped_meta = ds.meta
    clipped_meta.update(
        {
            "driver": "GTiff",
            "height": clipped_mask.shape[0],
            "width": clipped_mask.shape[1],
            "transform": transform,
            "nodata": no_data,
        }
    )

    # Crop array
    array_clip = array_copy[window.toslices()]

    # Fill no_data values for output array
    # Only 0s are falsy, so any other number (including np.nan) will evaluate to True in boolean arrays.
    # So, we need to explicitly set the fill value as False for boolean arrays.
    fill_value = no_data
    if array_copy.dtype == bool:
        fill_value = False

    array_clip[clipped_mask] = fill_value

    return array_clip, clipped_mask, clipped_meta


def partial_from_kwargs(func: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
    """Create a partial function from all of the kwargs that are found in the function's signature"""
    sig = signature(func)
    args = {p: kwargs.get(p) for p in sig.parameters if kwargs.get(p)}
    return partial(func, **args)


def create_ridge_area_raster(
    dem_ds: rasterio.DatasetReader,
    geometry: Polygon,
    classifier_funcs: tuple[BinaryClassifierFn, ...] = DEFAULT_CLASSIFIERS,
    denoiser_funcs: tuple[BinaryDenoiserFn, ...] = DEFAULT_DENOISERS,
    no_data_value: Any | None = None,
    **kwargs: Any,
) -> tuple[Array2D, ElevationArray2D, dict[Any, Any]]:
    """
    Main processing function to create the ridge area raster.

    This function uses the provided classifier_funcs and denoiser_funcs to classify the ridge and swale areas within the input DEM.

    Ridge Area Classification:
    --------------------------
    By default, scrollstats uses profile curvature (a measure of ridge convexity) and residual topography (a measure of ridge prominence) to classify ridge areas.
    Each classifier function is applied to the DEM, then the union of all the resulting binary arrays will be used for denoising.
    This means that the more classifier functions you use, the more conservative, but ideally more accurate, your ridge areas will be.

    If the user desires, they can provide their own classifier functions so long as the functions follow the pattern below

        classifier_func(ElevationArray2D, **kwargs) -> BinaryArray2D

    See scrollstats/delineation/raster_classifiers.py for the DEFAULT_CLASSIFIERS list of functions and their definitions.

    Clip Ridge and Swale Topography:
    --------------------------------
    In order to avoid edge-effects from the classifier functions, the area corresponding to the ridge and swale topography will be clipped from a larger DEM.
    The nodata value for the input DEM will be used unless no_data_value is specified.

    Image Denoising:
    ----------------
    Once the ridge areas are classified within the DEM as a binary array (1=ridge, 0=swale), scrollstats uses a series of denoising algorithms to clean up the result.
    By default, scrollstats uses binary closing and binary opening operations to efficiently remove small objects from the binary image, then it uses another filter to remove of any remaining image object smaller than a certain size (measured in px).
    Each classifier function is applied to the binary array in sequence, meaning that the output of the first classifier function is the input of the second, and so on.
    Therefore, a different ordering of the same list of denoiser functions may yield a different result.

    If the user desires, they can provide their own denoiser functions so long as the functions follow the pattern below

        denoiser_func(BinarryArray2D, **kwargs) -> BinaryArray2D

    See scrollstats/delineation/raster_denoisers.py for the DEFAULT_DENOISERS list of functions their definitions.

    Keyword Arguments for Image Processing Functions:
    -------------------------------------------------
    Any additional arguments required by the classifier_funcs or denoier_funcs can be provided to this function as keyword arguments
    Any keyword arguments provided to this function will be passed to a given classifier or denoiser function if the provided keyword matches a keyword in the function's signature.

    """

    # Read DEM as np.array - assumes single band DEM
    dem = dem_ds.read(1)

    # Set the output no data value
    if not no_data_value:
        no_data_value = dem_ds.nodata

    # Set temporary nodata values to absurd real number value for classifier functions which
    # may have mathematical errors with infinite values (np.nan)
    if np.isnan(dem_ds.nodata):
        no_data_mask = np.isnan(dem_ds.read(1))
    else:
        no_data_mask = dem == dem_ds.nodata
    dem[no_data_mask] = -99999

    # Classify the ridge and swale areas within the DEM with the provided classifier functions
    ## `partial_from_kwargs` creates a partial function which fills the parameters of each classifier function (except for the dem) from kwargs.
    ## This allows for us to simply loop over all of the classifier functions (which must return a BinarryArray2D) and apply them to the DEM
    ## The user must supply kwargs that exactly match the other parameter names in the classifier functions.
    binary_array = np.ones(dem.shape).astype(bool)
    for func in classifier_funcs:
        partial_func = partial_from_kwargs(func, **kwargs)
        transform = partial_func(dem)
        binary_array = transform & binary_array

    # Clip the DEM and Binary array to the bounds of the ridge and swale area
    dem_clip, _dem_mask, _dem_meta = clip_raster(
        dem_ds, geometry, no_data=no_data_value
    )
    binary_clip, binary_mask, binary_meta = clip_raster(
        dem_ds, geometry, array=binary_array, no_data=no_data_value
    )

    # Denoise the clipped binary array with the provided denoiser functions
    ## The denoiser functions are chained so that the output of the first is the input of the second and so on, so their order is important
    for func in denoiser_funcs:
        partial_func = partial_from_kwargs(func, **kwargs)
        binary_clip = partial_func(binary_clip)

    # Cast to float so that np.nan (or other floating point value) can be assigned as nodata value
    binary_clip = binary_clip.astype(float)
    binary_clip[binary_mask] = binary_meta["nodata"]

    return binary_clip, dem_clip, binary_meta


def create_ridge_area_raster_fs(
    dem_path: Path,
    geometry_path: Path,
    out_dir: Path,
    bend_id_dict: dict[str, str] | None = None,
    **kwargs: Any,
) -> tuple[Path, Path]:
    """File system interface for create_ridge_area_raster"""

    # create output folder if not exists
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(geometry_path)

    if bend_id_dict:
        col = next(iter(bend_id_dict.keys()))
        bend_id = bend_id_dict[col]
        geometry = gdf.set_index(col).loc[bend_id, "geometry"]
    else:
        geometry = gdf.loc[0, "geometry"]

    with rasterio.open(dem_path) as src:
        ridge_area_raster, cropped_dem, cropped_meta = create_ridge_area_raster(
            dem_ds=src, geometry=geometry, **kwargs
        )

        # Write arrays to disk
        binary_out_path = out_dir / f"{dem_path.stem}_ridge_area_raster.tif"
        dem_out_path = out_dir / f"{dem_path.stem}_clip.tif"

        with rasterio.open(binary_out_path, "w", **cropped_meta) as dst:
            dst.write(ridge_area_raster, 1)
            print(f"Wrote ridge area raster to disk: {binary_out_path}")

        with rasterio.open(dem_out_path, "w", **cropped_meta) as dst:
            dst.write(cropped_dem, 1)
            print(f"Wrote clipped DEM to disk: {dem_out_path}")

    return binary_out_path, dem_out_path
