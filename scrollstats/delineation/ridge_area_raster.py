from typing import Callable, Tuple
from functools import partial
from pathlib import Path

import numpy as np
import rasterio
import rasterio.mask
from scipy import ndimage
import geopandas as gpd
from shapely.geometry import Polygon
from rasterio import DatasetReader

from .curvature import quadratic_profile_curvature

# Define types
Array2D = np.ndarray[Tuple[int, int], np.dtype[any]]
ElevationArray2D = np.ndarray[Tuple[int, int], np.dtype[float]]
BinaryArray2D = np.ndarray[Tuple[int, int], np.dtype[bool]]

# Define functions as interfaces
## `BinaryClassifierFn`s and `BinaryDenoiserFn`s use the following signature
## They take a 2D array as input and output a binary 2D array
## Use partial functions fron the `functools` library to create wrapper functions which contain all input arguments other than the input 2D array
BinaryClassifierFn = Callable[[ElevationArray2D], BinaryArray2D]
BinaryDenoiserFn = Callable[[BinaryArray2D], BinaryArray2D]


# Define the mathematical operations to apply to the ElevationArray2D
# Profile curvature is defined in curvature.py 
def residual_topography(dem:ElevationArray2D, w:int) -> Array2D:
    """
    Using a moving window with side length `w`, subtract the focal mean from the central pixel value.
    """
    # Construct an array of processing windows from the input elevation array
    kernal = np.lib.stride_tricks.sliding_window_view(dem, (w,w))

    # Take the mean of each window
    means = kernal.mean(axis=(2,3))

    # `kernal` above only has the valid inner windows from the array, so `means` needs to be padded by w//2 all the way around
    padded_means = np.ones(dem.shape) * np.nan
    padded_means[w//2:-(w//2), w//2:-(w//2)] = means

    rt = dem - padded_means

    return rt

# Classifier Functions 
## classifier functions take an ElevationArray2D and any other args as input and return a BinaryArray2D as output
## use partial functions to supply other args to the classifier functions so that they may be used in `classify_raster()`
def profile_curvature_classifier(dem:ElevationArray2D, window:int, dx:float, threshold: int = 0) -> BinaryArray2D:
    """
    Calculates the profile curvature within a moving window with side length `window`.
    Returns a boolean array where True pixels are greater than `threshold`
    """
    profc = quadratic_profile_curvature(
          elevation=dem,
          window=window,
          dx=dx
    )
    
    return profc > threshold 

def residual_topography_classifier(dem:ElevationArray2D, window:int, threshold:int = 0) -> BinaryArray2D:
    """
    Calculates the profile curvature within a moving window with side length `window`.
    Returns a boolean array where True pixels are greater than `threshold`
    """
    rt = residual_topography(dem, window)

    return rt > threshold


def classify_raster(dem:ElevationArray2D, classifiers:list[BinaryClassifierFn]) -> BinaryArray2D:
    """
    Applies a series of functions (`classifiers`) to a DEM to classify the ridge areas.
    Each classifier takes an ElevationArray2D (2D float numpy array) as input and produces a BinaryArray2D (2D boolean numpy array) as an output.
    Partial functions are used to reduce the number of arguments needed for each classifier function to just the DEM array.

    Returns:
    The agreement (union) of all classifiers as a BinaryArray2D with the same shape as `dem` where True values represent ridge pixels.
    """
    out_array = np.ones(dem.shape).astype(bool)
    for func in classifiers:
        transform = func(dem)
        out_array = transform & out_array
    return out_array


# Denoiser functions
## denoiser functions take a BinaryArray2D and any other args as input and return a BinaryArray2D as output
## use partial functions to supply other args to the denoiser functions so that they may be used in `denoise_raster()`
def binary_flipper(binary_array:BinaryArray2D, func:Callable) -> BinaryArray2D:

    out_array = func(binary_array)
    out_array = ~func(~out_array)

    return out_array

def remove_small_feats(img:BinaryArray2D, size:int) -> BinaryArray2D:
    """
    Removes any patch/feature in a binary image that is below a certian pixel count

    Parameters
    ----------
    img : binary ndarray
    size : (int), minimum patch size needed to be kept in the image

    Returns
    -------
    out : binary ndarray
    """
    # Label all unique features in binary image
    label, _ = ndimage.label(img)

    # Get list of unique feat ids as well as pixel counts
    feats, counts = np.unique(label, return_counts=True)

    # list of feat ids that are too small
    ids = feats[counts < size]

    # Wipe out patches with id that is in `ids` list
    for id in ids:
        label[label == id] = 0

    # Convert all labels to 1
    label[label != 0] = 1

    return label.astype(bool)


def denoise_raster(binary_array:BinaryArray2D, denoisers:list[BinaryDenoiserFn]) -> BinaryArray2D:
    """
    Applies a series of functions (`denoisers`) to a BinaryArray2D (2D boolean numpy array) to refine the ridge area classification.
    Each denoiser is applied in succession, so the result of the first is used as input for the second and so on.
    Each denoiser takes as input and produces as output a BinaryArray2D.
    Partial functions are used to reduce the number of arguments needed for each denoiser function to just the input `binary_array`.

    Returns:
    The result of the series of denoising function as a BinaryArray2D with the same shape as the input `binary_array`
    """
    for func in denoisers:
        binary_array = func(binary_array)
    return binary_array


def clip_raster(ds:DatasetReader, geometry:Polygon, array=None, no_data=None):

    # Replace optional values
    if isinstance(array, np.ndarray):
        array_copy = array.copy()
    else:
        array_copy = ds.read(1)
    
    if not no_data:
        no_data = ds.nodata

    # For cropped_mask, True is area outside of geometry
    clipped_mask, transform, window= rasterio.mask.raster_geometry_mask(
        dataset=ds,
        shapes=[geometry],
        crop=True
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


def create_ridge_area_raster(dem_ds:DatasetReader, geometry:Polygon, **kwargs) -> tuple[Array2D, ElevationArray2D, dict]:
    """
    Main processing function to create the ridge area raster.

    """

    # Get kwargs
    window = kwargs.get("window")
    small_feats_size = kwargs.get("small_feats_size")
    dx = kwargs.get("dx")
    no_data = kwargs.get("no_data")


    dem = dem_ds.read(1)

    # Set nodata values to absurd real number value to avoid mathematical errors with infinite values (np.nan)
    no_data_mask = dem == dem_ds.nodata
    dem[no_data_mask] = -99999
    
    # Collect classifier functions
    classifier_funcs = [
        partial(profile_curvature_classifier, window=window, dx=dx),
        partial(residual_topography_classifier, window=window)
    ]
    # Classify the raster 
    binary_array = classify_raster(dem=dem, classifiers=classifier_funcs)

    # Clip the DEM and Binary array to the bounds of the bend
    dem_clip, _dem_mask, _dem_meta = clip_raster(dem_ds, geometry, no_data=no_data)
    binary_clip, binary_mask, binary_meta = clip_raster(dem_ds, geometry, array=binary_array, no_data=no_data)

    # Collect denoising functions 
    remove_small_feats_partial = partial(remove_small_feats, size=small_feats_size)
    remove_small_feats_partial_flip = partial(binary_flipper, func=remove_small_feats_partial)

    denoiser_funcs = [
        ndimage.binary_closing, 
        ndimage.binary_opening, 
        remove_small_feats_partial_flip
    ]

    # Denoise the binary array
    binary_clip_denoise = denoise_raster(binary_array=binary_clip, denoisers=denoiser_funcs)

    # Cast to float so that np.nan (or other floating point value) can be assigned as nodata value 
    binary_clip_denoise = binary_clip_denoise.astype(float)
    binary_clip_denoise[binary_mask] = binary_meta["nodata"]

    return binary_clip_denoise, dem_clip, binary_meta


def create_ridge_area_raster_fs(dem_path:Path, geometry_path:Path, out_dir:Path, bend_id_dict:dict[str:str] = None, **kwargs):
    """File system interface for create_ridge_area_raster"""

    gdf = gpd.read_file(geometry_path)

    if bend_id_dict:
        col = list(bend_id_dict.keys())[0]
        bend_id = bend_id_dict[col]
        geometry = gdf.set_index(col).loc[bend_id, "geometry"]
    else:
        geometry = gdf.loc[0, "geometry"]
    
    with rasterio.open(dem_path) as src:
        ridge_area_raster, cropped_dem, cropped_meta = create_ridge_area_raster(
            dem_ds = src,
            geometry=geometry,
            **kwargs
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
