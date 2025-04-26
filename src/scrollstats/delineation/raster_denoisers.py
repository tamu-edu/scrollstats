"""
Contains all functions responsible for denoising an input binary array ("denoiser functions") to more accurately represent the ridge and swale areas
Ridge areas will correspond to a value of True or 1, swale areas will correspond to a value of False or 0

Denoiser functions take a BinaryArray2D and any other kwargs as input and return a BinaryArray2D as output

    denoiser_func(BinaryArray2D, **kwargs) -> BinaryArray2D

"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .array_types import BinaryArray2D, BinaryDenoiserFn


# Denoiser functions
def binary_flipper(
    binary_array: BinaryArray2D, func: BinaryDenoiserFn
) -> BinaryArray2D:
    out_array = func(binary_array)
    out_array = ~func(~out_array)

    return out_array


def remove_small_feats(img: BinaryArray2D, size: int) -> BinaryArray2D:
    """
    Removes any patch/feature in a binary image that is below a certain size (measured in px)
    """
    # Label all unique features in binary image
    label, _ = ndimage.label(img)

    # Get list of unique feat ids as well as pixel counts
    feats, counts = np.unique(label, return_counts=True)

    # list of feat ids that are too small
    feat_ids = feats[counts < size]

    # Wipe out patches with id that is in `ids` list
    for feat_id in feat_ids:
        label[label == feat_id] = 0

    # Convert all labels to 1
    label[label != 0] = 1

    return label.astype(bool)


def remove_small_feats_w_flip(
    img: BinaryArray2D, small_feats_size: int
) -> BinaryArray2D:
    """
    Apply `remove_small_feats` to the ridge areas, then flip the values in the binary array to apply `remove_small_feats` to the swale areas
    """
    out_array = remove_small_feats(img, small_feats_size)
    out_array = ~remove_small_feats(~out_array, small_feats_size)

    return out_array


DEFAULT_DENOISERS = (
    ndimage.binary_closing,
    ndimage.binary_opening,
    remove_small_feats_w_flip,
)
