"""
Copyright (c) 2025 Andrew Vanderheiden. All rights reserved.

scrollstats: An open-source python library to calculate and extract morphometrics from scroll bar floodplains
"""

from __future__ import annotations

from ._version import version as __version__

__all__ = [
    "BendDataExtractor",
    "LineSmoother",
    "MultiTransect",
    "RidgeDataExtractor",
    "TransectDataExtractor",
    "__version__",
    "calc_ridge_amps",
    "calculate_ridge_metrics",
    "create_ridge_area_raster",
    "create_ridge_area_raster_fs",
    "create_transects",
    "map_amp_values",
]

from .delineation import (
    LineSmoother,
    create_ridge_area_raster,
    create_ridge_area_raster_fs,
)
from .ridge_metrics import (
    BendDataExtractor,
    RidgeDataExtractor,
    TransectDataExtractor,
    calc_ridge_amps,
    calculate_ridge_metrics,
    map_amp_values,
)
from .transecting import MultiTransect, create_transects
