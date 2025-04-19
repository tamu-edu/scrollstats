"""
Copyright (c) 2025 Andrew Vanderheiden. All rights reserved.

scrollstats: An open-source python library to calculate and extract morphometrics from scroll bar floodplains
"""

from __future__ import annotations

from ._version import version as __version__

__all__ = ["__version__"]

from .delineation import LineSmoother
from .delineation import create_ridge_area_raster
from .delineation import create_ridge_area_raster_fs

from .transecting import create_transects
from .transecting import MultiTransect

from .ridge_metrics import RidgeDataExtractor
from .ridge_metrics import TransectDataExtractor
from .ridge_metrics import BendDataExtractor
from .ridge_metrics import calc_ridge_amps
from .ridge_metrics import map_amp_values
from .ridge_metrics import calculate_ridge_metrics