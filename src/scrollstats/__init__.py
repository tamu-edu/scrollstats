"""
Copyright (c) 2025 Andrew Vanderheiden. All rights reserved.

scrollstats: An open-source python library to calculate and extract morphometrics from scroll bar floodplains
"""

from __future__ import annotations

import lazy_loader as lazy

from ._version import version as __version__  # noqa: F401

submodules = [
    "delineation",
    "ridge_metrics",
    "transecting",
]

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=submodules,
    submod_attrs={
        "delineation": [
            "LineSmoother",
            "create_ridge_area_raster",
            "create_ridge_area_raster_fs",
        ],
        "ridge_metrics": [
            "BendDataExtractor",
            "RidgeDataExtractor",
            "TransectDataExtractor",
            "calc_ridge_amps",
            "calculate_ridge_metrics",
            "map_amp_values",
        ],
        "transecting": ["MultiTransect", "create_transects"],
    },
)

__all__.append("__version__")
