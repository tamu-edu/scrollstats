from __future__ import annotations

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[],
    submod_attrs={
        "line_smoother": ["LineSmoother"],
        "raster_classifiers": ["quadratic_profile_curvature", "residual_topography"],
        "ridge_area_raster": [
            "create_ridge_area_raster",
            "create_ridge_area_raster_fs",
        ],
    },
)
