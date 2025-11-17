from __future__ import annotations

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[],
    submod_attrs={
        "calc_ridge_metrics": ["calculate_ridge_metrics"],
        "data_extractors": [
            "BendDataExtractor",
            "RidgeDataExtractor",
            "TransectDataExtractor",
        ],
        "ridge_amplitude": ["calc_ridge_amps", "map_amp_values"],
    },
)
