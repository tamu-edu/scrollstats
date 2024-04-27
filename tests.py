# Testing suite for ScrollStats

from pathlib import Path
import geopandas as gpd
from scrollstats import LineSmoother


RIDGES_PATH = Path("example_data/input/LBR_025_ridges_manual.geojson")


def check_line_smoother_density():
    """Ensure that LineSmoother generates LineStrings with a sufficient point density"""

    manual_ridges = gpd.read_file(RIDGES_PATH)

    spacing = 1
    window = 5
    ls = LineSmoother(manual_ridges, spacing=spacing, window=window)
    smooth_ridges = ls.execute()

    tolerance = 0.01
    deviance = smooth_ridges.geometry.apply(
        lambda x: abs((len(x.coords) / x.length) - spacing)
    )
    assert all(deviance < tolerance) 


if __name__ == "__main__":
    check_line_smoother_density()