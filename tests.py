# Testing suite for ScrollStats

from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio

from scrollstats import CalcResidualTopography, BinaryClassifier, RasterAgreementAssessor, RasterClipper, RasterDenoiser, LineSmoother, create_transects, calculate_ridge_metrics



# Data Paths
DEM_PATH = Path(f"example_data/input/LBR_025_dem.tif")
BEND_PATH = Path(f"example_data/input/LBR_025_bend.geojson")
PACKET_PATH = Path(f"example_data/input/LBR_025_packets.geojson")
CENTERLINE_PATH = Path(f"example_data/input/LBR_025_cl.geojson")
MANUAL_RIDGE_PATH = Path(f"example_data/input/LBR_025_ridges_manual.geojson")
OUTPUT_DIR = Path("example_data/output")

if not OUTPUT_DIR.is_dir():
    OUTPUT_DIR.mkdir(parents=True)

def check_line_smoother_density():
    """Ensure that LineSmoother generates LineStrings with a sufficient point density"""

    manual_ridges = gpd.read_file(MANUAL_RIDGE_PATH)

    spacing = 1
    window = 5
    ls = LineSmoother(manual_ridges, spacing=spacing, window=window)
    smooth_ridges = ls.execute()

    tolerance = 0.01
    deviance = smooth_ridges.geometry.apply(
        lambda x: abs((len(x.coords) / x.length) - spacing)
    )
    assert all(deviance < tolerance) 


def check_rt_transformation():
    """Check that the residual topography raster contains values both above and below zero"""
    rt = CalcResidualTopography(DEM_PATH, 45, OUTPUT_DIR)
    rt_path = rt.execute()

    with rasterio.open(rt_path) as src:
        array = src.read(1)
        assert ((array > 0).any() and (array < 0).any())


def check_binary_classification():
    """Check that the binary raster only contains 1, 0, and nan around the perimeter"""
    rt = CalcResidualTopography(DEM_PATH, 45, OUTPUT_DIR)
    rt_path = rt.execute()

    binclass = BinaryClassifier(rt_path, 0, OUTPUT_DIR)
    binclass_path = binclass.execute()

    with rasterio.open(binclass_path) as src:
        array = src.read(1)
        assert all(np.unique(array[~np.isnan(array)]) == np.array([0., 1.]))



# if __name__ == "__main__":
#     check_line_smoother_density()
#     check_rt_transformation()
#     check_binary_classification()