# Testing suite for ScrollStats

from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import LineString

from scrollstats import CalcResidualTopography, BinaryClassifier, RasterAgreementAssessor, RasterClipper, RasterDenoiser, LineSmoother, create_transects, RidgeDataExtractor, calculate_ridge_metrics



# Data Paths
DEM_PATH = Path(f"example_data/input/LBR_025_dem.tif")
BEND_PATH = Path(f"example_data/input/LBR_025_bend.geojson")
PACKET_PATH = Path(f"example_data/input/LBR_025_packets.geojson")
CENTERLINE_PATH = Path(f"example_data/input/LBR_025_cl.geojson")
MANUAL_RIDGE_PATH = Path(f"example_data/input/LBR_025_ridges_manual.geojson")
OUTPUT_DIR = Path("example_data/output")

if not OUTPUT_DIR.is_dir():
    OUTPUT_DIR.mkdir(parents=True)

def test_line_smoother_density():
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


def test_rt_transformation():
    """Check that the residual topography raster contains values both above and below zero"""
    rt = CalcResidualTopography(DEM_PATH, 45, OUTPUT_DIR)
    rt_path = rt.execute()

    with rasterio.open(rt_path) as src:
        array = src.read(1)
        assert ((array > 0).any() and (array < 0).any())


def test_binary_classification():
    """Check that the binary raster only contains 1, 0, and nan around the perimeter"""
    rt = CalcResidualTopography(DEM_PATH, 45, OUTPUT_DIR)
    rt_path = rt.execute()

    binclass = BinaryClassifier(rt_path, 0, OUTPUT_DIR)
    binclass_path = binclass.execute()

    with rasterio.open(binclass_path) as src:
        array = src.read(1)
        assert all(np.unique(array[~np.isnan(array)]) == np.array([0., 1.]))

# def test_raster_agreement()

def test_raster_clipper():
    """Test that the clipped raster has more nans after the clip"""
    geom = gpd.read_file(BEND_PATH)["geometry"][0]

    rc = RasterClipper(DEM_PATH, geom, OUTPUT_DIR)
    clip_path = rc.execute()

    with rasterio.open(DEM_PATH) as src:
        dem = src.read(1)
        dem[dem < 1e-10] = np.nan
        dem_nan_count = np.isnan(dem).sum()
    
    with rasterio.open(clip_path) as src:
        clip = src.read(1)
        clip_nan_count = np.isnan(clip).sum()

    assert dem_nan_count < clip_nan_count


# def test_raster_denoiser()

def test_create_transects():
    """Test that the transects intersect all of the ridges """

    centerline = gpd.read_file(CENTERLINE_PATH)
    ridges_manual = gpd.read_file(MANUAL_RIDGE_PATH)

    ls = LineSmoother(ridges_manual, 1, window=5)
    ridges_smooth = ls.execute()

    transects = create_transects(centerline, ridges_smooth, 100, 300, 200, 5)

    ridge_ids = ridges_smooth["ridge_id"]
    intersections = transects.overlay(ridges_smooth, keep_geom_type=False)
    ridge_ids_from_itx = intersections["ridge_id"].unique()

    assert [i for i in ridge_ids if i not in ridge_ids_from_itx] == []


def test_ridge_data_extractor():
    """Test that the ridge data extractor calculates expected ridge width, amplitude, and spacing"""
    
    # Assume regularly spaced ridges with a 30m peak-to-peak distance
    p2p = 30
    transect_substring = LineString([[0, 50], [0+p2p, 50], [0+2*p2p, 50]])
    position = 1

    # Model the DEM profile along the transect as a cosine wave 
    # Cosine wave is 60m in length with a wavelength of 30m  
    p = 2*np.pi / p2p
    x = np.arange(60)
    a = 1
    e = 5
    dem = a*np.cos(p*x) + e

    # Create a binary wave from the DEM where swales are 0 and ridges are 1
    bw = dem.copy()
    bw[bw <= e] = 0
    bw[bw > e] = 1

    # Mock a ridge line that intersects the middle vertex of the transect substring
    ridge_vals = {
        "ridge_id": ["r_999"],
        "bend_id": ["LBR_999"],
        "deposit_year": [np.nan],
        "geometry": [LineString([[30, 80], [30, 50], [30, 20]])] 
    }
    ridge = gpd.GeoDataFrame(data=ridge_vals)

    # Calculate the three core metrics (ridge amplitude, width, and spacing) at the transect-ridge intersection
    rde_data = RidgeDataExtractor(transect_substring, position, ridge, dem, bw).dump_data()

    assert rde_data["ridge_amp"] == a*2
    assert rde_data["ridge_width"] == p2p / 2
    assert rde_data["pre_mig_dist"] == p2p


# def test_calculate_ridge_metrics():
