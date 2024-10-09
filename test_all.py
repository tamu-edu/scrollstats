# Testing for ScrollStats

from pathlib import Path
import tempfile
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Point, LineString

from scrollstats import LineSmoother, create_transects, RidgeDataExtractor, TransectDataExtractor, BendDataExtractor, calculate_ridge_metrics

from scrollstats.delineation.ridge_area_raster import clip_raster

from scrollstats.delineation.raster_classifiers import (
    quadratic_profile_curvature,
    residual_topography
)

from scrollstats.delineation import (
    create_ridge_area_raster,
    create_ridge_area_raster_fs
)


# Data Paths
DEM_PATH = Path(f"example_data/input/LBR_025_dem.tif")
BEND_PATH = Path(f"example_data/input/LBR_025_bend.geojson")
PACKET_PATH = Path(f"example_data/input/LBR_025_packets.geojson")
CENTERLINE_PATH = Path(f"example_data/input/LBR_025_cl.geojson")
MANUAL_RIDGE_PATH = Path(f"example_data/input/LBR_025_ridges_manual.geojson")


def generate_waves(wavelength:int=30, amp:int=1, vert:int=5, signal_length:int = 60) -> np.ndarray:
    """
    Generate a 2D cosine wave with known properties for testing
    """

    p = 2*np.pi / wavelength
    x = np.arange(signal_length)
    
    dem_1d = amp*np.cos(p*x) + vert

    dem_2d = np.multiply(np.ones((signal_length, signal_length)), dem_1d)
    
    return dem_2d

def generate_ridges(y:int=30, rep:int=5, crs:str="EPSG:32139"):
    """Generate mock ridge lines with known spacing for testing """

    r = {
        "ridge_id": [f"r_{i:03d}" for i in range(0, rep-1)],
        "bend_id": ["LBR_999" for i in range(0, rep-1)],
        "deposit_year": [np.nan for i in range(0, rep-1)],
        "geometry": [LineString([[i, 0], [i, y*5]]) for i in range(y, y*5, y)]
    }
    return gpd.GeoDataFrame(data=r, geometry="geometry", crs=crs)


def generate_transects(y:int=30, rep:int=5, crs:str="EPSG:32139"):
    """Generate mock transects with known spacing for testing """

    t = {
        "transect_id":[f"t_{i:03d}" for i in range(0, rep-1)],
        "bend_id" :["LBR_999" for i in range(0, rep-1)],
        "geometry" : [LineString([[j, i] for j in range(0, y*rep, y)]) for i in range(y, y*rep, y)]
    }
    return gpd.GeoDataFrame(data=t, geometry="geometry", crs=crs)


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


def test_quadratic_profile_curvature():
    """Check that the profile curvature transformation accurately identifies ridge areas"""
    # Generate 2D cosine waves with known properties
    wavelength = 30
    amp = 1
    vert = 5  # vertical adjustment for wave from the x-axis (inflection point)
    dem = generate_waves(wavelength=wavelength, 
                         amp=amp,
                         vert=vert,
                         signal_length=wavelength*2)
    
    # Apply profile curvature to assess landscape convexity
    profc = quadratic_profile_curvature(dem, window=int(wavelength/2))

    # Select areas in the generated waves known to be ridges
    known_ridge_area = dem > vert
    known_swale_area = dem < vert

    # Assess if the known ridge/swale areas are +/-
    accurate_ridges = profc[known_ridge_area] > 0
    accurate_swales = profc[known_swale_area] < 0

    assert np.sum(accurate_ridges) > 0.9*accurate_ridges.size
    assert np.sum(accurate_swales) > 0.9*accurate_swales.size


def test_residual_topography():
    """Check that the residual topography transformation accurately identifies ridge areas"""
    # Generate 2D cosine waves with known properties
    wavelength = 30
    amp = 1
    vert = 5  # vertical adjustment for wave from the x-axis (inflection point)
    dem = generate_waves(wavelength=wavelength, 
                         amp=amp,
                         vert=vert,
                         signal_length=wavelength*2)
    
    # Apply residual topography to assess landscape prominence
    rt = residual_topography(dem, w=int(wavelength/2))
    rt_not_nan = ~np.isnan(rt)

    # Select areas in the generated waves known to be ridges
    known_ridge_area = dem > vert
    known_swale_area = dem < vert

    # Assess if the known ridge/swale areas are +/- (excluding nan areas)
    accurate_ridges = rt[known_ridge_area & rt_not_nan] > 0
    accurate_swales = rt[known_swale_area & rt_not_nan] < 0

    assert np.sum(accurate_ridges) > 0.9*accurate_ridges.size
    assert np.sum(accurate_swales) > 0.9*accurate_swales.size


def test_clip_raster():
    """Test if the array window shrunk as a result of the clip and if all values outside to the geometry are cast to np.nan """
    dem_ds = rasterio.open(DEM_PATH)
    gdf = gpd.read_file(BEND_PATH)
    geom = gdf.loc[0, "geometry"]

    array_clip, clipped_mask, clipped_meta = clip_raster(dem_ds, geom, no_data=np.nan)

    assert clipped_meta["width"] * clipped_meta["height"] < dem_ds.width * dem_ds.height
    assert np.all(np.isnan(array_clip[clipped_mask]))


def test_create_ridge_area_raster():
    """
    Test the following aspects of the create_ridge_area_raster function:
        1. no data is set correctly
        2. the output binary raster accurately identifies ridge areas
        3. the output dem raster window has shrunk as a result of the clip and if all values outside to the geometry are cast to the no data value 
    """
    wavelength = 30 # wavelength of cosine wave
    amp = 1 # amplitude of cosine wave
    vert = 5  # vertical adjustment of the cosine wave from x axis
    rep = 5  # repetitions of the cosine wave
    
    no_data_value = np.nan  # no data value set for input and output rasters

    # Mock DEM
    dem = generate_waves(wavelength, amp, vert, wavelength*rep)

    # Mock the bend area polygon by creating a circle in the middle of the area
    point = Point([i//2 for i in dem.shape])
    circle = point.buffer((wavelength/2)*(rep-1))

    # Write DEM to disk
    with tempfile.NamedTemporaryFile(suffix=".tiff") as fp:
        with rasterio.open(fp.name, "w", driver="GTiff", 
                        width=dem.shape[1], height=dem.shape[0], count=1, 
                        dtype=dem.dtype, crs="EPSG:32139", nodata=no_data_value) as dst:
            dst.write(dem, 1)

        dem_ds = rasterio.open(fp.name)
        
    binary_clip, dem_clip, binary_meta = create_ridge_area_raster(
        dem_ds = dem_ds, 
        geometry = circle, 
        no_data_value = no_data_value,
        window=int(wavelength/2), dx=1, small_feats_size=1)
        

    known_ridge_areas = dem_clip > vert
    no_data_area = (binary_clip != 0) & (binary_clip!=1)

    # Do all classified ridge pixels correspond to known ridge pixels?
    assert known_ridge_areas[binary_clip == 1].all()

    # Were nearly all of the known ridge pixels classified as ridge pixels?
    assert ((binary_clip == 1).sum() / known_ridge_areas.sum()) > 0.9

    # Was the no data value set correctly?
    if np.isnan(no_data_value):
        assert np.isnan(binary_clip[no_data_area]).all()
    else:
        assert (binary_clip[no_data_area] == no_data_value).all()



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
    """Test that the RidgeDataExtractor calculates expected ridge width, amplitude, and spacing"""
    
    # Generate a 1D DEM signal with 30m peak-to-peak distance (wavelength)
    p2p = 30
    vert = 5
    amp = 1
    signal_length = p2p*2
    dem = generate_waves(wavelength=p2p, vert=vert, signal_length=signal_length)[0]

    # Create a straight transect at position 1
    transect_substring = LineString([[0*p2p, 50], [1*p2p, 50], [2*p2p, 50]])
    position = 1

    # Create a binary wave from the DEM where swales are 0 and ridges are 1
    bw = dem.copy()
    bw[bw <= vert] = 0
    bw[bw > vert] = 1

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

    assert rde_data["ridge_amp"] == amp*2
    assert rde_data["ridge_width"] == p2p / 2
    assert rde_data["pre_mig_dist"] == p2p


def test_transect_data_extractor():
    """
    Test that the TransectDataExtractor calculates expected ridge width, amplitude, and spacing for an entire transect
    
    Mock ridges and transects are perpendicular to each other on a 30m grid.
    Mock DEM is a simple cosine wave with a 30m wavelength.
    """

    y = 30 # wavelength of cosine wave
    amp = 1 # amplitude of cosine wave
    vert = 5  # vertical adjustment of the cosine wave from x axis
    rep = 5  # repetitions of the cosine wave

    # Mock DEM
    dem = generate_waves(y, amp, 5, y*rep)

    # Mock ridges
    ridges = generate_ridges(y, rep)

    # Mock transects
    transects = generate_transects(y, rep)

    # Calculate ridge metrics for the first transect
    tde = TransectDataExtractor(
        **transects.loc[0, ["transect_id", "geometry"]], 
        dem_signal=dem[0], 
        bin_signal=(dem[0] > vert).astype(int),
        ridges=ridges
    )

    transect_metrics = tde.calc_ridge_metrics()

    assert all(transect_metrics["ridge_amp"] == amp*2)
    assert all(transect_metrics["ridge_width"] == y / 2)
    assert all(transect_metrics["pre_mig_dist"] == y)


def test_bend_data_extractor():
    """
    Test that the BendDataExtractor calculates expected ridge width, amplitude, and spacing for an entire bend.
    
    Mock ridges and transects are perpendicular to each other on a 30m grid.
    Mock DEM is a simple cosine wave with a 30m wavelength.
    Mock Binary Raster is a binary classification of DEM where 1s are values greater than a threshold and 0s are less than.
    Both DEM and Binary Raster are converted into temporary rasterio Dataset objects as required by BendDataExtractor
    """
        
    y = 30 # wavelength of cosine wave
    amp = 1 # amplitude of cosine wave
    vert = 5  # vertical adjustment of the cosine wave from x axis
    rep = 5  # repetitions of the cosine wave

    # Mock DEM and Binary Raster
    dem = generate_waves(y, amp, vert, y*rep)
    bin_arr = (dem > vert).astype(int)

    # Write DEM to disk
    with tempfile.NamedTemporaryFile(suffix=".tiff") as fp:
        with rasterio.open(fp.name, "w", driver="GTiff", 
                        width=dem.shape[1], height=dem.shape[0], count=1, 
                        dtype=dem.dtype, crs="EPSG:32139") as dst:
            dst.write(dem, 1)

            dem_ras = rasterio.open(fp.name)

    # Write Binary Raster to disk
    with tempfile.NamedTemporaryFile(suffix=".tiff") as fp:
        with rasterio.open(fp.name, "w", driver="GTiff", 
                        width=bin_arr.shape[1], height=bin_arr.shape[0], count=1, 
                        dtype=bin_arr.dtype, crs="EPSG:32139") as dst:
            dst.write(bin_arr, 1)
        
            bin_ras = rasterio.open(fp.name)

    # Mock ridges
    ridges = generate_ridges(y, rep)

    # Mock transects
    transects = generate_transects(y, rep)

    # Create BendDataExtractor to calculate metrics
    bde = BendDataExtractor(transects, bin_ras, dem_ras, ridges)

    assert all(bde.itx_metrics["ridge_amp"] == amp*2)
    assert all(bde.itx_metrics["ridge_width"] == y / 2)
    assert all(bde.itx_metrics["pre_mig_dist"] == y)