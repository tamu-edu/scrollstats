from pathlib import Path
import warnings

import rasterio
from shapely.errors import ShapelyDeprecationWarning
import matplotlib.pyplot as plt

from scrollstats import BendDataset, calculate_ridge_metrics


bend_ids = ["LBR_025", 
            "LBR_029", 
            "LBR_043", 
            "LBR_077",
            "MIS_005",
            "BEA_002"]

raster_paths = {"LBR_025":{
                    "dem":"/Users/avan/FLUD/BrazosScrolls/data/raster/dem/sb_1_025_clip.tif",
                    "bin":"/Users/avan/FLUD/BrazosScrolls/data/r10/agreement-clip-denoise/sb_1_025_agreement_45px_binclass_clip_dn_Buff100m_SmFt500m_ET80p.tif"
                }, 
                "LBR_029":{
                    "dem":"/Users/avan/FLUD/BrazosScrolls/data/raster/dem/sb_1_029_clip.tif",
                    "bin":"/Users/avan/FLUD/BrazosScrolls/data/r10/agreement-clip-denoise/sb_1_029_agreement_45px_binclass_clip_dn_Buff100m_SmFt500m_ET80p.tif"
                }, 
                "LBR_043":{
                    "dem":"/Users/avan/FLUD/BrazosScrolls/data/raster/dem/sb_1_043_clip.tif",
                    "bin":"/Users/avan/FLUD/BrazosScrolls/data/r10/agreement-clip-denoise/sb_1_043_agreement_45px_binclass_clip_dn_Buff100m_SmFt500m_ET80p.tif"
                }, 
                "LBR_077":{
                    "dem":"/Users/avan/FLUD/BrazosScrolls/data/raster/dem/sb_4_077_clip.tif",
                    "bin":"/Users/avan/FLUD/BrazosScrolls/data/r10/agreement-clip-denoise/sb_4_077_agreement_45px_binclass_clip_dn_Buff100m_SmFt500m_ET80p.tif"
                },
                "MIS_005":{
                    "dem":"/Users/avan/FLUD/BrazosScrolls/data/strick/raster/dem/m5_merged_clipped.tif",
                    "bin":"/Users/avan/FLUD/BrazosScrolls/data/strick/raster/agreement/m5_merged_agreement_15px_clip_binclass_dn_Buff0m_SmFt035m_ET80p.tif"
                },
                "BEA_002":{
                    "dem":"",
                    "bin":""
                }}

for bend in bend_ids:

    print(f"Bend: {bend}")
    print("=============")

    # Get raster info 
    local_crs = None  # need to use the raster crs for the vectors
    dem = None
    if Path(raster_paths[bend]["dem"]).is_file():
        dem = rasterio.open(raster_paths[bend]["dem"])
        local_crs = dem.crs

    bin_raster = None
    if Path(raster_paths[bend]["bin"]).is_file():
        bin_raster = rasterio.open(raster_paths[bend]["bin"])

    # Get vector info 
    bend_ds = BendDataset(bend)
    if local_crs:
        ridges = bend_ds.get_ridges().to_crs(local_crs)
        transects = bend_ds.get_transects().to_crs(local_crs)
        try:
            packets = bend_ds.get_packets().to_crs(local_crs)
        except ValueError:
            packets = None

    else:
        # If there is not a crs from the rasters (ie. no rasters) then simply reproject with the closest UTM
        ridges = bend_ds.get_ridges(True)
        transects = bend_ds.get_transects(True)
        try:
            packets = bend_ds.get_packets(True)
        except ValueError:
            packets = None



    # Calc metrics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ShapelyDeprecationWarning)
        rich_transects, itx = calculate_ridge_metrics(transects, bin_raster, dem, ridges)

    # Give feedback
    print(rich_transects.head())
    print("="*75)
    print("="*75)
    print(itx.head())
    

    # Make Plots
    fig, ax = plt.subplots(1,1, figsize=(10,8))
    itx.plot(ax=ax, column="ridge_amp", markersize="ridge_width", legend=True)
    ridges.plot(ax=ax, color="k", ls="--", lw=0.5, zorder=0)

    out_path = Path(f"/Users/avan/Desktop/TestPlots")
    if not out_path.is_dir():
        out_path.mkdir()                
    out_path = out_path / f"{bend}.png"
    plt.savefig(out_path, dpi=300)


