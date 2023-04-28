__all__ = ["DB_PW",
           "GLOBAL_GEOG_CRS",
           "RASTER_PATHS"]

# Password for DB
DB_PW = "your_db_password"

# Global-scale CRS used throughout the scripts
GLOBAL_GEOG_CRS = "EPSG:4326"

RASTER_PATHS = {"LBR_025":{
                    "dem":"data/raster/dem_clip/sb_1_025_newclip.tif",
                    "bin":"data/raster/bin_clip/sb_1_025_agreement_45px_binclass_clip_dn_Buff100m_SmFt500m_ET80p_newclip.tif"
                }, 
                "LBR_029":{
                    "dem":"data/raster/dem_clip/sb_1_029_newclip.tif",
                    "bin":"data/raster/bin_clip/sb_1_029_agreement_45px_binclass_clip_dn_Buff100m_SmFt500m_ET80p_newclip.tif"
                }, 
                "LBR_043":{
                    "dem":"data/raster/dem_clip/sb_1_043_newclip.tif",
                    "bin":"data/raster/bin_clip/sb_1_043_agreement_45px_binclass_clip_dn_Buff100m_SmFt500m_ET80p_newclip.tif"
                }, 
                "LBR_077":{
                    "dem":"data/raster/dem_clip/sb_4_077_newclip.tif",
                    "bin":"data/raster/bin_clip/sb_4_077_agreement_45px_binclass_clip_dn_Buff100m_SmFt500m_ET80p_newclip.tif"
                },
                "MIS_005":{
                    "dem":"data/raster/dem_clip/m5_merged_clipped.tif",
                    "bin":"data/raster/bin_clip/m5_merged_agreement_15px_clip_binclass_dn_Buff0m_SmFt035m_ET80p.tif"
                },
                "BEA_002":{
                    "dem":"",
                    "bin":""
                }}