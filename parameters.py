# Contains global variables such as file directories and other constants
from pathlib import Path

# Data directories
# Set DATA_DIR to the directory where you would like all ScrollStats data to be stored.
# All other directories within the `data` directory will be created automatically 
# ScrollStats expects all extent DEMs to be in the `data/input/dem` directory
# All other outputs of ScrollStats will be stored in the `data/output` directory
DATA_DIR = Path("data")

INPUT_DIR = DATA_DIR / "input"
DEM_DIR = INPUT_DIR / "dem"

OUTPUT_DIR = DATA_DIR / "output"

PROFC_DIR = OUTPUT_DIR / "profc"
PROFC_CLIP_DIR = OUTPUT_DIR / "profc_clip"
PROFC_BIN_DIR = OUTPUT_DIR  / "profc_bin"

RT_DIR = OUTPUT_DIR / "rt"
RT_CLIP_DIR = OUTPUT_DIR / "rt_clip"
RT_BIN_DIR = OUTPUT_DIR  / "rt_bin"

COMP_DIR = OUTPUT_DIR / "comp"
AGR_DIR = OUTPUT_DIR / "agr"
AGR_DENOISE_DIR = OUTPUT_DIR / "agr_denoise"

RASTER_DIRS = {"input":INPUT_DIR, 
               "output": OUTPUT_DIR,
               "dem": DEM_DIR, 
               "profc": PROFC_DIR,
               "profc_clip": PROFC_CLIP_DIR,
               "profc_bin": PROFC_BIN_DIR,
               "rt": RT_DIR,
               "rt_clip": RT_CLIP_DIR,
               "rt_bin": RT_BIN_DIR,
               "comp":COMP_DIR,
               "agr":AGR_DIR,
               "agr_denoise":AGR_DENOISE_DIR}

# Create raster directories if they dont exist
for RASTER_DIR in RASTER_DIRS.values():
    if not RASTER_DIR.exists():
        RASTER_DIR.mkdir(parents=True)


# Raster Processing Constants
RASTER_WINDOW_SIZE = 45    # Measured in img px
SMALL_FEATS_SIZE = 50      # Measured in img px
ELONGATION_THRESHOLD = 80  # Percentage 

# Vector Creation Constants
SMOOTHING_WINDOW_SIZE = 5  # Measured in vertices
VERTEX_SPACING = 1         # Measured in linear unit of dataset (meteres for example datasets)

## See `CreatingVectorData.ipynb` for explanation of constants below and how they are used in migration pathway creation
SHOOT_DISTANCE = 300       # Distance that the N1 coordinate will shoot out from point P1; easured in linear unit of dataset
SEARCH_DISTANCE = 200      # Buffer radius used to search for an N2 coordinate on R2; measured in linear unit of dataset
DEV_FROM_90 = 5            # Max angular deviation from 90° allowed when searching for an N2 coordinate on R2; measured in degrees

