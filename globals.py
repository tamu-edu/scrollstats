# Contains global variables such as file directories and other constants
from pathlib import Path

# Data directories
INPUT_DIR = Path("data/input")
DEM_DIR = INPUT_DIR / "dem"

OUTPUT_DIR = Path("data/output")

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
    if not RASTER_DIR.exists:
        RASTER_DIR.mkdir()


# Constants
WINDOW_SIZE = 45           # Measured in img px
SMALL_FEATS_SIZE = 50      # Measured in img px
ELONGATION_THRESHOLD = 80  # Percentage 
