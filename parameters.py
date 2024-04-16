# Contains global variables such as file directories and other constants
import os
import sys
from pathlib import Path

from GrassResources import get_grass_resources

# Data directories
# Set DATA_DIR to the directory where you would like all ScrollStats data to be stored.
# All other directories within the `data` directory will be created automatically 
DATA_DIR = Path("data")
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
GRASS_DIR = OUTPUT_DIR / "grassdata"

# Create data directories if they dont exist
for d in [INPUT_DIR, OUTPUT_DIR, GRASS_DIR]:
    if not d.exists():
        d.mkdir(parents=True)


# Set GRASS environment variables
# It is assumed that the user will be using the version of GRASS shipped with QGIS
# If the wants to specify which GRASS install to use, the path variables can be modified below
# GRASS_BASE = Path("auto")
# GRASS_BIN = Path("auto")
# GRASS_PYTHON = Path("auto")
# GRASS_VERSION = "auto"
GRASS_BASE = Path("/Applications/GRASS-7.8.app/Contents/Resources")
GRASS_BIN = Path("/Applications/GRASS-7.8.app/Contents/Resources/bin/grass78")
GRASS_PYTHON = Path("/Applications/GRASS-7.8.app/Contents/Resources/etc/python")
GRASS_VERSION = "78"

# Raster Processing Constants
RASTER_WINDOW_SIZE = 45    # Measured in img px
SMALL_FEATS_SIZE = 500     # Measured in img px
ELONGATION_THRESHOLD = 80  # Percentage 

# Vector Creation Constants
SMOOTHING_WINDOW_SIZE = 5  # Measured in vertices
VERTEX_SPACING = 1         # Measured in linear unit of dataset (meteres for example datasets)

## See `CreatingVectorData.ipynb` for explanation of constants below and how they are used in migration pathway creation
SHOOT_DISTANCE = 300       # Distance that the N1 coordinate will shoot out from point P1; easured in linear unit of dataset
SEARCH_DISTANCE = 200      # Buffer radius used to search for an N2 coordinate on R2; measured in linear unit of dataset
DEV_FROM_90 = 5            # Max angular deviation from 90° allowed when searching for an N2 coordinate on R2; measured in degrees


# ============================================================================




# Check if user has already defined GRASS paths; automatically find if not
if (str(GRASS_BASE) == "auto" or str(GRASS_BIN) == "auto" or str(GRASS_PYTHON) == "auto" or GRASS_VERSION == "auto"):
    GRASS_BASE, GRASS_BIN, GRASS_PYTHON, GRASS_VERSION = get_grass_resources()

# Set GRASS environment variables once located
os.environ["GISBASE"] = str(GRASS_BASE)
sys.path.append(str(GRASS_PYTHON)) 

