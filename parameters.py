# Contains global variables such as file directories and other constants
from pathlib import Path

# Data directories
# Set DATA_DIR to the directory where you would like all ScrollStats data to be stored.
# All other directories within the `data` directory will be created automatically 
DATA_DIR = Path("data")

INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

# Create data directories if they dont exist
for d in [INPUT_DIR, OUTPUT_DIR]:
    if not d.exists():
        d.mkdir(parents=True)


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

