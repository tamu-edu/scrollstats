# Contains global variables such as file directories and other constants
import os
import sys
from pathlib import Path

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
GRASS_BASE = Path("auto")
GRASS_BIN = Path("auto")
GRASS_PYTHON = Path("auto")


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

class GrassLocator:
    """
    Locates the GRASS GIS resources that shipped with QGIS in order to run headless GRASS sessions
    QGIS app structure differs based on the user's OS and the user may have multiple versions of QGIS installed.

    This class contains the necesarry logic to navigate the above difficulties and return the relevant paths for the most recent verison of QGIS available
    """
    def __init__(self):
        self.platform = sys.platform

        if not (self.platform.startswith("darwin")) or (self.platform.startswith("windows")):
            raise OSError("Automatic searching for GRASS installations is only supported on MacOS and Windows")

        self.qgis = self.find_qgis()
        self.grass_base = self.find_grass_base()
        self.grass_bin = self.find_grass_bin()
    
    def find_qgis(self) -> Path:
        """Finds the installation of QGIS depending on the platform of the user."""
        
        if self.platform.startswith("darwin"):
            apps = Path("/Applications")
            qgis = apps / "QGIS.app"

            # The user may have "QGIS-LTR" or multiple versions of QGIS installed
            if not qgis.exists():
                qgis_candidates = list(apps.glob("QGIS*.app"))
                if qgis_candidates:
                    qgis = qgis_candidates[-1]
                else:
                    raise OSError(f"QGIS was not found in {apps}")

            return qgis
        
        elif self.platform.startswith("windows"):
            program_files = Path(r"C:\Program Files")

            # Check for multiple versions of QGIS. If multiple, choose the most recent
            qgis_candidates = sorted(program_files.glob("QGIS*"))
            if qgis_candidates:
                qgis = qgis_candidates[-1]
                return qgis
            else:
                raise OSError(f"QGIS was not found in {program_files}")
            
    def find_grass_base(self) -> Path:
        """Find the grass directory shipped with QGIS"""
        
        if self.platform.startswith("darwin"):
            resources = self.qgis / "Contents" / "Resources"
            # Grab the versioned `grass##` directory, grabs the most recent version if multiple
            grass_base = sorted(resources.glob("grass*"))[-1]
            return grass_base
        
        elif self.platform.startswith("windows"):
            super_grass = self.qgis / "apps" / "grass"
            # Should only be one versioned `grass##` directory, but this grabs the most recent if there are multiple
            grass_base = sorted(super_grass.glob("grass*"))[-1]

            return grass_base
        
    def find_grass_bin(self) -> Path:
        """Find the grass binary used to execute grass commands"""

        if self.platform.startswith("darwin"):
            grass_bin = self.grass_base / "grass"

            return grass_bin
        
        elif self.platform.startswith("windows"):
            qgis_bin = self.qgis / "bin"
            # Should only be one versioned `grass##` binary, but this grabs the most recent if there are multiple
            grass_bin = sorted(qgis_bin.glob("grass*.bat"))[-1]

            return grass_bin
        
# Check if user has already defined GRASS paths; automatically find if not
if (str(GRASS_BASE) == "auto") or (str(GRASS_BIN) == "auto") or (str(GRASS_PYTHON) == "auto"):
    gl = GrassLocator()
    GRASS_BASE = gl.grass_base
    GRASS_BIN = gl.grass_bin
    GRASS_PYTHON = GRASS_BASE / "etc" / "python"

# Set GRASS environment variables once located
os.environ["GISBASE"] = str(GRASS_BASE)
sys.path.append(GRASS_PYTHON) 

