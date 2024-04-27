import sys
from pathlib import Path
from typing import Protocol, Tuple


class GrassLocator(Protocol):
    """
    Locates the GRASS GIS resources downloaded by the user
    GRASS is used to calculate the profile curvature raster with the r.param.scale tool
    """

    def get_grass_base() -> Path: ...

    def get_grass_bin() -> Path: ...

    def get_grass_py() -> Path: ...

    def get_grass_version() -> str: ...


class GrassLocator_MacOS:
    """
    Finds the GRASS GIS resources on MacOS.
    Implements the GrassLocator protocol defined above.
    """

    def __init__(self):
        self.grass_base = self.get_grass_base()
        self.grass_bin = self.get_grass_bin()

    def get_grass_base(self):
        """Find the base folder in the GRASS application that contains all grass resources"""

        apps = Path("/Applications")

        # Search for GRASS in the applitcations folder
        # The user may have multiple versions or none
        # Select the latest version if multiple
        grass_candidates = sorted(apps.glob("GRASS*.app"))

        try:
            grass_app = grass_candidates[-1]
        except IndexError:
            raise FileNotFoundError(f"GRASS GIS not found in {apps}")

        return grass_app / "Contents" / "Resources"

    def get_grass_bin(self):
        """Find the binary executable for GRASS GIS"""
        grass_base = self.grass_base
        bin_dir = grass_base / "bin"
        return next(bin_dir.glob("grass*"))

    def get_grass_py(self):
        """Find the python library shipped with GRASS"""
        grass_base = self.grass_base
        return grass_base / "etc" / "python"

    def get_grass_version(self) -> str:
        """Infers grass verison from the last characters of GRASS_BIN"""
        version_num = self.grass_bin.stem[-2:]
        if version_num.isnumeric():
            return version_num
        else:
            raise ValueError(
                f"Unable to infer grass verison from `GRASS_BIN`:{self.grass_bin}"
            )


def locate_grass_resources(gl: GrassLocator) -> Tuple[Path, Path, Path, str]:
    """Function used to enforce the GrassLocator protocol"""
    return (
        gl.get_grass_base(),
        gl.get_grass_bin(),
        gl.get_grass_py(),
        gl.get_grass_version(),
    )


def get_grass_resources() -> Tuple[Path, Path, Path, str]:
    if sys.platform.startswith("darwin"):
        gl = GrassLocator_MacOS()
    # TODO: add support for linux here with a GrassLocator_Linux()
    else:
        raise OSError(
            "ScrollStats can only automatically find GRASS resources on MacOS"
        )

    grass_base, grass_bin, grass_py, grass_version = locate_grass_resources(gl)

    # Check grass version before returning resource paths
    if not grass_version.startswith("7"):
        raise ValueError(
            f"ScrollStats is only compatible with GRASS 7.*. Detected version: {'.'.join(grass_version)}"
        )

    return (grass_base, grass_bin, grass_py, grass_version)


# class GrassLocator_QGIS:
#     """
#     Locates the GRASS GIS resources that shipped with QGIS in order to run headless GRASS sessions
#     QGIS app structure differs based on the user's OS and the user may have multiple versions of QGIS installed.

#     This class contains the necesarry logic to navigate the above difficulties and return the relevant paths for the most recent verison of QGIS available
#     """
#     def __init__(self):
#         self.platform = sys.platform

#         if not (self.platform.startswith("darwin") or self.platform.startswith("win")):
#             raise OSError("Automatic searching for GRASS installations is only supported on MacOS and Windows")

#         self.qgis = self.find_qgis()
#         self.grass_base = self.find_grass_base()
#         self.grass_bin = self.find_grass_bin()

#     def find_qgis(self) -> Path:
#         """Finds the installation of QGIS depending on the platform of the user."""

#         if self.platform.startswith("darwin"):
#             apps = Path("/Applications")
#             qgis = apps / "QGIS.app"

#             # The user may have "QGIS-LTR" or multiple versions of QGIS installed
#             if not qgis.exists():
#                 qgis_candidates = list(apps.glob("QGIS*.app"))
#                 if qgis_candidates:
#                     qgis = qgis_candidates[-1]
#                 else:
#                     raise OSError(f"QGIS was not found in {apps}")

#             return qgis

#         elif self.platform.startswith("win"):
#             program_files = Path(r"C:\Program Files")

#             # Check for multiple versions of QGIS. If multiple, choose the most recent
#             qgis_candidates = sorted(program_files.glob("QGIS*"))
#             if qgis_candidates:
#                 qgis = qgis_candidates[-1]
#                 return qgis
#             else:
#                 raise OSError(f"QGIS was not found in {program_files}")

#     def find_grass_base(self) -> Path:
#         """Find the grass directory shipped with QGIS"""

#         if self.platform.startswith("darwin"):
#             resources = self.qgis / "Contents" / "Resources"
#             # Grab the versioned `grass##` directory, grabs the most recent version if multiple
#             grass_base = sorted(resources.glob("grass*"))[-1]
#             return grass_base

#         elif self.platform.startswith("win"):
#             super_grass = self.qgis / "apps" / "grass"
#             # Should only be one versioned `grass##` directory, but this grabs the most recent if there are multiple
#             grass_base = sorted(super_grass.glob("grass*"))[-1]

#             return grass_base

#     def find_grass_bin(self) -> Path:
#         """Find the grass binary used to execute grass commands"""

#         if self.platform.startswith("darwin"):
#             grass_bin = self.grass_base / "grass"

#             return grass_bin

#         elif self.platform.startswith("win"):
#             qgis_bin = self.qgis / "bin"
#             # Should only be one versioned `grass##` binary, but this grabs the most recent if there are multiple
#             grass_bin = sorted(qgis_bin.glob("grass*.bat"))[-1]

#             return grass_bin
