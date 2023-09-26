# ScrollStats

An open-source python library to calculate and extract morphometrics from scrollbar floodplains.


# Getting Started

## Create conda environment
Create a conda environment from the provided `environment.yml` file with the following commands
```shell
# Navigate to this directory
cd path/to/scrollstats

# Create the environment
conda env create -f environment.yml
```

## Install QGIS3
To download the latest version of QGIS, visit the [QGIS download page](https://www.qgis.org/en/site/forusers/download.html) and download the version appropriate for your OS. 

ScrollStats was developed using QGIS 3.16.6. However, unless there are major changes to the QGIS python API, processing toolbox, or the GRASS toolbox, the single [QGIS processing script](scrollstats/delineation/profileCurvature_QGIS.py) in ScrollStats should still work.

To download a specific version of QGIS, visit the [QGIS download index page](https://download.qgis.org/downloads/) and select your desired version.   
## Install Postgress + PostGIS 
- TODO


# Using ScrollStats

## Create vector datasets
- TODO
## Create Raster datasets
The general raster processing pipeline is as follows:
1. Apply transformations to rasters
    - profile curvature  
    - residual topography
2. Clip transformed rasters to the boundaries of the ridge and swale topography
3. Apply binary classification to the rasters
    - 1 = Ridge; 0 = Swale
4. Assess where these two binary rasters agree in their classification
5. Denoise the Agreement raster to remove errant pixels

ScrollStats is designed to work on either many or single bends. ScrollStats expects the DEMs for all bends to be saved in the [data/input/dem](data/input/dem) folder.

ScrollStats uses interactive jupyter notebooks as the user interface, however the underlying [scripting library](scrollstats) can also be used on its own. Instructions on how to use the jupyter notebooks are described below

### Transform rasters
Use the [TransformRasters.ipynb](TransformRasters.ipynb) notebook to apply the residual topography transformation to the DEMs.

Use the [profile_curvature_QGIS.py]() script to calculate the profile curvature transformation of the dems. Instructions on how to use this script in the QGIS python console can be found in [TransformRasters.ipynb](TransformRasters.ipynb)

### Classify Rasters
Once raster transformation is complete, use the [ClassifyRasters.ipynb](ClassifyRasters.ipynb) to complete steps 2-5 of the raster porcessing pipeline described above. 

[ClassifyRasters.ipynb](ClassifyRasters.ipynb) results in a set of binary rasters, saved in [data/output/agr_denoise](data/output/agr_denoise), where 1s represent ridge areas and 0s represent swale areas. These binary rasters are then used in [NOTEBOOK_TBD.ipynb]() to calculate ridge metrics.

## Calculate Ridge Metrics
- TODO





# TO DO:
- [x] rewrite metric calculation for each itx
- [x] add transect creation modules
- [x] add delineation modules
- [x] incorporate the relative migration code
- [x] add example data 
- [x] add all figure creation scripts
- [ ] create a 'main' script that can handle delineation, transect creation, and metrics
- [x] assess use of sqlite for all data
- [x] figure out how to interface with db and local files
- [ ] create sql db generation file
