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
## Create data directory structure
The variable `DATA_DIR` in [parameters.py](parameters.py) defines where all data input and output will occur. By default ScrollStats will create a `data` folder in this directory with the following structure:
```
data
├── input
│   └── dem
└── output
    ├── agr
    ├── agr_denoise
    ├── comp
    ├── profc
    ├── profc_bin
    ├── profc_clip
    ├── rt
    ├── rt_bin
    └── rt_clip
```
ScrollStats requires that all extent DEMs be stored in the `data/input/dem` folder. All other derived outputs will be stored in the various `data/outputs` folders

## Set geoprocessing parameters
All geoprocessing parameters (such as window size) are kept in [parameters.py](parameters.py). Set all the parameters to the desired values before running any scripts. 

## Create vector datasets
ScrollStats uses 4 types of vector data as input:
- bend boundary
- packet boundaries
- ridges
- centerline

The bend boundary in this case defines the boundary encompasses the raised platform of ridge and swale topography between the channel and the relatively smooth floodplain. Each bend boundary should have a corresponding bend_id that can be used to relate data together. For example, the 25th bend on the Lower Brazos River after the city of Waco, TX was given the beind_id `LBR_025`.

The packet boundaries are polygons that fit perfectly within, and conver entirely, the bend boundary polygons. Packet boundaries encompass groups of ridges with similar trajectories. Packet boundaries should have bend_id column as well as a packet_id column. The bend_id can be used as a foreign key to relate the packets to their bend and the simple packet_id (ex. `p_01`) can be used to diffferentiate the packets within each bend. There is no guarantee of an inherent order with packets, but in gernal, they can and should be numbered incrementally from the most ancestral to the most recent.

Ridge polylines are manually created for each ridge on the bend. Ridge polylines can be created before the raster ridge classification process, however it is reccommended the binary ridge area rasters be used to help inform the creation of ridge polylines. Ridge polylines should have a bend_id column, a ridge id column, and optionally a packet_id column, if packets are used. The bend_id and optional packet_id columns can be used as foreign keys to relate ridges to the larger morphological features and the simple ridge_id (ex. `r_001`) can be used to differentiate ridges within a bend. To a greater extent than the packets, there is no guarantee of an inherent order to the ridges on a given bend. However, as seen in the accompanying manuscript, ridges often can be ordered within packets. Ordering the incremental ids within packets is not necesarry for running ScrollStats. 

The channel centerline polyline should not intersect the bend boundary polygon at all and should extend past the channel-ward edges of the bend boundary.

A high vertex density and high degree of "smoothness" are necesarry for both the channel centerline and ridge lines when creating the migration pathways. However, relatively sparse and coarse polylines can be densified and smoothed with the inlcluded [line_smoother.py](scrollstats/delineation/line_smoother.py) script. 

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
