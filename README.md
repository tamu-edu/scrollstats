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

ScrollStats was developed using QGIS 3.16.6. However, unless there are major changes to the QGIS python API, processing toolbox, or the GRASS toolbox, the single [QGIS processing script](scrollstats/delineation/profileCurvature_QGIS.py) in ScrollStats should still work on any version of QGIS 3.

To download a specific version of QGIS, visit the [QGIS download index page](https://download.qgis.org/downloads/) and select your desired version.   


# Using ScrollStats
Example usage of ScrollStats has been broken up in the four following jupyter notebooks:
- [CreateVectorDatasets.ipynb](CreateVectorDatasets.ipynb)
- [TransformRasters.ipynb](TransformRasters.ipynb)
- [ClassifyRasters.ipynb](ClassifyRasters.ipynb)
- [CalculateRidgeMetrics.ipynb](CalculateRidgeMetrics.ipynb)

These four notebooks all include detailed instruction on the intended use of the ScrollStats library with an included [example dataset](example_data) of a bend from the Lower Brazos River, TX. Once you are comfortable using the library from the notebooks, feel free to edit the code or make your own scripts to suit your needs. 

The four notebooks above are written to process 1 bend at a time. However, all of these operations are designed to be easily incorporated into a `for` loop for batch processing of multiple bends, if desired.


## Set geoprocessing parameters
All geoprocessing parameters (such as window size) are kept in [parameters.py](parameters.py). Set all the parameters to the desired values before running any scripts. 

## Create Raster datasets
The general raster processing pipeline is as follows:
1. Apply transformations to DEMs
    - profile curvature  
    - residual topography
2. Clip transformed rasters to the boundaries of the ridge and swale topography
3. Apply binary classification to the rasters
    - 1 = Ridge; 0 = Swale
4. Assess where these two binary rasters agree in their classification
5. Denoise the Agreement raster to remove errant pixels

Follow the instructions in [TransformRasters.ipynb](TransformRasters.ipynb) to create the transform rasters

ScrollStats uses interactive jupyter notebooks as the user interface, however the underlying [scripting library](scrollstats) can also be used on its own. Instructions on how to use the jupyter notebooks are described below

### Create Vector Datasets
Follow the instructions in [CreateVectorData.ipynb](CreateVectorData.ipynb) to create the vector datasets listed above

### Transform rasters
Use the [TransformRasters.ipynb](TransformRasters.ipynb) notebook to apply the residual topography transformation to the DEMs.

Use the [profile_curvature_QGIS.py]() script to calculate the profile curvature transformation of the dems. Instructions on how to use this script in the QGIS python console can be found in [TransformRasters.ipynb](TransformRasters.ipynb)

### Classify Rasters
Once raster transformation is complete, use the [ClassifyRasters.ipynb](ClassifyRasters.ipynb) to complete steps 2-5 of the raster porcessing pipeline described above. 

[ClassifyRasters.ipynb](ClassifyRasters.ipynb) results in a binary raster where 1s represent ridge areas and 0s represent swale areas. This binary raster is then used in [NOTEBOOK_TBD.ipynb]() to calculate ridge metrics.

### Calculate Ridge Metrics
- Once all of the vector datasets are created and the raster areas are delineated, use the [CalculateRidgeMetrics.ipynb](CalculateRidgeMetrics.ipynb) to calculate the ridge metrics