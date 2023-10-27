# ScrollStats

An open-source python library to calculate and extract morphometrics from scrollbar floodplains.


# Getting Started

### Create conda environment

Create a conda environment from the provided `environment.yml` file with the following commands
```shell
# Navigate to this directory
cd path/to/scrollstats

# Create the environmen
conda env create -f environment.yml
```


### Install QGIS3

To download the latest version of QGIS, visit the [QGIS download page](https://www.qgis.org/en/site/forusers/download.html) and download the version appropriate for your OS. 

ScrollStats was developed using QGIS 3.16.6. However, unless there are major changes to the QGIS python API, processing toolbox, or the GRASS toolbox, the single [QGIS processing script](scrollstats/delineation/profileCurvature_QGIS.py) in ScrollStats should still work on any version of QGIS 3.

To download a specific version of QGIS, visit the [QGIS download index page](https://download.qgis.org/downloads/) and select your desired version.   


# Using ScrollStats

ScrollStats uses interactive jupyter notebooks as the user interface, however the underlying [scripting library](scrollstats) can also be used on its own. Example usage of ScrollStats has been broken up in the four following notebooks:
- [CreateVectorDatasets.ipynb](CreateVectorDatasets.ipynb)
- [TransformRasters.ipynb](TransformRasters.ipynb)
- [ClassifyRasters.ipynb](ClassifyRasters.ipynb)
- [CalculateRidgeMetrics.ipynb](CalculateRidgeMetrics.ipynb)

These four notebooks all include detailed instruction on the intended use of the ScrollStats library with an included [example dataset](example_data) of a bend from the Lower Brazos River, TX. Once you are comfortable using the library from the notebooks, feel free to edit the code or make your own scripts to suit your needs. 

The four notebooks above are written to process 1 bend at a time. However, all of these operations are designed to be easily incorporated into a `for` loop for batch processing of multiple bends, if desired.




## The ScrollStats workflow

**1. Set geoprocessing parameters**

- All geoprocessing parameters (such as window size) are kept in [parameters.py](parameters.py). Set all the parameters to the desired values before running any scripts. 

**2. Create Vector Datasets**

- Create the following vector datasets to define key morphological features of the bend. Details of the vector data creation can be found in [CreateVectorDatasets.ipynb](CreateVectorDatasets.ipynb).

    - bend boundary
    - packet boundary
    - channel centerline
    - ridge lines

- All of the above vector datasets can be made before the next step of transeforming the DEM, however it is reccommended to use the binary ridge area raster (the output of [ClassifyRasters.ipynb](ClassifyRasters.ipynb)) to help inform the location of the ridge lines.

**3. Transform Rasters**

- The first step in the raster processing is applying geomorphic transformations (profile curvature and residual topography) to the extent DEM. Detailed steps of the transformation process are contained in the [TransformRasters.ipynb](TransformRasters.ipynb) notebook.

**4. Classify Rasters**

- Once raster transformation is complete, use the [ClassifyRasters.ipynb](ClassifyRasters.ipynb) notebook to apply a threshold at 0 to the transformed rasters and find the union of these two binary rasters. This results in a single binary raster where 1s represent ridge areas and 0s represent swale areas.

**5. Calculate Ridge Metrics**

- Once all of the vector datasets are created and the raster areas are delineated, use the [CalculateRidgeMetrics.ipynb](CalculateRidgeMetrics.ipynb) notebook to calculate the ridge metrics. These metrics include ridge amplitude, width, and migration distance from last ridge for every intersection of a ridge and migration pathway. 


# Contributing

Contribution to ScrollStats is welcome. There will forever be a "frozen" branch that contains the code exactly as it was at the time of publication, but it is the intent of the maintainer to accept community feedback and suggestions to the project.

**Submitting Feedback**
To submit feedback, please open an issue on this repository with the appropriate label. Currently used labels are:
- `documentation`: issues concerning the workflow or clarity of instructions
- `feature`: issues requesting or proposing new features for scrollstats
- `bug`: issues concerning errors in the code itself