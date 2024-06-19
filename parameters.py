# Contains global variables for convenient batch processing

# Raster Processing Constants
RASTER_WINDOW_SIZE = 45  # Measured in img px
SMALL_FEATS_SIZE = 500  # Measured in img px

# Vector Creation Constants
SMOOTHING_WINDOW_SIZE = 5  # Measured in vertices
VERTEX_SPACING = 1  # Measured in linear unit of dataset (meters for example datasets)

## See `CreatingVectorData.ipynb` for explanation of constants below and how they are used in migration pathway creation
SHOOT_DISTANCE = 300  # Distance that the N1 coordinate will shoot out from point P1; easured in linear unit of dataset
SEARCH_DISTANCE = 200  # Buffer radius used to search for an N2 coordinate on R2; measured in linear unit of dataset
DEV_FROM_90 = 5  # Max angular deviation from 90° allowed when searching for an N2 coordinate on R2; measured in degrees