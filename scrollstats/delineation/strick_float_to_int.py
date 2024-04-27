# This script converts a float dem (where the cell values are in meters) to an integer dem (where cell values are in cm).
# This is so the ImageJ program can open the file. Their doumnetation says it can open 32bit float images, but this is not happening for me


import sys, pathlib
import numpy as np
import rasterio


def float_to_int(a: np.array, dtype="int32"):
    """Cast the raster array to a integer type where cell values represent elevation in cm"""

    return (np.round(a, 2) * 100).astype(dtype).reshape(1, *a.shape)


if __name__ == "__main__":

    # Read in file
    in_path = pathlib.Path(sys.argv[1])

    # Define dtype of output
    dtype = "int8"

    # Defien out_path based on inpath and dtype
    out_name = f"_{dtype}".join([in_path.stem, in_path.suffix])
    out_path = in_path.with_name(out_name)

    # Read in raster and adjust profile for output
    ds = rasterio.open(in_path)
    profile = ds.profile
    profile["dtype"] = dtype
    profile["count"] = 1

    # Create int array
    a = float_to_int(ds.read(1), dtype=dtype)

    # Write to disk
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(a)
