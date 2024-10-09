
from typing import Tuple, Callable
import numpy as np

# Define types
Array2D = np.ndarray[Tuple[int, int], np.dtype[any]]
ElevationArray2D = np.ndarray[Tuple[int, int], np.dtype[float]]
BinaryArray2D = np.ndarray[Tuple[int, int], np.dtype[bool]]

# Define functions as interfaces
## `BinaryClassifierFn`s and `BinaryDenoiserFn`s use the following signature
## They take a 2D array as input and output a binary 2D array
## Use partial functions fron the `functools` library to create wrapper functions which contain all input arguments other than the input 2D array
BinaryClassifierFn = Callable[[ElevationArray2D], BinaryArray2D]
BinaryDenoiserFn = Callable[[BinaryArray2D], BinaryArray2D]