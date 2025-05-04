from __future__ import annotations

from typing import Any, Callable

from nptyping import NDArray, Shape

# Define array types
Array2D: NDArray[Shape["*, *"], Any]  # 2D array of any size of any dtype
ElevationArray2D: NDArray[Shape["*, *"], float]  # 2D array of any size of dtype float
BinaryArray2D: NDArray[Shape["*, *"], bool]  # 2D array of any size of dtype bool

# Define functions as interfaces
## `BinaryClassifierFn`s and `BinaryDenoiserFn`s use the following signature:
### They take a 2D array as input and output a binary 2D array
### Use partial functions from the `functools` library to create wrapper functions which contain all input arguments other than the input 2D array
BinaryClassifierFn: Callable[[ElevationArray2D], BinaryArray2D]
BinaryDenoiserFn: Callable[[BinaryArray2D], BinaryArray2D]
