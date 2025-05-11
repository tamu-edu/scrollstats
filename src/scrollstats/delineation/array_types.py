from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

from nptyping import Bool, Float, NDArray, Shape

# Define array types

## Array2D: 2D array of any size of any dtype
Array2D: TypeAlias = NDArray[Shape["*, *"], Any]

## ElevationArray2D: 2D array of any size of dtype float
ElevationArray2D: TypeAlias = NDArray[Shape["*, *"], Float]

## BinaryArray2D: 2D array of any size of dtype bool
BinaryArray2D: TypeAlias = NDArray[Shape["*, *"], Bool]

# Define functions as interfaces
## `BinaryClassifierFn`s and `BinaryDenoiserFn`s use the following signature:
### They take a 2D array as input and output a binary 2D array
### Use partial functions from the `functools` library to create wrapper functions which contain all input arguments other than the input 2D array
BinaryClassifierFn: TypeAlias = Callable[[ElevationArray2D], BinaryArray2D]
BinaryDenoiserFn: TypeAlias = Callable[[BinaryArray2D], BinaryArray2D]
