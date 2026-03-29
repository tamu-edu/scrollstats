from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from numpy import ndarray

# Define array types
# nptyping module was originally used to define numpy array types, but is incompatible with numpy v2.0 so it was removed
# Defined array types were preserved with generic `ndarray` to communicate returned array dtypes in type hints

## Array2D: 2D array of any size of any dtype
Array2D: TypeAlias = ndarray  # pylint: disable=C0103

## ElevationArray2D: 2D array of any size of dtype float
ElevationArray2D: TypeAlias = ndarray  # pylint: disable=C0103

## BinaryArray2D: 2D array of any size of dtype bool
BinaryArray2D: TypeAlias = ndarray  # pylint: disable=C0103

# Define functions as interfaces
## `BinaryClassifierFn`s and `BinaryDenoiserFn`s use the following signature:
### They take a 2D array as input and output a binary 2D array
### Use partial functions from the `functools` library to create wrapper functions which contain all input arguments other than the input 2D array

# Mypy does not allow for the complex Callable typing that's currently used (a callable with one typed input and any number of other inputs of any type)
## So, the difference between a BinaryClassifierFn and BinaryDenoiserFn are in name only

# Takes an ElevationArray2D and kwargs as input
BinaryClassifierFn: TypeAlias = Callable[..., BinaryArray2D]

# Takes a BinaryArray2D and kwargs as input
BinaryDenoiserFn: TypeAlias = Callable[..., BinaryArray2D]
