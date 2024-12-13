"""Type definitions for fractal field theory framework."""

from typing import Union, Dict, List, TypeVar, NewType
from sympy import Expr
import numpy as np

# Energy scales
Energy = NewType('Energy', float)  # GeV
Temperature = NewType('Temperature', float)  # Kelvin

# Field-related types
FieldConfig = NewType('FieldConfig', Expr)
WaveFunction = NewType('WaveFunction', Expr)

# Numerical types
RealValue = Union[float, np.float64]
ComplexValue = Union[complex, np.complex128]

# Analysis results
AnalysisResult = Dict[str, Union[RealValue, ComplexValue, Expr]]
ErrorEstimate = Dict[str, RealValue]

# Generic type for numerical arrays
Array = TypeVar('Array', np.ndarray, List[float]) 