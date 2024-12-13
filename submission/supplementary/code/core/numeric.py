"""Numerical computation utilities for fractal field theory."""

from typing import Union, Optional, Callable, Dict, Any
import numpy as np
from sympy import Expr, lambdify, N
from functools import wraps
import warnings
from .types import RealValue, ComplexValue, Array
from .constants import X, T

def make_numerical(expr: Expr, 
                  variables: Optional[list] = None) -> Callable:
    """
    Convert symbolic expression to fast numerical function.
    
    Args:
        expr: Symbolic expression to convert
        variables: List of variables to include (defaults to [X, T])
    
    Returns:
        Vectorized numerical function
    """
    if variables is None:
        variables = [X, T]
    return np.vectorize(lambdify(variables, expr, modules=['numpy']))

def evaluate_with_precision(expr: Expr,
                          subs: Optional[Dict[Any, float]] = None,
                          precision: int = 53) -> Union[RealValue, ComplexValue]:
    """
    Evaluate expression with specified precision.
    
    Args:
        expr: Expression to evaluate
        subs: Variable substitutions
        precision: Binary precision bits
    
    Returns:
        Numerical result
    """
    if subs is None:
        subs = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            result = N(expr.subs(subs), precision)
            return complex(result) if result.is_complex else float(result)
        except Exception as e:
            raise ValueError(f"Evaluation failed: {e}")

def check_numerical_bounds(array: Array,
                         bounds: Optional[tuple] = None) -> bool:
    """
    Check if numerical values are within acceptable bounds.
    
    Args:
        array: Array of values to check
        bounds: Optional (min, max) tuple
    
    Returns:
        True if values are within bounds
    """
    if not np.all(np.isfinite(array)):
        return False
    if bounds is not None:
        min_val, max_val = bounds
        return np.all((array >= min_val) & (array <= max_val))
    return True

def monitor_numerical_stability(func: Callable) -> Callable:
    """
    Decorator to monitor numerical stability of computations.
    
    Checks:
    1. Finite values
    2. Overflow/underflow
    3. NaN/Inf values
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        with np.errstate(all='raise'):
            try:
                result = func(*args, **kwargs)
                if isinstance(result, (np.ndarray, list)):
                    if not check_numerical_bounds(np.array(result)):
                        raise ValueError("Result contains invalid values")
                return result
            except FloatingPointError as e:
                raise ValueError(f"Numerical instability detected: {e}")
    return wrapper 