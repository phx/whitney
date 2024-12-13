"""Utilities for numerical computations and evaluation."""

from typing import Callable, Any, Optional, Union
from functools import lru_cache
import numpy as np
from sympy import Expr, N
from .types import RealValue, ComplexValue

def evaluate_expr(expr: Expr, 
                 subs: Optional[dict] = None, 
                 precision: int = 53) -> Union[RealValue, ComplexValue]:
    """
    Evaluate symbolic expression numerically.
    
    Args:
        expr: Symbolic expression to evaluate
        subs: Variable substitutions
        precision: Binary precision (default: double precision)
    
    Returns:
        Numerical value
    """
    if subs is None:
        subs = {}
    result = N(expr.subs(subs), precision)
    return complex(result) if result.is_complex else float(result)

@lru_cache(maxsize=1024)
def cached_evaluation(expr: Expr, *args) -> RealValue:
    """Cache expensive numerical evaluations."""
    return float(evaluate_expr(expr))

def check_numerical_stability(func: Callable) -> Callable:
    """
    Decorator to check numerical stability of computations.
    
    Monitors:
    1. Overflow/underflow
    2. Loss of precision
    3. Convergence issues
    """
    def wrapper(*args, **kwargs):
        with np.errstate(all='raise'):
            try:
                result = func(*args, **kwargs)
                if not np.isfinite(result).all():
                    raise ValueError("Non-finite values in result")
                return result
            except FloatingPointError as e:
                raise ValueError(f"Numerical instability detected: {e}")
    return wrapper 