"""Input validation utilities for fractal field theory framework."""

from typing import Any, Optional, Union, Dict
import numpy as np
from sympy import Expr
from .errors import ValidationError, PhysicsError
from .types import Energy, FieldConfig

def validate_energy(E: Energy, min_val: float = 0.0) -> None:
    """
    Validate energy scale parameter.
    
    Args:
        E: Energy value to validate
        min_val: Minimum allowed energy
        
    Raises:
        ValidationError: If energy is invalid
        PhysicsError: If energy violates physical constraints
    """
    if not isinstance(E, (int, float)):
        raise ValidationError(f"Energy must be numeric, got {type(E)}")
    if E <= min_val:
        raise PhysicsError(f"Energy must be > {min_val}, got {E}")

def validate_field_config(psi: FieldConfig) -> None:
    """
    Validate field configuration.
    
    Args:
        psi: Field configuration to validate
        
    Raises:
        ValidationError: If configuration is invalid
    """
    if not isinstance(psi, Expr):
        raise ValidationError("Field must be symbolic expression")
    if not psi.is_complex:
        raise ValidationError("Field must be complex-valued")

def validate_parameters(params: Dict[str, Any], 
                       bounds: Optional[Dict[str, tuple]] = None) -> None:
    """
    Validate numerical parameters.
    
    Args:
        params: Parameters to validate
        bounds: Optional parameter bounds
        
    Raises:
        ValidationError: If parameters are invalid
    """
    if bounds is None:
        bounds = {}
    
    for name, value in params.items():
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{name} must be numeric")
        if name in bounds:
            min_val, max_val = bounds[name]
            if not min_val <= value <= max_val:
                raise ValidationError(
                    f"{name} must be in [{min_val}, {max_val}], got {value}") 