"""Validation utilities for physics computations."""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from sympy import Expr
from .errors import ValidationError, PhysicsError
from .types import Energy, Momentum, WaveFunction

def validate_energy(energy: Union[float, Energy]) -> Energy:
    """
    Validate energy value.
    
    Args:
        energy: Energy value to validate
        
    Returns:
        Energy: Validated energy object
        
    Raises:
        ValidationError: If energy is invalid
    """
    if isinstance(energy, Energy):
        return energy
    try:
        return Energy(energy)
    except Exception as e:
        raise ValidationError(f"Invalid energy value: {e}")

def validate_momentum(momentum: Union[float, Momentum]) -> Momentum:
    """
    Validate momentum value.
    
    Args:
        momentum: Momentum value to validate
        
    Returns:
        Momentum: Validated momentum object
        
    Raises:
        ValidationError: If momentum is invalid
    """
    if isinstance(momentum, Momentum):
        return momentum
    try:
        return Momentum(momentum)
    except Exception as e:
        raise ValidationError(f"Invalid momentum value: {e}")

def validate_wavefunction(psi: Union[Expr, WaveFunction]) -> WaveFunction:
    """
    Validate wavefunction.
    
    Args:
        psi: Wavefunction to validate
        
    Returns:
        WaveFunction: Validated wavefunction
        
    Raises:
        ValidationError: If wavefunction is invalid
    """
    if isinstance(psi, WaveFunction):
        return psi
    try:
        return WaveFunction(psi)
    except Exception as e:
        raise ValidationError(f"Invalid wavefunction: {e}")

def validate_numeric_range(value: float, min_val: float, max_val: float, 
                         name: str = "value") -> float:
    """
    Validate numeric value is within range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of value for error messages
        
    Returns:
        float: Validated value
        
    Raises:
        ValidationError: If value is outside valid range
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric")
    if not min_val <= value <= max_val:
        raise ValidationError(
            f"{name} must be between {min_val} and {max_val}, got {value}"
        )
    return float(value)

def validate_config(config: Dict[str, Any], required_keys: List[str]) -> None:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
        required_keys: List of required keys
        
    Raises:
        ValidationError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValidationError("Configuration must be a dictionary")
    
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValidationError(f"Missing required configuration keys: {missing}")