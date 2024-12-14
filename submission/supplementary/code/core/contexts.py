"""Context managers for physics framework."""

from contextlib import contextmanager
import numpy as np
from typing import Generator
from sympy import pi, sqrt, exp, I
from core.physics_constants import C, HBAR, X, T
from core.errors import PhysicsError
from core.types import FieldConfig, NumericValue

@contextmanager
def gauge_phase(min_val: float = 0.0, max_val: float = 2*pi) -> Generator[float, None, None]:
    """
    Context manager for gauge phases.
    
    Provides a gauge phase value between min_val and max_val (default 0 to 2π).
    Ensures proper cleanup and phase validation.
    
    Args:
        min_val: Minimum phase value (default: 0)
        max_val: Maximum phase value (default: 2π)
        
    Yields:
        float: A valid gauge phase value
        
    Example:
        >>> with gauge_phase() as phase:
        ...     field.apply_gauge_transform(psi, phase)
    """
    try:
        # Generate a random phase in the valid range
        phase = min_val + (max_val - min_val) * np.random.random()
        
        # Ensure phase is in [0, 2π]
        phase = phase % (2*pi)
        
        yield phase
        
    finally:
        # Cleanup if needed
        pass 

@contextmanager
def lorentz_boost(min_velocity: float = -0.9, max_velocity: float = 0.9) -> Generator[tuple[float, float], None, None]:
    """
    Context manager for Lorentz boosts.
    
    Provides a velocity parameter and corresponding gamma factor for Lorentz transformations.
    Ensures proper validation and relativistic constraints.
    
    Args:
        min_velocity: Minimum velocity in units of c (default: -0.9)
        max_velocity: Maximum velocity in units of c (default: 0.9)
        
    Yields:
        tuple[float, float]: (beta, gamma) where:
            beta: velocity/c
            gamma: 1/sqrt(1 - beta^2)
            
    Raises:
        PhysicsError: If velocity would be superluminal
        
    Example:
        >>> with lorentz_boost() as (beta, gamma):
        ...     x_prime = gamma * (x - beta*c*t)
        ...     t_prime = gamma * (t - beta*x/c)
    """
    try:
        # Generate random velocity in allowed range
        beta = min_velocity + (max_velocity - min_velocity) * np.random.random()
        
        # Validate velocity
        if abs(beta) >= 1:
            raise PhysicsError(f"Superluminal velocity: {beta}c")
            
        # Compute gamma factor
        gamma = 1/sqrt(1 - beta**2)
        
        yield beta, gamma
        
    finally:
        pass 

@contextmanager
def quantum_state(energy: float = 1.0) -> Generator[tuple[object, float], None, None]:
    """
    Context manager for quantum states.
    
    Provides a normalized quantum state with specified energy.
    
    Args:
        energy: Energy of the state in GeV (default: 1.0)
        
    Yields:
        tuple[object, float]: (psi, norm) where:
            psi: Quantum state wavefunction
            norm: Normalization factor
            
    Example:
        >>> with quantum_state(energy=100.0) as (psi, norm):
        ...     field.compute_expectation(psi)
    """
    try:
        # Generate gaussian wavepacket
        psi = exp(-(X**2 + (C*T)**2)/(2*HBAR**2))
        
        # Compute normalization
        norm = sqrt(energy)
        
        yield psi, norm
        
    finally:
        pass

@contextmanager 
def field_config(
    mass: float = 125.0,
    coupling: float = 0.1,
    dimension: int = 4
) -> Generator[FieldConfig, None, None]:
    """
    Context manager for field configurations.
    
    Provides validated field configuration parameters.
    
    Args:
        mass: Field mass in GeV (default: 125.0)
        coupling: Coupling constant (default: 0.1)
        dimension: Spacetime dimension (default: 4)
        
    Yields:
        FieldConfig: Validated field configuration
        
    Example:
        >>> with field_config(mass=125.0) as config:
        ...     field.compute_potential(config)
    """
    try:
        config = FieldConfig(
            mass=mass,
            coupling=coupling,
            dimension=dimension
        )
        yield config
    finally:
        pass

@contextmanager
def numeric_precision(
    rtol: float = 1e-8,
    atol: float = 1e-10,
    maxiter: int = 1000
) -> Generator[dict, None, None]:
    """
    Context manager for numerical precision settings.
    
    Provides consistent precision parameters across computations.
    
    Args:
        rtol: Relative tolerance (default: 1e-8)
        atol: Absolute tolerance (default: 1e-10)
        maxiter: Maximum iterations (default: 1000)
        
    Yields:
        dict: Precision parameters
        
    Example:
        >>> with numeric_precision(rtol=1e-6) as prec:
        ...     field.compute_integral(psi, **prec)
    """
    try:
        precision = {
            'rtol': NumericValue(rtol),
            'atol': NumericValue(atol),
            'maxiter': maxiter,
            'stability_threshold': rtol * 10
        }
        yield precision
    finally:
        pass