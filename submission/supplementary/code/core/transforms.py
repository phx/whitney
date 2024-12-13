"""Coordinate and gauge transformations."""

from sympy import exp, I, sqrt, Matrix
from .physics_constants import C

def lorentz_boost(beta: float) -> Matrix:
    """
    Generate Lorentz boost matrix.
    
    Args:
        beta: Velocity parameter (v/c)
        
    Returns:
        2x2 Lorentz transformation matrix
    """
    gamma = 1/sqrt(1 - beta**2)
    return Matrix([
        [gamma, -gamma*beta],
        [-gamma*beta, gamma]
    ])

def gauge_transform(phase: float) -> exp:
    """
    Generate gauge transformation.
    
    Args:
        phase: Gauge transformation phase
        
    Returns:
        Gauge transformation operator
    """
    return exp(I * phase) 