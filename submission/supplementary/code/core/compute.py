"""Core computation functions for physics calculations."""

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from sympy import Expr, Symbol, integrate, exp
from .types import Energy, Momentum, WaveFunction, CrossSection
from .validation import (
    validate_energy, validate_momentum, validate_wavefunction,
    validate_numeric_range
)
from .errors import ComputationError, PhysicsError
from .physics_constants import ALPHA_VAL, X, E

def compute_cross_section(
    energy: Union[float, Energy],
    momentum: Union[float, Momentum],
    wavefunction: Union[Expr, WaveFunction]
) -> CrossSection:
    """
    Compute scattering cross section.
    
    Args:
        energy: Collision energy
        momentum: Incoming momentum
        wavefunction: Interaction wavefunction
        
    Returns:
        CrossSection: Computed cross section
        
    Raises:
        ComputationError: If computation fails
        PhysicsError: If parameters violate physical constraints
    """
    try:
        # Validate inputs
        E = validate_energy(energy)
        p = validate_momentum(momentum)
        psi = validate_wavefunction(wavefunction)
        
        # Compute matrix element
        M = integrate(psi * exp(-X**2/2), (X, -np.inf, np.inf))
        
        # Compute phase space factor
        phase_space = np.sqrt(1 - 4*p.value**2/E.value**2)
        
        # Compute cross section
        sigma = abs(M)**2 * phase_space / (32 * np.pi * E.value**2)
        
        return CrossSection(sigma)
        
    except Exception as e:
        raise ComputationError(f"Cross section computation failed: {e}")

def compute_correlation_function(
    x1: float,
    x2: float,
    energy: Union[float, Energy],
    wavefunction: Union[Expr, WaveFunction]
) -> float:
    """
    Compute two-point correlation function.
    
    Args:
        x1: First position
        x2: Second position
        energy: Energy scale
        wavefunction: Field configuration
        
    Returns:
        float: Correlation function value
        
    Raises:
        ComputationError: If computation fails
    """
    try:
        # Validate inputs
        E = validate_energy(energy)
        psi = validate_wavefunction(wavefunction)
        
        # Compute correlation
        dx = abs(x2 - x1)
        k = E.value * ALPHA_VAL
        
        # Evaluate wavefunction at both points
        psi1 = psi.evaluate_at(x1)
        psi2 = psi.evaluate_at(x2)
        
        # Compute correlation with proper normalization
        corr = np.real(np.conjugate(psi1) * psi2) * exp(-k * dx)
        
        return float(corr)
        
    except Exception as e:
        raise ComputationError(f"Correlation function computation failed: {e}")

def compute_branching_ratio(
    process: str,
    energy: Union[float, Energy],
    couplings: Dict[str, float]
) -> float:
    """
    Compute decay branching ratio.
    
    Args:
        process: Decay process identifier
        energy: Energy scale
        couplings: Coupling constants
        
    Returns:
        float: Branching ratio
        
    Raises:
        ComputationError: If computation fails
        PhysicsError: If process is not allowed
    """
    try:
        # Validate inputs
        E = validate_energy(energy)
        
        # Get process parameters
        if process not in ALLOWED_PROCESSES:
            raise PhysicsError(f"Unknown process: {process}")
            
        params = ALLOWED_PROCESSES[process]
        threshold = params["threshold"]
        
        if E.value < threshold:
            raise PhysicsError(f"Energy below threshold for {process}")
            
        # Compute phase space
        phase_space = compute_phase_space(E.value, params)
        
        # Compute matrix element
        coupling = couplings.get(process, 0.0)
        M_squared = compute_matrix_element_squared(coupling, params)
        
        # Compute width
        width = M_squared * phase_space / (32 * np.pi * E.value)
        
        # Get total width at this energy
        total_width = compute_total_width(E.value, couplings)
        
        # Compute branching ratio
        br = width / total_width
        
        return float(br)
        
    except Exception as e:
        raise ComputationError(f"Branching ratio computation failed: {e}")

# Process definitions and helper functions
ALLOWED_PROCESSES = {
    "Z->ee": {"threshold": 2 * 0.511e-3, "spin_avg": 3},  # e+e- threshold
    "Z->mumu": {"threshold": 2 * 0.106, "spin_avg": 3},   # μ+μ- threshold
    "Z->tautau": {"threshold": 2 * 1.777, "spin_avg": 3}, # τ+τ- threshold
}

def compute_phase_space(E: float, params: Dict) -> float:
    """Compute phase space factor."""
    threshold = params["threshold"]
    if E <= threshold:
        return 0.0
    return np.sqrt(1 - (threshold/E)**2)

def compute_matrix_element_squared(coupling: float, params: Dict) -> float:
    """Compute squared matrix element."""
    spin_avg = params["spin_avg"]
    return coupling**2 / spin_avg

def compute_total_width(E: float, couplings: Dict[str, float]) -> float:
    """Compute total width."""
    total = 0.0
    for process, params in ALLOWED_PROCESSES.items():
        if E > params["threshold"]:
            coupling = couplings.get(process, 0.0)
            phase_space = compute_phase_space(E, params)
            M_squared = compute_matrix_element_squared(coupling, params)
            total += M_squared * phase_space
    return total / (32 * np.pi * E)