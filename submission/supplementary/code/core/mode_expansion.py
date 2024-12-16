"""Mode expansion implementation."""

from typing import List, Optional, Tuple, Union, Dict
import numpy as np
from sympy import exp, I, pi, sqrt
from dataclasses import dataclass

from .types import Energy, WaveFunction
from .constants import ALPHA_VAL
from .physics_constants import HBAR, C
from .enums import ComputationMode

@dataclass
class ModeCoefficient:
    """Container for mode expansion coefficients."""
    n: int  # Mode number
    value: complex  # Coefficient value 
    energy: Energy  # Mode energy
    weight: float  # Statistical weight

class ModeExpansion:
    """Handles expansion of fields in fractal basis modes."""
    
    def __init__(self, alpha: float = ALPHA_VAL) -> None:
        self.alpha = alpha
        self.field = None  # Will be set after import 

    def compute_mode_coefficients(self, psi: WaveFunction, n_max: int = 100) -> List[ModeCoefficient]:
        """Expand field in fractal basis modes.
        
        Args:
            psi: Field configuration to expand
            n_max: Maximum mode number
            
        Returns:
            List of mode coefficients
        """
        coeffs = []
        for n in range(n_max):
            energy = self.field.compute_mode_energy(n)
            value = self._compute_overlap(psi, n)
            weight = self._compute_statistical_weight(n)
            coeffs.append(ModeCoefficient(n, value, energy, weight))
        return coeffs

    def _compute_overlap(self, psi: WaveFunction, n: int) -> complex:
        """Compute overlap with basis function."""
        basis_n = self.field.compute_basis_function(n)
        return self.field.compute_inner_product(psi, basis_n)

    def _compute_statistical_weight(self, n: int) -> float:
        """Compute statistical weight factor."""
        return exp(-self.alpha * n)

    def reconstruct_field(self, coeffs: List[ModeCoefficient]) -> WaveFunction:
        """Reconstruct field from mode coefficients.
        
        Args:
            coeffs: List of mode coefficients
            
        Returns:
            Reconstructed field configuration
        """
        psi = 0
        for coeff in coeffs:
            basis_n = self.field.compute_basis_function(coeff.n)
            psi += coeff.value * coeff.weight * basis_n
        return psi

    def compute_mode_energies(self, n_range: int) -> Dict[int, Energy]:
        """Compute energy spectrum for modes.
        
        Args:
            n_range: Number of modes to compute
            
        Returns:
            Dictionary mapping mode numbers to energies
            
        Examples:
            >>> modes = ModeExpansion()
            >>> energies = modes.compute_mode_energies(3)
            >>> len(energies)
            3
            >>> all(E > 0 for E in energies.values())  # Check positivity
            True
            >>> energies[0] < energies[1] < energies[2]  # Check ordering
            True
        """
        return {
            n: self.field.compute_mode_energy(n)
            for n in range(n_range)
        }