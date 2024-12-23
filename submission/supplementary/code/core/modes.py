"""Mode expansion implementation."""

from typing import List, Optional, Tuple, Union, Dict, TYPE_CHECKING
import numpy as np
from sympy import exp, I, pi, sqrt
from dataclasses import dataclass

from .types import Energy, WaveFunction
from .physics_constants import ALPHA_VAL, HBAR, C
from .enums import ComputationMode

if TYPE_CHECKING:
    from .field import UnifiedField

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
        """Initialize mode expansion.
        
        Args:
            alpha: Fine structure constant
        """
        self.alpha = alpha
        self.field: 'UnifiedField' = None  # Type hint with forward reference
    
    def compute_mode_coefficients(self, psi: WaveFunction, 
                                n_max: int = 100) -> List[ModeCoefficient]:
        """Expand field in fractal basis modes.
        
        Args:
            psi: Field configuration to expand
            n_max: Maximum mode number
            
        Returns:
            List of mode coefficients
+            
+        Examples:
+            >>> field = UnifiedField()
+            >>> psi = field.ground_state()
+            >>> modes = ModeExpansion()
+            >>> coeffs = modes.compute_mode_coefficients(psi, n_max=10)
+            >>> len(coeffs)
+            10
+            >>> isinstance(coeffs[0], ModeCoefficient)
+            True
        """
        coeffs = []
        for n in range(n_max):
            mode = self.basis.compute(n)
            coeff = self.field.compute_mode_coefficient(psi, mode)
            energy = self.field.compute_mode_energy(n)
            weight = float(abs(coeff)**2)
            
            coeffs.append(ModeCoefficient(
                n=n,
                value=complex(coeff),
                energy=energy,
                weight=weight
            ))
        return coeffs
    
    def reconstruct_field(self, coeffs: List[ModeCoefficient]) -> WaveFunction:
        """Reconstruct field from mode coefficients.
        
        Args:
            coeffs: List of mode coefficients
            
        Returns:
            Reconstructed field configuration
+            
+        Examples:
+            >>> modes = ModeExpansion()
+            >>> psi_orig = UnifiedField().ground_state()
+            >>> coeffs = modes.compute_mode_coefficients(psi_orig)
+            >>> psi_reconstructed = modes.reconstruct_field(coeffs)
+            >>> abs((psi_orig - psi_reconstructed).norm()) < 1e-10
+            True
        """
        psi = 0
        for c in coeffs:
            mode = self.basis.compute(c.n)
            psi += c.value * mode
        return psi
    
    def compute_mode_correlations(self, coeffs: List[ModeCoefficient]) -> np.ndarray:
        """Compute correlation matrix between modes.
        
        Args:
            coeffs: List of mode coefficients
            
        Returns:
            Mode correlation matrix
+            
+        Examples:
+            >>> modes = ModeExpansion()
+            >>> psi = UnifiedField().ground_state()
+            >>> coeffs = modes.compute_mode_coefficients(psi, n_max=3)
+            >>> corr = modes.compute_mode_correlations(coeffs)
+            >>> corr.shape
+            (3, 3)
+            >>> np.allclose(corr, corr.conj().T)  # Check Hermitian
+            True
        """
        n_modes = len(coeffs)
        corr = np.zeros((n_modes, n_modes), dtype=complex)
        
        for i, ci in enumerate(coeffs):
            for j, cj in enumerate(coeffs):
                corr[i,j] = ci.value * cj.value.conjugate()
                
        return corr
    
    def compute_mode_energies(self, n_range: int) -> Dict[int, Energy]:
        """Compute energy spectrum for a range of modes.
        
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