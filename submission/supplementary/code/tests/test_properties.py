"""Property-based tests for fractal field theory framework."""

import unittest
from hypothesis import given, strategies as st
import numpy as np
from sympy import exp
from core.basis import FractalBasis
from core.field import UnifiedField
from core.constants import ALPHA_VAL, X, T

class TestFrameworkProperties(unittest.TestCase):
    """Property-based test cases."""
    
    def setUp(self):
        """Initialize test environment."""
        self.basis = FractalBasis(alpha=ALPHA_VAL)
        self.field = UnifiedField(alpha=ALPHA_VAL)
    
    @given(st.integers(min_value=0, max_value=10),
           st.floats(min_value=1.0, max_value=1000.0))
    def test_basis_normalization_property(self, n, E):
        """Test that basis functions remain normalized for any valid input."""
        psi = self.basis.compute(n, E)
        norm = float(self.basis.check_orthogonality(n, n))
        self.assertAlmostEqual(norm, 1.0, places=5)
    
    @given(st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=5))
    def test_superposition_energy_property(self, indices):
        """Test that energy of superposition is well-defined."""
        # Create superposition of basis functions
        psi = sum(self.basis.compute(n) for n in indices)
        
        # Energy should be real and positive
        energy = self.field.compute_energy_density(psi)
        E_val = float(energy.subs([(X, 0), (T, 0)]))
        self.assertGreater(E_val, 0)
    
    @given(st.floats(min_value=0.0, max_value=2*np.pi))
    def test_gauge_invariance_property(self, theta):
        """Test gauge invariance for arbitrary phase rotations."""
        # Original field
        psi = self.basis.compute(0)
        
        # Gauge transformed field
        psi_transformed = psi * exp(1j * theta)
        
        # Compare observables
        E1 = self.field.compute_energy_density(psi)
        E2 = self.field.compute_energy_density(psi_transformed)
        
        diff = float(E1.subs([(X, 0), (T, 0)]) - E2.subs([(X, 0), (T, 0)]))
        self.assertLess(abs(diff), 1e-5)

if __name__ == '__main__':
    unittest.main() 