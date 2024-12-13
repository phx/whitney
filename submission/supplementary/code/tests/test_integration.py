"""Integration tests for fractal field theory framework."""

import unittest
import numpy as np
from sympy import exp, integrate, conjugate
from core.basis import FractalBasis
from core.field import UnifiedField
from core.constants import ALPHA_VAL, X, T, Z_MASS, ALPHA_REF, EXPERIMENTAL_DATA

class TestFrameworkIntegration(unittest.TestCase):
    """Integration test cases."""
    
    def setUp(self):
        """Initialize test environment."""
        self.basis = FractalBasis(alpha=ALPHA_VAL)
        self.field = UnifiedField(alpha=ALPHA_VAL)
    
    def test_basis_field_interaction(self):
        """Test interaction between basis functions and field configurations."""
        # Create field from basis function
        psi_basis = self.basis.compute(n=0)
        
        # Compute field equation
        field_eq = self.field.compute_field_equation(psi_basis)
        
        # Field equation should preserve basis function structure
        self.assertTrue(field_eq.has(exp(-X**2)))
        
        # Energy should be well-defined
        energy = self.field.compute_energy_density(psi_basis)
        self.assertTrue(float(energy.subs([(X, 0), (T, 0)])) > 0)
    
    def test_energy_conservation(self):
        """Test energy conservation across basis expansions."""
        # Create superposition of basis functions
        psi = sum(self.basis.compute(n) for n in range(3))
        
        # Track energy over time
        times = np.linspace(0, 1, 5)
        energies = []
        
        for t in times:
            E_t = float(self.field.compute_energy_density(psi).subs(T, t))
            energies.append(E_t)
        
        # Energy should be approximately constant
        for E in energies[1:]:
            self.assertAlmostEqual(E, energies[0], places=5)
    
    def test_gauge_invariance(self):
        """Test gauge invariance of field equations."""
        # Original field configuration
        psi = self.basis.compute(n=0)
        
        # Gauge transformed field (U(1) transformation)
        theta = 0.5  # Gauge parameter
        psi_transformed = psi * exp(1j * theta)
        
        # Compute observables
        E_original = self.field.compute_energy_density(psi)
        E_transformed = self.field.compute_energy_density(psi_transformed)
        
        # Physical observables should be invariant
        diff = float(integrate(E_original - E_transformed, (X, -float('inf'), float('inf'))))
        self.assertLess(abs(diff), 1e-6)
    
    def test_experimental_validation(self):
        """Compare predictions with experimental data."""
        for observable, (exp_val, exp_err) in EXPERIMENTAL_DATA.items():
            # Get prediction with uncertainty
            pred = self.field.compute_observable(observable)
            pred_val = pred['value']
            pred_err = pred['total_uncertainty']
            
            # Calculate pull value
            pull = (pred_val - exp_val) / np.sqrt(pred_err**2 + exp_err**2)
            
            # Verify prediction within 3σ
            self.assertLess(abs(pull), 3.0)

if __name__ == '__main__':
    unittest.main() 