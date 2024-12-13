"""Unit tests for FractalBasis class."""

import unittest
import numpy as np
from sympy import symbols, exp
from core.basis import FractalBasis
from core.field import UnifiedField
from core.constants import ALPHA_VAL, X, ALPHA_REF

class TestFractalBasis(unittest.TestCase):
    """Test cases for FractalBasis class."""
    
    def setUp(self):
        """Initialize test environment."""
        self.basis = FractalBasis(alpha=ALPHA_VAL)
    
    def test_basis_normalization(self):
        """Test if basis functions are properly normalized."""
        for n in range(3):  # Test first few basis functions
            psi = self.basis.compute(n)
            norm = float(self.basis.check_orthogonality(n, n))
            self.assertAlmostEqual(norm, 1.0, places=6) 
    
    def test_orthogonality(self):
        """Test orthogonality between different basis functions."""
        for n1 in range(3):
            for n2 in range(3):
                overlap = float(self.basis.check_orthogonality(n1, n2))
                if n1 == n2:
                    self.assertAlmostEqual(overlap, 1.0, places=6)
                else:
                    self.assertLess(overlap, 1e-6)
    
    def test_coupling_evolution(self):
        """Test gauge coupling evolution properties."""
        # Test coupling values at Z mass
        E_Z = 91.2  # GeV
        g1 = self.basis.coupling(1, E_Z)
        g2 = self.basis.coupling(2, E_Z)
        g3 = self.basis.coupling(3, E_Z)
        
        # Known values at Z mass
        self.assertAlmostEqual(g1, 0.358, places=3)
        self.assertAlmostEqual(g2, 0.652, places=3)
        self.assertAlmostEqual(g3, 1.221, places=3)
    
    def test_scaling_dimension(self):
        """Test scaling properties of basis functions."""
        # Test scaling dimension at reference energy
        E_ref = 1000.0  # GeV
        n = 0  # Ground state
        
        # Get scaling properties
        scaling = self.basis.analyze_scaling_dimension(n, E_ref)
        
        # Classical dimension should be 1 for scalar field
        self.assertEqual(scaling['classical_dimension'], 1.0)
        
        # Anomalous dimension should be small but non-zero
        self.assertGreater(scaling['anomalous_dimension'], 0)
        self.assertLess(scaling['anomalous_dimension'], 1)
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test negative basis index
        with self.assertRaises(ValueError):
            self.basis.compute(-1)
        
        # Test invalid gauge index
        with self.assertRaises(ValueError):
            self.basis.coupling(4, 1000)
        
        # Test negative energy
        with self.assertRaises(ValueError):
            self.basis.coupling(1, -100)
    
    def test_coupling_unification(self):
        """Verify gauge coupling unification at GUT scale."""
        field = UnifiedField()
        
        # Test at GUT scale
        E_gut = 2.0e16  # GeV
        g1 = field.compute_coupling(1, E_gut)
        g2 = field.compute_coupling(2, E_gut)
        g3 = field.compute_coupling(3, E_gut)
        
        # Verify unification
        assert abs(g1 - g2) < 1e-3, "g1 and g2 should unify"
        assert abs(g2 - g3) < 1e-3, "g2 and g3 should unify"
        assert abs(g1 - g3) < 1e-3, "g1 and g3 should unify"
        
        # Verify coupling value
        assert abs(g1 - 0.0376) < 1e-4, "Unified coupling should match prediction"
    
    def test_numerical_stability(self):
        """
        Test numerical stability of core computations.
        
        Verifies stability under small parameter variations:
        1. Coupling constant evolution
        2. Field configurations
        3. Observable calculations
        
        Raises:
            AssertionError: If computations show numerical instability
        """
        field = UnifiedField()
        
        # Test coupling evolution stability
        E_test = 1000.0  # GeV
        coupling_results = []
        
        for _ in range(100):
            # Vary energy slightly
            E_perturbed = E_test * (1 + np.random.normal(0, 1e-6))
            coupling_results.append(field.compute_coupling(1, E_perturbed))
        
        coupling_std = np.std(coupling_results)
        self.assertLess(coupling_std, 1e-6, 
                        f"Coupling evolution unstable: σ={coupling_std}")
        
        # Test observable stability
        obs_results = []
        for _ in range(100):
            # Vary parameters within uncertainties
            alpha_perturbed = ALPHA_VAL * (1 + np.random.normal(0, 1e-6))
            field_perturbed = UnifiedField(alpha=alpha_perturbed)
            obs_results.append(
                field_perturbed.compute_observable('sin2_theta_W')['value']
            )
        
        obs_std = np.std(obs_results)
        self.assertLess(obs_std, 1e-6,
                        f"Observable computation unstable: σ={obs_std}")