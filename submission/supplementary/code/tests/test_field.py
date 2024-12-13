"""Unit tests for UnifiedField class."""

import unittest
import numpy as np
from sympy import symbols, exp, diff, integrate
from core.field import UnifiedField
from core.constants import (
    ALPHA_VAL, Z_MASS, ALPHA_REF,
    g1_REF, g2_REF, g3_REF,
    X, T
)

class TestUnifiedField(unittest.TestCase):
    """Test cases for UnifiedField class."""
    
    def setUp(self):
        """Initialize test environment."""
        self.field = UnifiedField(alpha=ALPHA_VAL)
    
    def test_field_equation(self):
        """Test field equation structure and properties."""
        # Create simple test field configuration
        psi = exp(-X**2/2) * exp(-T**2/2)
        
        # Compute field equation
        field_eq = self.field.compute_field_equation(psi)
        
        # Test equation properties
        self.assertTrue(field_eq.has(diff(psi, T, 2)))  # Has time derivative
        self.assertTrue(field_eq.has(diff(psi, X, 2)))  # Has space derivative
        self.assertTrue(field_eq.has(psi))              # Has field term
    
    def test_energy_conservation(self):
        """Test energy conservation for static configurations."""
        # Static gaussian configuration
        psi = exp(-X**2/2)
        
        # Compute energy at different times
        E_t0 = float(self.field.compute_energy_density(psi).subs(T, 0))
        E_t1 = float(self.field.compute_energy_density(psi).subs(T, 1))
        
        # Energy should be constant for static configuration
        self.assertAlmostEqual(E_t0, E_t1, places=6)
    
    def test_symmetry_analysis(self):
        """Test symmetry properties of field configurations."""
        # Symmetric configuration
        psi_sym = exp(-X**2/2)
        analysis = self.field.analyze_field_configuration(psi_sym)
        
        # Parity should be preserved
        self.assertLess(analysis['parity_violation'], 1e-6)
        
        # Time reversal should be preserved
        self.assertLess(analysis['time_reversal_violation'], 1e-6)
    
    def test_potential_positivity(self):
        """Test that potential energy is positive definite."""
        # Test various field configurations
        test_fields = [
            exp(-X**2/2),
            X * exp(-X**2/2),
            exp(-X**2/2) * exp(-T**2/2)
        ]
        
        for psi in test_fields:
            potential = self.field._compute_potential(psi)
            # Evaluate at some points
            for x in [-1.0, 0.0, 1.0]:
                val = float(potential.subs(X, x).subs(T, 0))
                self.assertGreaterEqual(val, 0)
    
    def test_coupling_evolution(self):
        """Test gauge coupling evolution."""
        # Test at Z mass
        g1 = self.field.compute_coupling(1, Z_MASS[0])
        
        # Should match reference value within uncertainty
        self.assertLess(abs(g1 - g1_REF[0]), g1_REF[1])
        
        # Test with uncertainties
        result = self.field.coupling_with_uncertainty(1, Z_MASS[0])
        self.assertIn('value', result)
        self.assertIn('statistical_error', result)
        self.assertIn('total_uncertainty', result)
    
    def test_theoretical_predictions(self):
        """Validate key theoretical predictions from paper."""
        field = UnifiedField(alpha=ALPHA_VAL)
        
        # Test unification prediction (Eq. 4.5)
        E_gut = 2.0e16  # GUT scale
        g1 = field.compute_coupling(1, E_gut)
        g2 = field.compute_coupling(2, E_gut)
        g3 = field.compute_coupling(3, E_gut)
        
        # Couplings should unify at GUT scale
        self.assertAlmostEqual(g1, g2, places=4)
        self.assertAlmostEqual(g2, g3, places=4)
        
        # Test fractal dimension (Eq. 3.12)
        dim = field.calculate_fractal_dimension(E=1000.0)
        self.assertAlmostEqual(dim, 4.0 + ALPHA_VAL, places=5)
        
        # Test holographic entropy bound (Eq. 6.8)
        S = field.compute_holographic_entropy(E=1000.0)
        S_bound = field.compute_entropy_bound(E=1000.0)
        self.assertLess(S, S_bound)
        
        def test_weak_mixing_angle_lep(self):
            """Verify weak mixing angle against LEP data."""
            field = UnifiedField(alpha=ALPHA_VAL)
            
            # LEP data points from paper Table 4.1
            lep_data = {
                # Energy (GeV): (sin²θW, uncertainty)
                91.2: (0.23122, 0.00003),   # Z peak
                94.3: (0.23159, 0.00041),   # Z+2
                89.4: (0.23098, 0.00026),   # Z-2
                161.3: (0.23196, 0.00043),  # LEP2
                207.0: (0.23228, 0.00064)   # LEP2 max
            }
            
            # Test prediction at each energy
            for energy, (exp_val, exp_err) in lep_data.items():
                # Override energy in weak mixing angle calculation
                field.Z_MASS_VAL = energy
                pred = field._compute_weak_mixing_angle()
                
                # Compute pull value
                pull = (pred - exp_val) / exp_err
                
                # Verify prediction within 2σ
                self.assertLess(abs(pull), 2.0,
                    f"Prediction at {energy} GeV deviates by {pull}σ")
                
                # Verify precision matches paper claim
                rel_error = abs(pred - exp_val) / exp_val
                self.assertLess(rel_error, 0.001,  # 0.1% precision
                    f"Precision at {energy} GeV is {rel_error*100}%")
        
        def test_coupling_unification(self):
            """Verify gauge coupling unification at GUT scale."""
            field = UnifiedField(alpha=ALPHA_VAL)
            
            # Test points from paper Table 3.2
            test_points = {
                # Energy (GeV): (g1, g2, g3)
                91.2: (0.357, 0.652, 1.221),      # Z mass (measured)
                1e3: (0.368, 0.642, 1.087),       # 1 TeV
                1e6: (0.387, 0.626, 0.953),       # 1000 TeV
                1e12: (0.435, 0.608, 0.782),      # Intermediate
                2e16: (0.519, 0.519, 0.519),      # GUT scale
            }
            
            # Verify running couplings
            for energy, (g1_exp, g2_exp, g3_exp) in test_points.items():
                g1 = field.compute_coupling(1, energy)
                g2 = field.compute_coupling(2, energy)
                g3 = field.compute_coupling(3, energy)
                
                # Check against expected values
                self.assertAlmostEqual(g1, g1_exp, places=3,
                    msg=f"g1 mismatch at E={energy} GeV")
                self.assertAlmostEqual(g2, g2_exp, places=3,
                    msg=f"g2 mismatch at E={energy} GeV")
                self.assertAlmostEqual(g3, g3_exp, places=3,
                    msg=f"g3 mismatch at E={energy} GeV")
                
                # Verify threshold corrections
                if energy >= 1e15:  # Near GUT scale
                    # Threshold corrections from paper Eq. 3.15
                    delta = abs((g1 - g2)/g1) + abs((g2 - g3)/g2)
                    self.assertLess(delta, 0.01,  # 1% threshold corrections
                        msg=f"Large threshold corrections at E={energy} GeV")
        
        def test_b_physics_predictions(self):
            """Validate B-physics predictions against experimental data."""
            field = UnifiedField(alpha=ALPHA_VAL)
            
            # Test branching ratios from paper Table 4.2
            br_tests = {
                'BR_Bs_mumu': {
                    'prediction': (3.09e-9, 0.12e-9),  # Theory prediction
                    'experiment': (2.8e-9, 0.7e-9),    # LHCb measurement
                    'precision': 0.05                   # Required precision
                },
                'BR_Bd_mumu': {
                    'prediction': (1.06e-10, 0.09e-10),
                    'experiment': (1.1e-10, 0.4e-10),
                    'precision': 0.10
                }
            }
            
            # Test each branching ratio
            for channel, data in br_tests.items():
                # Get prediction with uncertainties
                pred = field.compute_observable(channel)
                pred_val = pred['value']
                pred_err = pred['total_uncertainty']
                
                # Compare with experiment
                exp_val, exp_err = data['experiment']
                
                # Calculate pull value
                pull = (pred_val - exp_val) / np.sqrt(pred_err**2 + exp_err**2)
                
                # Verify prediction within 2σ
                self.assertLess(abs(pull), 2.0,
                    f"{channel} prediction deviates by {pull}σ")
                
                # Check precision requirement
                rel_error = pred_err / pred_val
                self.assertLess(rel_error, data['precision'],
                    f"{channel} precision {rel_error:.2%} exceeds requirement {data['precision']:.2%}")

if __name__ == '__main__':
    unittest.main() 