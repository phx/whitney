"""Unit tests for numerical stability."""

import unittest
import numpy as np
from core.stability import (analyze_perturbation, check_convergence,
                          verify_error_bounds)

class TestStability(unittest.TestCase):
    """Test cases for stability analysis."""
    
    def test_perturbation_analysis(self):
        """Test stability under parameter perturbations."""
        def test_func(x: float) -> float:
            return x**2
            
        params = {'x': 1.0}
        results = analyze_perturbation(test_func, params)
        
        self.assertAlmostEqual(results['mean'], 1.0, places=2)
        self.assertLess(results['std'], 1e-5)
    
    def test_convergence(self):
        """Test convergence detection."""
        # Test converging sequence
        seq = [1.0 + 0.1/n for n in range(1, 100)]
        self.assertTrue(check_convergence(seq))
        
        # Test non-converging sequence
        seq = [(-1)**n for n in range(100)]
        self.assertFalse(check_convergence(seq))
    
    def test_error_bounds(self):
        """Test error bound verification."""
        # Generate samples
        samples = np.random.normal(1.0, 0.1, 1000)
        
        # Test valid bounds
        self.assertTrue(verify_error_bounds(1.0, 0.2, samples))
        
        # Test invalid bounds
        self.assertFalse(verify_error_bounds(1.0, 0.05, samples)) 