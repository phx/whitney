"""Tests for numerical stability analysis."""

import pytest
import numpy as np
from core.stability import (
    analyze_perturbation,
    check_convergence,
    verify_error_bounds
)
from core.types import NumericValue
from core.errors import StabilityError

@pytest.mark.stability
class TestPerturbationAnalysis:
    """Test perturbation analysis functionality."""
    
    def test_basic_perturbation(self):
        """Test basic perturbation analysis."""
        def test_func(x: float) -> NumericValue:
            return NumericValue(x**2, abs(2*x)*0.01)
            
        params = {'x': 1.0}
        results = analyze_perturbation(test_func, params)
        
        assert 'mean' in results
        assert 'std' in results
        assert 'max_dev' in results
        assert 'condition' in results
        assert abs(results['mean'] - 1.0) < 0.1
    
    def test_unstable_function(self):
        """Test detection of unstable functions."""
        def unstable_func(x: float) -> NumericValue:
            if x > 1.0:
                return NumericValue(float('inf'), None)
            return NumericValue(x, 0.1)
            
        with pytest.raises(StabilityError):
            analyze_perturbation(unstable_func, {'x': 1.1})
    
    @pytest.mark.parametrize('epsilon', [1e-4, 1e-5, 1e-6])
    def test_perturbation_size(self, epsilon):
        """Test effect of perturbation size."""
        def test_func(x: float) -> NumericValue:
            return NumericValue(x, 0.01*abs(x))
            
        results = analyze_perturbation(
            test_func,
            {'x': 1.0},
            epsilon=epsilon
        )
        assert results['std'] < epsilon

@pytest.mark.stability
class TestConvergence:
    """Test convergence checking functionality."""
    
    def test_basic_convergence(self):
        """Test basic convergence detection."""
        # Converging sequence
        values = [NumericValue(1.0 + 0.1/n, 0.01/n) for n in range(1, 11)]
        assert check_convergence(values, threshold=1e-3)
        
        # Non-converging sequence
        values = [NumericValue((-1)**n, 0.1) for n in range(10)]
        assert not check_convergence(values, threshold=1e-3)
    
    def test_early_convergence(self):
        """Test early convergence detection."""
        values = [NumericValue(1.0, 0.01) for _ in range(3)]
        assert check_convergence(values, threshold=1e-6)
    
    def test_slow_convergence(self):
        """Test slow convergence detection."""
        values = [NumericValue(1.0 + 1.0/n, 0.1/n) for n in range(1, 101)]
        assert check_convergence(values, threshold=1e-2)

@pytest.mark.stability
class TestErrorBounds:
    """Test error bound verification."""
    
    def test_basic_error_bounds(self):
        """Test basic error bound verification."""
        nominal = 1.0
        error_est = 0.1
        samples = np.random.normal(nominal, error_est/2, 1000)
        
        assert verify_error_bounds(nominal, error_est, samples)
    
    def test_invalid_bounds(self):
        """Test detection of invalid error bounds."""
        nominal = 1.0
        error_est = 0.01  # Too small
        samples = np.random.normal(nominal, 0.1, 1000)  # Larger spread
        
        assert not verify_error_bounds(nominal, error_est, samples)
    
    @pytest.mark.parametrize('confidence', [0.68, 0.95, 0.99])
    def test_confidence_levels(self, confidence):
        """Test different confidence levels."""
        nominal = 1.0
        error_est = 0.1
        samples = np.random.normal(nominal, error_est/2, 1000)
        
        assert verify_error_bounds(
            nominal,
            error_est,
            samples,
            confidence=confidence
        )