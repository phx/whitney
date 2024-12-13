"""Tests for utility functions."""

import pytest
import numpy as np
from sympy import Symbol, exp
from core.utils import (
    evaluate_expr,
    cached_evaluation,
    check_numerical_stability,
    get_memory_usage,
    profile_computation
)
from core.types import NumericValue
from core.errors import ComputationError, StabilityError

@pytest.mark.numeric
class TestExpressionEvaluation:
    """Test expression evaluation utilities."""
    
    def test_basic_evaluation(self):
        """Test basic expression evaluation."""
        x = Symbol('x')
        expr = x**2 + 2*x + 1
        result = evaluate_expr(expr, {'x': 2.0})
        assert np.isclose(result, 9.0)
    
    def test_complex_expression(self):
        """Test complex expression evaluation."""
        x, y = Symbol('x'), Symbol('y')
        expr = exp(-((x-y)/10.0)**2)
        result = evaluate_expr(expr, {'x': 0.0, 'y': 0.0})
        assert np.isclose(result, 1.0)
    
    def test_invalid_input(self):
        """Test error handling for invalid input."""
        x = Symbol('x')
        expr = x**2
        
        with pytest.raises(ComputationError):
            evaluate_expr(expr, {'y': 1.0})  # Missing variable
            
        with pytest.raises(ComputationError):
            evaluate_expr(expr, {'x': 'invalid'})  # Invalid value

@pytest.mark.numeric
class TestCaching:
    """Test caching functionality."""
    
    def test_basic_caching(self):
        """Test basic caching behavior."""
        @cached_evaluation
        def expensive_func(x):
            return x**2
        
        # First call computes
        result1 = expensive_func(2.0)
        # Second call should use cache
        result2 = expensive_func(2.0)
        
        assert result1 == result2
        assert expensive_func.cache_info().hits == 1
    
    def test_cache_size(self):
        """Test cache size limiting."""
        @cached_evaluation(maxsize=2)
        def func(x):
            return x**2
        
        # Fill cache
        func(1.0)
        func(2.0)
        func(3.0)  # Should evict oldest entry
        
        # Check cache behavior
        assert func.cache_info().currsize <= 2
        assert func(3.0) == 9.0  # Should be cached
        assert func(1.0) == 1.0  # Should recompute

@pytest.mark.numeric
class TestStabilityChecks:
    """Test numerical stability checking."""
    
    def test_basic_stability(self):
        """Test basic stability checking."""
        def stable_func(x):
            return NumericValue(x**2, abs(2*x)*0.01)
            
        result = check_numerical_stability(stable_func, {'x': 1.0})
        assert result['stable']
        assert 'condition_number' in result
    
    def test_unstable_computation(self):
        """Test detection of unstable computations."""
        def unstable_func(x):
            if x > 1.0:
                return NumericValue(float('inf'), None)
            return NumericValue(x, 0.1)
            
        with pytest.raises(StabilityError):
            check_numerical_stability(unstable_func, {'x': 1.1})
    
    @pytest.mark.parametrize('threshold', [1e-6, 1e-8, 1e-10])
    def test_stability_threshold(self, threshold):
        """Test different stability thresholds."""
        def func(x):
            return NumericValue(x, threshold/2)
            
        result = check_numerical_stability(
            func,
            {'x': 1.0},
            threshold=threshold
        )
        assert result['stable']

@pytest.mark.performance
class TestPerformanceMonitoring:
    """Test performance monitoring utilities."""
    
    def test_memory_usage(self):
        """Test memory usage tracking."""
        initial = get_memory_usage()
        
        # Allocate some memory
        large_array = np.zeros((1000, 1000))
        final = get_memory_usage()
        
        if initial >= 0 and final >= 0:  # Memory tracking available
            assert final >= initial
    
    def test_computation_profiling(self):
        """Test computation profiling."""
        @profile_computation
        def slow_func():
            np.random.random((1000, 1000))
            return True
        
        result = slow_func()
        profile = slow_func.get_profile()
        
        assert result is True
        assert 'time' in profile
        assert 'memory' in profile
        assert profile['time'] > 0