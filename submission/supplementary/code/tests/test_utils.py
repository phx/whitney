"""Tests for utility functions."""

import pytest
import numpy as np
from functools import partial
from core.utils import (
    evaluate_expr,
    cached_evaluation,
    check_numerical_stability,
    profile_computation,
    propagate_errors,
    batch_process
)
from core.errors import ComputationError, StabilityError

def test_evaluate_expr():
    """Test expression evaluation."""
    # Test basic arithmetic
    expr = "x**2 + 2*x + 1"
    x = 2.0
    result = evaluate_expr(expr, {'x': x})
    assert result == x**2 + 2*x + 1
    
    # Test with multiple variables
    expr = "a*x**2 + b*x + c"
    params = {'x': 2.0, 'a': 1.0, 'b': 2.0, 'c': 1.0}
    result = evaluate_expr(expr, params)
    assert result == params['a']*params['x']**2 + params['b']*params['x'] + params['c']
    
    # Test error handling
    with pytest.raises(ComputationError):
        evaluate_expr("1/0", {})
    
    with pytest.raises(ComputationError):
        evaluate_expr("undefined_var", {})

def test_cached_evaluation():
    """Test computation caching."""
    call_count = 0
    
    @cached_evaluation(maxsize=2)
    def expensive_computation(x):
        nonlocal call_count
        call_count += 1
        return x**2
    
    # First call should compute
    result1 = expensive_computation(2.0)
    assert result1 == 4.0
    assert call_count == 1
    
    # Second call with same input should use cache
    result2 = expensive_computation(2.0)
    assert result2 == 4.0
    assert call_count == 1  # Unchanged
    
    # Different input should compute
    result3 = expensive_computation(3.0)
    assert result3 == 9.0
    assert call_count == 2
    
    # Test cache size limit
    expensive_computation(4.0)  # Should evict 2.0
    expensive_computation(2.0)  # Should recompute
    assert call_count == 4

def test_numerical_stability():
    """Test numerical stability checking."""
    # Stable computation
    def stable_func(x):
        return np.sin(x)
    
    result = check_numerical_stability(stable_func(1.0))
    assert result
    
    # Unstable computation
    def unstable_func(x):
        return 1e10 * (x - 1.0)
    
    with pytest.raises(StabilityError):
        check_numerical_stability(unstable_func(1.0))
    
    # Test with custom threshold
    result = check_numerical_stability(unstable_func(1.0), 
                                     threshold=1e12)
    assert not result

def test_profile_computation():
    """Test computation profiling."""
    @profile_computation
    def test_func():
        x = 0
        for i in range(1000):
            x += i
        return x
    
    result = test_func()
    
    # Check profile data
    assert hasattr(test_func, 'profile_data')
    assert 'execution_time' in test_func.profile_data
    assert 'memory_usage' in test_func.profile_data
    assert test_func.profile_data['execution_time'] > 0
    assert test_func.profile_data['memory_usage'] > 0

def test_error_propagation():
    """Test error propagation utilities."""
    def func(x, y):
        return x**2 + y**2
    
    values = {'x': 1.0, 'y': 2.0}
    errors = {'x': 0.1, 'y': 0.2}
    
    # Test error propagation
    result = propagate_errors(func, values, errors)
    assert 'value' in result
    assert 'error' in result
    assert np.isclose(result['value'], 5.0)  # 1² + 2²
    
    # Error should combine x and y uncertainties
    theoretical_error = np.sqrt(
        (2*values['x']*errors['x'])**2 +
        (2*values['y']*errors['y'])**2
    )
    assert np.isclose(result['error'], theoretical_error)

def test_batch_processing():
    """Test batch processing utilities."""
    data = np.arange(100)
    
    def process_batch(batch):
        return np.sum(batch)
    
    # Test with different batch sizes
    result1 = batch_process(data, process_batch, batch_size=10)
    result2 = batch_process(data, process_batch, batch_size=20)
    
    assert len(result1) == 10
    assert len(result2) == 5
    assert np.sum(result1) == np.sum(result2)
    assert np.sum(result1) == np.sum(data)