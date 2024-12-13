"""Tests for NumericValue class."""

import pytest
import numpy as np
from core.types import NumericValue

def test_numeric_value_initialization():
    """Test basic initialization."""
    # Real values
    x = NumericValue(1.0)
    assert x.value == 1.0
    assert x.uncertainty is None
    
    # With uncertainty
    x = NumericValue(1.0, 0.1)
    assert x.value == 1.0
    assert x.uncertainty == 0.1
    
    # Complex values
    z = NumericValue(1+1j, 0.1)
    assert z.value == 1+1j
    assert z.uncertainty == 0.1

def test_numeric_value_validation():
    """Test value validation."""
    # Invalid value types
    with pytest.raises(TypeError):
        NumericValue("not a number")
    
    # Invalid uncertainties
    with pytest.raises(ValueError):
        NumericValue(1.0, -0.1)  # Negative uncertainty
    
    with pytest.raises(TypeError):
        NumericValue(1.0, "0.1")  # Wrong type
        
    # Non-finite values
    with pytest.raises(ValueError):
        NumericValue(float('inf'))
    
    with pytest.raises(ValueError):
        NumericValue(float('nan'))

def test_numeric_value_range_validation():
    """Test range validation."""
    # Valid range
    x = NumericValue(1.0, valid_range=(0, 2))
    assert x.value == 1.0
    
    # Outside range
    with pytest.raises(ValueError):
        NumericValue(3.0, valid_range=(0, 2))
    
    # Complex value range
    z = NumericValue(1+1j, valid_range=(0, 2))  # Magnitude ~1.414
    assert abs(z.value) < 2.0
    
    with pytest.raises(ValueError):
        NumericValue(2+2j, valid_range=(0, 2))  # Magnitude ~2.828

def test_numeric_value_arithmetic():
    """Test arithmetic operations."""
    x = NumericValue(2.0, 0.2)
    y = NumericValue(3.0, 0.3)
    
    # Addition
    z = x + y
    assert z.value == 5.0
    assert np.isclose(z.uncertainty, np.sqrt(0.2**2 + 0.3**2))
    
    # Multiplication
    z = x * y
    assert z.value == 6.0
    # Check relative uncertainty propagation
    rel_unc = np.sqrt((0.2/2.0)**2 + (0.3/3.0)**2)
    assert np.isclose(z.uncertainty, 6.0 * rel_unc)

def test_numeric_value_complex_arithmetic():
    """Test complex arithmetic."""
    z1 = NumericValue(1+1j, 0.1)
    z2 = NumericValue(2-1j, 0.2)
    
    # Addition
    z = z1 + z2
    assert z.value == (3+0j)
    assert np.isclose(z.uncertainty, np.sqrt(0.1**2 + 0.2**2))
    
    # Multiplication
    z = z1 * z2
    assert z.value == (3+1j)
    # Check uncertainty propagation for complex values
    assert z.uncertainty is not None 

def test_numeric_value_division():
    """Test division operation."""
    # Division by scalar
    x = NumericValue(6.0, 0.2)
    y = x / 2
    assert y.value == 3.0
    assert np.isclose(y.uncertainty, 0.1)  # Uncertainty scales linearly
    
    # Division by NumericValue
    x = NumericValue(6.0, 0.2)
    y = NumericValue(2.0, 0.1)
    z = x / y
    assert z.value == 3.0
    # Check relative uncertainty propagation
    rel_unc = np.sqrt((0.2/6.0)**2 + (0.1/2.0)**2)
    assert np.isclose(z.uncertainty, 3.0 * rel_unc)
    
    # Division without uncertainties
    x = NumericValue(6.0)
    y = NumericValue(2.0)
    z = x / y
    assert z.value == 3.0
    assert z.uncertainty is None