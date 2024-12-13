"""Tests for validation utilities."""

import pytest
import numpy as np
from sympy import Symbol, exp
from core.validation import (
    validate_energy, validate_momentum, validate_wavefunction,
    validate_numeric_range, validate_config
)
from core.types import Energy, Momentum, WaveFunction
from core.errors import ValidationError

def test_validate_energy():
    """Test energy validation."""
    # Valid cases
    assert isinstance(validate_energy(100.0), Energy)
    assert isinstance(validate_energy(Energy(100.0)), Energy)
    
    # Invalid cases
    with pytest.raises(ValidationError):
        validate_energy(-1.0)
    with pytest.raises(ValidationError):
        validate_energy("invalid")

def test_validate_momentum():
    """Test momentum validation."""
    # Valid cases
    assert isinstance(validate_momentum(50.0), Momentum)
    assert isinstance(validate_momentum(Momentum(50.0)), Momentum)
    
    # Invalid cases
    with pytest.raises(ValidationError):
        validate_momentum("invalid")

def test_validate_wavefunction():
    """Test wavefunction validation."""
    x = Symbol('x')
    psi = exp(-x**2)
    
    # Valid cases
    assert isinstance(validate_wavefunction(psi), WaveFunction)
    assert isinstance(validate_wavefunction(WaveFunction(psi)), WaveFunction)
    
    # Invalid cases
    with pytest.raises(ValidationError):
        validate_wavefunction("invalid")

def test_validate_numeric_range():
    """Test numeric range validation."""
    # Valid cases
    assert validate_numeric_range(5.0, 0.0, 10.0) == 5.0
    assert validate_numeric_range(0.0, 0.0, 10.0) == 0.0
    assert validate_numeric_range(10.0, 0.0, 10.0) == 10.0
    
    # Invalid cases
    with pytest.raises(ValidationError):
        validate_numeric_range(-1.0, 0.0, 10.0)
    with pytest.raises(ValidationError):
        validate_numeric_range(11.0, 0.0, 10.0)
    with pytest.raises(ValidationError):
        validate_numeric_range("invalid", 0.0, 10.0)

def test_validate_config():
    """Test configuration validation."""
    config = {
        'energy': 100.0,
        'momentum': 50.0,
        'mode': 'numeric'
    }
    
    # Valid cases
    validate_config(config, ['energy', 'momentum'])
    validate_config(config, ['energy'])
    
    # Invalid cases
    with pytest.raises(ValidationError):
        validate_config(config, ['invalid_key'])
    with pytest.raises(ValidationError):
        validate_config("invalid", ['energy']) 