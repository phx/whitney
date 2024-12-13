"""Tests for computation functions."""

import pytest
import numpy as np
from sympy import Symbol, exp
from core.compute import (
    compute_cross_section, compute_correlation_function,
    compute_branching_ratio
)
from core.types import Energy, Momentum, WaveFunction
from core.errors import ComputationError, PhysicsError
from core.physics_constants import ALPHA_VAL, X

@pytest.fixture
def test_wavefunction():
    """Create test wavefunction."""
    return WaveFunction(exp(-X**2))

def test_compute_cross_section(test_wavefunction):
    """Test cross section computation."""
    # Valid case
    sigma = compute_cross_section(
        energy=100.0,
        momentum=50.0,
        wavefunction=test_wavefunction
    )
    assert sigma.value > 0
    
    # Invalid cases
    with pytest.raises(ComputationError):
        compute_cross_section(-100.0, 50.0, test_wavefunction)
    with pytest.raises(ComputationError):
        compute_cross_section(100.0, -50.0, test_wavefunction)

def test_compute_correlation_function(test_wavefunction):
    """Test correlation function computation."""
    # Valid cases
    corr = compute_correlation_function(0.0, 1.0, 100.0, test_wavefunction)
    assert isinstance(corr, float)
    assert -1 <= corr <= 1
    
    # Test correlation properties
    corr_zero = compute_correlation_function(0.0, 0.0, 100.0, test_wavefunction)
    assert np.isclose(corr_zero, 1.0)
    
    # Test symmetry
    corr_12 = compute_correlation_function(1.0, 2.0, 100.0, test_wavefunction)
    corr_21 = compute_correlation_function(2.0, 1.0, 100.0, test_wavefunction)
    assert np.isclose(corr_12, corr_21)

def test_compute_branching_ratio():
    """Test branching ratio computation."""
    couplings = {
        "Z->ee": 0.1,
        "Z->mumu": 0.1,
        "Z->tautau": 0.1
    }
    
    # Valid cases
    br = compute_branching_ratio("Z->ee", 100.0, couplings)
    assert 0 <= br <= 1
    
    # Sum of branching ratios should be 1
    brs = [
        compute_branching_ratio(p, 100.0, couplings)
        for p in ["Z->ee", "Z->mumu", "Z->tautau"]
    ]
    assert np.isclose(sum(brs), 1.0)
    
    # Invalid cases
    with pytest.raises(PhysicsError):
        compute_branching_ratio("invalid", 100.0, couplings)
    with pytest.raises(PhysicsError):
        compute_branching_ratio("Z->ee", 1e-6, couplings)  # Below threshold 