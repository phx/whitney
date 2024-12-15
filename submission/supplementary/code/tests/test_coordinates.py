"""Tests for coordinate system implementation and invariance properties."""

import pytest
try:
    from hypothesis import given, strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    from unittest.mock import patch
    HAS_HYPOTHESIS = False
    # Mock hypothesis decorator if not available
    def given(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    st = None

import numpy as np
from sympy import (
    exp, I, pi, sqrt, 
    diff, integrate, conjugate,
    evalf
)

from core.field import UnifiedField
from core.basis import FractalBasis
from core.types import (
    Energy, 
    FieldConfig, 
    WaveFunction
)
from core.errors import PhysicsError
from core.contexts import (
    gauge_phase,
    lorentz_boost,
    quantum_state,
    numeric_precision
)
from core.physics_constants import (
    X, T, C, HBAR,
    ALPHA_VAL,
    P
)

@pytest.fixture
def field():
    """Create UnifiedField instance for testing."""
    return UnifiedField(alpha=ALPHA_VAL)

@pytest.fixture
def basis():
    """Create FractalBasis instance for testing."""
    return FractalBasis(alpha=ALPHA_VAL)

@pytest.fixture
def constants(request):
    """Get physics constants from conftest."""
    return {'X': X, 'T': T, 'P': P, 'HBAR': HBAR, 'C': C}

@pytest.mark.physics
class TestLorentzInvariance:
    """Test Lorentz invariance properties."""
    
    @given(st.floats(min_value=-0.9, max_value=0.9))
    def test_energy_density_invariance(self, beta):
        """Test that energy density transforms correctly."""
        field = UnifiedField(alpha=ALPHA_VAL)
        with gauge_phase() as phase, \
             quantum_state(energy=1.0) as (psi, _), \
             numeric_precision(rtol=1e-10) as prec:
            psi_gauge = field.apply_gauge_transform(psi, phase)
            E1 = field.compute_energy_density(psi_gauge)
            
            # Apply Lorentz boost
            psi_boosted = field.apply_lorentz_transform(psi_gauge, beta)
            E2 = field.compute_energy_density(psi_boosted)
            
            # Energy density should transform correctly
            gamma = 1/sqrt(1 - beta**2)
            assert abs(E2 - gamma**2 * E1) < prec['rtol']
    
    @given(st.floats(min_value=-0.9, max_value=0.9))
    def test_causality_invariance(self, beta):
        """Test that causality is preserved under boosts."""
        field = UnifiedField(alpha=ALPHA_VAL)
        with gauge_phase() as phase, \
             quantum_state(energy=1.0) as (psi, _), \
             numeric_precision(rtol=1e-10) as prec:
            psi_gauge = field.apply_gauge_transform(psi, phase)
            assert field.check_causality(psi_gauge)
            
            # Causality should be preserved under boosts
            psi_boosted = field.apply_lorentz_transform(psi_gauge, beta)
            assert field.check_causality(psi_boosted)

    def test_invalid_boost(self, field, constants):
        """Test that superluminal boosts are rejected."""
        X, T, C, HBAR = constants['X'], constants['T'], constants['C'], constants['HBAR']
        psi = exp(-(X**2 + (C*T)**2)/(2*HBAR**2))
        
        with pytest.raises(PhysicsError):
            field.apply_lorentz_transform(psi, 1.1)  # v > c

class TestCoordinateSystem:
    """Test coordinate system properties."""
    
    def test_basic_properties(self, field, constants):
        """Test basic coordinate system properties."""
        X, T, C = constants['X'], constants['T'], constants['C']
        
        # Test at origin
        psi_origin = exp(0)
        assert field.check_causality(psi_origin)
        
        # Test at light cone
        psi_lightcone = exp(-(X + C*T)*(X - C*T)/(2*constants['HBAR']**2))
        assert field.check_causality(psi_lightcone)
        
        # Test at spatial infinity
        psi_infinity = exp(-abs(X))
        E = field.compute_energy_density(psi_infinity)
        assert E.is_real
        assert E >= 0

class TestNumericalStability:
    """Test numerical stability of coordinate transformations."""
    
    @given(st.floats(min_value=0.99, max_value=0.999999))
    def test_high_boost_stability(self, beta, field, constants):
        """Test numerical stability at high boosts."""
        X, T, C, HBAR = constants['X'], constants['T'], constants['C'], constants['HBAR']
        psi = exp(-(X**2 + (C*T)**2)/(2*HBAR**2))
        
        # Apply high boost
        psi_boosted = field.apply_lorentz_transform(psi, beta)
        
        # Should maintain reasonable values
        E = field.compute_energy_density(psi_boosted)
        assert not np.isinf(float(E.evalf()))
        assert not np.isnan(float(E.evalf()))
    
    @given(st.integers(min_value=100, max_value=1000))
    def test_large_coordinate_stability(self, scale, field, constants):
        """Test stability at large coordinate values."""
        X, T, C, HBAR = constants['X'], constants['T'], constants['C'], constants['HBAR']
        # Test at large distances
        psi = exp(-(X**2 + (C*T)**2)/(2*scale*HBAR**2))
        
        # Should still give physical results
        E = field.compute_energy_density(psi)
        assert E.is_real
        assert E >= 0
        assert field.check_causality(psi)