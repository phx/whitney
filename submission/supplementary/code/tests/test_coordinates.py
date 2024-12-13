"""Tests for coordinate system implementation and invariance properties."""

import pytest
from hypothesis import given, strategies as st
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
from core.transforms import lorentz_boost, gauge_transform
from core.physics_constants import (
    ALPHA_VAL,
    HBAR, C,
    X, T, P
)
from core.errors import PhysicsError

@pytest.fixture
def field():
    """Create UnifiedField instance for testing."""
    return UnifiedField(alpha=ALPHA_VAL)

@pytest.fixture
def basis():
    """Create FractalBasis instance for testing."""
    return FractalBasis(alpha=ALPHA_VAL)

class TestLorentzInvariance:
    """Test Lorentz invariance of field theory."""
    
    @given(st.floats(min_value=-0.9, max_value=0.9))
    def test_energy_density_invariance(self, beta, field):
        """Test that energy density is Lorentz invariant."""
        # Create test field configuration
        psi = exp(-(X**2 + (C*T)**2)/(2*HBAR**2))
        
        # Compute original energy density
        E1 = field.compute_energy_density(psi)
        
        # Apply Lorentz boost
        psi_boosted = field.apply_lorentz_transform(psi, beta)
        E2 = field.compute_energy_density(psi_boosted)
        
        # Energy density should be invariant
        assert abs(E1 - E2) < 1e-10
    
    @given(st.floats(min_value=-0.9, max_value=0.9))
    def test_causality_invariance(self, beta, field):
        """Test that causality is preserved under Lorentz transformations."""
        psi = exp(-(X**2 + (C*T)**2)/(2*HBAR**2))
        
        # Check causality in original frame
        assert field.check_causality(psi)
        
        # Check causality in boosted frame
        psi_boosted = field.apply_lorentz_transform(psi, beta)
        assert field.check_causality(psi_boosted)

    def test_invalid_boost(self, field):
        """Test that superluminal boosts are rejected."""
        psi = exp(-(X**2 + (C*T)**2)/(2*HBAR**2))
        
        with pytest.raises(PhysicsError):
            field.apply_lorentz_transform(psi, 1.1)  # v > c

class TestGaugeInvariance:
    """Test gauge invariance properties."""
    
    @given(st.floats(min_value=0, max_value=2*pi))
    def test_observables_invariance(self, phase, field):
        """Test that physical observables are gauge invariant."""
        psi = exp(-(X**2 + (C*T)**2)/(2*HBAR**2))
        
        # Apply gauge transformation
        psi_transformed = field.apply_gauge_transform(psi, phase)
        
        # Energy density should be invariant
        E1 = field.compute_energy_density(psi)
        E2 = field.compute_energy_density(psi_transformed)
        assert abs(E1 - E2) < 1e-10
        
        # Causality should be invariant
        c1 = field.check_causality(psi)
        c2 = field.check_causality(psi_transformed)
        assert c1 == c2

class TestBasisTransformations:
    """Test basis function transformation properties."""
    
    @given(
        st.integers(min_value=0, max_value=5),
        st.floats(min_value=-0.9, max_value=0.9)
    )
    def test_basis_lorentz_covariance(self, n, beta, basis):
        """Test that basis functions transform covariantly."""
        # Compute basis function
        psi = basis.compute(n, Energy(1.0))
        
        # Apply Lorentz transformation
        psi_boosted = basis.apply_lorentz_transform(psi, beta)
        
        # Should remain normalized
        assert abs(basis.compute_inner_product(psi_boosted, psi_boosted) - 1.0) < 1e-10
        
        # Should preserve orthogonality
        if n > 0:
            psi_0 = basis.compute(0, Energy(1.0))
            psi_0_boosted = basis.apply_lorentz_transform(psi_0, beta)
            overlap = basis.compute_inner_product(psi_boosted, psi_0_boosted)
            assert abs(overlap) < 1e-10 

class TestEdgeCases:
    """Test coordinate system edge cases."""
    
    def test_near_lightspeed_boost(self, field):
        """Test behavior near lightspeed."""
        psi = exp(-(X**2 + (C*T)**2)/(2*HBAR**2))
        
        # Test very close to c
        beta = 0.9999999
        gamma = 1/sqrt(1 - beta**2)
        
        # Should still preserve causality
        psi_boosted = field.apply_lorentz_transform(psi, beta)
        assert field.check_causality(psi_boosted)
        
        # Energy should scale correctly with gamma
        E1 = field.compute_energy_density(psi)
        E2 = field.compute_energy_density(psi_boosted)
        assert abs(E2/E1 - gamma) < 1e-6
    
    @given(st.integers(min_value=1, max_value=100))
    def test_large_gauge_transform(self, n, field):
        """Test large gauge transformations (multiple of 2π)."""
        psi = exp(-(X**2 + (C*T)**2)/(2*HBAR**2))
        
        # Apply large gauge transformation
        phase = 2*pi*n
        psi_transformed = field.apply_gauge_transform(psi, phase)
        
        # Should return to original state
        assert abs(psi_transformed - psi) < 1e-10
    
    def test_coordinate_singularities(self, field):
        """Test behavior at coordinate singularities."""
        # Test at origin
        psi_origin = exp(0)
        assert field.check_causality(psi_origin)
        
        # Test at light cone
        psi_lightcone = exp(-(X + C*T)*(X - C*T)/(2*HBAR**2))
        assert field.check_causality(psi_lightcone)
        
        # Test at spatial infinity
        psi_infinity = exp(-abs(X))
        E = field.compute_energy_density(psi_infinity)
        assert E.is_real
        assert E >= 0

class TestNumericalStability:
    """Test numerical stability of coordinate transformations."""
    
    @given(st.floats(min_value=0.99, max_value=0.999999))
    def test_high_boost_stability(self, beta, field):
        """Test numerical stability at high boosts."""
        psi = exp(-(X**2 + (C*T)**2)/(2*HBAR**2))
        
        # Apply high boost
        psi_boosted = field.apply_lorentz_transform(psi, beta)
        
        # Should maintain reasonable values
        E = field.compute_energy_density(psi_boosted)
        assert not np.isinf(float(E.evalf()))
        assert not np.isnan(float(E.evalf()))
    
    @given(st.integers(min_value=100, max_value=1000))
    def test_large_coordinate_stability(self, scale, field):
        """Test stability at large coordinate values."""
        # Test at large distances
        psi = exp(-(X**2 + (C*T)**2)/(2*scale*HBAR**2))
        
        # Should still give physical results
        E = field.compute_energy_density(psi)
        assert E.is_real
        assert E >= 0
        assert field.check_causality(psi)