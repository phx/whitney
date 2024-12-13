"""Tests for field correlation functions."""

import pytest
from hypothesis import given, strategies as st
import numpy as np
from sympy import (
    exp, I, pi, sqrt, integrate, conjugate, diff,
    Symbol, oo, Function
)
from core.field import UnifiedField
from core.types import Energy, FieldConfig, WaveFunction
from core.physics_constants import X, T, P, HBAR, C
from core.errors import PhysicsError

@pytest.fixture
def field():
    """Create UnifiedField instance for testing."""
    return UnifiedField(alpha=0.1)

@pytest.fixture
def test_state():
    """Create test field configuration."""
    return exp(-(X**2 + (C*T)**2)/(2*HBAR**2))

class TestTwoPointFunctions:
    """Test two-point correlation functions."""
    
    def test_vacuum_correlator(self, field, test_state):
        """Test vacuum two-point function."""
        x1, x2 = X + 1, X - 1
        t1, t2 = T + 0.5, T - 0.5
        
        # Compute <0|φ(x₁,t₁)φ(x₂,t₂)|0>
        correlator = field.compute_correlator(
            test_state,
            [(x1, t1), (x2, t2)]
        )
        
        # Should decay with spacelike separation
        separation = sqrt((x1 - x2)**2 - C**2*(t1 - t2)**2)
        assert abs(correlator) <= exp(-separation/HBAR)
    
    def test_cluster_decomposition(self, field, test_state):
        """Test cluster decomposition principle."""
        # Points with large separation
        x1, x2 = X + 100, X - 100
        t1, t2 = T, T
        
        correlator = field.compute_correlator(
            test_state,
            [(x1, t1), (x2, t2)]
        )
        
        # Should factorize at large distances
        product = (
            field.compute_correlator(test_state, [(x1, t1)]) *
            field.compute_correlator(test_state, [(x2, t2)])
        )
        
        assert abs(correlator - product) < 1e-10

class TestSpectralFunctions:
    """Test spectral decomposition of correlators."""
    
    @given(st.floats(min_value=1.0, max_value=10.0))
    def test_källén_lehmann(self, mass, field):
        """Test Källén-Lehmann spectral representation."""
        # Create massive field state
        config = FieldConfig(mass=mass, coupling=0.1, dimension=1)
        psi = field.compute_field(config)
        
        # Compute spectral function
        rho = field.compute_spectral_function(psi)
        
        # Should be positive and normalized
        assert rho >= 0
        norm = integrate(rho, (P, -oo, oo))
        assert abs(norm - 1) < 1e-10
    
    def test_dispersion_relations(self, field, test_state):
        """Test dispersion relations."""
        # Compute retarded propagator
        G_R = field.compute_retarded_propagator(test_state)
        
        # Verify causality through dispersion relations
        omega = Symbol('omega')
        k = Symbol('k')
        
        # Kramers-Kronig relations
        re_G = integrate(G_R.real, (omega, -oo, oo))
        im_G = integrate(G_R.imag, (omega, -oo, oo))
        
        assert abs(re_G) < 1e-10  # Real part should vanish
        assert im_G > 0  # Imaginary part should be positive

class TestWardIdentities:
    """Test Ward identities from gauge invariance."""
    
    def test_current_conservation(self, field, test_state):
        """Test current conservation Ward identity."""
        # Compute current-field correlator
        j0, j1 = field.compute_gauge_current(test_state)
        
        # Ward identity: ∂_μ<j^μ(x)φ(y)> = 0
        d_t_corr = diff(
            field.compute_correlator(test_state, [(j0, T)]),
            T
        )
        d_x_corr = diff(
            field.compute_correlator(test_state, [(j1, X)]),
            X
        )
        
        assert abs(d_t_corr + d_x_corr) < 1e-10
    
    def test_charge_conservation(self, field, test_state):
        """Test charge conservation."""
        # Compute charge operator
        Q = integrate(
            field.compute_gauge_current(test_state)[0],
            (X, -oo, oo)
        )
        
        # Should commute with Hamiltonian
        H = field._compute_hamiltonian()
        commutator = Q*H - H*Q
        
        assert abs(commutator) < 1e-10