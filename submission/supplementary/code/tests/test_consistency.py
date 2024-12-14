"""Tests for theoretical consistency checks."""

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
    diff, integrate
)
from core.field import UnifiedField
from core.basis import FractalBasis
from core.types import Energy
from core.errors import PhysicsError
from core.physics_constants import (
    ALPHA_VAL,
    Z_MASS
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
    from core.physics_constants import X, T, P, HBAR, C
    return {'X': X, 'T': T, 'P': P, 'HBAR': HBAR, 'C': C}

class TestConsistencyChecks:
    """Test theoretical consistency requirements."""
    
    @given(st.floats(min_value=0.1, max_value=10.0))
    def test_energy_positivity(self, energy, field, constants):
        """Test energy positivity condition."""
        if not HAS_HYPOTHESIS:
            energy = 1.0  # Default value when hypothesis not available
            
        X, T, C, HBAR = constants['X'], constants['T'], constants['C'], constants['HBAR']
        psi = exp(-(X**2 + (C*T)**2)/(2*HBAR**2))
        
        E = field.compute_energy_density(psi)
        assert E >= 0
        
        # Test with negative mass parameter
        with pytest.raises(PhysicsError):
            field.compute_energy_density(-1.0 * psi)
    
    def test_causality(self, field, constants):
        """Test causality constraints."""
        X, T, C = constants['X'], constants['T'], constants['C']
        
        # Causal configuration
        psi_causal = exp(-(X**2 - (C*T)**2))
        assert field.check_causality(psi_causal)
        
        # Acausal configuration
        psi_acausal = exp(X*T)  # Mixes space and time incorrectly
        assert not field.check_causality(psi_acausal)
    
    def test_unitarity(self, field):
        """Test unitarity of S-matrix."""
        # Generate test states
        energies = [100.0, 200.0, 300.0]
        states = []
        for E in energies:
            states.append(field.compute_basis_state(E))
        
        # Compute S-matrix elements
        S = field.compute_s_matrix(states)
        
        # Check unitarity: S†S = 1
        S_dag_S = S.conjugate().T @ S
        assert np.allclose(S_dag_S, np.eye(len(states)), atol=1e-6)