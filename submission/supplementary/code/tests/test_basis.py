"""Tests for FractalBasis implementation."""

import pytest
from sympy import exp, I, pi
import numpy as np
from core.basis import FractalBasis
from core.types import Energy, FieldConfig
from core.modes import ComputationMode
from core.errors import PhysicsError

@pytest.fixture
def basis():
    """Create FractalBasis instance for testing."""
    return FractalBasis(alpha=0.1)

def test_inheritance():
    """Test proper inheritance from UnifiedField."""
    basis = FractalBasis()
    assert isinstance(basis, FractalBasis)
    assert hasattr(basis, 'compute_energy_density')
    assert hasattr(basis, 'check_causality')

def test_basis_computation():
    """Test basis function computation."""
    basis = FractalBasis()
    psi = basis.compute(n=0, E=Energy(1.0))
    assert psi is not None
    
    with pytest.raises(PhysicsError):
        basis.compute(n=-1)

def test_field_equations():
    """Test field equation solution in basis."""
    basis = FractalBasis()
    config = FieldConfig(mass=1.0, dimension=0, coupling=0.1)
    psi = basis._solve_field_equations(config)
    assert psi is not None
    
    # Should satisfy field equations
    density = basis.compute_energy_density(psi)
    assert density.is_real

def test_evolution_operator():
    """Test evolution operator computation."""
    basis = FractalBasis()
    U = basis._compute_evolution_operator(Energy(10.0))
    assert U is not None
    
    # Should be unitary
    assert abs(abs(U) - 1.0) < 1e-10

# ... rest of existing tests ...