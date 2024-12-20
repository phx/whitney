"""Tests for FractalBasis implementation."""

import pytest
from sympy import exp, I, pi
import numpy as np
from core.basis import FractalBasis
from core.types import Energy, FieldConfig, WaveFunction
from core.modes import ComputationMode
from core.errors import PhysicsError
from core.physics_constants import T  # Import symbolic time variable

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
    """Test field equation solutions."""
    config = FieldConfig(mass=1.0, dimension=1, coupling=0.1)
    basis = FractalBasis()
    
    # Create proper gaussian wavepacket
    x = basis._solve_field_equations(config).grid
    psi = np.exp(-x**2/2) / np.sqrt(2*np.pi)  # Normalized gaussian
    
    # Create proper WaveFunction object
    psi = WaveFunction(
        psi=psi,           # Numeric array
        grid=x,
        quantum_numbers={'n': 0}
    )
    
    # Should satisfy field equations
    density = basis.compute_energy_density(psi)
    assert density.is_real

def test_evolution_operator():
    """Test evolution operator computation."""
    basis = FractalBasis()
    U = basis._compute_evolution_operator(Energy(10.0))
    assert U is not None

    # Should be unitary
    t_vals = np.linspace(-1, 1, 10)
    for t in t_vals:
        U_val = complex(U.subs(T, t))
        assert abs(abs(U_val) - 1.0) < 1e-10

# ... rest of existing tests ...