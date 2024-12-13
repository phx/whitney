"""Tests for FractalBasis implementation."""

import pytest
import numpy as np
from core.basis import FractalBasis
from core.constants import ALPHA_VAL
from core.errors import PhysicsError
from core.types import WaveFunction

@pytest.fixture
def basis():
    """Create a test basis instance."""
    return FractalBasis(alpha=ALPHA_VAL, max_level=3)

def test_basis_initialization(basis):
    """Test basis initialization."""
    assert isinstance(basis, FractalBasis)
    assert hasattr(basis, 'alpha')
    assert basis.alpha == pytest.approx(ALPHA_VAL)
    assert basis.max_level == 3
    
    # Test invalid initialization
    with pytest.raises(PhysicsError):
        FractalBasis(alpha=-1.0)  # Invalid coupling
    
    with pytest.raises(ValueError):
        FractalBasis(alpha=ALPHA_VAL, max_level=0)  # Invalid level

def test_basis_functions(basis):
    """Test basis function generation."""
    x = np.linspace(-5, 5, 100)
    
    # Test ground state
    psi_0 = basis.compute_basis_function(0, x)
    assert isinstance(psi_0, WaveFunction)
    assert np.all(np.isfinite(psi_0.psi))
    assert psi_0.quantum_numbers['n'] == 0
    
    # Test excited states
    for n in range(1, 4):
        psi_n = basis.compute_basis_function(n, x)
        assert isinstance(psi_n, WaveFunction)
        assert np.all(np.isfinite(psi_n.psi))
        assert psi_n.quantum_numbers['n'] == n

def test_orthonormality(basis):
    """Test basis orthonormality."""
    x = np.linspace(-10, 10, 200)
    dx = x[1] - x[0]
    
    # Generate basis functions
    basis_functions = [
        basis.compute_basis_function(n, x)
        for n in range(4)
    ]
    
    # Test orthonormality
    for i, psi_i in enumerate(basis_functions):
        for j, psi_j in enumerate(basis_functions):
            overlap = np.sum(np.conj(psi_i.psi) * psi_j.psi) * dx
            if i == j:
                assert abs(overlap - 1.0) < 1e-6  # Normalized
            else:
                assert abs(overlap) < 1e-6  # Orthogonal

def test_completeness(basis):
    """Test basis completeness."""
    x = np.linspace(-5, 5, 100)
    
    # Test function to expand
    def test_func(x):
        return np.exp(-x**2/2) * np.cos(2*x)
    
    # Expand in basis
    coeffs = basis.compute_expansion_coefficients(test_func, x)
    reconstruction = basis.reconstruct_function(coeffs, x)
    
    # Test reconstruction accuracy
    error = np.max(np.abs(test_func(x) - reconstruction))
    assert error < 1e-3  # Should reconstruct within tolerance

def test_fractal_properties(basis):
    """Test fractal scaling properties."""
    x = np.linspace(-5, 5, 100)
    psi = basis.compute_basis_function(0, x)
    
    # Test scaling property
    lambda_scale = 2.0
    scaled_x = x * lambda_scale
    scaled_psi = basis.compute_basis_function(0, scaled_x)
    
    # Check scaling relation
    scale_factor = lambda_scale**basis.scaling_dimension
    assert np.allclose(
        scaled_psi.psi * scale_factor,
        psi.psi,
        rtol=1e-5
    )

def test_error_handling(basis):
    """Test error handling."""
    x = np.linspace(-5, 5, 100)
    
    # Test invalid quantum number
    with pytest.raises(ValueError):
        basis.compute_basis_function(-1, x)
    
    with pytest.raises(ValueError):
        basis.compute_basis_function(basis.max_level + 1, x)
    
    # Test invalid grid
    with pytest.raises(ValueError):
        basis.compute_basis_function(0, np.array([]))  # Empty grid