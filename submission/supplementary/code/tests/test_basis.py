"""Tests for fractal basis functions."""

import pytest
import numpy as np
from sympy import exp, I
from core.basis import FractalBasis
from core.types import Energy, WaveFunction
from core.modes import ComputationMode
from core.errors import PhysicsError, ValidationError
from core.physics_constants import ALPHA_VAL, Z_MASS, X

@pytest.fixture
def basis():
    """Create test basis instance."""
    return FractalBasis(alpha=ALPHA_VAL, mode=ComputationMode.SYMBOLIC)

class TestBasisFunctions:
    """Test basis function computation and properties."""
    
    def test_basic_function(self, basis):
        """Test basic basis function computation."""
        n = 0  # Ground state
        E = Energy(Z_MASS)
        
        psi = basis.compute_basis_function(n, E)
        
        # Should be normalized Gaussian-like
        assert isinstance(psi, WaveFunction)
        assert psi.contains(exp(-X**2))  # Contains Gaussian factor
        
        # Test normalization
        norm = basis.compute_inner_product(psi, psi)
        assert np.isclose(norm, 1.0, rtol=1e-6)
    
    def test_orthonormality(self, basis):
        """Test basis function orthonormality."""
        E = Energy(Z_MASS)
        
        # Test <ψₙ|ψₙ> = 1
        for n in range(3):
            psi_n = basis.compute_basis_function(n, E)
            norm = basis.compute_inner_product(psi_n, psi_n)
            assert np.isclose(norm, 1.0, rtol=1e-6)
        
        # Test <ψₘ|ψₘ> = 0 for n ≠ m
        for n in range(2):
            for m in range(n+1, 3):
                psi_n = basis.compute_basis_function(n, E)
                psi_m = basis.compute_basis_function(m, E)
                overlap = basis.compute_inner_product(psi_n, psi_m)
                assert np.isclose(overlap, 0.0, atol=1e-6)
    
    def test_completeness(self, basis):
        """Test basis completeness relation."""
        E = Energy(Z_MASS)
        test_points = np.linspace(-1, 1, 5)
        
        # Test resolution of identity
        for x in test_points:
            # Sum over first few basis functions
            sum_val = 0
            for n in range(5):
                psi = basis.compute_basis_function(n, E)
                sum_val += abs(psi.evaluate_at(x))**2
            
            # Should approach 1 for complete basis
            assert sum_val > 0.9  # Allow some truncation error
    
    @pytest.mark.parametrize("n,E", [
        (-1, 100.0),  # Invalid index
        (0, -100.0),  # Invalid energy
        (0, 0.0),     # Zero energy
    ])
    def test_invalid_parameters(self, basis, n, E):
        """Test error handling for invalid parameters."""
        with pytest.raises((ValidationError, PhysicsError)):
            basis.compute_basis_function(n, Energy(E))
    
    def test_gauge_transformation(self, basis):
        """Test gauge transformation properties."""
        n = 0
        E = Energy(Z_MASS)
        psi = basis.compute_basis_function(n, E)
        
        # Apply U(1) transformation
        phase = exp(I * np.pi/4)
        psi_transformed = basis.apply_gauge_transformation(psi, phase)
        
        # Check norm preservation
        norm = basis.compute_inner_product(psi_transformed, psi_transformed)
        assert np.isclose(norm, 1.0, rtol=1e-6)
        
        # Check phase relation
        ratio = psi_transformed / psi
        assert np.isclose(abs(ratio), 1.0, rtol=1e-6)