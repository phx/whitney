"""Tests for gauge transformations."""

import pytest
import numpy as np
from hypothesis import given, strategies as st
from sympy import exp, I, pi, sqrt, integrate, conjugate, oo, Matrix, diff
from core.field import UnifiedField
from core.basis import FractalBasis
from core.types import Energy
from core.errors import PhysicsError, GaugeError
from core.contexts import gauge_phase, lorentz_boost
from core.physics_constants import (
    X, T, C, HBAR,
    ALPHA_VAL,
    Z_MASS
)

@pytest.fixture
def su2_generators():
    """Create SU(2) generators."""
    sigma_x = Matrix([[0, 1], [1, 0]])
    sigma_y = Matrix([[0, -I], [I, 0]])
    sigma_z = Matrix([[1, 0], [0, -1]])
    return [sigma_x/2, sigma_y/2, sigma_z/2]

@pytest.fixture
def field():
    """Create UnifiedField instance for testing."""
    return UnifiedField(alpha=0.1)

@pytest.fixture
def test_state():
    """Create test field configuration."""
    return exp(-(X**2 + (C*T)**2)/(2*HBAR**2))

@pytest.mark.physics
class TestU1GaugeTransformations:
    """Test U(1) gauge transformations."""
    
    @given(st.floats(min_value=0.1, max_value=10.0))
    def test_phase_rotation(self, energy, field):
        """Test phase rotation of wavefunction."""
        with gauge_phase() as phase:
            psi = exp(-(field.X**2 + field.T**2))
            psi_transformed = field.apply_gauge_transform(psi, phase)
            
            # Check phase rotation
            assert field.check_gauge_invariance(psi, phase)

class TestNonAbelianTransformations:
    """Test non-abelian gauge transformations."""
    
    def test_su2_transformation(self, field, test_state, su2_generators):
        """Test SU(2) gauge transformation."""
        params = [pi/4, pi/3, pi/6]
        psi_transformed = field.apply_nonabelian_gauge_transform(
            test_state, su2_generators, params
        )
        
        # Should preserve norm
        assert abs(abs(psi_transformed) - abs(test_state)) < 1e-10
    
    def test_generator_param_mismatch(self, field, test_state, su2_generators):
        """Test error on generator/parameter mismatch."""
        params = [pi/4, pi/3]  # One parameter missing
        with pytest.raises(GaugeError):
            field.apply_nonabelian_gauge_transform(test_state, su2_generators, params)

class TestGaugeCurrents:
    """Test gauge current conservation."""
    
    def test_current_conservation(self, field, test_state):
        """Test conservation of gauge current."""
        j0, j1 = field.compute_gauge_current(test_state)
        
        # Verify current conservation ∂_μj^μ = 0
        d_t_j0 = diff(j0, T)
        d_x_j1 = diff(j1, X)
        divergence = d_t_j0 + d_x_j1
        
        assert abs(divergence) < 1e-10
    
    @given(st.floats(min_value=0.1, max_value=10.0))
    def test_current_gauge_covariance(self, energy, field, test_state):
        """Test gauge covariance of current."""
        with gauge_phase() as phase:
            # Original currents
            j0, j1 = field.compute_gauge_current(test_state)
            
            # Transformed currents
            psi_transformed = field.apply_gauge_transform(test_state, phase)
            j0_transformed, j1_transformed = field.compute_gauge_current(psi_transformed)
            
            # Currents should transform covariantly
            assert abs(j0_transformed - j0) < 1e-10
            assert abs(j1_transformed - j1) < 1e-10

class TestObservables:
    """Test gauge invariance of physical observables."""
    
    def test_energy_density_invariance(self, field, test_state):
        """Test gauge invariance of energy density."""
        with gauge_phase() as phase:
            E = field.compute_energy_density(test_state)
            assert field.check_gauge_invariance(test_state, field.compute_energy_density)
    
    def test_charge_density_invariance(self, field, test_state):
        """Test gauge invariance of charge density."""
        def charge_density(psi):
            return abs(psi)**2
        
        with gauge_phase() as phase:
            assert field.check_gauge_invariance(test_state, charge_density) 