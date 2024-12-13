"""Tests for gauge transformations and invariance."""

import pytest
from hypothesis import given, strategies as st
import numpy as np
from sympy import (
    exp, I, pi, sqrt, Matrix,
    diff, conjugate, integrate, oo
)
from core.field import UnifiedField
from core.types import Energy, FieldConfig, WaveFunction
from core.physics_constants import X, T, P, HBAR, C
from core.errors import GaugeError

@pytest.fixture
def field():
    """Create UnifiedField instance for testing."""
    return UnifiedField(alpha=0.1)

@pytest.fixture
def test_state():
    """Create test field configuration."""
    return exp(-(X**2 + (C*T)**2)/(2*HBAR**2))

class TestU1GaugeTransformations:
    """Test U(1) gauge transformation properties."""
    
    @given(st.floats(min_value=0, max_value=2*pi))
    def test_phase_rotation(self, phase, field, test_state):
        """Test U(1) phase rotation."""
        psi_transformed = field.apply_gauge_transform(test_state, phase)
        
        # Phase should be applied correctly
        expected = test_state * exp(I * phase)
        assert abs(psi_transformed - expected) < 1e-10
    
    def test_invalid_phase(self, field, test_state):
        """Test invalid phase values are rejected."""
        with pytest.raises(GaugeError):
            field.apply_gauge_transform(test_state, -1.0)
        with pytest.raises(GaugeError):
            field.apply_gauge_transform(test_state, 3*pi)

class TestNonAbelianTransformations:
    """Test non-abelian gauge transformations."""
    
    @pytest.fixture
    def su2_generators(self):
        """Create SU(2) generators."""
        sigma_x = Matrix([[0, 1], [1, 0]])
        sigma_y = Matrix([[0, -I], [I, 0]])
        sigma_z = Matrix([[1, 0], [0, -1]])
        return [sigma_x/2, sigma_y/2, sigma_z/2]
    
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
    
    @given(st.floats(min_value=0, max_value=2*pi))
    def test_current_gauge_covariance(self, phase, field, test_state):
        """Test gauge covariance of current."""
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
        E = field.compute_energy_density(test_state)
        assert field.check_gauge_invariance(test_state, field.compute_energy_density)
    
    def test_charge_density_invariance(self, field, test_state):
        """Test gauge invariance of charge density."""
        def charge_density(psi):
            return abs(psi)**2
        
        assert field.check_gauge_invariance(test_state, charge_density) 