"""Property-based tests for field theory implementation."""

import pytest
from hypothesis import given, strategies as st
import numpy as np
from sympy import exp, I, pi, sqrt
from core.field import UnifiedField
from core.basis import FractalBasis
from core.types import Energy, FieldConfig
from core.physics_constants import X, T, P
from core.errors import PhysicsError

# Strategies for property-based testing
@st.composite
def field_configs(draw):
    """Generate valid field configurations."""
    return FieldConfig(
        mass=draw(st.floats(min_value=1.0, max_value=1000.0)),
        coupling=draw(st.floats(min_value=0.0, max_value=1.0)),
        dimension=draw(st.integers(min_value=0, max_value=10))
    )

@st.composite
def wave_functions(draw):
    """Generate valid wave functions."""
    k = draw(st.floats(min_value=0.1, max_value=10.0))
    return exp(-k * X**2/2)

# Property Tests
class TestFieldProperties:
    """Test fundamental field theory properties."""
    
    @given(field_configs())
    def test_energy_positivity(self, config):
        """Test that energy density is positive."""
        field = UnifiedField(alpha=0.1)
        psi = field.compute_field(config)
        density = field.compute_energy_density(psi)
        assert density.is_real
        assert density >= 0
    
    @given(wave_functions())
    def test_causality(self, psi):
        """Test causality constraint."""
        field = UnifiedField()
        assert field.check_causality(psi)
    
    @given(st.floats(min_value=0.0, max_value=2*pi))
    def test_gauge_invariance(self, phase):
        """Test gauge invariance of observables."""
        field = UnifiedField()
        psi = exp(-X**2/2)
        
        # Apply gauge transformation
        psi_transformed = psi * exp(I * phase)
        
        # Energy density should be invariant
        E1 = field.compute_energy_density(psi)
        E2 = field.compute_energy_density(psi_transformed)
        assert abs(E1 - E2) < 1e-10

class TestBasisProperties:
    """Test fractal basis properties."""
    
    @given(st.integers(min_value=0, max_value=5),
           st.integers(min_value=0, max_value=5))
    def test_orthogonality(self, n, m):
        """Test basis function orthogonality."""
        basis = FractalBasis()
        psi_n = basis.compute(n, Energy(1.0))
        psi_m = basis.compute(m, Energy(1.0))
        
        # <ψₙ|ψₘ> = δₙₘ
        overlap = basis.compute_inner_product(psi_n, psi_m)
        expected = 1.0 if n == m else 0.0
        assert abs(overlap - expected) < 1e-10
    
    @given(st.integers(min_value=0, max_value=5),
           st.floats(min_value=1.0, max_value=100.0))
    def test_scaling_relation(self, n, E):
        """Test basis function scaling properties."""
        basis = FractalBasis()
        psi = basis.compute(n, Energy(E))
        
        # Test scaling dimension
        scaled = basis.compute(n, Energy(2*E))
        ratio = scaled / psi
        assert abs(abs(ratio) - 2**basis.scaling_dimension) < 1e-10 