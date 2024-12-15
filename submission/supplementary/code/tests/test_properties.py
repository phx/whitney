"""Property-based tests for field theory implementation."""

import pytest
from hypothesis import given, strategies as st
import numpy as np
from sympy import exp, I, pi, sqrt, integrate, conjugate, oo
from core.field import UnifiedField
from core.types import Energy, FieldConfig
from core.errors import PhysicsError
from core.contexts import gauge_phase, lorentz_boost
from core.physics_constants import (
    X, T, C, HBAR,
    ALPHA_VAL
)

@pytest.fixture
def field():
    """Create UnifiedField instance for testing."""
    return UnifiedField(alpha=0.1)

class TestFieldProperties:
    """Test fundamental properties of quantum fields."""
    
    @given(st.floats(min_value=0.1, max_value=10.0))
    def test_energy_positivity(self, mass):
        """Test that energy is always positive."""
        field = UnifiedField(alpha=0.1)
        config = FieldConfig(mass=mass, coupling=0.1, dimension=1)
        psi = field.compute_field(config)
        E = field.compute_energy_density(psi)
        assert E >= 0
    
    @given(st.floats(min_value=-10.0, max_value=10.0))
    @pytest.mark.timeout(5)  # 5 second timeout
    def test_norm_conservation(self, time):
        """Test that norm is conserved under time evolution."""
        field = UnifiedField(alpha=ALPHA_VAL)
        psi = exp(-(X**2 + (C*T)**2)/(2*HBAR**2))
        field.state = psi
        
        # Evolve to given time
        evolved = field.evolve(Energy(time))
        
        # Check norm conservation with infinite limits
        norm1 = integrate(conjugate(psi) * psi, (X, -oo, oo))
        norm2 = integrate(conjugate(evolved) * evolved, (X, -oo, oo))
        assert abs(norm2 - norm1) < 1e-10
        
        # Also verify with finite limits as numerical check
        LIMIT = 10.0
        finite_norm1 = integrate(conjugate(psi) * psi, (X, -LIMIT, LIMIT))
        finite_norm2 = integrate(conjugate(evolved) * evolved, (X, -LIMIT, LIMIT))
        assert abs(finite_norm2 - finite_norm1) < 1e-10

class TestSymmetryProperties:
    """Test symmetry properties of the field theory."""
    
    @given(st.floats(min_value=0.1, max_value=10.0))
    def test_phase_invariance(self, energy):
        """Test U(1) phase invariance."""
        field = UnifiedField(alpha=0.1)
        with gauge_phase() as phase:
            psi = field.compute_basis_state(energy=1.0)
            psi_transformed = field.apply_gauge_transform(psi, phase)
            
            # Observable quantities should be invariant
            E1 = field.compute_energy_density(psi)
            E2 = field.compute_energy_density(psi_transformed)
            assert abs(E1 - E2) < 1e-10
    
    @given(st.floats(min_value=-0.99, max_value=0.99))
    def test_lorentz_invariance(self, velocity, field):
        """Test Lorentz invariance of the theory."""
        psi = exp(-(X**2 + (C*T)**2)/(2*HBAR**2))
        
        # Apply Lorentz boost
        gamma = 1/sqrt(1 - velocity**2/C**2)
        x_prime = gamma * (X - velocity*T)
        t_prime = gamma * (T - velocity*X/C**2)
        
        psi_boosted = psi.subs([(X, x_prime), (T, t_prime)])
        
        # Energy-momentum tensor should transform covariantly
        T00 = field.compute_energy_density(psi)
        T00_boosted = field.compute_energy_density(psi_boosted)
        
        # Account for Lorentz transformation of energy density
        assert abs(T00_boosted - gamma**2 * T00) < 1e-10

class TestLocalityProperties:
    """Test locality and causality properties."""
    
    @given(st.floats(min_value=1.0, max_value=10.0))
    def test_microcausality(self, separation, field):
        """Test that field commutators vanish outside light cone."""
        psi = exp(-(X**2 + (C*T)**2)/(2*HBAR**2))
        field.state = psi
        
        # Points with spacelike separation
        x1, x2 = X + separation, X - separation
        t1, t2 = T, T
        
        # Compute commutator
        commutator = field.compute_commutator([(x1, t1), (x2, t2)])
        
        # Should vanish for spacelike separation
        assert abs(commutator) < 1e-10
    
    @given(st.floats(min_value=0.1, max_value=5.0))
    def test_cluster_decomposition(self, distance, field):
        """Test cluster decomposition principle."""
        psi = exp(-(X**2 + (C*T)**2)/(2*HBAR**2))
        
        # Points with large spatial separation
        x1, x2 = X + distance, X - distance
        t = T
        
        # Compute correlation function
        corr = field.compute_correlator(psi, [(x1, t), (x2, t)])
        
        # Should decay with distance
        assert abs(corr) <= exp(-distance/HBAR)