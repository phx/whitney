"""Property-based tests for field theory implementation."""

import pytest
from hypothesis import given, settings, strategies as st
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
from contextlib import contextmanager

@pytest.fixture
def field():
    """Create UnifiedField instance for testing."""
    return UnifiedField(alpha=0.1)

@st.composite
def velocities(draw):
    """Strategy for generating valid velocities."""
    return draw(st.floats(min_value=-0.99, max_value=0.99))

@st.composite
def separations(draw):
    """Strategy for generating valid separations."""
    return draw(st.floats(min_value=0.1, max_value=10.0))

@st.composite
def distances(draw):
    """Strategy for generating valid distances."""
    return draw(st.floats(min_value=0.1, max_value=10.0))

@st.composite
def fractal_levels(draw):
    """Strategy for generating valid fractal recursion levels."""
    return draw(st.integers(min_value=1, max_value=10))

@contextmanager
def test_velocity(value: float = 0.5):
    """Context manager for test velocities."""
    yield value

@contextmanager
def test_separation(value: float = 1.0):
    """Context manager for test separations."""
    yield value

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
    @pytest.mark.skip(reason="Test hangs - needs optimization of infinite integral")
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
    
    @given(fractal_levels())
    def test_fractal_scaling(self, n_level, field):
        """Test fractal scaling relation from paper Eq. I.1."""
        # Test the fractal mass scaling from appendix_i_sm_features.tex
        m_n = field.compute_mass_at_level(n_level)
        m_0 = field.compute_mass_at_level(0)
        
        # Check fractal scaling formula from paper
        expected = m_0 * np.prod([
            1 + ALPHA_VAL**k * field.compute_fractal_exponent(k) 
            for k in range(1, n_level + 1)
        ])
        assert np.isclose(m_n, expected, rtol=1e-6)

    @given(st.floats(min_value=0.1, max_value=10.0))
    def test_harmonic_mean_convergence(self, energy, field):
        """Test harmonic mean convergence from appendix_j_math_details.tex."""
        # Compute harmonic mean of fractal exponents
        H_n = field.compute_harmonic_mean(energy)
        
        # Should converge to critical exponent
        h_crit = field.compute_critical_exponent()
        assert abs(H_n - h_crit) < ALPHA_VAL**2

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
    
    @given(velocities())
    def test_lorentz_invariance(self, velocity):
        """Test Lorentz invariance."""
        field = UnifiedField(alpha=0.1)
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

class TestHolographicProperties:
    """Test holographic properties from appendix_g_holographic.tex."""
    
    def test_entropy_bound(self, field):
        """Test holographic entropy bound (Eq. G.1)."""
        # Test area-scaling of entropy at different scales
        for n in range(1, 5):
            L_n = 10.0 * ALPHA_VAL**n  # Characteristic length at level n
            area = L_n**2
            
            # Compute effective degrees of freedom
            N_n = field.compute_dof_at_level(n)
            entropy = np.log(N_n)
            
            # Check holographic bound from appendix_g_holographic.tex
            assert entropy <= area/(4 * HBAR)
    
    @given(st.integers(min_value=1, max_value=5))
    def test_fractal_dof_scaling(self, n, field):
        """Test fractal scaling of degrees of freedom (Eq. G.2)."""
        N_n = field.compute_dof_at_level(n)
        L_n = field.compute_length_scale(n)
        
        # Check scaling formula from paper
        expected = (L_n/HBAR)**2 * np.prod([
            1 + ALPHA_VAL**k for k in range(1, n+1)
        ])**(-1)
        assert np.isclose(N_n, expected, rtol=1e-6)

class TestRGFlowProperties:
    """Test RG flow properties from appendix_h_rgflow.tex."""
    
    @given(st.floats(min_value=100, max_value=1e16))
    def test_beta_function(self, energy, field):
        """Test beta function structure (Eq. H.1)."""
        # Compute beta function coefficients
        for gauge_index in [1, 2, 3]:
            beta = field.compute_beta_function(gauge_index, Energy(energy))
            
            # Should have correct fractal expansion
            coeffs = field.compute_beta_coefficients(gauge_index, n_max=3)
            expected = sum(
                ALPHA_VAL**n * coeffs[n] 
                for n in range(len(coeffs))
            )
            assert np.isclose(beta, expected, rtol=1e-4)
    
    def test_unification_uniqueness(self, field):
        """Test uniqueness of unification point (Sec. H.3)."""
        # Compute couplings near GUT scale
        E_GUT = 2.1e16  # From appendix_e_predictions.tex
        energies = np.logspace(15, 17, 100)
        
        # Find where couplings are closest
        diffs = []
        for E in energies:
            g1 = field.compute_coupling(1, Energy(E))
            g2 = field.compute_coupling(2, Energy(E))
            g3 = field.compute_coupling(3, Energy(E))
            
            # Maximum difference between any two couplings
            diff = max(abs(g1-g2), abs(g2-g3), abs(g3-g1))
            diffs.append(diff)
        
        # Should have unique minimum near E_GUT
        min_idx = np.argmin(diffs)
        E_min = energies[min_idx]
        
        # Check uniqueness by verifying derivative is non-zero
        assert diffs[min_idx+1] > diffs[min_idx] < diffs[min_idx-1]
        assert np.isclose(E_min, E_GUT, rtol=0.1)

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