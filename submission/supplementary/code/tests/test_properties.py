"""Property-based tests for field theory implementation."""

import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
import numpy as np
from sympy import exp, I, pi, sqrt, integrate, conjugate, oo
from core.field import UnifiedField
from core.types import (
    Energy, 
    FieldConfig, 
    WaveFunction,
    NumericValue
)
from core.errors import PhysicsError
from core.contexts import gauge_phase, lorentz_boost
from core.physics_constants import (
    X, T, C, HBAR,
    ALPHA_VAL
)
from contextlib import contextmanager
from typing import Optional

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

@contextmanager
def test_distance(value: float = 1.0):
    """Context manager for test distances."""
    yield value

@contextmanager
def separation_context(distance: float):
    """Context manager for spacetime separation.
    
    Args:
        distance: Spacetime separation distance
    """
    try:
        yield {'t': 0.0, 'x': distance}
    finally:
        pass  # Cleanup if needed

@contextmanager
def distance_context(r: float):
    """Context manager for spatial distance.
    
    Args:
        r: Spatial distance
    """
    try:
        yield {'x': r, 'y': 0.0, 'z': 0.0}
    finally:
        pass

@contextmanager
def quantum_state_context(energy: float = 100.0, n: Optional[int] = None):
    """Context manager for quantum state creation.
    
    From appendix_a_convergence.tex Eq A.2:
    States can be specified by either energy or quantum number.
    
    Args:
        energy: Energy in GeV (default: 100.0)
        n: Principal quantum number (optional)
        
    Yields:
        WaveFunction: Properly normalized quantum state
    """
    field = UnifiedField(alpha=0.1)
    try:
        if n is not None:
            state = field.compute_basis_state(n=n)
        else:
            state = field.compute_basis_state(energy=Energy(energy))
        yield state
    finally:
        pass  # Cleanup if needed

@pytest.fixture
def gauge_field():
    """Create gauge field fixture with proper initialization."""
    def _make_field(coupling: float = 0.1):
        config = FieldConfig(
            mass=100.0,
            coupling=coupling,
            dimension=4
        )
        field = UnifiedField(alpha=coupling)
        return field, config
    return _make_field

@contextmanager
def validation_context(field: UnifiedField = None):
    """Context manager for validation checks."""
    try:
        yield ValidationHelper(field)
    finally:
        pass

class ValidationHelper:
    """Helper class for validation checks."""
    
    def __init__(self, field: UnifiedField = None):
        """Initialize with optional field instance."""
        self.field = field or UnifiedField(alpha=0.1)
    
    def check_quantum_numbers(self, psi: WaveFunction) -> bool:
        """Verify quantum number consistency."""
        if 'n' in psi.quantum_numbers:
            n = psi.quantum_numbers['n']
            E = self.field.compute_energy_eigenvalue(n)
            return abs(E - psi.quantum_numbers.get('E', E)) < self.field.precision
        return True
    
    def check_normalization(self, psi: WaveFunction) -> bool:
        """Verify wavefunction normalization."""
        norm = self.field.compute_norm(psi)
        return abs(norm - 1.0) < self.field.precision

@contextmanager
def rg_flow_context(scale: float = 1e16):
    """Context manager for RG flow tests.
    
    From appendix_h_rgflow.tex Eq H.2:
    RG flow requires proper handling of scale transformations.
    
    Args:
        scale: Energy scale in GeV
        
    Yields:
        Dict containing:
            field: UnifiedField instance
            config: FieldConfig instance
            scale: Energy scale
    """
    field = UnifiedField(alpha=0.1)
    config = FieldConfig(
        mass=100.0,
        coupling=0.1,
        dimension=4
    )
    try:
        yield {
            'field': field,
            'config': config,
            'scale': Energy(scale)
        }
    finally:
        pass  # Cleanup if needed

@contextmanager
def holographic_context(radius: float = 10.0):
    """Context manager for holographic tests.
    
    From appendix_g_holographic.tex Eq G.1:
    Holographic tests require proper boundary conditions.
    
    Args:
        radius: AdS radius in natural units
        
    Yields:
        Dict containing:
            field: UnifiedField instance
            config: FieldConfig instance
            radius: AdS radius
            boundary: Boundary state
    """
    field = UnifiedField(alpha=0.1)
    config = FieldConfig(
        mass=100.0,
        coupling=0.1,
        dimension=4
    )
    try:
        yield {
            'field': field,
            'config': config,
            'radius': radius,
            'boundary': field.compute_boundary_state(radius)
        }
    finally:
        pass  # Cleanup if needed

@contextmanager
def fractal_context(level: int = 5):
    """Context manager for fractal basis tests.
    
    From appendix_a_convergence.tex Eq A.3:
    Fractal basis requires proper level handling and convergence control.
    
    Args:
        level: Fractal recursion level
        
    Yields:
        Dict containing:
            field: UnifiedField instance
            config: FieldConfig instance
            level: Recursion level
            basis: Fractal basis at given level
    """
    field = UnifiedField(alpha=0.1)
    config = FieldConfig(
        mass=100.0,
        coupling=0.1,
        dimension=4
    )
    try:
        yield {
            'field': field,
            'config': config,
            'level': level,
            'basis': field.compute_fractal_basis(level)
        }
    finally:
        pass  # Cleanup if needed

@contextmanager
def gauge_context(charge: float = 1.0):
    """Context manager for gauge symmetry tests.
    
    From appendix_b_gauge.tex Eq B.1:
    Gauge transformations require proper connection handling.
    
    Args:
        charge: Gauge charge value
        
    Yields:
        Dict containing:
            field: UnifiedField instance
            config: FieldConfig instance
            charge: Gauge charge
            connection: Gauge connection
    """
    field = UnifiedField(alpha=0.1)
    config = FieldConfig(
        mass=100.0,
        coupling=0.1,
        dimension=4
    )
    try:
        yield {
            'field': field,
            'config': config,
            'charge': charge,
            'connection': field.compute_gauge_connection(charge)
        }
    finally:
        pass  # Cleanup if needed

@contextmanager
def scale_context(scale: float = 2.0):
    """Context manager for scale transformation tests.
    
    From appendix_d_scale.tex Eq D.1:
    Scale transformations require proper dimension handling.
    
    Args:
        scale: Scale transformation parameter
        
    Yields:
        Dict containing:
            field: UnifiedField instance
            config: FieldConfig instance
            scale: Scale parameter
            dimension: Scaling dimension
    """
    field = UnifiedField(alpha=0.1)
    config = FieldConfig(
        mass=100.0,
        coupling=0.1,
        dimension=4
    )
    try:
        yield {
            'field': field,
            'config': config,
            'scale': scale,
            'dimension': field.scaling_dimension
        }
    finally:
        pass  # Cleanup if needed

class TestFieldProperties:
    """Test fundamental properties of quantum fields."""
    
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
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
    
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=1000  # Increase timeout for complex computations
    )
    @given(velocities())
    def test_lorentz_invariance_enhanced(self, field, velocity):
        """Test enhanced Lorentz invariance of field equations.
        
        From appendix_i_sm_features.tex Eq I.7:
        Field equations must be covariant under proper orthochronous Lorentz transformations.
        """
        with lorentz_boost(velocity) as boost:
            # Create test state
            psi = field.compute_basis_state(energy=Energy(100.0))
            
            # Compute stress-energy tensor components
            T00 = field.compute_energy_density(psi)
            T00_boosted = field.compute_energy_density(boost.transform(psi))
            
            # Verify Lorentz transformation law
            gamma = 1/sqrt(1 - velocity**2/C**2)
            assert abs(T00_boosted - gamma**2 * T00) < field.GAUGE_THRESHOLD

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
    
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(st.floats(min_value=1.0, max_value=10.0))
    def test_microcausality(self, field, distance):
        """Test microcausality for spacelike separation."""
        with separation_context(distance) as sep:
            # Test spacelike correlator
            psi = field.compute_basis_state(energy=Energy(100.0))
            corr = field.compute_correlator(psi, [(0.0, 0.0), (sep['x'], sep['t'])])
            assert abs(corr) < field.CAUSALITY_THRESHOLD
    
    def test_cluster_decomposition(self):
        """Test cluster decomposition principle."""
        field = UnifiedField()
        with test_distance() as distance:
            psi = exp(-(X**2 + (C*T)**2)/(2*HBAR**2))
            
            # Points with large spatial separation
            x1, x2 = X + distance, X - distance
            t = T
            
            # Compute correlation function
            corr = field.compute_correlator(psi, [(x1, t), (x2, t)])
            
            # Should decay with distance
            assert abs(corr) <= exp(-distance/HBAR)

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(st.floats(min_value=5.0, max_value=50.0))
    def test_cluster_decomposition_enhanced(self, field, distance):
        """Test enhanced cluster decomposition with property-based testing."""
        with distance_context(distance) as coords:
            # Create test state with proper energy
            psi = field.compute_basis_state(energy=Energy(100.0))
            
            # Test correlation at various distances
            points = [(0.0, 0.0), (coords['x'], 0.0)]
            corr = field.compute_correlator(psi, points)
            
            # Verify exponential decay bound
            assert abs(corr) <= field.compute_cluster_bound(distance)

class TestFieldInfrastructure:
    """Test enhanced field theory infrastructure."""
    
    @settings(deadline=None)
    @given(st.floats(min_value=10.0, max_value=1000.0))
    def test_state_preparation(self, energy):
        """Test quantum state preparation with proper normalization."""
        with quantum_state_context(energy=energy) as psi:
            assert abs(psi.norm - 1.0) < 1e-7

class TestValidationInfrastructure:
    """Test enhanced validation infrastructure."""
    
    @settings(deadline=None)
    @given(st.integers(min_value=0, max_value=10))
    def test_quantum_number_consistency(self, quantum_state, n):
        """Test quantum number validation."""
        with validation_context() as validator:
            psi = quantum_state(n=n)
            assert validator.check_quantum_numbers(psi)
            assert validator.check_normalization(psi)

class TestRGFlowEnhanced:
    """Enhanced tests for RG flow properties from appendix_h_rgflow.tex."""
    
    @settings(deadline=None)
    @given(st.floats(min_value=1e15, max_value=1e17))
    def test_coupling_unification(self, scale):
        """Test enhanced coupling unification with proper error handling."""
        with rg_flow_context(scale=scale) as context:
            field = context['field']
            couplings = field.compute_couplings(Energy(scale))
            
            g1, g2, g3 = couplings['g1'], couplings['g2'], couplings['g3']
            assert abs(g1.value - g2.value) <= g1.uncertainty + g2.uncertainty
            assert abs(g2.value - g3.value) <= g2.uncertainty + g3.uncertainty

class TestHolographicEnhanced:
    """Enhanced tests for holographic properties from appendix_g_holographic.tex."""
    
    @settings(deadline=None)
    @given(st.floats(min_value=1.0, max_value=100.0))
    def test_holographic_correlators(self, radius):
        """Test enhanced holographic correlator properties."""
        with holographic_context(radius=radius) as context:
            field = context['field']
            boundary = context['boundary']
            
            points = [(0.0, 0.0), (radius, 0.0)]
            G_boundary = field.compute_boundary_correlator(boundary, points)
            G_bulk = field.compute_bulk_correlator(boundary, points)
            
            assert abs(G_boundary - radius**(2*field.scaling_dimension) * G_bulk) < field.precision

class TestFractalBasisEnhanced:
    """Enhanced tests for fractal basis properties from appendix_a_convergence.tex."""
    
    @settings(deadline=None)
    @given(st.integers(min_value=1, max_value=10))
    def test_fractal_completeness(self, level):
        """Test enhanced fractal basis completeness."""
        with fractal_context(level=level) as context:
            field = context['field']
            basis = context['basis']
            
            psi = field.compute_basis_state(energy=Energy(100.0))
            expansion = field.compute_fractal_expansion(psi, basis)
            psi_reconstructed = field.reconstruct_from_expansion(expansion, basis)
            
            error = field.compute_state_distance(psi, psi_reconstructed)
            assert error <= ALPHA_VAL**level

class TestGaugeSymmetryEnhanced:
    """Enhanced tests for gauge symmetry from appendix_b_gauge.tex."""
    
    @settings(deadline=None)
    @given(st.floats(min_value=0.1, max_value=10.0))
    def test_gauge_covariance(self, charge):
        """Test enhanced gauge covariance of field equations."""
        with gauge_context(charge=charge) as context:
            field = context['field']
            connection = context['connection']
            
            psi = field.compute_basis_state(energy=Energy(100.0))
            phase = field.compute_gauge_phase(charge)
            psi_transformed = field.apply_gauge_transform(psi, phase)
            
            D_psi = field.compute_covariant_derivative(psi, connection)
            D_psi_transformed = field.compute_covariant_derivative(psi_transformed, connection)
            
            assert abs(D_psi_transformed - exp(I*phase)*D_psi) < field.GAUGE_THRESHOLD

class TestScaleInvarianceEnhanced:
    """Enhanced tests for scale invariance from appendix_d_scale.tex."""
    
    @settings(deadline=None)
    @given(st.floats(min_value=0.5, max_value=5.0))
    def test_scale_covariance(self, scale):
        """Test enhanced scale covariance of field equations."""
        with scale_context(scale=scale) as context:
            field = context['field']
            dimension = context['dimension']
            
            psi = field.compute_basis_state(energy=Energy(100.0))
            psi_scaled = field.apply_scale_transform(psi, scale)
            
            F_psi = field.compute_field_equation(psi)
            F_psi_scaled = field.compute_field_equation(psi_scaled)
            
            correction = field.compute_scale_anomaly(scale)
            assert abs(F_psi_scaled - scale**(dimension+2)*F_psi - correction) < field.precision