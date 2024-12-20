"""Tests for UnifiedField base class."""

import pytest
from sympy import exp, I, Symbol, pi, diff
import numpy as np
from core.field import UnifiedField
from core.types import Energy, FieldConfig, WaveFunction, NumericValue
from core.physics_constants import X, T, P, HBAR, C, M_PLANCK, G
from core.modes import ComputationMode
from core.errors import PhysicsError, ValidationError
from typing import Dict, Optional, Union, List, Tuple
from math import factorial

@pytest.fixture
def field():
    """Create UnifiedField instance for testing."""
    return UnifiedField(alpha=0.1)

def test_field_initialization():
    """Test UnifiedField initialization."""
    field = UnifiedField(alpha=0.1)
    assert field.alpha == 0.1
    assert field.state is None
    
    with pytest.raises(ValidationError):
        UnifiedField(alpha=-1.0)

def test_energy_density():
    """Test energy density computation."""
    field = UnifiedField()
    psi = exp(-X**2/2)
    density = field.compute_energy_density(psi)
    assert density.is_real

def test_causality():
    """Test causality check."""
    field = UnifiedField()
    
    # Causal configuration
    psi_causal = exp(-X**2/2 - T**2/2)
    assert field.check_causality(psi_causal)
    
    # Non-causal configuration
    psi_noncausal = exp(X*T)  # Mixing space and time incorrectly
    assert not field.check_causality(psi_noncausal)

def test_field_evolution():
    """Test field evolution."""
    field = UnifiedField()
    
    # Set initial state
    field.state = exp(-X**2/2)
    
    # Evolve to new energy
    evolved = field.evolve(Energy(10.0))
    assert evolved is not None
    
    with pytest.raises(PhysicsError):
        field.evolve(Energy(-1.0))

class TestUnifiedFieldCore:
    """Test core UnifiedField functionality."""
    
    @pytest.fixture
    def field(self):
        return UnifiedField()
        
    def test_field_initialization(self, field):
        """Test basic field initialization."""
        assert field.alpha > 0
        assert field.dimension == 4
        assert field.N_STABLE_MAX > 0
        assert field.precision > 0
        
    def test_invalid_initialization(self):
        """Test invalid initialization parameters."""
        with pytest.raises(ValidationError):
            UnifiedField(alpha=-1)
        with pytest.raises(ValidationError):
            UnifiedField(dimension=0)
            
    def test_sympy_conversion(self, field):
        """Test WaveFunction <-> Sympy conversion."""
        # Create test wavefunction
        psi = exp(-field.X**2/(2*HBAR)) * exp(-I*field.T)
        wf = WaveFunction(psi=psi, grid=np.linspace(-10, 10, 100))
        
        # Test conversion to sympy
        expr = field._to_sympy_expr(wf)
        assert isinstance(expr, Symbol)
        
        # Test conversion back
        wf2 = field._from_sympy_expr(expr)
        assert isinstance(wf2, WaveFunction)
        assert wf2.psi == wf.psi
        
    def test_dark_matter_density(self, field):
        """Test dark matter density computation."""
        # Test valid radius
        rho = field.compute_dark_matter_density(20.0)
        assert isinstance(rho, NumericValue)
        assert rho.value > 0
        assert rho.uncertainty > 0
        
        # Test invalid radius
        with pytest.raises(ValueError):
            field.compute_dark_matter_density(-1.0)
            
    def test_lorentz_invariance(self, field):
        """Test Lorentz transformation properties."""
        # Create test state
        psi = field.compute_field(Energy(100.0))
        
        # Apply boost
        beta = 0.5
        psi_boosted = field.apply_lorentz_transform(psi, beta)
        
        # Verify properties
        assert isinstance(psi_boosted, WaveFunction)
        with pytest.raises(PhysicsError):
            field.apply_lorentz_transform(psi, 2.0)  # v > c
            
    def test_causality(self, field):
        """Test causality constraints."""
        # Create causal configuration
        E = Energy(100.0)
        psi = field.compute_field(E)
        
        # Verify causality
        assert field.check_causality(psi)
        
    def test_energy_conservation(self, field):
        """Test energy conservation."""
        E = Energy(100.0)
        psi = field.compute_field(E)
        
        # Compute energy density
        rho = field.compute_energy_density(psi)
        assert isinstance(rho, NumericValue)
        assert rho.value > 0
        
    @pytest.mark.parametrize("mass", [1.0, 10.0, 100.0])
    def test_field_equations(self, field, mass):
        """Test field equation solutions."""
        config = FieldConfig(mass=mass, coupling=field.alpha)
        psi = field._solve_field_equations(config)
        assert isinstance(psi, WaveFunction)

class TestQuantumFieldTheory:
    """Test quantum field theory computations."""
    
    @pytest.fixture
    def field(self):
        return UnifiedField()

    def test_neutrino_angles(self, field):
        """Test neutrino mixing angle computation."""
        angles = field.compute_neutrino_angles(
            include_uncertainty=True,
            rtol=1e-8,
            atol=1e-10
        )
        assert isinstance(angles, dict)
        assert 'theta_12' in angles
        assert 'theta_23' in angles
        assert 'theta_13' in angles
        for angle in angles.values():
            assert isinstance(angle, NumericValue)
            assert 0 <= angle.value <= pi/2

    def test_fermion_masses(self, field):
        """Test fermion mass spectrum computation."""
        masses = field.compute_fermion_masses(rtol=1e-8, atol=1e-10)
        assert isinstance(masses, dict)
        # Check key fermions
        assert 'electron' in masses
        assert 'muon' in masses
        assert 'tau' in masses
        assert 'top' in masses
        # Verify mass hierarchy
        assert masses['electron'].value < masses['muon'].value
        assert masses['muon'].value < masses['tau'].value

    def test_coupling_evolution(self, field):
        """Test gauge coupling evolution."""
        energies = np.logspace(2, 16, 100)  # From 100 GeV to 10^16 GeV
        couplings = field.evolve_coupling(energies)
        assert len(couplings) == len(energies)
        # Verify asymptotic freedom
        assert couplings[-1] < couplings[0]

    def test_gut_scale(self, field):
        """Test GUT scale computation."""
        E_gut = field.compute_gut_scale(rtol=1e-8)
        assert isinstance(E_gut, Energy)
        assert 1e15 < E_gut.value < 1e17  # Expected GUT scale range
        assert E_gut.uncertainty > 0

    def test_vertex_corrections(self, field):
        """Test vertex correction computations."""
        for n in range(5):  # Test first few orders
            factor = field._compute_vertex_factor(n)
            assert isinstance(factor, complex)
            # Verify suppression with order
            assert abs(factor) < field.alpha**n

    def test_lsz_reduction(self, field):
        """Test LSZ reduction factors."""
        for n in range(5):
            factor = field._compute_lsz_factor(n)
            assert isinstance(factor, complex)
            # Verify expected phase behavior
            assert abs(factor) <= 1.0

    def test_correlator(self, field):
        """Test n-point correlation functions."""
        # Create test state
        psi = field.compute_field(Energy(100.0))
        
        # Test points
        points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        
        # Compute correlator
        corr = field.compute_correlator(
            psi,
            points,
            include_uncertainty=True
        )
        assert isinstance(corr, NumericValue)
        assert corr.uncertainty > 0

    def test_scattering_amplitudes(self, field):
        """Test scattering amplitude computation."""
        energies = np.array([100.0, 200.0, 300.0])
        momenta = np.array([90.0, 180.0, 270.0])
        
        amplitudes = field.compute_amplitudes(energies, momenta)
        assert len(amplitudes) == len(energies)
        for amp in amplitudes:
            assert isinstance(amp, complex)

    @pytest.mark.parametrize("gauge_index", [1, 2, 3])
    def test_gauge_couplings(self, field, gauge_index):
        """Test gauge coupling computation."""
        energy = Energy(1000.0)  # 1 TeV
        g = field.compute_coupling(gauge_index, energy)
        assert isinstance(g, float)
        assert 0 < g < 1  # Weak coupling regime

class TestTheoreticalPredictions:
    """Test mathematical theorems and predictions from appendices."""
    
    @pytest.fixture
    def field(self):
        return UnifiedField()

    def test_fractal_dimension(self, field):
        """
        Test fractal dimension prediction.
        From appendix_d_scale.tex Eq D.8:
        The effective dimension should scale as D_eff = 4 - α²
        """
        D_eff = field.compute_effective_dimension()
        assert isinstance(D_eff, NumericValue)
        expected = 4 - field.alpha**2
        assert abs(D_eff.value - expected) < 1e-6

    def test_mass_hierarchy(self, field):
        """
        Test fermion mass hierarchy prediction.
        From appendix_i_sm_features.tex Eq I.8:
        m_n/m_{n+1} ≈ exp(-π/α)
        """
        config = FieldConfig(
            mass=125.0,  # Higgs mass in GeV
            coupling=0.1,  # Standard coupling
            dimension=4,  # 4D spacetime
            max_level=10,  # Sufficient for convergence
            precision=1e-8  # High precision
        )
        masses = field.compute_fermion_masses(config, rtol=1e-8)
        ratio = masses['muon'].value / masses['tau'].value
        expected = np.exp(-np.pi/field.alpha)
        assert abs(ratio - expected) < 0.1  # 10% accuracy

    def test_coupling_unification(self, field):
        """
        Test coupling unification theorem.
        From appendix_h_rgflow.tex Eq H.7:
        All gauge couplings should meet at GUT scale
        """
        E_gut = field.compute_gut_scale()
        g1 = field.compute_coupling(1, E_gut)
        g2 = field.compute_coupling(2, E_gut)
        g3 = field.compute_coupling(3, E_gut)
        # All couplings should be equal at GUT scale
        assert abs(g1 - g2) < 1e-3
        assert abs(g2 - g3) < 1e-3

    def test_holographic_entropy(self, field):
        """
        Test holographic entropy scaling.
        From appendix_g_holographic.tex Eq G.4:
        S ∝ A/4G_N where A is area
        """
        radius = 1.0  # Planck units
        entropy = field.compute_entropy(radius)
        area = 4 * np.pi * radius**2
        expected = area/(4 * G)  # G is Newton's constant
        assert abs(entropy.value/expected - 1) < 0.1

    def test_quantum_corrections(self, field):
        """
        Test quantum corrections to classical predictions.
        From appendix_e_predictions.tex Eq E.9
        """
        # Test first few orders of quantum corrections
        for n in range(5):
            correction = field.compute_quantum_correction(n)
            # Should decrease factorially with order
            assert abs(correction) < 1.0/factorial(n)

    def test_beta_function(self, field):
        """
        Test beta function coefficients.
        From appendix_h_rgflow.tex Eq H.3
        """
        beta = field.compute_beta_function()
        # First coefficient should be negative (asymptotic freedom)
        assert beta[0] < 0
        # Should have expected number of coefficients
        assert len(beta) == field.N_STABLE_MAX

    def test_ward_identity(self, field):
        """
        Test Ward identity conservation.
        From appendix_b_gauge.tex Eq B.12
        """
        # Create test state with proper quantum numbers
        test_psi = WaveFunction(
            psi=exp(-X**2/(2*HBAR)) * exp(-I*T/HBAR),  # Gaussian packet
            grid=(-10, 10, 100),  # Spatial grid
            quantum_numbers={'n': 0, 'l': 0, 'm': 0}  # Ground state
        )
        
        # Call compute_noether_current with test state
        current = field.compute_noether_current(test_psi)
        
        # Verify current conservation with enhanced precision
        div_j = diff(current[0], T) + C * diff(current[1], X)
        assert abs(float(div_j)) < field.precision * field.alpha

    def test_unitarity_bounds(self, field):
        """
        Test unitarity bounds on scattering amplitudes.
        From appendix_j_math_details.tex Eq J.15
        """
        energies = np.logspace(2, 4, 10)
        for E in energies:
            # Create initial and final state wavefunctions
            psi1 = field._solve_field_equations(
                FieldConfig(mass=E, dimension=1, coupling=0.1)
            )
            psi2 = field._solve_field_equations(
                FieldConfig(mass=E, dimension=1, coupling=0.1)
            )
            
            # Compute scattering amplitude
            amplitude = field.compute_scattering_amplitude(psi1, psi2)
            
            # From appendix_j_math_details.tex Eq J.15:
            # The unitarity bound includes energy dependence
            bound = np.sqrt(16*np.pi/E)  # S-matrix unitarity bound
            
            # Account for form factor suppression
            bound *= np.exp(-E/(4*M_PLANCK))
            
            assert abs(amplitude) <= bound, (
                f"Unitarity violated at E={E} GeV: "
                f"|A|={abs(amplitude):.3e} > {bound:.3e}"
            )

    def test_fractal_recursion(self, field):
        """
        Test fractal recursion relations.
        From appendix_l_simplification.tex Eq L.4
        """
        # Test recursion relation for first few levels
        for n in range(3):
            F_n = field.compute_fractal_coefficient(n)
            F_n1 = field.compute_fractal_coefficient(n+1)
            ratio = abs(F_n1/F_n)
            # Should follow geometric progression
            assert abs(ratio - field.alpha) < 1e-6

    def test_dark_matter_profile(self, field):
        """
        Test dark matter density profile prediction.
        From appendix_e_predictions.tex Eq E.5
        """
        # Test density scaling at different radii
        r_vals = np.logspace(-1, 2, 10)
        for r in r_vals:
            rho = field.compute_dark_matter_density(r)
            # Should follow modified NFW profile
            r_s = 20.0  # Scale radius
            x = r/r_s
            expected = 1/(x * (1 + x)**2)
            ratio = rho.value/expected
            # Allow for fractal corrections
            assert 0.1 < ratio < 10.0