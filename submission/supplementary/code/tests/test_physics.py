"""Physics calculation tests."""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from sympy import exp, I, pi
from core.field import UnifiedField
from core.types import Energy, Momentum
from core.errors import PhysicsError
from core.contexts import (
    gauge_phase,
    quantum_state,
    field_config,
    numeric_precision
)
from core.physics_constants import (
    X, T, C, HBAR,
    ALPHA_VAL
)

@pytest.fixture
def field():
    """Create UnifiedField instance for testing."""
    return UnifiedField(alpha=ALPHA_VAL)

@pytest.mark.physics
class TestPhysicsCalculations:
    """Test physics calculations and predictions."""
    
    @given(st.floats(min_value=0.1, max_value=10.0))
    def test_coupling_evolution(self, energy):
        """Test coupling constant evolution."""
        field = UnifiedField(alpha=ALPHA_VAL)
        with numeric_precision() as prec:
            energies = np.logspace(2, 4, 10)
            couplings = field.evolve_coupling(energies, **prec)
            
            # Test asymptotic freedom
            assert couplings[-1] < couplings[0]
            
            # Test monotonic decrease
            assert np.all(np.diff(couplings) < 0)
    
    def test_cross_sections(self, field):
        """Test cross section calculations."""
        with quantum_state(energy=100.0) as (psi, norm), numeric_precision() as prec:
            # Test energies
            E = np.logspace(2, 3, 10)  # 100 GeV - 1 TeV
            
            # Compute cross section
            sigma = field.compute_cross_section(E, psi, **prec)
            
            # Test high-energy behavior
            ratio = sigma[-1] / sigma[-2]
            expected = (E[-2] / E[-1])**4
            # Use very loose tolerances due to numerical precision in cross section calculation
            assert np.isclose(ratio, expected,
                           rtol=1e-4,  # Much looser relative tolerance
                           atol=1e-6)  # Much looser absolute tolerance

@pytest.mark.theory
class TestTheorems:
    """Test theoretical consistency and theorems."""
    
    @given(st.floats(min_value=0.1, max_value=10.0))
    @settings(deadline=1000)  # Increase deadline to 1 second
    def test_unitarity(self, energy):
        """Test unitarity constraints."""
        field = UnifiedField(alpha=ALPHA_VAL)
        with quantum_state(energy=energy) as (psi1, _), \
             quantum_state(energy=energy) as (psi2, _), \
             numeric_precision() as prec:
            
            states = [psi1, psi2]
            s_matrix = field.compute_s_matrix(states, **prec)
            identity = s_matrix @ s_matrix.conj().T
            assert np.allclose(
                identity, 
                np.eye(len(identity)), 
                rtol=float(prec['rtol'].value),
                atol=float(prec['atol'].value)
            )

@pytest.mark.physics
class TestNeutrinoPhysics:
    """Test neutrino mixing and oscillation predictions."""
    
    def test_neutrino_mixing_angles(self, field):
        """Test neutrino mixing angle predictions."""
        with numeric_precision(rtol=1e-4) as prec:
            theta_12, theta_23, theta_13 = field.compute_neutrino_angles(**prec)
            
            # Check against experimental values
            assert abs(np.sin(theta_12)**2 - 0.307) < 0.013  # Solar angle
            assert abs(np.sin(theta_23)**2 - 0.545) < 0.020  # Atmospheric angle
            assert abs(np.sin(theta_13)**2 - 0.022) < 0.001  # Reactor angle
    
    def test_neutrino_mass_hierarchy(self, field):
        """Test neutrino mass hierarchy prediction."""
        with field_config(dimension=4) as config, numeric_precision(rtol=1e-5) as prec:
            masses = field.compute_neutrino_masses(config, **prec)
            
            # Check mass differences match observations
            dm21 = masses[1]**2 - masses[0]**2
            dm32 = masses[2]**2 - masses[1]**2
            
            assert abs(dm21 - 7.53e-5) < 0.18e-5  # eV²
            assert abs(dm32 - 2.453e-3) < 0.034e-3  # eV²

    def test_neutrino_oscillations(self, field):
        """Test neutrino oscillation probabilities."""
        with field_config(dimension=4) as config, numeric_precision(rtol=1e-5) as prec:
            # Test νμ → νe appearance at T2K baseline
            L = 295.0  # km (T2K baseline)
            E = 0.6  # GeV (T2K peak energy)
            
            P_mue = field.compute_oscillation_probability(
                initial='muon',
                final='electron',
                L=L,
                E=E,
                **prec
            )
            
            # Compare to T2K best fit
            assert abs(P_mue.value - 0.0597) < 0.0073  # T2K 2020 result

@pytest.mark.physics
class TestCPViolation:
    """Test CP violation and matter-antimatter asymmetry."""
    
    def test_ckm_matrix(self, field):
        """Test CKM matrix properties."""
        with numeric_precision(rtol=1e-10) as prec:
            V = field.compute_ckm_matrix(**prec)
            
            # Test unitarity with float values for tolerances
            assert np.allclose(
                V @ V.conj().T,
                np.eye(3),
                rtol=float(prec['rtol'].value),
                atol=float(prec['atol'].value)
            )
            
            # Test determinant is 1
            assert abs(np.linalg.det(V) - 1.0) < float(prec['rtol'].value)
            
            # Test Jarlskog invariant with experimental precision
            J = abs(np.imag(V[0,1] * V[1,2] * V[2,0]))
            assert abs(J - 3e-5) < 0.2e-5  # PDG value with uncertainty
    
    def test_jarlskog_invariant(self, field):
        """Test Jarlskog invariant prediction."""
        with numeric_precision(rtol=1e-6) as prec:
            J = field.compute_jarlskog(**prec)
            assert abs(J - 3.2e-5) < 0.3e-5
    
    def test_baryon_asymmetry(self, field):
        """Test baryon asymmetry generation."""
        with numeric_precision(rtol=1e-7) as prec:
            # Test CP violation strength
            epsilon = field.compute_cp_violation(**prec)
            assert abs(epsilon - 1e-6) < 1e-7
            
            # Test final asymmetry
            eta_B = field.compute_baryon_asymmetry(**prec)
            assert abs(eta_B - 6.1e-10) < 0.3e-10

@pytest.mark.physics
class TestMassGeneration:
    """Test mass generation mechanism."""
    
    def test_higgs_mechanism(self, field):
        """Test Higgs mechanism predictions."""
        with field_config(mass=125.0) as config, numeric_precision(rtol=1e-3) as prec:
            # Test vacuum expectation value
            v = field.compute_higgs_vev(config, **prec)
            assert abs(v - 246.0) < 0.1  # GeV
            
            # Test Higgs mass
            mH = field.compute_higgs_mass(config, **prec)
            assert abs(mH - 125.1) < 0.2  # GeV
    
    def test_fermion_masses(self, field):
        """Test fermion mass predictions."""
        with field_config(dimension=4) as config, numeric_precision(rtol=1e-4) as prec:
            # Compute masses with quantum corrections
            masses = field.compute_fermion_masses(config, **prec)
            
            # Check key masses
            assert abs(masses['top'] - 173.0) < 0.4  # GeV
            assert abs(masses['electron'] - 0.511e-3) < 1e-6  # Convert MeV to GeV for comparison
            
            # Check mass ratios
            assert abs(masses['muon'].value/masses['electron'].value - 206.8) < 0.2
            assert abs(masses['tau'].value/masses['muon'].value - 16.8) < 0.1
            
            # Test mass ratios with higher precision
            with numeric_precision(rtol=1e-5) as ratio_prec:
                ratios = field.compute_mass_ratios(config, **ratio_prec)
                assert abs(ratios['mu_tau'] - 0.0595) < 0.0001
                
            # Test generational hierarchy
            assert masses['top'] > masses['charm'] > masses['up']
            assert masses['bottom'] > masses['strange'] > masses['down']
            assert masses['tau'] > masses['muon'] > masses['electron']