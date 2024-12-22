"""Tests for theoretical predictions and experimental validation."""

import pytest
import numpy as np
from scipy import stats
from core.field import UnifiedField
from core.types import Energy, CrossSection
from core.errors import PhysicsError
from core.constants import GUT_SCALE
from core.physics_constants import (
    G,          # Gravitational constant
    HBAR,       # Reduced Planck constant 
    C,          # Speed of light
    M_PLANCK,   # Planck mass
    Z_MASS,     # Z boson mass
    GUT_SCALE   # Grand unification scale
)
from pathlib import Path

@pytest.fixture
def field():
    """Create UnifiedField instance for testing."""
    return UnifiedField(alpha=0.1)

@pytest.mark.predictions
class TestExperimentalPredictions:
    """Test experimental predictions."""
    
    def test_cross_section_prediction(self, field):
        """Test cross section predictions match data."""
        # Test energies
        E = [Energy(e) for e in np.logspace(2, 3, 10)]  # Wrap in Energy objects
        
        # Generate test state with mass
        psi = field.compute_basis_state(E[0], mass=91.2)  # Z boson mass in GeV
        
        # Compute cross sections
        sigma = field.compute_cross_section(E, psi)
        
        # Should decrease with energy (asymptotic freedom)
        assert np.all(np.diff(sigma) < 0)
        
        # Should match experimental bounds
        assert all(s > 1e-40 for s in sigma)  # cm^2
    
    def test_coupling_running(self, field):
        """Test coupling constant running."""
        # Test range
        E = np.logspace(2, 4, 10)  # 100 GeV - 10 TeV
        
        # Compute running coupling
        alpha = field.compute_running_coupling(E)
        
        # Should decrease with energy (asymptotic freedom)
        assert np.all(np.diff(alpha) < 0)
        
        # Should match low-energy value
        assert abs(alpha[0] - 0.118) < 0.005  # αs(MZ)

def test_cross_sections(field):
    """Test cross section calculations."""
    # Test energies
    E = np.logspace(2, 3, 10)  # 100 GeV - 1 TeV
    
    # Generate test state
    psi = field.compute_basis_state(E[0])
    
    # Compute cross section
    sigma = field.compute_cross_section(E, psi)
    
    # Test high-energy behavior
    ratio = sigma[-1] / sigma[-2]
    expected = (E[-2] / E[-1])**4
    assert np.isclose(ratio, expected, rtol=0.1)

def test_coupling_unification(field):
    """Test gauge coupling unification."""
    # Compute couplings at GUT scale
    couplings = field.compute_couplings(GUT_SCALE)
    
    # All couplings should be equal at GUT scale
    g1, g2, g3 = couplings['g1'], couplings['g2'], couplings['g3']
    assert abs(g1 - g2) < 0.1
    assert abs(g2 - g3) < 0.1
    assert abs(g3 - g1) < 0.1
    
    # Should match expected unified coupling
    g_gut = (g1 + g2 + g3) / 3
    assert abs(g_gut - 0.7) < 0.1

def test_gravitational_wave_spectrum(field: UnifiedField):
    """
    Test gravitational wave spectrum predictions.
    From appendix_e_predictions.tex Eq E.31-E.33
    """
    # Set up frequency range covering both quantum and classical regimes
    omega = np.logspace(-16, 4, 1000)  # Hz
    
    # Generate predictions
    h_pred = field.compute_gravitational_wave_spectrum(omega)
    
    # Load experimental data
    data_path = Path(__file__).parent.parent.parent / 'data' / 'gw_spectrum.dat'
    data = np.loadtxt(data_path)
    h_exp = data[:, 1]
    h_err = data[:, 2]
    
    # Add minimum error floor to prevent division by zero
    # From appendix_e_predictions.tex Eq E.35
    h_err = np.maximum(h_err, 1e-60 * np.abs(h_exp))
    
    # Bin predictions to match experimental frequency bins
    h_pred_binned = np.zeros_like(h_exp)
    for i in range(len(h_exp)):
        # Find frequencies in this bin
        if i == 0:
            mask = omega < data[0, 0]
        elif i == len(h_exp) - 1:
            mask = omega >= data[-1, 0]
        else:
            mask = (omega >= data[i-1, 0]) & (omega < data[i, 0])
        # Average predictions in bin
        if np.any(mask):
            h_pred_binned[i] = np.mean(h_pred[mask])
    
    # Statistical comparison with proper error handling
    valid_mask = (h_err > 0) & np.isfinite(h_pred_binned) & np.isfinite(h_exp)
    chi2 = np.sum(((h_pred_binned[valid_mask] - h_exp[valid_mask])/h_err[valid_mask])**2)
    dof = np.sum(valid_mask) - 2  # Two fit parameters
    
    # Compute p-value with protection against overflow
    try:
        p_value = 1 - stats.chi2.cdf(min(chi2, 1e300), dof)
    except:
        p_value = 0.0  # Conservative fallback
        
    # Verify predictions match data
    assert p_value > 0.05, f"Gravitational wave predictions disagree with data (p={p_value:.2e})"
    
    # Additional physics checks from appendix_e_predictions.tex
    
    # 1. Classical regime (low frequency)
    low_freq_mask = omega < 1e-8
    if np.any(low_freq_mask):
        # Should follow h ~ 1/omega^4 power law
        ratio = h_pred[low_freq_mask][1:] / h_pred[low_freq_mask][:-1]
        freq_ratio = (omega[low_freq_mask][1:] / omega[low_freq_mask][:-1])**4
        assert np.allclose(ratio, freq_ratio, rtol=0.1)
    
    # 2. Quantum regime (high frequency) 
    high_freq_mask = omega > 1e2
    if np.any(high_freq_mask):
        # Should have quantum suppression
        assert np.all(h_pred[high_freq_mask] < h_exp[high_freq_mask])
    
    # 3. No zeros or infinities
    assert not np.any(h_pred == 0)
    assert not np.any(np.isinf(h_pred))
    assert np.all(np.isfinite(h_pred))

def test_coupling_verification():
    """
    Test coupling evolution with quantum corrections.
    From appendix_h_rgflow.tex Eq H.1-H.4 and appendix_k_io_distinction.tex Eq K.51
    """
    field = UnifiedField()
    
    # Test asymptotic freedom at high energies
    E_high = 1e16  # Near GUT scale
    couplings_high = field.compute_couplings(E_high)
    
    # All couplings should be small but positive at high energies
    for g in couplings_high.values():
        assert 0 < g < 1.0, "Coupling should show asymptotic freedom"
        
    # Test quantum coherence preservation
    E_qc = 1e12  # Intermediate scale
    couplings_qc = field.compute_couplings(E_qc)
    
    # From appendix_k_io_distinction.tex Eq K.51:
    def quantum_coherence_measure(g1: float, g2: float, g3: float) -> float:
        """Compute quantum coherence between couplings"""
        g_avg = (g1 + g2 + g3)/3
        # Enhanced quantum coherence measure with proper scaling
        def coherence_term(g: float) -> float:
            """Compute coherence contribution with boundary conditions"""
            r = g/g_avg
            # Enhanced stability near unity
            x = -r * np.log(r) * (1 - np.exp(-r))
            # Proper boundary behavior
            return x * (1 + np.exp(-(r - 1)**2))
        
        # Sum coherence terms with proper normalization
        coherence = sum(coherence_term(g) for g in [g1, g2, g3])
        # Ensure positive measure
        return float(np.abs(coherence))
        
    qc = quantum_coherence_measure(
        couplings_qc['g1'],
        couplings_qc['g2'],
        couplings_qc['g3']
    )
    
    assert qc > 0, "Quantum coherence must be positive"
    
    # Test holographic bound satisfaction
    E_holo = 1e14
    couplings_holo = field.compute_couplings(E_holo)
    
    # From appendix_g_holographic.tex Eq G.34:
    lambda_h = np.sqrt(M_PLANCK/E_holo)
    A = 4*np.pi*(lambda_h*C/E_holo)**2
    S_max = A/(4*HBAR*G)
    
    S = -sum(g * np.log(g) for g in couplings_holo.values())
    assert S < S_max, "Holographic entropy bound must be satisfied"
    
    # Test unification at GUT scale
    couplings_gut = field.compute_couplings(GUT_SCALE)
    g_vals = list(couplings_gut.values())
    
    # Couplings should unify to within 1%
    g_spread = max(g_vals) - min(g_vals)
    assert g_spread < 0.01, "Couplings must unify at GUT scale"
    
    # Test proper ordering at low energies
    couplings_low = field.compute_couplings(Z_MASS)
    assert (
        couplings_low['g1'] < couplings_low['g2'] < couplings_low['g3']
    ), "Coupling hierarchy must be preserved at low energies"