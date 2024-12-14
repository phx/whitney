"""Tests for theoretical predictions and experimental validation."""

import pytest
import numpy as np
from core.field import UnifiedField
from core.types import Energy, CrossSection
from core.errors import PhysicsError
from core.constants import GUT_SCALE

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
        E = np.logspace(2, 3, 10)  # 100 GeV - 1 TeV
        
        # Generate test state
        psi = field.compute_basis_state(E[0])
        
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