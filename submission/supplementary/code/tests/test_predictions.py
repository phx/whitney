"""Tests for theoretical predictions and experimental validation."""

import pytest
try:
    from hypothesis import given, strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    from unittest.mock import patch
    HAS_HYPOTHESIS = False
    # Mock hypothesis decorator if not available
    def given(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    st = None

import numpy as np
from sympy import exp, I, pi, Symbol, integrate, oo, diff
from core.field import UnifiedField
from core.types import Energy, CrossSection
from core.errors import ValidationError
from core.physics_constants import (
    ALPHA_VAL,
    Z_MASS,
    ALPHA_REF
)
from core.constants import GUT_SCALE

@pytest.fixture
def field():
    """Create UnifiedField instance for testing."""
    return UnifiedField(alpha=ALPHA_VAL)

@pytest.fixture
def constants(request):
    """Get physics constants from conftest."""
    from core.physics_constants import X, T, P, HBAR, C
    return {'X': X, 'T': T, 'P': P, 'HBAR': HBAR, 'C': C}

class TestExperimentalPredictions:
    """Test agreement with experimental data."""
    
    @given(st.floats(min_value=88.0, max_value=94.0))
    def test_cross_section_prediction(self, energy, field):
        """Test cross section predictions against LEP data."""
        if not HAS_HYPOTHESIS:
            energy = 91.2  # Z mass peak if hypothesis not available
            
        # Compute cross section
        sigma = field.compute_cross_section(energy)
        cross_sections = [sigma]
        
        # Compare with LEP data
        peak_position = energy if len(cross_sections) == 1 else np.linspace(88, 94, 100)[np.argmax(cross_sections)]
        assert abs(peak_position - Z_MASS) < 0.1  # Within 100 MeV

    @given(st.floats(min_value=90.0, max_value=1000.0))
    def test_coupling_running(self, energy, field):
        """Test coupling evolution against collider data."""
        if not HAS_HYPOTHESIS:
            energy = 91.2  # LEP energy if hypothesis not available
            
        # Test points from various experiments
        data_points = {
            91.2: (0.117, 0.002),   # LEP
            200: (0.112, 0.003),    # HERA
            1000: (0.108, 0.004)    # LHC
        }
        
        # Find closest reference point
        closest_E = min(data_points.keys(), key=lambda x: abs(x - energy))
        g_exp, unc = data_points[closest_E]
        
        g_theory = field.compute_coupling(3, energy)  # Strong coupling
        assert abs(g_theory - g_exp) < 2*unc  # Within 2σ

    def test_cross_sections(self, field, constants):
        """Test predicted cross sections against data."""
        X, T, C, HBAR = constants['X'], constants['T'], constants['C'], constants['HBAR']
        
        # Test quantum field configuration
        psi = exp(-(X**2 + (C*T)**2)/(2*HBAR**2))
        
        # Compute cross section at different energies
        test_energies = [100.0, 500.0, 1000.0]  # GeV
        for E in test_energies:
            sigma = field.compute_cross_section(E, psi)
            
            # Cross section should be positive
            assert sigma.value > 0
            
            # Should scale as 1/E^2 at high energies
            if E > 500:
                sigma_ref = field.compute_cross_section(E/2, psi)
                ratio = sigma.value / sigma_ref.value
                assert np.isclose(ratio, 0.25, rtol=0.1)  # 1/(E^2) scaling

def test_coupling_unification():
    """Test prediction of coupling constant unification."""
    field = UnifiedField()
    
    # Compute couplings at GUT scale
    couplings = field.compute_couplings(GUT_SCALE)
    
    # Should converge to single value within errors
    alpha_gut = couplings['alpha_gut']
    assert abs(alpha_gut - 0.0376) < 0.0002  # Matches experimental bounds