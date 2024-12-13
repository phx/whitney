"""Experimental predictions and validation."""

import numpy as np
from core.field import UnifiedField
from core.constants import Z_MASS, ALPHA_REF

def test_cross_section_prediction():
    """Test cross section predictions against LEP data."""
    field = UnifiedField()
    
    # LEP energies
    energies = np.linspace(88, 94, 100)  # GeV
    
    # Z resonance peak
    cross_sections = []
    for E in energies:
        sigma = field.compute_cross_section(E)
        cross_sections.append(sigma)
    
    # Compare with LEP data
    peak_position = energies[np.argmax(cross_sections)]
    assert abs(peak_position - Z_MASS) < 0.1  # Within 100 MeV

def test_coupling_running():
    """Test coupling evolution against collider data."""
    field = UnifiedField()
    
    # Test points from various experiments
    data_points = {
        # Energy (GeV): (coupling, uncertainty)
        91.2: (0.117, 0.002),   # LEP
        200: (0.112, 0.003),    # HERA
        1000: (0.108, 0.004)    # LHC
    }
    
    for E, (g_exp, unc) in data_points.items():
        g_theory = field.compute_coupling(3, E)  # Strong coupling
        assert abs(g_theory - g_exp) < 2*unc  # Within 2σ 