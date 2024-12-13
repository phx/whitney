"""Physics validation tests for fractal field theory."""

import numpy as np
import pytest
from core.constants import (
    ALPHA_REF, Z_MASS, W_MASS, SIN2_THETA_W,
    g1_REF, g2_REF, g3_REF,
    GAMMA_1, GAMMA_2, GAMMA_3
)
from core.field import UnifiedField
from core.basis import FractalBasis

def test_weinberg_angle():
    """Test Weinberg angle prediction (Eq. 2.6 in paper)."""
    sin2_theta = g1_REF**2 / (g1_REF**2 + g2_REF**2)
    assert abs(sin2_theta - SIN2_THETA_W) < 1e-5

def test_gauge_coupling_unification():
    """Test gauge coupling unification at high energy (Eq. 3.11)."""
    field = UnifiedField()
    
    # Test at unification scale (from paper)
    E_GUT = 2e16  # GeV
    
    # Compute couplings
    g1 = field.compute_coupling(1, E_GUT)
    g2 = field.compute_coupling(2, E_GUT)
    g3 = field.compute_coupling(3, E_GUT)
    
    # Should unify to within theoretical uncertainty
    assert abs(g1 - g2) < 1e-3
    assert abs(g2 - g3) < 1e-3

def test_fractal_scaling_relations():
    """Test fractal scaling relations (Eq. 3.8)."""
    basis = FractalBasis()
    
    # Test scaling at different energies
    E1, E2 = 100.0, 1000.0  # GeV
    
    for gauge_index in [1, 2, 3]:
        g1 = basis.coupling(gauge_index, E1)
        g2 = basis.coupling(gauge_index, E2)
        
        # Verify scaling relation
        ratio = np.log(g2/g1) / np.log(E2/E1)
        gamma = {1: GAMMA_1, 2: GAMMA_2, 3: GAMMA_3}[gauge_index]
        
        assert abs(ratio + gamma) < 1e-4

def test_mass_generation():
    """Test dynamical mass generation (Eq. 2.5)."""
    field = UnifiedField()
    
    # Create test field configuration
    psi = field.compute_basis_function(n=0, E=Z_MASS)
    
    # Compute mass term
    energy = field.compute_energy_density(psi)
    
    # Mass should be positive and finite
    assert energy > 0
    assert np.isfinite(energy)

def test_beta_functions():
    """Test beta function predictions (Eq. 3.10)."""
    basis = FractalBasis()
    
    # Test at reference scale
    E = Z_MASS
    
    for gauge_index in [1, 2, 3]:
        beta = basis.compute_beta_function(gauge_index, E)
        g = basis.coupling(gauge_index, E)
        
        # Beta function should have correct sign
        gamma = {1: GAMMA_1, 2: GAMMA_2, 3: GAMMA_3}[gauge_index]
        expected_sign = -np.sign(gamma)
        assert np.sign(beta) == expected_sign
        
        # Magnitude should be reasonable
        assert abs(beta/g) < 0.1  # Perturbative regime

@pytest.mark.parametrize("E", [10.0, 100.0, 1000.0])
def test_energy_conservation(E):
    """Test energy conservation in field evolution."""
    field = UnifiedField()
    psi = field.compute_basis_function(n=0, E=E)
    
    # Evolve field
    t = np.linspace(0, 10, 100)
    evolution = field.evolve_field(psi, t)
    
    # Energy should be conserved
    energies = evolution['energy']
    assert np.allclose(energies, energies[0], rtol=1e-4) 