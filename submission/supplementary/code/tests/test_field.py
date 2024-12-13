"""Tests for UnifiedField implementation."""

import pytest
import numpy as np
from core.field import UnifiedField
from core.constants import ALPHA_VAL, M_Z, M_PLANCK
from core.errors import PhysicsError
from core.types import Energy, Momentum

@pytest.fixture
def field():
    """Create a test field instance."""
    return UnifiedField(alpha=ALPHA_VAL)

@pytest.fixture
def energy_points():
    """Generate test energy points."""
    return np.logspace(np.log10(M_Z), np.log10(M_PLANCK), 100)

def test_field_initialization(field):
    """Test field initialization."""
    assert isinstance(field, UnifiedField)
    assert hasattr(field, 'alpha')
    assert field.alpha == pytest.approx(ALPHA_VAL)
    
    # Test invalid initialization
    with pytest.raises(PhysicsError):
        UnifiedField(alpha=-1.0)  # Invalid coupling
    
    with pytest.raises(PhysicsError):
        UnifiedField(alpha=1.0)  # Coupling too large

def test_coupling_evolution(field, energy_points):
    """Test gauge coupling evolution."""
    # Test U(1) coupling
    g1 = field.compute_coupling(1, energy_points)
    assert len(g1) == len(energy_points)
    assert np.all(g1 > 0)  # Couplings must be positive
    assert np.all(np.isfinite(g1))  # Must be finite
    
    # Test SU(2) coupling
    g2 = field.compute_coupling(2, energy_points)
    assert len(g2) == len(energy_points)
    assert np.all(g2 > 0)
    assert np.all(np.isfinite(g2))
    
    # Test SU(3) coupling
    g3 = field.compute_coupling(3, energy_points)
    assert len(g3) == len(energy_points)
    assert np.all(g3 > 0)
    assert np.all(np.isfinite(g3))
    
    # Test coupling unification
    high_E = np.array([M_PLANCK])
    g1_gut = field.compute_coupling(1, high_E)
    g2_gut = field.compute_coupling(2, high_E)
    g3_gut = field.compute_coupling(3, high_E)
    
    assert abs(g1_gut - g2_gut) < 0.1  # Should unify within 10%
    assert abs(g2_gut - g3_gut) < 0.1

def test_field_equations(field):
    """Test field equation solutions."""
    x = np.linspace(-10, 10, 100)
    phi = field.compute_field_configuration(x)
    
    assert len(phi) == len(x)
    assert np.all(np.isfinite(phi))
    
    # Test energy conservation
    energy = field.compute_energy(x, phi)
    assert energy > 0
    assert np.isfinite(energy)
    
    # Test field equations
    eom = field.check_field_equations(x, phi)
    assert np.all(np.abs(eom) < 1e-6)  # Should satisfy equations of motion

def test_observables(field):
    """Test observable calculations."""
    # Test weak mixing angle
    result = field.compute_observable('sin2_theta_W')
    assert 'value' in result
    assert 'total_uncertainty' in result
    assert 0.2 < result['value'] < 0.24  # Physical range
    
    # Test branching ratio
    BR = field.compute_branching_ratio('Bs_to_mumu')
    assert 'value' in BR
    assert 'error' in BR
    assert 1e-9 < BR['value'] < 1e-8  # Physical range
    
    # Test correlation function
    r_points = np.linspace(0, 10, 100)
    corr = field.compute_correlation_function(r_points)
    assert len(corr) == len(r_points)
    assert np.all(np.isfinite(corr))