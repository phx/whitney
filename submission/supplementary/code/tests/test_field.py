"""Tests for UnifiedField base class."""

import pytest
from sympy import exp, I
import numpy as np
from core.field import UnifiedField
from core.types import Energy, FieldConfig, WaveFunction
from core.physics_constants import X, T, P
from core.modes import ComputationMode
from core.errors import PhysicsError, ValidationError

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