"""Tests for detector and measurement implementation."""

import pytest
import numpy as np
from core.detector import Detector
from core.field import UnifiedField
from core.types import Energy, WaveFunction
from core.physics_constants import HBAR, C, Z_MASS

def test_detector_response():
    """Test detector response function."""
    detector = Detector()
    field = UnifiedField()
    psi = field.compute(n=0, E=Energy(Z_MASS))
    
    # Measure response
    response = detector.measure(psi)
    assert 0 <= response.value <= 1
    assert response.uncertainty < 0.1

def test_energy_resolution():
    """Test energy resolution."""
    detector = Detector()
    E = Z_MASS
    resolution = detector.compute_resolution(E)
    # Check ΔE/E scaling
    assert resolution.value/E < 0.01

def test_detection_efficiency():
    """Test detection efficiency."""
    detector = Detector()
    field = UnifiedField()
    psi = field.compute(n=0, E=Energy(Z_MASS))
    
    # Check efficiency
    eff = detector.compute_efficiency(psi)
    assert 0.5 <= eff.value <= 1.0

@pytest.mark.parametrize("E", [
    1.0,  # Low energy
    Z_MASS,  # Intermediate
    1000.0  # High energy
])
def test_energy_reconstruction(E):
    """Test energy reconstruction."""
    detector = Detector()
    field = UnifiedField()
    psi = field.compute(n=0, E=Energy(E))
    
    # Reconstruct energy
    E_rec = detector.reconstruct_energy(psi)
    assert abs(E_rec.value - E)/E < 0.1