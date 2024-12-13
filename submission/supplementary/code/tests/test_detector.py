"""Tests for the detector simulation module."""

import pytest
import numpy as np
from core.types import Energy, Momentum, RealValue
from core.detector import Detector
from core.errors import PhysicsError

@pytest.fixture
def standard_detector():
    """Create a standard detector configuration for testing."""
    return Detector(
        resolution={
            'energy': 0.1,  # 10% energy resolution
            'momentum': 0.05,  # 5% momentum resolution
            'position': 0.001  # 1mm position resolution
        },
        acceptance={
            'eta': (-2.5, 2.5),  # Pseudorapidity range
            'pt_min': 20.0,  # Minimum pT in GeV
            'phi': (0, 2*np.pi)  # Full azimuthal coverage
        }
    )

def test_detector_initialization():
    """Test detector initialization and parameter validation."""
    # Test valid initialization
    detector = standard_detector()
    assert detector.resolution['energy'] == 0.1
    assert detector.acceptance['eta'] == (-2.5, 2.5)
    
    # Test invalid resolution
    with pytest.raises(ValueError):
        Detector(
            resolution={'energy': -0.1},  # Invalid negative resolution
            acceptance={'eta': (-2.5, 2.5)}
        )
    
    # Test invalid acceptance
    with pytest.raises(ValueError):
        Detector(
            resolution={'energy': 0.1},
            acceptance={'eta': (2.5, -2.5)}  # Invalid range
        )

def test_detector_response():
    """Test detector response simulation."""
    detector = standard_detector()
    
    # Test energy measurement
    true_energy = Energy(100.0)
    measured = detector.simulate_measurement(true_energy)
    assert isinstance(measured['energy'], RealValue)
    assert measured['energy'].uncertainty is not None
    assert np.isclose(
        measured['energy'].uncertainty,
        true_energy.value * detector.resolution['energy'],
        rtol=1e-6
    )
    
    # Test momentum measurement
    true_momentum = Momentum(50.0)
    measured = detector.simulate_measurement(true_momentum)
    assert isinstance(measured['momentum'], RealValue)
    assert measured['momentum'].uncertainty is not None
    assert np.isclose(
        measured['momentum'].uncertainty,
        true_momentum.value * detector.resolution['momentum'],
        rtol=1e-6
    )

def test_detector_efficiency():
    """Test detector efficiency calculations."""
    detector = standard_detector()
    
    # Test efficiency in acceptance
    pt = RealValue(30.0)  # Above pt threshold
    eta = RealValue(0.0)  # Central region
    eff = detector.compute_efficiency(pt, eta)
    assert eff.value > 0.0
    assert eff.value <= 1.0
    
    # Test efficiency outside acceptance
    pt = RealValue(10.0)  # Below pt threshold
    eta = RealValue(3.0)  # Outside eta range
    eff = detector.compute_efficiency(pt, eta)
    assert eff.value == 0.0

def test_detector_acceptance():
    """Test detector acceptance checks."""
    detector = standard_detector()
    
    # Test pt acceptance
    assert detector.check_acceptance(
        pt=RealValue(25.0),
        eta=RealValue(0.0)
    )
    assert not detector.check_acceptance(
        pt=RealValue(15.0),  # Below threshold
        eta=RealValue(0.0)
    )
    
    # Test eta acceptance
    assert detector.check_acceptance(
        pt=RealValue(30.0),
        eta=RealValue(2.0)
    )
    assert not detector.check_acceptance(
        pt=RealValue(30.0),
        eta=RealValue(3.0)  # Outside range
    )

def test_detector_error_handling():
    """Test detector error handling."""
    detector = standard_detector()
    
    # Test invalid energy
    with pytest.raises(PhysicsError):
        detector.simulate_measurement(Energy(-100.0))
    
    # Test invalid momentum
    with pytest.raises(PhysicsError):
        detector.simulate_measurement(Momentum(-50.0))
    
    # Test invalid efficiency inputs
    with pytest.raises(ValueError):
        detector.compute_efficiency(
            pt=RealValue(float('nan')),
            eta=RealValue(0.0)
        )

def test_detector_calibration():
    """Test detector calibration and recalibration."""
    detector = standard_detector()
    
    # Test initial calibration constants
    assert 'energy_scale' in detector.calibration
    assert 'position_offset' in detector.calibration
    
    # Test calibration with reference data
    reference_data = {
        'energy': [100.0, 200.0, 300.0],
        'measured': [98.5, 197.0, 295.5],
        'uncertainties': [2.0, 3.0, 4.0]
    }
    
    detector.calibrate(reference_data)
    
    # Verify calibration improved energy scale
    assert np.isclose(detector.calibration['energy_scale'], 0.985, rtol=1e-3)
    
    # Test applying calibration
    true_energy = Energy(150.0)
    measured = detector.simulate_measurement(true_energy)
    expected = 150.0 * detector.calibration['energy_scale']
    assert np.isclose(measured['energy'].value, expected, rtol=1e-3)
    
    # Test calibration stability
    for _ in range(3):
        detector.calibrate(reference_data)
        new_scale = detector.calibration['energy_scale']
        assert np.isclose(new_scale, 0.985, rtol=1e-3)

def test_resolution_scaling():
    """Test detector resolution scaling with energy/momentum."""
    detector = standard_detector()
    
    # Test energy resolution scaling (typically ~ 1/√E)
    e1 = Energy(100.0)
    e2 = Energy(400.0)
    
    res1 = detector.get_resolution(e1)
    res2 = detector.get_resolution(e2)
    
    # Verify resolution improves with √E
    assert np.isclose(res1/res2, 2.0, rtol=1e-2)
    
    # Test momentum resolution scaling
    p1 = Momentum(50.0)
    p2 = Momentum(200.0)
    
    res1 = detector.get_resolution(p1)
    res2 = detector.get_resolution(p2)
    
    # Verify resolution scaling with momentum
    assert res2 > res1  # Resolution gets worse at higher momentum

def test_systematic_uncertainties():
    """Test handling of detector systematic uncertainties."""
    detector = standard_detector()
    
    # Test systematic uncertainty sources
    systematics = detector.get_systematics()
    assert 'energy_scale' in systematics
    assert 'resolution' in systematics
    assert 'acceptance' in systematics
    
    # Test systematic uncertainty propagation
    energy = Energy(100.0, 2.0)
    measurement = detector.simulate_measurement(
        energy, 
        include_systematics=True
    )
    
    # Verify systematics are included
    assert measurement['energy'].systematics is not None
    assert 'energy_scale' in measurement['energy'].systematics
    
    # Test total uncertainty calculation
    total_unc = np.sqrt(
        measurement['energy'].uncertainty**2 +
        sum(s**2 for s in measurement['energy'].systematics.values())
    )
    assert total_unc > measurement['energy'].uncertainty  # Systematics increase uncertainty