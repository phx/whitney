"""Tests for detector simulation and response."""

import pytest
import numpy as np
from core.detector import Detector
from core.types import Energy, Momentum, CrossSection
from core.errors import ValidationError, PhysicsError
from core.constants import Z_MASS

@pytest.mark.physics
class TestDetectorResponse:
    """Test detector response and efficiency."""
    
    @pytest.fixture
    def detector(self):
        """Create test detector instance."""
        config = {
            'resolution': 0.01,  # 1% resolution
            'acceptance': (-2.5, 2.5),  # Pseudorapidity range
            'threshold': 10.0,  # GeV
            'efficiency': 0.9  # 90% efficiency
        }
        return Detector(**config)
    
    def test_energy_resolution(self, detector):
        """Test energy resolution scaling."""
        E = Energy(100.0)  # 100 GeV
        measured = detector.measure_energy(E)
        
        # Resolution should scale as σ/E = a/√E
        expected_sigma = E.value * detector.resolution / np.sqrt(E.value/Z_MASS)
        assert measured.uncertainty <= expected_sigma
    
    def test_momentum_resolution(self, detector):
        """Test momentum resolution."""
        p = Momentum(50.0, theta=np.pi/4, phi=0)
        measured = detector.measure_momentum(p)
        
        # Check components are measured correctly
        assert np.isclose(measured.magnitude, p.magnitude, rtol=0.1)
        assert np.isclose(measured.theta, p.theta, atol=0.01)
        assert np.isclose(measured.phi, p.phi, atol=0.01)
    
    @pytest.mark.parametrize('eta', [-2.0, 0.0, 2.0])
    def test_acceptance(self, detector, eta):
        """Test detector acceptance."""
        p = Momentum(50.0, eta=eta, phi=0)
        assert detector.in_acceptance(p)
        
        # Test outside acceptance
        p_out = Momentum(50.0, eta=3.0, phi=0)
        assert not detector.in_acceptance(p_out)

@pytest.mark.physics
class TestDetectorEfficiency:
    """Test detector efficiency calculations."""
    
    @pytest.fixture
    def detector(self):
        return Detector(efficiency=0.9)
    
    def test_basic_efficiency(self, detector):
        """Test basic efficiency calculation."""
        n_events = 1000
        detected = sum(detector.trigger() for _ in range(n_events))
        efficiency = detected / n_events
        
        assert np.isclose(efficiency, detector.efficiency, rtol=0.1)
    
    def test_energy_dependent_efficiency(self, detector):
        """Test energy-dependent efficiency."""
        E_low = Energy(15.0)  # Just above threshold
        E_high = Energy(1000.0)  # Well above threshold
        
        eff_low = detector.compute_efficiency(E_low)
        eff_high = detector.compute_efficiency(E_high)
        
        assert eff_high > eff_low  # Higher efficiency at higher energy

@pytest.mark.physics
class TestCalibration:
    """Test detector calibration."""
    
    @pytest.fixture
    def detector(self):
        return Detector()
    
    def test_energy_calibration(self, detector):
        """Test energy calibration."""
        # Generate calibration data
        true_energies = np.linspace(10, 1000, 100)
        measured = [detector.measure_energy(Energy(E)) for E in true_energies]
        
        # Perform calibration
        detector.calibrate(true_energies, measured)
        
        # Test calibrated measurement
        E_test = Energy(500.0)
        result = detector.measure_calibrated_energy(E_test)
        assert np.isclose(result.value, E_test.value, rtol=0.02)
    
    def test_cross_section_measurement(self, detector):
        """Test cross section measurement."""
        sigma = CrossSection(1e-3)  # 1 fb
        n_events = 1000
        luminosity = 1e4  # 10/fb
        
        measured = detector.measure_cross_section(
            n_events,
            luminosity,
            efficiency=0.9
        )
        
        assert isinstance(measured, CrossSection)
        assert measured.uncertainty is not None

@pytest.mark.physics
class TestSystematics:
    """Test systematic uncertainty handling."""
    
    @pytest.fixture
    def detector(self):
        return Detector(systematics={'energy_scale': 0.01})
    
    def test_systematic_uncertainties(self, detector):
        """Test systematic uncertainty propagation."""
        E = Energy(100.0)
        result = detector.measure_energy(E, include_systematics=True)
        
        # Total uncertainty should include systematics
        assert result.uncertainty > E.value * detector.resolution
    
    def test_correlation_handling(self, detector):
        """Test correlated systematic uncertainties."""
        E1 = Energy(100.0)
        E2 = Energy(200.0)
        
        r1 = detector.measure_energy(E1, include_systematics=True)
        r2 = detector.measure_energy(E2, include_systematics=True)
        
        correlation = detector.compute_correlation(r1, r2)
        assert 0 <= correlation <= 1  # Should be positively correlated