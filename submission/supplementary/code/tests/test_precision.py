"""Tests for precision control and validation."""

import pytest
import numpy as np
from core.precision import (
    validate_precision,
    PrecisionMeasurement,
    PhaseLockLoop,
    FrequencyComb
)
from core.types import NumericValue, Energy
from core.errors import PrecisionError
from core.field import UnifiedField

@pytest.mark.precision
class TestPrecisionValidation:
    """Test precision validation functionality."""
    
    def test_basic_validation(self):
        """Test basic precision validation."""
        value = NumericValue(1.0, 0.01)
        assert validate_precision(value, 1.0, rtol=0.02)
        assert not validate_precision(value, 1.0, rtol=0.001)
    
    def test_validation_without_uncertainty(self):
        """Test validation fails without uncertainty."""
        value = NumericValue(1.0)
        with pytest.raises(PrecisionError, match="Cannot validate.*uncertainty"):
            validate_precision(value, 1.0)
    
    def test_absolute_tolerance(self):
        """Test absolute tolerance handling."""
        value = NumericValue(1e-6, 1e-7)
        assert validate_precision(value, 1e-6, atol=1e-6)
        assert not validate_precision(value, 1e-6, atol=1e-8)

@pytest.mark.precision
class TestPrecisionMeasurement:
    """Test precision measurement system."""
    
    @pytest.fixture
    def field(self):
        """Create test field instance."""
        return UnifiedField()
    
    @pytest.fixture
    def measurement(self, field):
        """Create precision measurement instance."""
        return PrecisionMeasurement(field)
    
    def test_coupling_measurement(self, measurement):
        """Test high-precision coupling measurement."""
        g, uncertainty = measurement.measure_coupling(1000.0, 1)
        assert isinstance(g, float)
        assert isinstance(uncertainty, float)
        assert uncertainty < 1e-10  # High precision requirement
    
    def test_pll_stabilization(self):
        """Test phase-locked loop stabilization."""
        pll = PhaseLockLoop()
        value = 1.0
        stabilized = pll.stabilize(value)
        assert abs(stabilized - value) < 1e-6
    
    def test_frequency_comb(self):
        """Test frequency comb precision enhancement."""
        comb = FrequencyComb()
        value = 1.0
        enhanced = comb.enhance_precision(value)
        assert abs(enhanced - value) < 1e-12  # Enhanced precision

@pytest.mark.precision
class TestPrecisionScaling:
    """Test precision scaling with energy."""
    
    @pytest.fixture
    def measurement(self):
        """Create measurement system."""
        field = UnifiedField()
        return PrecisionMeasurement(field)
    
    @pytest.mark.parametrize('energy', [
        100.0,  # Low energy
        1000.0,  # Medium energy
        10000.0  # High energy
    ])
    def test_precision_energy_scaling(self, measurement, energy):
        """Test how precision scales with energy."""
        g, uncertainty = measurement.measure_coupling(energy, 1)
        # Uncertainty should scale with energy
        assert uncertainty < 1e-12 * energy
    
    def test_precision_limits(self, measurement):
        """Test precision measurement limits."""
        # Very high energy should have larger uncertainty
        g_high, unc_high = measurement.measure_coupling(1e6, 1)
        # Very low energy should have smaller uncertainty
        g_low, unc_low = measurement.measure_coupling(10.0, 1)
        
        assert unc_high > unc_low 