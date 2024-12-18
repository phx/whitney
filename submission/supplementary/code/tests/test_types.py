"""Tests for the type system and custom physics types."""

import pytest
import numpy as np
from core.types import (
    RealValue, Energy, Momentum, CrossSection, 
    BranchingRatio, FieldConfig, WaveFunction,
    NumericValue,  # Add NumericValue import
    ensure_numeric_value  # Add ensure_numeric_value import
)
from core.errors import PhysicsError, ValidationError

@pytest.mark.types
class TestRealValue:
    """Test base RealValue type functionality."""
    
    def test_initialization(self):
        """Test RealValue initialization and validation."""
        # Valid initialization
        x = RealValue(1.0, 0.1)
        assert x.value == 1.0
        assert x.uncertainty == 0.1
        
        # No uncertainty
        x = RealValue(1.0)
        assert x.uncertainty is None
        
        # Invalid values
        with pytest.raises(ValueError):
            RealValue(float('nan'))
        with pytest.raises(ValueError):
            RealValue(float('inf'))
    
    def test_arithmetic(self):
        """Test arithmetic operations with uncertainty propagation."""
        x = RealValue(10.0, 1.0)
        y = RealValue(5.0, 0.5)
        
        # Addition
        z = x + y
        assert z.value == 15.0
        assert np.isclose(z.uncertainty, np.sqrt(1.0**2 + 0.5**2))
        
        # Multiplication
        z = x * y
        assert z.value == 50.0
        rel_unc = np.sqrt((1.0/10.0)**2 + (0.5/5.0)**2)
        assert np.isclose(z.uncertainty, 50.0 * rel_unc)

@pytest.mark.types
class TestPhysicsTypes:
    """Test physics-specific type implementations."""
    
    def test_energy_validation(self):
        """Test Energy type validation."""
        # Valid energy
        e = Energy(100.0, 5.0)
        assert e.value == 100.0
        assert e.units == "GeV"
        
        # Negative energy
        with pytest.raises(ValueError):
            Energy(-100.0)
        
        # Unit conversion
        e_tev = e.to_units("TeV")
        assert e_tev.units == "TeV"
        assert e_tev.value == 0.1
        assert np.isclose(e_tev.uncertainty, 0.005)
    
    def test_momentum_validation(self):
        """Test Momentum type validation."""
        # Valid momentum
        p = Momentum(100.0, 5.0)
        assert p.value == 100.0
        assert p.units == "GeV/c"
        
        # Negative momentum
        with pytest.raises(ValueError):
            Momentum(-100.0)
        
        # Unit conversion
        p_tev = p.to_units("TeV/c")
        assert p_tev.units == "TeV/c"
        assert p_tev.value == 0.1
        assert np.isclose(p_tev.uncertainty, 0.005)

@pytest.mark.types
class TestConfigTypes:
    """Test configuration and analysis types."""
    
    def test_field_config(self):
        """Test FieldConfig validation."""
        # Valid configuration
        config = FieldConfig(
            alpha=0.1,
            dimension=4,
            parameters={'mass': 125.0}
        )
        config.validate()
        
        # Invalid coupling
        with pytest.raises(ValueError):
            FieldConfig(alpha=1.5, dimension=4, parameters={}).validate()
        
        # Invalid dimension
        with pytest.raises(ValueError):
            FieldConfig(alpha=0.1, dimension=0, parameters={}).validate()
    
    def test_wave_function(self):
        """Test WaveFunction operations."""
        # Create test wavefunction
        psi = np.array([1.0, 1.0j, -1.0, -1.0j]) / 2.0
        grid = np.linspace(-1, 1, 4)
        wf = WaveFunction(psi, grid, {'n': 0, 'l': 0})
        
        # Test normalization
        wf.normalize()
        assert np.isclose(np.sum(np.abs(wf.psi)**2), 1.0)
        
        # Test zero wavefunction
        wf.psi = np.zeros_like(wf.psi)
        with pytest.raises(PhysicsError):
            wf.normalize()

def test_numeric_value_uncertainty_propagation():
    """Test uncertainty propagation in arithmetic operations."""
    x = NumericValue(1.0, 0.1)
    y = NumericValue(2.0, 0.2)
    
    # Test multiplication with uncertainties
    result = x * y
    assert np.isclose(result.value, 2.0)
    assert np.isclose(result.uncertainty, 0.3)  # Using error propagation rules
    
    # Test relative uncertainty
    rel_uncert = x.relative_uncertainty
    assert np.isclose(rel_uncert, 0.1)

def test_numeric_value_numpy_compatibility():
    """Test NumericValue compatibility with numpy operations."""
    arr = np.array([1.0])
    x = ensure_numeric_value(arr)
    assert isinstance(x, NumericValue)
    assert x.value == 1.0
    
    # Test numpy scalar types
    y = ensure_numeric_value(np.float64(2.0))
    assert isinstance(y, NumericValue)
    assert y.value == 2.0

def test_numeric_value_validation():
    """Test input validation for NumericValue."""
    with pytest.raises(ValueError):
        NumericValue(float('inf'))
    
    with pytest.raises(ValueError):
        NumericValue(float('nan'))
    
    with pytest.raises(TypeError):
        ensure_numeric_value(np.array([1.0, 2.0]))

@pytest.mark.types
class TestNumericValueEnhancements:
    """Test enhanced NumericValue functionality."""
    
    def test_complex_conversion(self):
        """Test enhanced complex number handling."""
        # Test real value
        x = NumericValue(1.0, 0.1)
        z = complex(x)
        assert z == (1.0 + 0j)
        
        # Test complex value
        x = NumericValue(1.0 + 1.0j, 0.1)
        z = complex(x)
        assert z == (1.0 + 1.0j)
        
        # Test phase property
        assert np.isclose(x.phase, np.pi/4)

    def test_numpy_conversion(self):
        """Test numpy array conversion."""
        x = NumericValue(1.0, 0.1)
        arr = x.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        assert np.isclose(arr, 1.0)
        
        # Test complex value
        x = NumericValue(1.0 + 1.0j, 0.1)
        arr = x.to_numpy()
        assert arr.dtype == np.complex128
        assert np.isclose(arr, 1.0 + 1.0j)

    def test_measurement_creation(self):
        """Test creation from physical measurements."""
        # Test positive error
        x = NumericValue.from_measurement(100.0, 0.5)
        assert x.value == 100.0
        assert x.uncertainty == 0.5
        
        # Test negative error handling
        x = NumericValue.from_measurement(100.0, -0.5)
        assert x.uncertainty == 0.5  # Should take absolute value
        
        # Test zero error
        x = NumericValue.from_measurement(100.0, 0.0)
        assert x.uncertainty == 0.0