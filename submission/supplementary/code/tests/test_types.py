"""Tests for the type system and custom physics types."""

import pytest
import numpy as np
from core.types import (
    RealValue, Energy, Momentum, CrossSection, 
    BranchingRatio, FieldConfig, WaveFunction
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