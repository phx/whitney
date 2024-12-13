"""Tests for unified field theory implementation."""

import pytest
import numpy as np
from core.field import UnifiedField
from core.types import Energy, FieldConfig, WaveFunction
from core.modes import ComputationMode
from core.errors import PhysicsError, ValidationError
from core.constants import (
    ALPHA_VAL, Z_MASS, g1_REF, g2_REF, g3_REF,
    GAMMA_1, GAMMA_2, GAMMA_3
)

@pytest.mark.physics
class TestFieldEvolution:
    """Test field evolution and dynamics."""
    
    @pytest.fixture
    def field(self):
        """Create test field instance."""
        return UnifiedField(alpha=ALPHA_VAL, mode=ComputationMode.MIXED)
    
    def test_field_initialization(self, field):
        """Test field initialization."""
        assert isinstance(field, UnifiedField)
        assert field.alpha == pytest.approx(ALPHA_VAL)
        assert hasattr(field, 'basis')
    
    def test_field_evolution(self, field):
        """Test time evolution of field."""
        # Initial state
        psi0 = np.zeros(10)
        psi0[0] = 1.0  # Ground state
        
        # Evolution parameters
        times = np.linspace(0, 10, 100)
        evolution = field.evolve_field(
            initial_state=psi0,
            times=times,
            precision=1e-6
        )
        
        assert 'state' in evolution
        assert 'energy' in evolution
        assert 'error' in evolution
        assert len(evolution['state']) == len(times)
        
        # Test energy conservation
        energies = evolution['energy']
        assert np.allclose(energies, energies[0], rtol=1e-5)
    
    def test_gauge_invariance(self, field):
        """Test gauge invariance of evolution."""
        psi0 = np.zeros(10)
        psi0[0] = 1.0
        
        # Original evolution
        times = np.linspace(0, 1, 10)
        evolution1 = field.evolve_field(psi0, times)
        
        # Gauge transformed evolution
        gauge_phase = np.exp(1j * np.pi/4)
        psi0_transformed = psi0 * gauge_phase
        evolution2 = field.evolve_field(psi0_transformed, times)
        
        # Physical observables should be unchanged
        assert np.allclose(
            np.abs(evolution1['state']),
            np.abs(evolution2['state']),
            rtol=1e-6
        )

@pytest.mark.physics
class TestCouplingEvolution:
    """Test coupling constant evolution."""
    
    @pytest.fixture
    def field(self):
        return UnifiedField()
    
    def test_coupling_running(self, field):
        """Test running coupling constants."""
        # Test at Z mass
        g1 = field.compute_coupling(1, Energy(Z_MASS))
        g2 = field.compute_coupling(2, Energy(Z_MASS))
        g3 = field.compute_coupling(3, Energy(Z_MASS))
        
        # Compare with reference values
        assert np.isclose(g1, g1_REF, rtol=1e-3)
        assert np.isclose(g2, g2_REF, rtol=1e-3)
        assert np.isclose(g3, g3_REF, rtol=1e-3)
    
    def test_beta_functions(self, field):
        """Test beta function computation."""
        E = Energy(1000.0)  # 1 TeV
        
        beta1 = field.compute_beta_function(1, E)
        beta2 = field.compute_beta_function(2, E)
        beta3 = field.compute_beta_function(3, E)
        
        # Beta functions should have correct signs
        assert beta1 > 0  # U(1) is asymptotically free
        assert beta2 < 0  # SU(2) is asymptotically free
        assert beta3 < 0  # SU(3) is asymptotically free
    
    @pytest.mark.parametrize('gauge_index', [1, 2, 3])
    def test_anomalous_dimensions(self, field, gauge_index):
        """Test anomalous dimension computation."""
        E = Energy(1000.0)
        gamma = field.compute_anomalous_dimension(gauge_index, E)
        
        # Compare with theoretical values
        gamma_ref = {
            1: GAMMA_1,
            2: GAMMA_2,
            3: GAMMA_3
        }[gauge_index]
        
        assert np.isclose(gamma, gamma_ref, rtol=1e-3)

@pytest.mark.physics
class TestFieldConfiguration:
    """Test field configuration handling."""
    
    @pytest.fixture
    def field(self):
        return UnifiedField()
    
    def test_config_validation(self, field):
        """Test field configuration validation."""
        # Valid config
        valid_config = FieldConfig(
            mass=125.0,
            coupling=0.1,
            dimension=4
        )
        assert field.validate_config(valid_config)
        
        # Invalid configs
        with pytest.raises(ValidationError):
            field.validate_config(FieldConfig(mass=-1.0))  # Negative mass
            
        with pytest.raises(ValidationError):
            field.validate_config(FieldConfig(coupling=-0.1))  # Negative coupling
    
    def test_energy_validation(self, field):
        """Test energy scale validation."""
        # Valid energy
        assert field.validate_energy(Energy(100.0))
        
        # Invalid energies
        with pytest.raises(PhysicsError):
            field.validate_energy(Energy(-100.0))  # Negative energy
            
        with pytest.raises(PhysicsError):
            field.validate_energy(Energy(0.0))  # Zero energy