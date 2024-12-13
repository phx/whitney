"""Physics calculation tests."""

import pytest
import numpy as np
from core.field import UnifiedField
from core.types import Energy, Momentum

@pytest.mark.physics
class TestPhysicsCalculations:
    """Test physics calculations and predictions."""
    
    def test_coupling_evolution(self, standard_field):
        """Test coupling constant evolution."""
        energies = np.logspace(2, 4, 10)
        couplings = standard_field.evolve_coupling(energies)
        
        # Test asymptotic freedom
        assert couplings[-1] < couplings[0]
        
        # Test monotonic decrease
        assert np.all(np.diff(couplings) < 0)
    
    def test_cross_sections(self, standard_field, physics_data):
        """Test cross-section calculations."""
        energies = physics_data['energies']
        cross_sections = standard_field.compute_cross_sections(energies)
        
        # Test high-energy behavior (power law)
        ratio = cross_sections[-1] / cross_sections[-2]
        expected = (energies[-2] / energies[-1])**4
        assert np.isclose(ratio, expected, rtol=0.1)

@pytest.mark.theory
class TestTheorems:
    """Test theoretical consistency and theorems."""
    
    def test_unitarity(self, standard_field):
        """Test unitarity constraints."""
        # Generate phase space points
        momenta = np.array([[100.0, 0.0, 0.0, 0.0],
                           [0.0, 100.0, 0.0, 0.0]])
        
        # Test S-matrix unitarity
        s_matrix = standard_field.compute_s_matrix(momenta)
        identity = s_matrix @ s_matrix.conj().T
        assert np.allclose(identity, np.eye(len(identity)), atol=1e-6)