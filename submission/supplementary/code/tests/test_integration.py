"""Integration tests for the fractal field theory framework."""

import pytest
import numpy as np
from core.field import UnifiedField
from core.basis import FractalBasis
from core.detector import Detector
from core.types import Energy, Momentum, CrossSection
from core.numeric import integrate_phase_space
from core.stability import check_convergence
from core.errors import ComputationError
from core.physics_constants import (
    G, HBAR, C, M_PLANCK  # Add these imports
)

@pytest.mark.integration
class TestPhysicsWorkflow:
    """Test complete physics workflow."""
    
    def test_cross_section_calculation(self, standard_field, physics_data):
        """Test complete cross-section calculation workflow."""
        # 1. Configure field
        field = standard_field
        
        # 2. Set up phase space
        energies = physics_data['energies']
        momenta = physics_data['momenta']
        
        # 3. Calculate matrix elements
        amplitudes = field.compute_amplitudes(energies, momenta)
        assert len(amplitudes) == len(energies)
        
        # 4. Integrate cross sections
        cross_sections = []
        for E, M in zip(energies, amplitudes):
            sigma = integrate_phase_space(
                lambda p, q: abs(M(p, q))**2,
                limits=[(0, E), (-E, E)]
            )
            cross_sections.append(CrossSection(sigma))
        
        # 5. Verify high-energy behavior
        ratios = [cs1.value/cs2.value for cs1, cs2 in zip(cross_sections[:-1], cross_sections[1:])]
        assert all(r > 1 for r in ratios)  # Cross section decreases with energy

@pytest.mark.integration
class TestDetectorSimulation:
    """Test detector simulation workflow."""
    
    def test_detector_reconstruction(self, standard_detector, physics_data):
        """Test detector reconstruction pipeline."""
        # 1. Generate particles
        true_energies = [Energy(E) for E in physics_data['energies']]
        true_momenta = [Momentum(p) for p in physics_data['momenta']]
        
        # 2. Simulate detector response
        measurements = []
        for E, p in zip(true_energies, true_momenta):
            # Check acceptance
            if standard_detector.check_acceptance(pt=p, eta=0.0):
                # Simulate measurement
                meas = standard_detector.simulate_measurement(
                    energy=E,
                    include_systematics=True
                )
                measurements.append(meas)
        
        # 3. Verify reconstruction
        for true_E, meas in zip(true_energies, measurements):
            # Energy within resolution
            assert abs(meas['energy'].value - true_E.value) < true_E.value * 0.2
            
            # Uncertainties properly propagated
            assert meas['energy'].uncertainty is not None
            assert meas['energy'].systematics is not None

@pytest.mark.integration
class TestNumericalWorkflow:
    """Test numerical computation workflow."""
    
    def test_field_evolution(self, standard_field, numeric_precision):
        """Test complete field evolution workflow."""
        # 1. Set up initial conditions
        psi0 = np.zeros(10)
        psi0[0] = 1.0  # Ground state
        
        # 2. Evolve field
        times = np.linspace(0, 10, 100)
        evolution = standard_field.evolve_field(
            initial_state=psi0,
            times=times,
            **numeric_precision
        )
        
        # 3. Check conservation laws
        energies = evolution['energy']
        assert check_convergence(
            energies,
            target=energies[0],
            **numeric_precision
        )
        
        # 4. Verify stability
        assert evolution['stable']
        assert evolution['max_error'] < numeric_precision['stability_threshold']

@pytest.mark.integration
class TestAnalysisWorkflow:
    """Test analysis workflow."""
    
    def test_coupling_unification(self, standard_field, physics_data):
        """Test coupling unification analysis."""
        # 1. Calculate running couplings
        energies = np.logspace(2, 16, 100)  # 100 GeV to 10^16 GeV
        g1 = standard_field.compute_coupling(1, energies)
        g2 = standard_field.compute_coupling(2, energies)
        g3 = standard_field.compute_coupling(3, energies)
        
        # 2. Find unification scale
        diffs = np.max([abs(g1-g2), abs(g2-g3), abs(g3-g1)], axis=0)
        unification_scale = energies[np.argmin(diffs)]
        
        # 3. Verify unification
        assert unification_scale > 1e15  # Above 10^15 GeV
        assert min(diffs) < 0.1  # Couplings meet within 10%

@pytest.mark.integration
def test_gravitational_wave_coherence():
    """
    Test quantum coherence preservation in gravitational waves.
    From appendix_k_io_distinction.tex Eq K.51-K.53
    """
    field = UnifiedField()
    
    # Test frequencies
    omega = np.logspace(-16, 4, 1000)
    
    # Compute spectrum
    h = field.compute_gravitational_wave_spectrum(omega)
    
    # Verify quantum coherence preservation
    for i in range(len(omega)-1):
        # Coherence measure from appendix_k_io_distinction.tex Eq K.52
        dh = abs(h[i+1] - h[i]) / abs(h[i])
        dw = abs(omega[i+1] - omega[i]) / omega[i]
        coherence = dh/dw
        
        # Should satisfy quantum coherence bound
        assert coherence < np.exp(omega[i]/(2*M_PLANCK*C**2))