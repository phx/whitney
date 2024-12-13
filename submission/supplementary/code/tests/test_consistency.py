"""Framework consistency and stability tests."""

import numpy as np
from core.field import UnifiedField
from core.basis import FractalBasis

def test_numerical_precision():
    """Verify numerical precision meets physics requirements."""
    basis = FractalBasis()
    
    # Test at multiple scales
    for E in [10, 100, 1000]:  # GeV
        # Compute observables
        result = basis.compute_with_errors(n=1, E=E)
        
        # Check error bounds
        assert result['total_error'] < 1e-8
        assert result['normalization_error'] < 1e-10
        assert result['truncation_error'] < 1e-9

def test_gauge_invariance():
    """Verify gauge invariance of physical observables."""
    field = UnifiedField()
    
    # Original configuration
    psi = field.compute_basis_function(n=0)
    E1 = field.compute_energy_density(psi)
    
    # Gauge transformed configuration
    phase = np.exp(1j * 0.5)
    psi_transformed = psi * phase
    E2 = field.compute_energy_density(psi_transformed)
    
    # Physical observables should be invariant
    assert abs(E1 - E2) < 1e-12 