"""Tests for n-point correlation functions and scaling properties."""

import numpy as np
from core.field import UnifiedField
from core.basis import FractalBasis
from core.constants import Z_MASS, ALPHA_REF

def test_two_point_function():
    """
    Test two-point correlation function.
    Verifies scaling behavior and cluster decomposition.
    """
    field = UnifiedField()
    
    # Compute at different separations
    separations = np.logspace(-1, 2, 20)  # From 0.1 to 100 GeV^-1
    correlations = []
    
    psi = field.compute_basis_function(n=0, E=Z_MASS)
    
    for r in separations:
        # Compute <ψ(x)ψ(x+r)>
        corr = field.compute_correlation(psi, r)
        correlations.append(corr)
    
    # Test scaling behavior
    # Should fall off as r^(-2Δ) where Δ is scaling dimension
    scaling_dim = -np.log(correlations[-1]/correlations[0]) / np.log(separations[-1]/separations[0])
    expected_dim = 1.0 + ALPHA_REF  # From paper Eq. 4.8
    
    assert abs(scaling_dim - expected_dim) < 0.01

def test_higher_order_correlators():
    """Test higher-order correlation functions."""
    field = UnifiedField()
    psi = field.compute_basis_function(n=0, E=Z_MASS)
    
    # Three-point function
    r1, r2 = 1.0, 2.0
    three_point = field.compute_three_point_correlation(psi, r1, r2)
    
    # Test factorization in large separation limit
    two_point_1 = field.compute_correlation(psi, r1)
    two_point_2 = field.compute_correlation(psi, r2)
    
    # Should approximately factorize
    assert abs(three_point - two_point_1 * two_point_2) < 0.1

def test_scaling_violations():
    """Test scaling violation patterns predicted by theory."""
    basis = FractalBasis()
    
    # Compute anomalous dimensions at different scales
    energies = np.logspace(1, 4, 20)  # 10 GeV to 10 TeV
    dimensions = []
    
    for E in energies:
        dim = basis.analyze_scaling_dimension(n=1, E=E)
        dimensions.append(dim['total_dimension'])
    
    # Test logarithmic running
    log_derivs = np.diff(dimensions) / np.diff(np.log(energies))
    
    # Should be approximately constant (up to higher orders)
    assert np.std(log_derivs) < 0.01 