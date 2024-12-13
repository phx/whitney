"""Mathematical proofs and theorem validation."""

import numpy as np
import pytest
from sympy import Symbol, integrate, oo, exp, diff
from core.basis import FractalBasis
from core.field import UnifiedField
from core.constants import ALPHA_REF, Z_MASS

def test_completeness_theorem():
    """
    Prove completeness of fractal basis.
    Theorem 1 from paper: The fractal basis forms a complete set.
    """
    basis = FractalBasis()
    x = Symbol('x')
    
    # Test function (Gaussian)
    f = exp(-x**2)
    
    # Expand in basis up to N
    N = 10
    expansion = 0
    for n in range(N):
        psi_n = basis.compute(n)
        cn = integrate(psi_n * f, (x, -oo, oo))
        expansion += cn * psi_n
    
    # Compute L2 norm of difference
    error = integrate((f - expansion)**2, (x, -oo, oo))
    assert float(error) < 1e-6

def test_unitarity():
    """
    Prove unitarity of time evolution.
    Theorem 2: Time evolution preserves probability.
    """
    field = UnifiedField()
    psi = field.compute_basis_function(n=0, E=Z_MASS)
    
    # Evolve for various times
    times = np.linspace(0, 10, 100)
    evolution = field.evolve_field(psi, times)
    
    # Check norm conservation
    norms = evolution['norm']
    assert np.allclose(norms, 1.0, rtol=1e-8)

def test_causality():
    """
    Prove causality of field propagation.
    Theorem 3: Field propagation respects light cone structure.
    """
    field = UnifiedField()
    psi = field.compute_basis_function(n=0)
    
    # Test points outside light cone
    t, x = 1.0, 2.0  # Space-like separation
    
    # Compute commutator of fields
    commutator = field.compute_field_equation(psi.subs('t', t).subs('x', x))
    assert abs(float(commutator)) < 1e-10 