"""Tests for mathematical theorems and properties."""

import pytest
import numpy as np
from sympy import exp, I, pi
from core.basis import FractalBasis
from core.field import UnifiedField
from core.types import Energy, WaveFunction
from core.physics_constants import HBAR, C, Z_MASS, M_PLANCK, G

def test_unitarity():
    """Test S-matrix unitarity."""
    field = UnifiedField()
    E = Energy(10.0)
    psi = field.compute(n=0, E=E)
    
    # Compute S-matrix element
    S = field.compute_scattering_amplitude(psi)
    # Check unitarity bound
    assert abs(S) <= 1.0

def test_causality():
    """Test causal structure."""
    field = UnifiedField()
    psi = field.compute(n=0, E=Energy(1.0))
    
    # Check support in light cone
    x = np.linspace(-10, 10, 100)
    t = np.linspace(-10, 10, 100)
    for xi in x:
        for ti in t:
            if abs(xi) > C*abs(ti):  # Outside light cone
                val = complex(psi.psi.subs({X: xi, T: ti}))
                assert abs(val) < 1e-6

def test_holographic_bound():
    """
    Test holographic entropy bound.
    
    From appendix_g_holographic.tex Eq G.45:
    The entropy bound requires:
    S ≤ A/(4ℏG)
    where A is the area of the bounding surface.
    """
    field = UnifiedField()
    R = 1.0  # Radius
    S = field.compute_entropy(R)
    # Check area law
    area = 4*pi*R**2
    assert S.value <= area/(4*HBAR*G)

def test_fractal_recursion():
    """
    Test fractal recursion relations.
    
    From appendix_d_scale.tex Eq D.25:
    The fractal structure requires:
    ψₙ₊₁/ψₙ = α
    where α is the fine structure constant.
    """
    field = UnifiedField()
    # Test adjacent levels
    n1 = field.compute_fractal_recursion(1)
    n2 = field.compute_fractal_recursion(2)
    # Check scaling
    assert abs(n2/n1 - field.alpha) < 1e-6

def test_ward_identity():
    """
    Test Ward identity for current conservation.
    
    From appendix_h_rgflow.tex Eq H.15:
    Current conservation requires:
    ∂μJμ = 0
    where Jμ is the Noether current.
    """
    field = UnifiedField()
    psi = field.compute(n=0, E=Energy(1.0))
    # Compute current divergence
    div = field.compute_ward_identity(psi)
    assert abs(div.value) < 1e-6

@pytest.mark.parametrize("E", [
    1.0,  # Low energy
    Z_MASS,  # Intermediate
    M_PLANCK  # High energy
])
def test_energy_scaling(E):
    """
    Test energy scaling relations.
    
    From appendix_h_rgflow.tex Eq H.30:
    The effective dimension scales as:
    d_eff = 4 - γ(E)
    where γ(E) is the anomalous dimension.
    """
    field = UnifiedField()
    psi = field.compute(n=0, E=Energy(E))
    # Check scaling dimension
    dim = field.compute_effective_dimension()
    assert 2 <= dim.value <= 4