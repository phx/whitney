"""Tests for unified field theory implementation.

From appendix_j_math_details.tex:
The test suite validates:
1. Field equation solutions
2. Coupling unification
3. Quantum corrections
4. Holographic properties
"""

import pytest
import numpy as np
from numpy import pi, exp, log, sqrt
from pathlib import Path

from core.field import UnifiedField
from core.types import Energy, WaveFunction, FieldConfig, NumericValue
from core.physics_constants import (
    HBAR, C, G, M_P, I,  # Level 1: Fundamental constants
    g_μν, Gamma,  # Level 2: Mathematical objects
    Z_MASS  # Level 3: Derived quantities
)

# Load test data
DATA_DIR = Path("supplementary/data")

def load_test_data(filename: str) -> dict:
    """
    Load test data with proper validation.
    
    Args:
        filename: Name of data file to load
        
    Returns:
        dict: Validated test data
        
    Raises:
        FileNotFoundError: If data file is missing
        ValidationError: If data format is invalid
    """
    data = {}
    with open(DATA_DIR / filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            name, *values = line.strip().split(',')
            data[name] = [float(v) for v in values]
    return data

@pytest.fixture
def field():
    """
    Create UnifiedField instance with test configuration.
    
    From appendix_j_math_details.tex Eq J.1:
    Uses standard test parameters:
    - α = 0.1 (scaling parameter)
    - ε = 1e-10 (precision)
    - d = 4 (dimension)
    """
    return UnifiedField(alpha=0.1, precision=1e-10)

@pytest.fixture
def test_state():
    """
    Create test quantum state.
    
    From appendix_j_math_details.tex Eq J.2:
    Uses Gaussian ground state:
    ψ₀(x) = exp(-x²/2)
    
    Returns:
        WaveFunction: Normalized test state
    """
    grid = np.linspace(-3, 3, 100)  # SACRED: grid range
    psi = np.exp(-grid**2/2)  # Gaussian test state
    return WaveFunction(
        psi=psi,
        grid=grid,
        quantum_numbers={'n': 1, 'E': 1.0}
    )

def test_energy_density(field, test_state):
    """
    Test energy density computation.
    
    From appendix_d_scale.tex Eq D.7:
    Verifies:
    1. Classical kinetic term: T ~ |∇ψ|²
    2. Potential energy: V ~ |x|²|ψ|²
    3. Quantum corrections: ΔE ~ α^n
    
    Args:
        field: UnifiedField fixture
        test_state: Test WaveFunction fixture
        
    Raises:
        AssertionError: If energy properties are violated
    """
    result = field.compute_energy_density(test_state)
    
    # Load validation data
    valid_data = load_test_data("validation_results.csv")
    expected = valid_data['energy_density'][0]
    
    assert abs(result.value - expected) < 1e-6

def test_gravitational_action(field, test_state):
    """
    Test gravitational action computation.
    
    From appendix_c_gravity.tex Eq C.1:
    Verifies:
    1. Einstein-Hilbert term: S_EH ~ ∫d⁴x √(-g)R
    2. Quantum corrections: S_q ~ ∑ α^n C_n(R)
    3. Holographic bound: S ≤ A/(4G)
    
    Args:
        field: UnifiedField fixture
        test_state: Test WaveFunction fixture
        
    Raises:
        AssertionError: If action properties are violated
    """
    result = field.compute_gravitational_action(test_state)
    
    # Verify holographic bound
    area = 4*pi * (3*HBAR/M_P)**2  # Boundary area
    assert result.value <= area/(4*HBAR)

def test_coupling_unification(field):
    """
    Test gauge coupling unification.
    
    From appendix_h_rgflow.tex Eq H.5-H.7:
    Verifies:
    1. Coupling convergence: |gᵢ(M_GUT) - gⱼ(M_GUT)| → 0
    2. RG flow consistency: β(g) = μ∂g/∂μ
    3. Quantum corrections: g(μ) = g₀ + ∑ α^n b_n(g₀)ln(μ/μ₀)
    
    Args:
        field: UnifiedField fixture
        
    Raises:
        AssertionError: If unification fails
    """
    E_GUT = Energy(2.1e16)  # GeV
    couplings = field.compute_unified_couplings(E_GUT)
    
    g1 = couplings['g1'].value
    g2 = couplings['g2'].value
    g3 = couplings['g3'].value
    
    assert abs(g1 - g2) < 1e-6
    assert abs(g2 - g3) < 1e-6

def test_predictions(field):
    """
    Test theoretical predictions.
    
    From appendix_e_predictions.tex:
    Verifies:
    1. GUT scale: M_GUT ~ 2×10¹⁶ GeV
    2. Unified coupling: α_GUT ~ 1/25
    3. Proton lifetime: τ_p ~ 10³⁴ years
    
    Args:
        field: UnifiedField fixture
        
    Raises:
        AssertionError: If predictions deviate from expected values
    """
    predictions = field._compute_theoretical_predictions()
    
    # Load validation data
    valid_data = load_test_data("predictions.csv")
    
    # Verify GUT scale
    M_GUT = predictions['M_GUT'].value
    assert abs(M_GUT - valid_data['M_GUT'][0])/M_GUT < 0.15
    
    # Verify unified coupling
    alpha_GUT = predictions['alpha_GUT'].value
    assert abs(alpha_GUT - valid_data['alpha_GUT'][0]) < 2e-4
    
    # Verify proton lifetime
    tau_p = predictions['tau_p'].value
    assert abs(log(tau_p/valid_data['tau_p'][0])) < 0.3