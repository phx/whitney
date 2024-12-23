"""Tests for fractal basis implementation."""

import pytest
import numpy as np
from numpy import pi, exp, log, sqrt
from pathlib import Path

from core.basis import FractalBasis
from core.types import WaveFunction, BasisConfig, NumericValue
from core.physics_constants import HBAR, C, I

# Load test data
DATA_DIR = Path("supplementary/data")

def load_test_data(filename: str) -> dict:
    """Load test data with proper validation."""
    data = {}
    with open(DATA_DIR / filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            name, *values = line.strip().split(',')
            data[name] = [float(v) for v in values]
    return data

@pytest.fixture
def basis():
    """Create FractalBasis instance with test configuration."""
    return FractalBasis(alpha=0.1, precision=1e-10)

@pytest.fixture
def test_state():
    """Create test quantum state."""
    grid = np.linspace(-3, 3, 100)  # SACRED: grid range
    psi = np.exp(-grid**2/2)  # Gaussian test state
    return WaveFunction(
        psi=psi,
        grid=grid,
        quantum_numbers={'n': 1, 'E': 1.0}
    )

def test_basis_initialization(basis):
    """Test basis initialization."""
    assert basis.alpha == 0.1
    assert basis.precision == 1e-10
    assert basis.max_level == 10
    assert basis.dimension == 4
    assert basis.scaling_dimension == 1.0

def test_basis_functions(basis):
    """
    Test basis function computation.
    
    From appendix_b_basis.tex Eq B.1-B.3:
    Verifies:
    1. Proper scaling behavior: ψₙ(x) = α^(n/2)ψ₀(αⁿx)
    2. Correct normalization: ∫|ψₙ|² = 1
    3. Grid range preservation: x ∈ [-3,3]
    """
    config = BasisConfig(dimension=4, precision=1e-10)
    functions = basis.compute_basis_functions(config)
    
    # Check number of functions
    assert len(functions) == basis.max_level
    
    # Verify scaling
    for n, mode in enumerate(functions):
        # Check grid range
        assert np.allclose(mode.grid, np.linspace(-3, 3, 100))
        
        # Verify normalization
        norm = np.sum(np.abs(mode.psi)**2) * (mode.grid[1] - mode.grid[0])
        assert abs(norm - 1.0) < 1e-6
        
        # Check scaling behavior
        expected_scale = basis.alpha**n
        max_amp = np.max(np.abs(mode.psi))
        assert abs(max_amp - expected_scale) < 1e-6

def test_state_projection(basis, test_state):
    """
    Test quantum state projection.
    
    From appendix_b_basis.tex Eq B.7-B.9:
    Verify projection coefficients.
    """
    coeffs = basis.project_state(test_state)
    
    # Load validation data
    valid_data = load_test_data("wavelet_analysis.csv")
    
    # Check coefficient properties
    total_prob = sum(abs(c)**2 for c in coeffs.values())
    assert abs(total_prob - 1.0) < 1e-6
    
    # Verify scaling behavior
    for n, c_n in coeffs.items():
        expected = basis.alpha**(n * basis.scaling_dimension)
        assert abs(abs(c_n) - expected) < valid_data['scaling'][0]

def test_fractal_analysis(basis, test_state):
    """
    Test fractal scaling analysis.
    
    From appendix_b_basis.tex Eq B.15-B.17:
    Verify fractal dimension computation.
    """
    results = basis.analyze_fractal_scaling(test_state)
    
    # Load validation data
    stats = load_test_data("statistical_analysis.csv")
    
    # Check fractal dimension
    D_f = results['fractal_dimension'].value
    assert 1.0 < D_f < 2.0
    assert abs(D_f - stats['expected_dimension'][0]) < 3*results['fractal_dimension'].uncertainty
    
    # Verify scaling symmetry
    assert results['scaling_symmetry'].value < stats['symmetry_threshold'][0]
    
    # Check coherence length
    assert results['coherence_length'].value > stats['min_coherence_length'][0]

def test_wavelet_analysis(basis, test_state):
    """
    Test wavelet transform analysis.
    
    From appendix_b_basis.tex Eq B.29-B.31:
    Verify wavelet properties.
    """
    results = basis.analyze_wavelet_transform(test_state)
    
    # Load validation data
    thresholds = load_test_data("wavelet_analysis.csv")
    
    # Check localization
    assert results['localization'].value < thresholds['max_localization'][0]
    
    # Verify resolution of unity
    assert abs(results['resolution'].value - 1.0) < thresholds['resolution_tolerance'][0]
    
    # Check admissibility
    adm = results['admissibility'].value
    assert thresholds['min_admissibility'][0] < adm < thresholds['max_admissibility'][0]

def test_error_handling(basis):
    """Test error handling and validation."""
    with pytest.raises(ValueError):
        FractalBasis(alpha=-0.1)  # Invalid scaling parameter
        
    with pytest.raises(ValueError):
        FractalBasis(dimension=1)  # Invalid dimension
        
    grid = np.linspace(-3, 3, 100)  # SACRED: grid range
    invalid_state = WaveFunction(
        psi=np.zeros_like(grid),  # Invalid zero state
        grid=grid,
        quantum_numbers={'n': 1, 'E': 1.0}
    )
    
    with pytest.raises(ValueError):
        basis.project_state(invalid_state) 