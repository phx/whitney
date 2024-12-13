"""Common test fixtures and configuration."""

import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pytest
import numpy as np
from core.field import UnifiedField
from core.basis import FractalBasis
from core.detector import Detector
from core.types import Energy, Momentum, CrossSection
from core.constants import ALPHA_VAL, M_Z, M_PLANCK

@pytest.fixture(scope="session")
def test_grid():
    """Create standard test grid."""
    return np.linspace(-10, 10, 100)

@pytest.fixture(scope="session")
def energy_points():
    """Create standard energy points."""
    return np.logspace(np.log10(M_Z), np.log10(M_PLANCK), 100)

@pytest.fixture
def field():
    """Create test field instance."""
    return UnifiedField(alpha=ALPHA_VAL)

@pytest.fixture
def basis():
    """Create test basis instance."""
    return FractalBasis(alpha=ALPHA_VAL, max_level=3)

@pytest.fixture
def detector():
    """Create test detector instance."""
    return Detector(
        resolution={
            'energy': 0.1,
            'position': 0.001
        },
        acceptance={
            'eta': (-2.5, 2.5),
            'pt': 20.0
        }
    )

@pytest.fixture
def test_data_generator():
    """Generate test data."""
    def _generate(n_samples=1000):
        energies = np.random.uniform(10, 1000, n_samples)
        momenta = np.random.uniform(0, 500, n_samples)
        return {
            'energies': [Energy(e) for e in energies],
            'momenta': [Momentum(p) for p in momenta]
        }
    return _generate

@pytest.fixture
def test_covariance():
    """Generate test covariance matrix."""
    def _generate(size=3):
        # Generate random positive definite matrix
        A = np.random.randn(size, size)
        return A @ A.T
    return _generate 

@pytest.fixture(scope="session")
def standard_field():
    """Create a standard unified field configuration for testing."""
    return UnifiedField(
        alpha=0.1,
        dimension=4,
        parameters={
            'mass': 125.0,
            'coupling': 0.1,
            'scale': 1000.0
        }
    )

@pytest.fixture(scope="session")
def standard_basis():
    """Create a standard fractal basis for testing."""
    return FractalBasis(
        dimension=4,
        max_level=3,
        symmetry='SO(4)'
    )

@pytest.fixture
def physics_data():
    """Generate standard physics test data."""
    return {
        'energies': [100.0, 200.0, 500.0, 1000.0],
        'momenta': [50.0, 150.0, 300.0, 800.0],
        'cross_sections': [45.0, 32.0, 12.0, 5.0],
        'uncertainties': {
            'statistical': [2.0, 1.5, 0.8, 0.4],
            'systematic': [1.0, 0.8, 0.4, 0.2]
        }
    }

@pytest.fixture
def numeric_precision():
    """Set up numerical precision requirements."""
    return {
        'rtol': 1e-6,  # Relative tolerance
        'atol': 1e-8,  # Absolute tolerance
        'maxiter': 1000,  # Maximum iterations
        'stability_threshold': 1e-4  # Stability criterion
    }