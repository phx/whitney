"""Common test fixtures and configuration."""

import os
import sys
import pytest
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.detector import Detector
from core.types import Energy, Momentum, CrossSection
from core.physics_constants import (
    ALPHA_VAL, Z_MASS,
    g1_REF, g2_REF, g3_REF
)
from core.types import FieldConfig
from core.modes import ComputationMode
from typing import Dict, Any, List, Tuple, Optional
from core.physics_constants import (
    HBAR, C, G, X, T, P, E,  # Add E to imports
    g1_REF, g2_REF, g3_REF
)
from core.field import UnifiedField
from core.basis import FractalBasis

# Physical constants for tests
M_PLANCK = 1.22e19  # Planck mass in GeV

@pytest.fixture(scope="session")
def test_grid() -> np.ndarray:
    """
    Create standard test grid.
    
    Returns:
        np.ndarray: Evenly spaced points from -10 to 10
    """
    return np.linspace(-10, 10, 100)

@pytest.fixture(scope="session")
def energy_points() -> np.ndarray:
    """
    Create standard energy points.
    
    Returns:
        np.ndarray: Log-spaced points from Z mass to Planck mass
    """
    return np.logspace(np.log10(Z_MASS), np.log10(M_PLANCK), 100)

@pytest.fixture
def detector() -> Detector:
    """
    Create test detector instance.
    
    Returns:
        Detector: Configured detector instance for testing
    """
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
def test_data_generator() -> callable:
    """
    Generate test data function.
    
    Returns:
        callable: Function that generates test data
    """
    def _generate(n_samples: int = 1000) -> Dict[str, List]:
        energies = np.random.uniform(10, 1000, n_samples)
        momenta = np.random.uniform(0, 500, n_samples)
        return {
            'energies': [Energy(e) for e in energies],
            'momenta': [Momentum(p) for p in momenta]
        }
    return _generate

@pytest.fixture
def test_covariance() -> callable:
    """
    Generate test covariance matrix function.
    
    Returns:
        callable: Function that generates covariance matrices
    """
    def _generate(size: int = 3) -> np.ndarray:
        A = np.random.randn(size, size)
        return A @ A.T
    return _generate 

@pytest.fixture
def numeric_precision() -> Dict[str, float]:
    """
    Set up numerical precision requirements.
    
    Returns:
        Dict[str, float]: Precision parameters
    """
    return {
        'rtol': 1e-6,
        'atol': 1e-8,
        'maxiter': 1000,
        'stability_threshold': 1e-4
    }

@pytest.fixture
def test_config() -> Dict[str, Any]:
    """
    Basic test configuration.
    
    Returns:
        Dict[str, Any]: Test configuration parameters
    """
    return {
        'alpha': ALPHA_VAL,
        'energy': Z_MASS,
        'mode': ComputationMode.MIXED
    }

@pytest.fixture
def field_config() -> FieldConfig:
    """
    Test field configuration.
    
    Returns:
        FieldConfig: Standard field configuration for testing
    """
    return FieldConfig(
        mass=125.0,
        coupling=0.1,
        dimension=4
    )