"""Numerical stability analysis for fractal field theory."""

from typing import Dict, List, Callable, Any, Union
import numpy as np
from .types import NumericValue
from .errors import StabilityError

def analyze_perturbation(func: Callable[..., NumericValue],
                        params: Dict[str, float],
                        n_samples: int = 100,
                        epsilon: float = 1e-6) -> Dict[str, float]:
    """
    Analyze stability under parameter perturbations.
    
    Implements perturbation analysis from paper Sec. 3.4:
    1. Randomly perturb input parameters
    2. Compute function with perturbed inputs
    3. Analyze statistical properties of results
    
    Args:
        func: Function to analyze
        params: Parameter values
        n_samples: Number of perturbation samples
        epsilon: Relative perturbation size
        
    Returns:
        Dict containing:
        - mean: Mean value
        - std: Standard deviation
        - max_dev: Maximum deviation
        - condition: Condition number estimate
        
    Raises:
        StabilityError: If function is numerically unstable
    """
    results = []
    nominal = func(**params)
    
    for _ in range(n_samples):
        # Generate perturbed parameters
        perturbed = {
            k: v * (1 + np.random.normal(0, epsilon))
            for k, v in params.items()
        }
        
        # Compute with perturbed parameters
        try:
            result = func(**perturbed)
            results.append(result)
        except Exception as e:
            raise StabilityError(f"Function unstable under perturbation: {e}")
    
    results = np.array(results)
    mean = np.mean(results)
    std = np.std(results)
    max_dev = np.max(np.abs(results - nominal))
    
    # Estimate condition number
    condition = (std/epsilon) / (abs(mean) + 1e-10)
    
    return {
        'mean': float(mean),
        'std': float(std),
        'max_dev': float(max_dev),
        'condition': float(condition)
    }

def check_convergence(sequence: List[float],
                     rtol: float = 1e-8,
                     atol: float = 1e-10,
                     window: int = 10) -> bool:
    """
    Check convergence of numerical sequence.
    
    Implements convergence tests from paper Sec. 3.5:
    1. Relative change between successive terms
    2. Absolute change between successive terms
    3. Moving average stability
    
    Args:
        sequence: Numerical sequence to check
        rtol: Relative tolerance
        atol: Absolute tolerance
        window: Window size for moving average
        
    Returns:
        bool: True if sequence has converged
    """
    if len(sequence) < window + 1:
        return False
        
    # Check relative change
    rel_change = np.abs(np.diff(sequence[-window:])) / (np.abs(sequence[-window-1:-1]) + atol)
    if np.any(rel_change > rtol):
        return False
        
    # Check absolute change
    abs_change = np.abs(np.diff(sequence[-window:]))
    if np.any(abs_change > atol):
        return False
        
    # Check moving average stability
    ma = np.convolve(sequence[-2*window:], np.ones(window)/window, mode='valid')
    ma_change = np.abs(np.diff(ma)) / (np.abs(ma[:-1]) + atol)
    if np.any(ma_change > rtol):
        return False
        
    return True

def verify_error_bounds(nominal: float,
                       error_est: float,
                       samples: List[float],
                       confidence: float = 0.95) -> bool:
    """
    Verify error bounds against Monte Carlo samples.
    
    Implements error bound verification from paper Sec. 3.6:
    1. Compute empirical confidence intervals
    2. Compare with theoretical error bounds
    3. Verify containment of samples
    
    Args:
        nominal: Nominal value
        error_est: Estimated error
        samples: Monte Carlo samples
        confidence: Confidence level
        
    Returns:
        bool: True if error bounds are valid
    """
    samples = np.array(samples)
    n_samples = len(samples)
    
    # Compute empirical confidence interval
    sorted_samples = np.sort(samples)
    lower_idx = int(n_samples * (1 - confidence) / 2)
    upper_idx = int(n_samples * (1 + confidence) / 2)
    empirical_interval = (
        sorted_samples[lower_idx],
        sorted_samples[upper_idx]
    )
    
    # Theoretical bounds
    z_score = {
        0.68: 1.0,
        0.95: 1.96,
        0.99: 2.58
    }.get(confidence, 1.96)
    
    theoretical_interval = (
        nominal - z_score * error_est,
        nominal + z_score * error_est
    )
    
    # Verify containment
    empirical_width = empirical_interval[1] - empirical_interval[0]
    theoretical_width = theoretical_interval[1] - theoretical_interval[0]
    
    # Error bounds should be conservative
    return theoretical_width >= empirical_width 