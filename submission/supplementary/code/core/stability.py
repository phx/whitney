"""Numerical stability analysis for fractal field theory."""

from typing import Dict, List, Callable, Any, Union
import numpy as np
from .types import NumericValue, Energy
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

def check_convergence(
    values: List[NumericValue],
    threshold: float = 1e-6
) -> bool:
    """
    Check if sequence of values has converged.
    
    Args:
        values: Sequence of numeric values
        threshold: Convergence threshold
        
    Returns:
        bool: True if converged
    """
    if len(values) < 2:
        return False
        
    # Compare last two values
    diff = abs(values[-1].value - values[-2].value)
    return diff < threshold

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