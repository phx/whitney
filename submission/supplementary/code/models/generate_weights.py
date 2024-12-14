"""Generate initial weights for neural network models."""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
from scipy.stats import chi2
from dataclasses import dataclass

from core.field import UnifiedField
from core.basis import FractalBasis
from core.types import Energy, WaveFunction, ErrorEstimate
from core.physics_constants import (
    ALPHA_VAL,
    Z_MASS,
    ALPHA_REF
)

@dataclass
class BasisWeight:
    """Container for basis function weights."""
    n: int  # Mode number
    value: complex  # Weight value
    energy: Energy  # Associated energy scale
    error: float  # Numerical error estimate

class WeightGenerator:
    """Generate physically motivated initial weights."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize weight generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.field = UnifiedField(alpha=ALPHA_VAL)
        self.basis = FractalBasis(alpha=ALPHA_VAL)
        if seed is not None:
            np.random.seed(seed)
    
    def generate_basis_weights(self, n_modes: int = 100) -> np.ndarray:
        """Generate weights based on basis function coefficients.
        
        Args:
            n_modes: Number of basis modes to use
            
        Returns:
            Array of weights shaped for network layers
        """
        weights = []
        for n in range(n_modes):
            psi = self.basis.compute(n)
            coeff = self.field.compute_mode_coefficient(psi)
            weights.append(float(coeff))
        return np.array(weights)
    
    def generate_coupling_weights(self, energies: List[float]) -> np.ndarray:
        """Generate weights based on coupling evolution.
        
        Args:
            energies: List of energy scales to sample
            
        Returns:
            Array of weights based on coupling values
        """
        weights = []
        for E in energies:
            alpha = self.field.compute_coupling(3, E)
            weights.append(float(alpha))
        return np.array(weights)
    
    def save_weights(self, weights: Dict[str, np.ndarray], path: Union[str, Path]) -> None:
        """Save generated weights to file.
        
        Args:
            weights: Dictionary of weight arrays
            path: Path to save weights file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert arrays to lists for JSON serialization
        weights_json = {k: v.tolist() for k, v in weights.items()}
        
        with open(path, 'w') as f:
            json.dump(weights_json, f, indent=2) 

def compute_basis_weights(
    n_max: int = 100,
    alpha: float = ALPHA_VAL,
    target_precision: float = 1e-10
) -> List[BasisWeight]:
    """Compute weights for fractal basis functions."""
    basis = FractalBasis(alpha=alpha)
    weights = []
    
    for n in range(n_max):
        # Compute weight through fractal recursion
        value = basis.compute_weight(n)
        energy = basis.compute_mode_energy(n)
        error = basis.estimate_truncation_error(n)
        
        weights.append(BasisWeight(
            n=n,
            value=value,
            energy=energy,
            error=error
        ))
        
        # Check convergence
        if error < target_precision:
            break
            
    return weights

def optimize_weights(
    weights: List[BasisWeight],
    experimental_data: Dict[str, float],
    target_precision: float = 1e-10,
    learning_rate: float = 0.01,
    max_iterations: int = 1000
) -> List[BasisWeight]:
    """Optimize weights against experimental data.
    
    Args:
        weights: Initial basis weights
        experimental_data: Dictionary of measured values
        target_precision: Target precision for optimization
        learning_rate: Learning rate for gradient descent
        max_iterations: Maximum optimization iterations
        
    Returns:
        Optimized weights matching experimental constraints
    """
    field = UnifiedField()
    
    for iteration in range(max_iterations):
        # Compute predictions
        predictions = field.compute_observables(weights)
        
        # Compute chi-squared
        chi2 = sum((predictions[key] - experimental_data[key])**2 
                  for key in experimental_data)
        
        if chi2 < target_precision:
            break
            
        # Update weights using gradient descent
        gradients = field.compute_weight_gradients(weights, experimental_data)
        for w, g in zip(weights, gradients):
            w.value -= learning_rate * g
            
    return weights

def estimate_systematic_errors(
    weights: List[BasisWeight],
    n_samples: int = 1000
) -> Dict[str, ErrorEstimate]:
    """Estimate systematic uncertainties in weight determination.
    
    Args:
        weights: Computed basis weights
        n_samples: Number of Monte Carlo samples
        
    Returns:
        Dictionary of error estimates for each observable
    """
    field = UnifiedField()
    results = {}
    
    # Monte Carlo error estimation
    for _ in range(n_samples):
        # Vary weights within uncertainties
        varied_weights = [
            BasisWeight(
                n=w.n,
                value=w.value * (1 + np.random.normal(0, w.error)),
                energy=w.energy,
                error=w.error
            )
            for w in weights
        ]
        
        # Compute observables
        obs = field.compute_observables(varied_weights)
        
        # Accumulate results
        for key, val in obs.items():
            if key not in results:
                results[key] = []
            results[key].append(val)
            
    # Compute statistical properties
    return {
        key: ErrorEstimate(
            mean=np.mean(vals),
            std=np.std(vals),
            systematic=np.std(vals) / np.sqrt(n_samples)
        )
        for key, vals in results.items()
    }

def validate_weights(
    weights: List[BasisWeight],
    confidence_level: float = 0.95
) -> Dict[str, bool]:
    """Validate weights against experimental constraints.
    
    Args:
        weights: Computed basis weights
        confidence_level: Statistical confidence level
        
    Returns:
        Dictionary of validation results for each observable
    """
    field = UnifiedField()
    
    # Key experimental constraints
    constraints = {
        'alpha_gut': (0.0376, 0.0002),      # GUT coupling
        'proton_lifetime': (1.6e34, 0.3e34), # Years
        'weinberg_angle': (0.231, 0.001),    # At MZ
        'hierarchy_ratio': (1e-17, 1e-18)    # MW/MP
    }
    
    # Compute predictions
    predictions = field.compute_observables(weights)
    errors = estimate_systematic_errors(weights)
    
    # Validate each observable
    results = {}
    for key, (exp_val, exp_err) in constraints.items():
        pred_val = predictions[key]
        pred_err = errors[key].systematic
        
        # Compute chi-squared
        chi2 = ((pred_val - exp_val) / np.sqrt(pred_err**2 + exp_err**2))**2
        p_value = 1 - chi2.cdf(chi2, df=1)
        
        # Check if within confidence level
        results[key] = p_value > (1 - confidence_level)
        
    return results

def visualize_weight_distribution(
    weights: List[BasisWeight],
    save_path: Optional[str] = None
) -> None:
    """Visualize the distribution of basis weights.
    
    Args:
        weights: List of basis weights
        save_path: Optional path to save plot
    """
    import matplotlib.pyplot as plt
    
    # Extract data
    ns = [w.n for w in weights]
    values = [abs(w.value) for w in weights]
    energies = [w.energy for w in weights]
    errors = [w.error for w in weights]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Weight magnitude vs mode number
    ax1.errorbar(ns, values, yerr=errors, fmt='o', label='Weight magnitude')
    ax1.set_yscale('log')
    ax1.set_xlabel('Mode number n')
    ax1.set_ylabel('|Weight|')
    ax1.set_title('Weight Distribution')
    ax1.grid(True)
    
    # Energy scale distribution
    ax2.scatter(ns, energies, c=values, cmap='viridis', label='Energy scale')
    ax2.set_yscale('log')
    ax2.set_xlabel('Mode number n')
    ax2.set_ylabel('Energy (GeV)')
    ax2.set_title('Energy Scale Distribution')
    ax2.colorbar(label='Weight magnitude')
    ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_validation_results(
    results: Dict[str, bool],
    errors: Dict[str, ErrorEstimate],
    save_path: Optional[str] = None
) -> None:
    """Plot validation results with uncertainties.
    
    Args:
        results: Dictionary of validation results
        errors: Dictionary of error estimates
        save_path: Optional path to save plot
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each observable
    observables = list(results.keys())
    y_pos = np.arange(len(observables))
    
    # Color based on validation result
    colors = ['green' if results[obs] else 'red' for obs in observables]
    
    # Plot with error bars
    for i, obs in enumerate(observables):
        ax.barh(i, errors[obs].mean, xerr=errors[obs].std, 
                color=colors[i], alpha=0.6, label=obs)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(observables)
    ax.set_xlabel('Value')
    ax.set_title('Observable Predictions with Uncertainties')
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()