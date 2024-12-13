#!/usr/bin/env python3
"""
Comprehensive validation of the entire framework.
"""

import numpy as np
from core.field import UnifiedField
from core.constants import EXPERIMENTAL_DATA, E_PLANCK, LAMBDA_QCD
from core.errors import ValidationError, PhysicsError
import pandas as pd
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_framework() -> None:
    """Run comprehensive framework validation."""
    logger.info("Starting comprehensive validation...")
    
    # Initialize framework
    field = UnifiedField()
    
    # 1. Physical Bounds
    logger.info("Checking physical bounds...")
    validate_physical_bounds(field)
    
    # 2. Experimental Agreement
    logger.info("Validating experimental predictions...")
    validate_experimental_agreement(field)
    
    # 3. Numerical Stability
    logger.info("Testing numerical stability...")
    validate_numerical_stability(field)
    
    # 4. Conservation Laws
    logger.info("Verifying conservation laws...")
    validate_conservation_laws(field)
    
    logger.info("All validations passed successfully!")

def validate_physical_bounds(field: UnifiedField) -> None:
    """Verify all quantities respect physical bounds."""
    # Test energy scales
    E_test = np.logspace(-1, 19, 100)
    for E in E_test:
        try:
            field.validate_energy_scale(E)
        except (ValueError, PhysicsError) as e:
            if E >= E_PLANCK or E <= LAMBDA_QCD:
                # Expected failure
                continue
            raise ValidationError(f"Unexpected energy scale validation error: {e}")

def validate_experimental_agreement(field: UnifiedField) -> None:
    """Verify agreement with experimental data."""
    for obs, (exp_val, exp_err) in EXPERIMENTAL_DATA.items():
        pred = field.compute_observable(obs)
        pull = (pred['value'] - exp_val) / np.sqrt(
            pred['total_uncertainty']**2 + exp_err**2
        )
        if abs(pull) > 3:  # More than 3σ deviation
            raise ValidationError(
                f"Observable {obs} deviates by {pull:.1f}σ from experiment"
            )

def validate_numerical_stability(field: UnifiedField) -> None:
    """Verify numerical stability of computations."""
    # Test coupling evolution
    E_test = 1000.0
    results = []
    for _ in range(100):
        E_perturbed = E_test * (1 + np.random.normal(0, 1e-6))
        results.append(field.compute_coupling(1, E_perturbed))
    
    std_dev = np.std(results)
    if std_dev > 1e-6:
        raise ValidationError(f"Numerical instability detected: σ={std_dev}")

def validate_conservation_laws(field: UnifiedField) -> None:
    """Verify conservation laws are respected."""
    # Energy conservation
    psi = field.compute_basis_function(n=0, E=100.0)
    E_initial = field.compute_energy_density(psi)
    
    # Evolve field
    t_points = np.linspace(0, 10, 100)
    evolution = field.evolve_field(psi, t_points)
    
    # Check energy conservation
    E_final = field.compute_energy_density(evolution['field_values'][-1])
    if abs(E_final - E_initial) > 1e-6 * abs(E_initial):
        raise ValidationError("Energy conservation violated")

if __name__ == '__main__':
    try:
        validate_framework()
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise 