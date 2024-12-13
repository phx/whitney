"""Tests for numerical methods and stability."""

import pytest
import numpy as np
from core.numeric import integrate_phase_space, solve_field_equations
from core.stability import check_convergence, analyze_perturbation
from core.precision import set_precision, get_precision_info

@pytest.mark.numeric
class TestNumericalMethods:
    """Test numerical integration and solving methods."""
    
    def test_phase_space_integration(self, numeric_precision):
        """Test phase space integration accuracy."""
        def test_integrand(p, q):
            return np.exp(-((p-q)/10.0)**2)
        
        result = integrate_phase_space(
            test_integrand,
            limits=[(0, 100), (0, 100)],
            **numeric_precision
        )
        
        # Known analytic result for Gaussian integral
        expected = 100.0  # Approximate for large limits
        assert np.isclose(result, expected, 
                         rtol=numeric_precision['rtol'],
                         atol=numeric_precision['atol'])
    
    def test_field_equation_solver(self, standard_field, numeric_precision):
        """Test field equation numerical solver."""
        initial_conditions = {
            'field': np.zeros(10),
            'derivative': np.ones(10),
            'time': np.linspace(0, 1, 10)
        }
        
        solution = solve_field_equations(
            field=standard_field,
            initial=initial_conditions,
            **numeric_precision
        )
        
        # Test energy conservation
        energies = solution['energy']
        assert np.allclose(energies, energies[0], 
                          rtol=numeric_precision['rtol'])

@pytest.mark.numeric
class TestNumericalStability:
    """Test numerical stability and convergence."""
    
    def test_convergence_criteria(self, numeric_precision):
        """Test convergence checking."""
        # Generate test sequence
        sequence = [1.0 + 0.1/n for n in range(1, 11)]
        
        # Should converge to 1.0
        converged = check_convergence(
            sequence,
            target=1.0,
            **numeric_precision
        )
        assert converged
        
        # Divergent sequence should fail
        divergent = [n**2 for n in range(1, 11)]
        assert not check_convergence(
            divergent,
            target=1.0,
            **numeric_precision
        )
    
    def test_perturbation_stability(self, standard_field, numeric_precision):
        """Test stability under perturbations."""
        # Base configuration
        base_config = {
            'mass': 125.0,
            'coupling': 0.1
        }
        
        # Perturbed configurations
        perturbations = [
            {'mass': 125.1, 'coupling': 0.1},
            {'mass': 125.0, 'coupling': 0.11},
            {'mass': 124.9, 'coupling': 0.09}
        ]
        
        stability = analyze_perturbation(
            field=standard_field,
            base_config=base_config,
            perturbations=perturbations,
            **numeric_precision
        )
        
        assert stability['stable']
        assert stability['max_deviation'] < numeric_precision['stability_threshold']

@pytest.mark.numeric
class TestPrecisionControl:
    """Test precision control and validation."""
    
    def test_precision_settings(self):
        """Test precision control functions."""
        # Set custom precision
        custom_precision = {
            'float_precision': 'float64',
            'int_precision': 'int64',
            'min_precision': 1e-8
        }
        set_precision(**custom_precision)
        
        # Verify settings
        current = get_precision_info()
        assert current['float_precision'] == custom_precision['float_precision']
        assert current['min_precision'] == custom_precision['min_precision'] 