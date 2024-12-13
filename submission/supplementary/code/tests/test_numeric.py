"""Tests for numerical computation utilities."""

import pytest
import numpy as np
from sympy import Symbol, exp
from core.numeric import integrate_phase_space, solve_field_equations
from core.types import NumericValue, Energy, RealValue
from core.errors import ComputationError

@pytest.mark.numeric
class TestPhaseSpaceIntegration:
    """Test phase space integration functionality."""
    
    def test_basic_integration(self):
        """Test basic integration of simple function."""
        x = Symbol('x')
        f = x**2
        result = integrate_phase_space(f, (0, 1))
        assert isinstance(result, NumericValue)
        assert np.isclose(result.value, 1/3, rtol=1e-6)
        assert result.uncertainty is not None
    
    def test_gaussian_integration(self):
        """Test integration of Gaussian function."""
        x = Symbol('x')
        f = exp(-x**2)
        result = integrate_phase_space(f, (-np.inf, np.inf))
        assert np.isclose(result.value, np.sqrt(np.pi), rtol=1e-6)
    
    def test_precision_control(self):
        """Test integration with different precision requirements."""
        x = Symbol('x')
        f = x**3
        
        # High precision
        result_high = integrate_phase_space(f, (0, 1), precision=1e-8)
        # Normal precision
        result_normal = integrate_phase_space(f, (0, 1), precision=1e-4)
        
        assert result_high.uncertainty < result_normal.uncertainty
    
    def test_error_handling(self):
        """Test error handling in integration."""
        x = Symbol('x')
        
        # Invalid limits
        with pytest.raises(ComputationError):
            integrate_phase_space(x, (1, 0))
        
        # Invalid function
        with pytest.raises(ComputationError):
            integrate_phase_space(None, (0, 1))

@pytest.mark.numeric
class TestFieldEquations:
    """Test field equation solver."""
    
    def test_field_equation_solution(self):
        """Test solving basic field equations."""
        config = {
            'mass': 125.0,
            'coupling': 0.1
        }
        energy = Energy(1000.0)  # 1 TeV
        
        result = solve_field_equations(config, energy)
        assert isinstance(result, NumericValue)
        assert result.value > 0
        assert result.uncertainty is not None
    
    def test_convergence_check(self):
        """Test convergence of field equation solutions."""
        config = {
            'mass': 125.0,
            'coupling': 0.1
        }
        energy = Energy(1000.0)
        
        result = solve_field_equations(config, energy, max_iter=1000)
        assert result.uncertainty < 0.01 * abs(result.value)  # 1% precision
    
    def test_invalid_config(self):
        """Test error handling for invalid configurations."""
        with pytest.raises(ComputationError):
            solve_field_equations({}, Energy(100.0))
        
        with pytest.raises(ComputationError):
            solve_field_equations({'mass': -1.0}, Energy(100.0))

    @pytest.mark.parametrize('energy', [
        Energy(100.0),
        Energy(1000.0),
        Energy(10000.0)
    ])
    def test_energy_scaling(self, energy):
        """Test solution behavior at different energy scales."""
        config = {
            'mass': 125.0,
            'coupling': 0.1
        }
        result = solve_field_equations(config, energy)
        assert result.value != 0 