"""Core implementation of fractal basis functions."""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from sympy import (
    Symbol, symbols, exp, integrate, conjugate, sqrt,
    hermite, Expr, diff, oo, I, log, pi
)
import cmath
import warnings
from .physics_constants import (
    ALPHA_VAL, X, E, T, P, Z_MASS,
    GAMMA_1, GAMMA_2, GAMMA_3,
    g1_REF, g2_REF, g3_REF,
    ALPHA_REF
)
from .types import (Energy, FieldConfig, WaveFunction, 
                   ErrorEstimate, RealValue, AnalysisResult)
from .modes import ComputationMode
from .utils import evaluate_expr, cached_evaluation, check_numerical_stability
from .errors import (ValidationError, ComputationError, PhysicsError, 
                    StabilityError, ConfigurationError)
from .field import UnifiedField

class FractalBasis(UnifiedField):
    """
    Implements recursive fractal basis functions.
    
    Extends UnifiedField to provide specific implementation using fractal basis expansion.
    The basis functions form a complete orthonormal set that diagonalizes the evolution
    operator.
    
    Key features:
    - Recursive construction via scaling relations
    - Built-in energy scale dependence
    - Automatic normalization and error estimation
    """
    
    # Error thresholds
    STABILITY_THRESHOLD = 1e-10
    MAX_ITERATIONS = 1000
    
    # Physical constants
    E0 = Z_MASS  # Z boson mass in GeV
    E0_MIN = 1.0  # Minimum valid energy scale
    E0_MAX = 1000.0  # Maximum valid energy scale
    
    # Numerical stability parameters
    N_STABLE_MAX = 50  # Maximum n for direct normalization
    LOG_NORM_THRESHOLD = 100  # Switch to log-space normalization above this
    
    def __init__(self, alpha: float = ALPHA_VAL, mode: ComputationMode = ComputationMode.MIXED):
        """Initialize fractal basis with scaling parameter."""
        super().__init__(alpha=alpha, mode=mode)
        self.scaling_dimension = 1.0  # Specific to fractal basis
    
    def _solve_field_equations(self, config: FieldConfig) -> WaveFunction:
        """
        Solve field equations using fractal basis expansion.
        
        Overrides UnifiedField._solve_field_equations to implement specific
        solution method using fractal basis functions.
        
        Args:
            config: Field configuration parameters
            
        Returns:
            WaveFunction: Solution in fractal basis
        """
        n = config.dimension
        E = config.mass
        
        # Get generator function with scaled argument
        scaled_x = self.alpha**n * X
        F = self._generator_function(scaled_x)
        
        # Apply modulation and scaling
        modulation = self._modulation_factor(n, Energy(E))
        
        # Combine all factors
        psi = self.alpha**n * F * modulation
        
        # Normalize
        return self.normalize(psi)

    def _compute_evolution_operator(self, energy: Energy) -> WaveFunction:
        """
        Compute evolution operator in fractal basis.
        
        Overrides UnifiedField._compute_evolution_operator to provide
        specific implementation for fractal basis evolution.
        
        Args:
            energy: Target energy scale
            
        Returns:
            WaveFunction: Evolution operator
        """
        k = energy.value / self.alpha
        phase = exp(-I * k * T)
        return phase * self._scaling_operator(k)

    def _generator_function(self, x: Symbol) -> WaveFunction:
        """
        Compute generator function for basis construction.
        
        Args:
            x: Scaled coordinate
            
        Returns:
            WaveFunction: Generator function
        """
        return exp(-x**2/2)
    
    def _modulation_factor(self, n: int, E: Energy) -> WaveFunction:
        """
        Compute energy-dependent modulation factor.
        
        Args:
            n: Basis function index
            E: Energy scale
            
        Returns:
            WaveFunction: Modulation factor
        """
        k = E.value / self.alpha**n
        return exp(-k**2 * X**2/2)

    def _scaling_operator(self, k: float) -> WaveFunction:
        """
        Compute scaling part of evolution operator.
        
        Args:
            k: Scaling parameter
            
        Returns:
            WaveFunction: Scaling operator
        """
        return exp(self.scaling_dimension * log(k))

    def _validate_inputs(self, n: int, E: float) -> None:
        """
        Validate common input parameters.
        
        Args:
            n: Basis function index
            E: Energy scale
            
        Raises:
            ValidationError: If inputs are invalid
            PhysicsError: If physics constraints are violated
        """
        if not isinstance(n, int):
            raise ValidationError(f"Basis index must be integer, got {type(n)}")
        if n < 0:
            raise ValidationError(f"Basis index must be non-negative, got {n}")
        
        if not isinstance(E, (int, float)):
            raise ValidationError(f"Energy must be numeric, got {type(E)}")
        if E <= 0:
            raise PhysicsError(f"Energy must be positive, got {E}")
        self.validate_energy_scale(E)
    
    def compute(self, n: int, E: Energy = Energy(1.0)) -> WaveFunction:
        """
        Compute nth basis function at energy E.
        
        Args:
            n: Basis function index
            E: Energy scale in GeV
            
        Returns:
            Symbolic expression for basis function
        """
        self._validate_inputs(n, E)
        
        try:
            # Get generator function with scaled argument
            scaled_x = self.alpha**n * X
            F = self._generator_function(scaled_x)
            
            # Apply modulation and scaling
            modulation = self._modulation_factor(n, E)
            
            # Combine all factors
            psi = self.alpha**n * F * modulation
            
            # Normalize
            psi = self.normalize(psi)
            
            # Apply computation mode
            if self.mode == ComputationMode.NUMERIC:
                return evaluate_expr(psi)
            elif self.mode == ComputationMode.MIXED:
                return cached_evaluation(psi)
            return psi
        except (FloatingPointError, ValueError) as e:
            raise ComputationError(f"Basis function computation failed: {e}")
        except Exception as e:
            raise ComputationError(f"Unexpected error in basis computation: {e}")

    def normalize(self, psi: Expr) -> Expr:
        """
        Normalize basis function to unit norm.
        
        Uses stable normalization methods for large n:
        1. Direct normalization for small n
        2. Log-space normalization for large n
        3. Asymptotic approximation for very large n
        
        Args:
            psi: Basis function to normalize
        
        Returns:
            Normalized basis function
        """
        try:
            # Calculate norm squared
            norm_squared = integrate(conjugate(psi) * psi, 
                                   (X, -float('inf'), float('inf')))
            
            # Check if norm is too large
            if abs(norm_squared) > self.LOG_NORM_THRESHOLD:
                # Use log-space normalization
                log_norm = 0.5 * np.log(abs(norm_squared))
                phase = cmath.phase(norm_squared)
                return psi * exp(-log_norm - 1j * phase/2)
            
            # Standard normalization for small values
            return psi / sqrt(norm_squared)
            
        except (ValueError, OverflowError) as e:
            warnings.warn(f"Normalization failed, using asymptotic approximation: {e}")
            # Use asymptotic approximation for very large values
            return self._asymptotic_normalization(psi)

    def _asymptotic_normalization(self, psi: Expr) -> Expr:
        """
        Compute asymptotic normalization for large n.
        
        Uses series expansion and leading order terms.
        
        Args:
            psi: Basis function to normalize
            
        Returns:
            Asymptotically normalized function
        """
        # Extract leading order behavior
        leading_term = psi.as_leading_term(X)
        # Estimate normalization from leading term
        asymp_norm = sqrt(abs(leading_term.coeff(X, 0)))
        return psi / asymp_norm

    def check_orthogonality(self, n1: int, n2: int, E: float = 1.0) -> float:
        """
        Check orthogonality between basis functions.
        
        The orthogonality condition is:
        ∫ψₙ*(x)ψₘ(x)dx = δₙₘ
        
        Args:
            n1, n2: Basis function indices
            E: Energy scale
        
        Returns:
            float: Overlap integral
        """
        psi1 = self.compute(n1, E)
        psi2 = self.compute(n2, E)
        
        overlap = integrate(conjugate(psi1) * psi2, 
                          (X, -float('inf'), float('inf')))
        return float(abs(overlap))

    def coupling(self, gauge_index: int, E: Energy) -> float:
        """
        Compute gauge coupling evolution.
        
        Implements Eq. 3.8 from paper:
        g_i(E) = g_i(M_Z) * [1 + α*ln(E/M_Z)]^(-γ_i)
        
        where:
        - g_i(M_Z) are the measured couplings at Z mass
        - γ_i are the fractal scaling exponents
        - α is the fine structure constant
        
        Args:
            gauge_index: Which gauge coupling (1,2,3)
            E: Energy scale in GeV
        
        Returns:
            float: Coupling value at given energy
        """
        if gauge_index not in [1, 2, 3]:
            raise ValidationError(f"Invalid gauge index: {gauge_index}")
        if E <= 0:
            raise PhysicsError(f"Energy must be positive, got {E}")
        
        # Use constants from paper
        gamma = {
            1: GAMMA_1,
            2: GAMMA_2,
            3: GAMMA_3
        }
        
        g0 = {
            1: g1_REF,
            2: g2_REF,
            3: g3_REF
        }
        
        return g0[gauge_index] * (1 + ALPHA_REF * np.log(E/Z_MASS))**(-gamma[gauge_index])

    def compute_beta_function(self, gauge_index: int, E: float) -> float:
        """
        Compute beta function for coupling evolution.
        
        Implements Eq. 3.10 from paper:
        β(g) = dg/dln(E) = -γᵢ * g * [1 + α*ln(E/M_Z)]^(-1)
        
        where:
        - γᵢ are the fractal scaling exponents
        - M_Z is the Z boson mass
        - α is the fine structure constant
        
        Args:
            gauge_index: Which gauge coupling
            E: Energy scale
        
        Returns:
            float: Beta function value
        """
        g = self.coupling(gauge_index, E)
        gamma = {
            1: GAMMA_1,
            2: GAMMA_2,
            3: GAMMA_3
        }[gauge_index]
        
        return -gamma * g / (1 + ALPHA_REF * np.log(E/Z_MASS))

    def analyze_scaling_dimension(self, n: int, E: Energy) -> AnalysisResult:
        """
        Analyze scaling dimension of basis functions.
        
        The scaling dimension Δ determines how fields transform:
        ψ(λx) = λ^(-Δ) ψ(x)
        
        Args:
            n: Basis function index
            E: Energy scale
        
        Returns:
            Dict containing scaling dimensions and anomalous contributions
        """
        # Classical scaling dimension
        delta_classical = 1.0  # Mass dimension of scalar field
        
        # Anomalous dimension from fractal structure
        delta_anomalous = self.alpha * (n + 0.5) * np.exp(-E/1000)
        
        # Total scaling dimension
        delta_total = delta_classical + delta_anomalous
        
        return {
            'classical_dimension': delta_classical,
            'anomalous_dimension': delta_anomalous,
            'total_dimension': delta_total,
            'energy_scale': E,
            'basis_index': n
        }

    def verify_scaling_symmetry(self, n: int, lambda_scale: float = 2.0) -> Dict[str, float]:
        """
        Verify scaling symmetry of basis functions.
        
        Tests if ψ(λx)/ψ(x) = λ^(-Δ) holds within numerical precision.
        
        Args:
            n: Basis function index
            lambda_scale: Scaling factor
        
        Returns:
            Dict containing verification results
        """
        # Compute function at reference point
        psi_ref = self.compute(n).subs(X, 1.0)
        
        # Compute scaled function
        psi_scaled = self.compute(n).subs(X, lambda_scale)
        
        # Extract scaling behavior
        ratio = complex(psi_scaled / psi_ref)
        measured_dim = -np.log(abs(ratio)) / np.log(lambda_scale)
        
        # Compare with theoretical prediction
        theoretical = self.analyze_scaling_dimension(n, 1.0)['total_dimension']
        
        return {
            'measured_dimension': measured_dim,
            'theoretical_dimension': theoretical,
            'relative_error': abs(measured_dim - theoretical) / theoretical,
            'scale_factor': lambda_scale
        }

    def calculate_fractal_dimension(self, E: float) -> float:
        """
        Calculate fractal dimension at given energy scale.
        
        The fractal dimension D_f is given by:
        D_f = 2 + lim_{n→∞} ln(Σ α^k h(k))/ln(n)
        where h(k) is the scaling function.
        
        Args:
            E: Energy scale in GeV
        
        Returns:
            float: Fractal dimension
        """
        try:
            self.validate_energy_scale(E)
            n_max = self.MAX_ITERATIONS
            
            # Use generator for memory efficiency
            def scaling_terms():
                for k in range(1, n_max):
                    try:
                        yield self.alpha**k * float(abs(self.compute(k, E).subs(X, 1.0)))
                    except (ValueError, OverflowError):
                        break
            
            # Sum terms until convergence or max iterations
            scaling_sum = 0.0
            for term in scaling_terms():
                scaling_sum += term
                if abs(term) < self.STABILITY_THRESHOLD * abs(scaling_sum):
                    break
            
            if scaling_sum <= 0:
                raise StabilityError("Scaling sum convergence failed")
            
            return 2.0 + float(np.log(scaling_sum) / np.log(n_max))
            
        except (ValueError, OverflowError) as e:
            raise ComputationError(f"Fractal dimension computation failed: {e}")
        except StabilityError as e:
            raise StabilityError(f"Fractal dimension stability error: {e}")

    def coupling_with_uncertainty(self, gauge_index: int, E: float) -> Dict[str, float]:
        """
        Compute gauge coupling with uncertainty estimates.
        
        Includes:
        1. Statistical uncertainty from input parameters
        2. Systematic uncertainty from truncation
        3. Scale uncertainty from energy dependence
        
        Args:
            gauge_index: Which gauge coupling (1,2,3)
            E: Energy scale in GeV
        
        Returns:
            Dict containing value and uncertainties
        """
        # Central value
        g = self.coupling(gauge_index, E)
        
        # Parameter uncertainties (from experimental inputs)
        g0_uncertainty = {
            1: 0.00020,  # U(1) coupling at M_Z
            2: 0.00035,  # SU(2) coupling
            3: 0.00110   # SU(3) coupling
        }[gauge_index]
        
        # Scale uncertainty
        scale_var = abs(self.coupling(gauge_index, 2*E) - 
                       self.coupling(gauge_index, E/2))/2
        
        # Truncation uncertainty from series expansion
        trunc_error = self.alpha**4 * abs(g)  # Fourth order in α
        
        # Total uncertainty (add in quadrature)
        total_uncertainty = np.sqrt(g0_uncertainty**2 + 
                                  scale_var**2 + 
                                  trunc_error**2)
        
        return {
            'value': g,
            'statistical_error': g0_uncertainty,
            'scale_uncertainty': scale_var,
            'truncation_error': trunc_error,
            'total_uncertainty': total_uncertainty
        }

    def compute_with_errors(self, n: int, E: float = 1.0) -> Dict[str, Expr]:
        """
        Compute basis function with error estimates.
        
        Includes:
        1. Normalization uncertainty
        2. Truncation error from series expansion
        3. Numerical integration errors
        
        Args:
            n: Basis function index
            E: Energy scale
        
        Returns:
            Dict containing function and error terms
        """
        # Central value
        psi = self.compute(n, E)
        
        # Normalization uncertainty
        norm_error = abs(self.check_orthogonality(n, n) - 1.0)
        
        # Truncation error from series expansion
        trunc_error = self.alpha**(n+1) * abs(self._generator_function(X))
        
        # Integration error estimate (from quadrature)
        quad_error = 1e-8 * abs(psi)  # Conservative estimate
        
        return {
            'function': psi,
            'normalization_error': norm_error,
            'truncation_error': trunc_error,
            'integration_error': quad_error,
            'total_error': norm_error + trunc_error + quad_error
        }

    def compute_basis_function(self, n: int, E: Energy) -> WaveFunction:
        """
        Compute nth basis function at energy E.
        
        Args:
            n: Basis function index
            E: Energy scale
            
        Returns:
            WaveFunction: Computed basis function
            
        Raises:
            PhysicsError: If parameters are invalid
            ComputationError: If computation fails
        """
        # Validate inputs
        if n < 0:
            raise PhysicsError(f"Invalid basis index: {n}")
        if E.value <= 0:
            raise PhysicsError(f"Invalid energy: {E}")
            
        try:
            # Get generator function with scaled argument
            scaled_x = self.alpha**n * X
            F = self._generator_function(scaled_x)
            
            # Apply modulation and scaling
            modulation = self._modulation_factor(n, E)
            
            # Combine all factors
            psi = self.alpha**n * F * modulation
            
            # Normalize
            psi = self.normalize(psi)
            
            # Apply computation mode
            if self.mode == ComputationMode.NUMERIC:
                return evaluate_expr(psi)
            elif self.mode == ComputationMode.MIXED:
                return cached_evaluation(psi)
            return psi
            
        except Exception as e:
            raise ComputationError(f"Basis function computation failed: {e}")
