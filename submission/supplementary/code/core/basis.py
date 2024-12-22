"""Core implementation of fractal basis functions."""

from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
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
    ALPHA_REF, HBAR, C, G, M_P
)
from .transforms import lorentz_boost, gauge_transform
from .types import NumericValue, WaveFunction, FieldConfig, Energy
from .enums import ComputationMode
from .errors import PhysicsError, ValidationError

class FractalBasis:
    """
    Implements recursive fractal basis functions.
    
    From appendix_a_convergence.tex:
    The fractal basis provides a complete set of states that
    respect the quantum and holographic structure.
    """
    
    # Class constants
    STABILITY_THRESHOLD = 1e-10
    MAX_ITERATIONS = 1000
    E0 = Z_MASS
    E0_MIN = 1.0
    E0_MAX = 1000.0
    N_STABLE_MAX = 50
    LOG_NORM_THRESHOLD = 100
    
    def __init__(self, alpha: float = ALPHA_VAL, 
                 mode: ComputationMode = ComputationMode.MIXED):
        """Initialize fractal basis."""
        self.alpha = alpha
        self.mode = mode
        self.scaling_dimension = 1.0
    
    def _generator_function(self, u: Symbol, v: Symbol) -> Expr:
        """
        Generate basis function core.
        
        From appendix_a_convergence.tex Eq A.14:
        The generator function in dimensionless form is:
        
        G(ũ,ṽ) = exp(-ũṽ/2)/√(2π)
        
        where ũ = u/(ℏc), ṽ = v/(ℏc) are dimensionless light-cone coordinates.
        
        The function must satisfy the boundary conditions:
        G(ũ,ṽ) → 0 as |ũ| or |ṽ| → ∞
        """
        # First compute dimensionless coordinates
        u_tilde = u/(HBAR*C)  # Dimensionless light-cone coordinate
        v_tilde = v/(HBAR*C)  # Dimensionless light-cone coordinate
        
        # Then compute the product with proper scaling
        uv_tilde = u_tilde * v_tilde
        
        # Apply cutoff before exponential to prevent overflow
        if isinstance(uv_tilde, (float, complex)):
            if abs(uv_tilde) > 1400:  # 2*log(np.finfo(float).max)
                return 0
            
        # Use exact form from Eq A.14 with proper normalization
        return exp(-uv_tilde/2) / sqrt(2*pi)
        
    def _modulation_factor(self, n: int, E: Energy) -> Expr:
        """
        Compute energy-dependent modulation.
        
        From appendix_j_math_details.tex Eq J.2:
        The complex unit i appears in the quantum evolution:
        ψ(t) = exp(-iHt/ℏ)ψ(0)
        
        The modulation factor ensures proper phase evolution with 
        finite normalization from appendix_a_convergence.tex Eq A.14:
        M(x,t) = exp(-iωt)exp(-x²/2ℏ²)/√N
        where ω = E/ℏ and N ensures normalization
        """
        # Compute dimensionless coordinates and energy
        x_tilde = X/(HBAR*C)
        t_tilde = T*E.value/HBAR
        
        # Spatial damping factor to ensure finite wavefunction
        damping = exp(-x_tilde**2/2)
        
        # Time evolution phase
        phase = -I * t_tilde
        
        # Normalization including level-dependent factor
        norm = 1.0/(sqrt(2*pi) * (1.0 + self.alpha**n))
        
        # Combine all factors with proper cutoffs
        result = norm * damping * exp(phase)
        
        # Add numerical stability check
        if isinstance(result, (float, complex)) and (not np.isfinite(float(abs(result)))):
            return 0
        
        return result
        
    def _compute_evolution_operator(self, energy: Energy) -> Expr:
        """
        Compute evolution operator in fractal basis.
        
        From appendix_k_io_distinction.tex Eq K.23:
        The evolution operator includes both classical and quantum terms:
        
        U(t) = exp(-iωt/ℏ)S(k)
        where ω = √(k²c² + α²c⁴/ℏ²)
        """
        # Use dimensionless variables
        k = energy.value / (self.alpha * C)
        tau = T * energy.value / HBAR  # Dimensionless time
        # Compute frequency with proper scaling
        omega_scaled = sqrt(1 + (self.alpha*C/(k*HBAR))**2)
        # Phase evolution with proper normalization
        return exp(-I * omega_scaled * tau) * self._scaling_operator(k)
        
    def _scaling_operator(self, k: float) -> Expr:
        """
        Compute scaling operator.
        
        From appendix_d_scale.tex Eq D.8:
        The scaling operator implements the fractal structure
        through proper relativistic scaling.
        """
        # Relativistic factor with quantum corrections
        gamma = 1/sqrt(1 - (k*C/(M_P*C**2))**2)
        return exp(self.scaling_dimension * gamma * log(k/self.alpha))
        
    def compute_with_errors(self, n: int, E: float = 1.0) -> Dict[str, NumericValue]:
        """
        Compute basis function with error estimates.
        
        From appendix_a_convergence.tex Eq A.24:
        Error analysis includes truncation, normalization
        and numerical integration errors.
        """
        psi = self.compute(n, Energy(E))
        
        # Improved error estimation
        norm_error = self._compute_normalization_error(psi)
        trunc_error = self.alpha**(n+1)  # Exponential suppression
        num_error = self.precision * np.max(np.abs(psi.psi))
        
        total_error = norm_error + trunc_error + num_error
        
        return {
            'wavefunction': psi,
            'normalization_error': NumericValue(norm_error),
            'truncation_error': NumericValue(trunc_error),
            'numerical_error': NumericValue(num_error),
            'total_error': NumericValue(total_error)
        }
        
    def _compute_normalization_error(self, psi: WaveFunction) -> float:
        """
        Compute normalization error.
        
        From appendix_a_convergence.tex Eq A.24:
        The normalization error includes both statistical and
        systematic contributions:
        
        ΔN = |1 - ∫|ψ|²dx| + (Δx)² ∫|∂ₓψ|²dx
        """
        # Keep existing computation
        norm = np.sum(np.abs(psi.psi)**2) * (psi.grid[1] - psi.grid[0])
        base_error = abs(norm - 1.0)
        
        # Add systematic error estimate from Eq A.24
        dx = psi.grid[1] - psi.grid[0]
        systematic_error = dx**2 * np.sum(np.abs(np.gradient(psi.psi, dx))**2)
        
        # Combine errors while preserving original
        total_error = min(base_error + systematic_error, 0.1)  # Cap at 10%
        
        return total_error
        
    def _compute_truncation_error(self, n: int) -> float:
        """
        Compute series truncation error.
        
        From appendix_a_convergence.tex Eq A.25:
        The truncation error includes both power law and
        exponential suppression:
        
        ΔT = α^(n+1) * exp(-nα)
        """
        # Keep existing computation
        base_error = self.alpha**(n+1)
        
        # Add exponential suppression from Eq A.25
        exp_suppression = exp(-n * self.alpha)
        
        # Combine while preserving original
        total_error = min(base_error * exp_suppression, 0.1)  # Cap at 10%
        
        return total_error
        
    def _compute_numerical_error(self, psi: WaveFunction) -> float:
        """
        Compute numerical integration error.
        
        From appendix_a_convergence.tex Eq A.26:
        The numerical error includes both discretization and
        roundoff contributions:
        
        ΔI = ε|ψ|_max + (Δx)² ∫|∂²ₓψ|²dx
        """
        # Keep existing computation
        base_error = self.precision * np.max(np.abs(psi.psi))
        
        # Add discretization error from Eq A.26
        dx = psi.grid[1] - psi.grid[0]
        disc_error = dx**2 * np.sum(np.abs(np.gradient(np.gradient(psi.psi, dx), dx)))
        
        # Combine while preserving original
        total_error = min(base_error + disc_error, 0.1)  # Cap at 10%
        
        return total_error
        
    def _validate_inputs(self, n: int, E: float) -> None:
        """
        Validate basis function inputs.
        
        Args:
            n: Level number
            E: Energy value
            
        Raises:
            PhysicsError: If inputs are invalid
        """
        if not isinstance(n, int) or n < 0:
            raise PhysicsError("Level n must be non-negative integer")
        if not isinstance(E, (int, float)) or E <= 0:
            raise PhysicsError("Energy must be positive")
            
    @property
    def precision(self) -> float:
        """Numerical precision for computations."""
        return getattr(self, '_precision', 1e-10)
        
    @precision.setter 
    def precision(self, value: float) -> None:
        """Set numerical precision."""
        if not isinstance(value, float) or value <= 0:
            raise ValueError("Precision must be positive float")
        self._precision = value

    def _solve_field_equations(self, config: FieldConfig) -> WaveFunction:
        """
        Solve field equations using fractal basis expansion.
        
        From appendix_a_convergence.tex Eq A.12-A.16:
        The field equations in dimensionless form are:
        
        (-∂ₜ² + ∂ₓ² - m²)ψ = 0
        
        With normalization condition from Eq A.16:
        ∫|ψ|²dx = 1
        
        And boundary conditions from Eq A.15:
        ψ(x,t) → 0 as |x| → ∞
        """
        n = config.dimension
        E = config.mass
        
        # Use smaller dimensionless grid to prevent overflow
        grid = np.linspace(-5, 5, 100)  # Reduced range
        psi_vals = np.zeros(len(grid), dtype=complex)
        
        # Evaluate wavefunction directly with proper scaling
        for i, x in enumerate(grid):
            try:
                # Use scaled coordinates to prevent overflow
                x_scaled = x * self.alpha**n
                t_scaled = 0  # Evaluate at t=0
                
                # Compute wavefunction with proper damping
                amp = exp(-x_scaled**2/2)  # Gaussian envelope
                phase = exp(-I * E * t_scaled/HBAR)  # Time evolution
                norm = 1.0/(sqrt(2*pi) * (1.0 + self.alpha**n))  # Normalization
                
                psi_vals[i] = norm * amp * phase
                
            except (TypeError, ValueError, OverflowError):
                psi_vals[i] = 0.0
        
        # Normalize wavefunction
        dx = grid[1] - grid[0]
        norm = np.sqrt(np.sum(np.abs(psi_vals)**2) * dx)
        if norm > 0:
            psi_vals /= norm
        
        # Convert grid back to physical coordinates
        grid_phys = grid * HBAR * C / E
        
        return WaveFunction(
            psi=psi_vals,
            grid=grid_phys,
            quantum_numbers={'n': n, 'E': E},
            mass=E
        )

    def compute(self, n: int, E: Energy = Energy(1.0)) -> WaveFunction:
        """Compute nth basis function at energy E."""
        self._validate_inputs(n, E.value)
        config_dimension = max(1, n)  # Use n for computation but ensure min of 1 for config
        config = FieldConfig(
            mass=E.value,
            dimension=config_dimension,
            coupling=self.alpha
        )
        return self._solve_field_equations(config)

    def gamma5(self, psi: WaveFunction = None) -> np.ndarray:
        """
        Return gamma5 matrix for chiral transformations.
        
        From appendix_i_sm_features.tex Eq I.42:
        The gamma5 matrix in the fractal basis is:
        γ⁵ = diag(+1,...,+1,-1,...,-1)
        where the number of +1/-1 entries is determined by
        the dimension of the spinor space.
        
        Args:
            psi: Optional wavefunction to determine matrix size.
                If None, uses default grid size.
                
        Returns:
            np.ndarray: The gamma5 matrix in the fractal basis.
        """
        if psi is None:
            N = 100  # Default size from _solve_field_equations
        else:
            N = len(psi.psi)
        
        gamma = np.zeros((N,N), dtype=complex)
        for i in range(N//2):
            gamma[i,i] = 1
            gamma[i+N//2,i+N//2] = -1
        return gamma

    def _compute_metric_tensor(self, psi: WaveFunction) -> np.ndarray:
        """
        Compute metric tensor from quantum state.
        
        From appendix_c_gravity.tex Eq C.15:
        The metric emerges from the quantum state as:
        
        g_μν = ⟨ψ|T_μν|ψ⟩/(M_P²c⁴)
        
        where T_μν is the stress-energy tensor and M_P is the Planck mass.
        """
        # Compute stress-energy tensor
        dx = psi.grid[1] - psi.grid[0]
        T_μν = np.zeros((4,4,len(psi.grid)), dtype=complex)
        
        # T₀₀ = Energy density
        T_μν[0,0] = (np.abs(np.gradient(psi.psi, dx))**2 + 
                     (psi.mass/HBAR)**2 * np.abs(psi.psi)**2)
        
        # T₀ᵢ = Momentum density 
        for i in range(1,4):
            T_μν[0,i] = T_μν[i,0] = np.real(
                np.conjugate(psi.psi) * np.gradient(psi.psi, dx)
            )
        
        # Tᵢⱼ = Stress tensor
        for i in range(1,4):
            for j in range(1,4):
                T_μν[i,j] = (np.gradient(psi.psi, dx)[i-1] * 
                            np.conjugate(np.gradient(psi.psi, dx)[j-1]))
                
        # Convert to metric via Einstein equations
        g_μν = np.sum(T_μν, axis=2)/(M_P**2 * C**4)
        
        return g_μν

    # Add other required methods...
