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
    ALPHA_REF, HBAR, C, G, M_PLANCK
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
        
        From appendix_a_convergence.tex Eq A.12:
        The generator function determines the fractal structure
        through light-cone coordinates:
        
        G(u,v) = exp(-uv/2ℏc)/√(2πℏc)
        """
        # Lorentz invariant combination with proper scaling
        s = u*v  # Spacetime interval
        # Use dimensionless form to avoid overflow
        s_scaled = s/(2*HBAR*C)
        return exp(-s_scaled) / sqrt(2*pi)
        
    def _modulation_factor(self, n: int, E: Energy) -> Expr:
        """
        Compute energy-dependent modulation.
        
        From appendix_d_scale.tex Eq D.15:
        The modulation factor ensures proper energy scaling:
        
        M(x,t) = exp(-k²(x² - c²t²)/(2ℏ²))/√k
        where k = E/(αⁿc)
        """
        # Use dimensionless wavevector to avoid overflow
        k = E.value / (self.alpha**n * C)
        # Scale coordinates properly
        x_scaled = X * sqrt(k/(HBAR*C))
        t_scaled = T * C * sqrt(k/(HBAR*C))
        # Add proper normalization
        return exp(-(x_scaled**2 - t_scaled**2)/2) / sqrt(k)
        
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
        gamma = 1/sqrt(1 - (k*C/(M_PLANCK*C**2))**2)
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
        
        From appendix_a_convergence.tex Eq A.12:
        The field equations in dimensionless form are:
        
        (-∂ₜ² + ∂ₓ² - m²)ψ = 0
        
        With the scaling relations from appendix_d_scale.tex Eq D.8:
        x → x/(mᵢc), t → t/(mᵢc²), ψ → ψ√(mᵢc)
        """
        n = config.dimension
        E = config.mass
        
        # Use dimensionless variables from Eq D.8
        tau = T * E / (HBAR * C)  # Dimensionless time
        xi = X * E / (HBAR * C)   # Dimensionless position
        
        # Compute light-cone coordinates (Eq A.13)
        u = (tau + xi)/sqrt(2)
        v = (tau - xi)/sqrt(2)
        
        # Scale coordinates with quantum corrections (Eq D.15)
        scaled_u = self.alpha**n * u 
        scaled_v = self.alpha**n * v
        
        # Get generator function with proper normalization (Eq A.14)
        F = self._generator_function(scaled_u, scaled_v)
        
        # Apply modulation with proper phase (Eq D.16)
        modulation = self._modulation_factor(n, Energy(E))
        
        # Combine all factors with correct normalization (Eq A.15)
        psi = (self.alpha**n * F * modulation * exp(-I*tau)) / sqrt(2*pi)
        
        # Create dimensionless grid (Eq D.8)
        grid = np.linspace(-5, 5, 100)  # Reduced range to avoid overflow
        psi_vals = np.zeros(len(grid), dtype=complex)
        
        # Evaluate with proper error handling
        for i, x in enumerate(grid):
            try:
                # Convert to physical coordinates (Eq D.8)
                x_phys = x * HBAR * C / E
                val = complex(psi.subs({X: x_phys, T: 0}))
                if np.isfinite(val):
                    psi_vals[i] = val
            except (TypeError, ValueError):
                continue
            
        # Normalize with proper integration measure (Eq A.16)
        dx = grid[1] - grid[0]
        norm = np.sqrt(np.sum(np.abs(psi_vals)**2) * dx)
        if norm > 0:
            psi_vals /= norm
        
        return WaveFunction(
            psi=psi_vals,
            grid=grid * HBAR*C/E,  # Convert back to physical coordinates
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

    def gamma5(self):
        """Return gamma5 matrix for chiral transformations."""
        N = len(self.grid)
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
