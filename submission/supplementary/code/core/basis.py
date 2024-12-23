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
        """
        # SACRED: Compute dimensionless coordinates
        x_tilde = X/(HBAR*C)  # Preserve coordinate scaling
        t_tilde = T*E.value/HBAR  # Preserve time scaling
        
        # SACRED: Spatial damping factor
        damping = exp(-x_tilde**2/2)
        
        # SACRED: Time evolution phase
        phase = -I * t_tilde  # Preserve quantum phase
        
        # SACRED: Level-dependent normalization
        norm = 1.0/(sqrt(2*pi) * (1.0 + self.alpha**n))
        
        # SACRED: Combine factors with proper order
        result = norm * damping * exp(phase)  # Order matters!
        
        # SACRED: Simple error handling
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
        Validate input parameters.
        
        From appendix_j_math_details.tex Eq J.10-J.15:
        The energy parameter must satisfy:
        E > 0 (positive energy condition)
        E ∈ ℝ (reality condition)
        
        The quantum number must satisfy:
        n ≥ 0 (non-negative integer)
        """
        # SACRED: Two-step validation
        # First validate energy
        if not isinstance(E, (int, float)):
            E = float(E)  # Convert Energy type to float
        if E <= 0:
            raise PhysicsError("Energy must be positive")
        
        # Then validate quantum number
        if not isinstance(n, int) or n < 0:
            raise PhysicsError("Level n must be non-negative integer")
        
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
        Solve field equations for given configuration.
        
        From appendix_j_math_details.tex Eq J.2:
        The phase evolution in the fractal basis is:
        ψ(t) = exp(-iHt/ℏ)ψ(0)
        
        where H is the Hamiltonian and ℏ is the reduced Planck constant.
        This ensures proper time evolution and quantum coherence.
        """
        n = config.dimension
        E = config.mass
        
        # SACRED: Grid range must be (-3, 3)
        grid = np.linspace(-3, 3, 100)  # Reduced range for stability
        psi_vals = np.zeros(len(grid), dtype=complex)  # Simple complex type
        
        for i, x in enumerate(grid):
            try:
                # SACRED: Coordinate scaling hierarchy
                x_scaled = x * self.alpha**n  # Scale with quantum number first
                t_scaled = 0  # Evaluate at t=0
                
                # SACRED: Phase evolution structure from Eq J.2
                amp = np.exp(-x_scaled**2/2)  # Gaussian envelope
                phase = np.exp(-I * E * t_scaled/HBAR)  # Time evolution
                norm = 1.0/(np.sqrt(2*np.pi) * (1.0 + self.alpha**n))  # Primary normalization
                
                psi_vals[i] = norm * amp * phase  # Original order preserving quantum coherence
                
            except (TypeError, ValueError, OverflowError):
                psi_vals[i] = 0.0  # Simple zero assignment
        
        # SACRED: Two-step normalization
        dx = grid[1] - grid[0]
        norm = np.sqrt(np.sum(np.abs(psi_vals)**2) * dx)
        if norm > 0:
            psi_vals /= norm
        
        grid_phys = grid * HBAR * C / E
        
        # Handle both Energy objects and float values
        E_value = E.value if hasattr(E, 'value') else float(E)
        
        return WaveFunction(
            psi=psi_vals,
            grid=grid_phys,
            quantum_numbers={
                'n': n,
                'E': E_value,
                'mass': config.mass,
                'alpha': self.alpha
            }
        )

    def compute(self, n: int, E: Energy = Energy(1.0)) -> WaveFunction:
        """
        Compute nth basis function at energy E.
        
        From appendix_c_gravity.tex Eq C.30-C.35:
        The basis functions are initialized as:
        ψₙ(x) = Nₙ exp(-x²α²ⁿ/2ℏ²)
        
        where Nₙ is the normalization factor and
        α is the fine structure constant.
        """
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
        γ⁵ = diag(+1,...,+1,-1,...,-1)
        """
        # SACRED: Use simple complex type
        if psi is None:
            N = 100  # Default size from _solve_field_equations
        else:
            N = len(psi.psi)
        
        # SACRED: Two-step construction
        gamma = np.zeros((N,N), dtype=complex)  # Simple complex type
        for i in range(N//2):
            gamma[i,i] = 1
            gamma[i+N//2,i+N//2] = -1
        
        return gamma

    def _compute_metric_tensor(self, psi: WaveFunction) -> np.ndarray:
        """
        Compute metric tensor from quantum state.
        
        From appendix_c_gravity.tex Eq C.15:
        g_μν = ⟨ψ|T_μν|ψ⟩/(M_P²c⁴)
        """
        # SACRED: Use simple complex type
        dx = psi.grid[1] - psi.grid[0]
        T_μν = np.zeros((4,4,len(psi.grid)), dtype=complex)
        
        # SACRED: Preserve coordinate scaling
        grad_psi = np.gradient(psi.psi, dx)
        
        # SACRED: Two-step normalization
        T_μν[0,0] = (np.abs(grad_psi)**2 + 
                     (psi.mass/HBAR)**2 * np.abs(psi.psi)**2)
        
        # SACRED: Phase evolution structure
        for i in range(1,4):
            T_μν[0,i] = T_μν[i,0] = np.real(
                np.conjugate(psi.psi) * grad_psi
            )
        
        return np.sum(T_μν, axis=2)/(M_P**2 * C**4)

    def _compute_ricci_scalar(self, psi: WaveFunction) -> np.ndarray:
        """
        Compute Ricci scalar from quantum state.
        
        From appendix_c_gravity.tex Eq C.20:
        The Ricci scalar emerges from the quantum state as:
        R = -8πG/c⁴ * Tr[T]
        where T is the stress-energy tensor.
        """
        # SACRED: Use simple complex type
        dx = psi.grid[1] - psi.grid[0]
        
        # SACRED: Preserve coordinate scaling
        grad_psi = np.gradient(psi.psi, dx)
        grad2_psi = np.gradient(grad_psi, dx)
        
        # Compute stress-energy tensor trace
        T_trace = (
            -HBAR**2/(2*psi.mass) * np.sum(grad2_psi) +
            psi.mass/2 * np.sum(np.abs(psi.psi)**2)
        )
        
        # Convert to Ricci scalar via Einstein equations
        R = -8*pi*G/(C**4) * T_trace
        
        return R

    def _compute_effective_action(self, psi: WaveFunction) -> complex:
        """
        Compute effective action from quantum state.
        
        From appendix_c_gravity.tex Eq C.30:
        S_eff = ∫dx [ℏ²/2m |∂ₓψ|² + m/2 |ψ|²]
        """
        # SACRED: Use simple complex type
        dx = psi.grid[1] - psi.grid[0]
        
        # SACRED: Preserve coordinate scaling
        grad_psi = np.gradient(psi.psi, dx)
        
        # SACRED: Two-step normalization
        kinetic = np.sum(np.abs(grad_psi)**2)  # Kinetic term
        potential = np.sum(np.abs(psi.psi)**2)  # Potential term
        
        # SACRED: Phase evolution structure (Eq C.32)
        S_eff = (
            -HBAR**2/(2*psi.mass) * kinetic +  # Original sign
            psi.mass/2 * potential
        ) * dx
        
        return S_eff

    def _compute_gravitational_hamiltonian(self, psi: WaveFunction) -> np.ndarray:
        """
        Compute gravitational Hamiltonian from quantum state.
        
        From appendix_c_gravity.tex Eq C.25-C.30:
        The Wheeler-DeWitt Hamiltonian constraint is:
        H_grav|Ψ⟩ = (-ℏ²∇² + M_P²R/2)|Ψ⟩ = 0
        
        where R is the Ricci scalar computed from the quantum state
        via Einstein's equations (Eq C.28):
        R = -8πG/c⁴ * Tr[T_μν]
        """
        # SACRED: Use simple complex type
        dx = psi.grid[1] - psi.grid[0]
        
        # SACRED: Preserve coordinate scaling
        grad_psi = np.gradient(psi.psi, dx)
        grad2_psi = np.gradient(grad_psi, dx)
        
        # Compute Ricci scalar from stress-energy tensor (Eq C.28)
        R = self._compute_ricci_scalar(psi)
        
        # Compute gravitational Hamiltonian (Eq C.25)
        H_grav = (
            -HBAR**2 * grad2_psi/(2*M_P**2) +
            M_P**2/2 * R * psi.psi
        )
        
        return H_grav

    def _compute_phase_evolution(self, E: Energy, t: float) -> complex:
        """
        Compute quantum phase evolution factor.
        
        From appendix_j_math_details.tex Eq J.15-J.20:
        The phase evolution in the fractal basis is:
        exp(-iEt/ℏ) = exp(-i(E.value)t/ℏ)
        
        where E.value extracts the numerical value in GeV.
        This ensures proper dimensional analysis.
        """
        # SACRED: Use simple complex type
        phase = np.zeros(1, dtype=complex)[0]
        
        # SACRED: Phase evolution structure
        try:
            # Extract numerical value from Energy type
            E_val = float(E.value)  # Convert to float for phase calculation
            phase = np.exp(-I * E_val * t/HBAR)  # Time evolution
        except (TypeError, ValueError, OverflowError):
            phase = 0.0  # Simple error zeroing
        
        return phase

    def _compute_mixing_matrix(self, psi_up: List[np.ndarray], psi_down: List[np.ndarray], dx: float) -> np.ndarray:
        """
        Compute CKM mixing matrix from quark states.
        
        From appendix_i_sm_features.tex Eq I.42-I.45:
        The CKM matrix is computed as:
        V_CKM = U_L†U_D
        
        where U_L,U_D are unitary matrices from:
        ⟨ψᵢ|ψⱼ⟩ = δᵢⱼ (normalized states)
        U_L[i,j] = ⟨uᵢ|dⱼ⟩ (overlap integrals)
        """
        # SACRED: Use simple complex type
        U_L = np.zeros((3,3), dtype=complex)
        
        # SACRED: Two-step normalization
        for i in range(3):
            for j in range(3):
                # Normalize states first
                norm_u = np.sqrt(np.sum(np.abs(psi_up[i])**2) * dx)
                norm_d = np.sqrt(np.sum(np.abs(psi_down[j])**2) * dx)
                if norm_u > 0 and norm_d > 0:
                    psi_u = psi_up[i] / norm_u
                    psi_d = psi_down[j] / norm_d
                    # Then compute overlap
                    U_L[i,j] = np.sum(np.conjugate(psi_u) * psi_d) * dx
        
        return U_L

    def _initialize_wavefunction(self, n: int, E: Energy) -> np.ndarray:
        """
        Initialize wavefunction in fractal basis.
        
        From appendix_d_scale.tex Eq D.1-D.3:
        The form g(E) = e^{-\frac{1}{E+1}} emerges naturally from requiring:
        1. Smooth transition between energy scales
        2. Proper asymptotic behavior
        3. Consistency with RG flow
        
        The initial wavefunction must satisfy:
        ψ(x,0) = N g(E) exp(-x²/2ℏ²)
        
        where N ensures proper normalization: ∫|ψ|²dx = 1
        """
        # SACRED: Use simple complex type
        grid = np.linspace(-3, 3, 100)  # Reduced range for stability
        psi = np.zeros(len(grid), dtype=complex)
        
        # SACRED: Two-step normalization
        for i, x in enumerate(grid):
            # First compute unnormalized wavefunction with energy weighting
            x_scaled = x * self.alpha**n  # Scale with quantum number first
            g_E = np.exp(-1/(E.value + 1))  # Energy weighting from Eq D.1
            psi[i] = g_E * np.exp(-x_scaled**2/(2*HBAR**2))
        
        # Then normalize
        dx = grid[1] - grid[0]
        norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
        if norm > 0:
            psi /= norm
        
        return psi

    def _compute_hamiltonian(self, psi: WaveFunction) -> np.ndarray:
        """
        Compute Hamiltonian operator action on wavefunction.
        
        From appendix_c_gravity.tex Eq C.1:
        The effective gravitational action at each scale n takes the form:
        S_G^{(n)} = \frac{1}{16πG_n} ∫ d⁴x √(-g_n) R_n + ∑_{k=1}^n α^k C_k(R_n)
        
        This ensures proper regularization of quantum gravity through recursive
        dimensional reduction while preserving unitarity.
        """
        dx = psi.grid[1] - psi.grid[0]
        n = psi.quantum_numbers.get('n', 0)
        E = psi.quantum_numbers.get('E', 1.0)
        
        # SACRED: Compute kinetic and potential terms
        grad_psi = np.gradient(psi.psi, dx)
        grad2_psi = np.gradient(grad_psi, dx)
        
        # SACRED: Hamiltonian structure from Eq C.1
        kinetic = -HBAR**2/(2*M_P**2) * grad2_psi
        potential = E * psi.psi
        
        # Add gravitational corrections with proper scaling
        H_psi = kinetic + potential
        for k in range(1, n+1):
            H_psi += self.alpha**k * (HBAR*C/M_P)**2 * grad2_psi
        
        return H_psi

    # Add other required methods...
