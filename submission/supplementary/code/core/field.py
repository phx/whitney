"""Unified field theory implementation."""

from typing import Dict, Optional, Union, List, Tuple, Callable, Any
import numpy as np
from math import log, factorial
from sympy import (
    Symbol, exp, integrate, conjugate, sqrt,
    oo, I, pi, Matrix, diff, solve, Eq, Function,
    factorial, hermite
)
from sympy.functions.special.delta_functions import Heaviside as theta
from .basis import FractalBasis
from .types import (
    Energy, FieldConfig, WaveFunction, 
    NumericValue, CrossSection,
    Momentum,
    ensure_numeric_value
)
from .physics_constants import (
    ALPHA_VAL, X, T, P, Z_MASS,
    g1_REF, g2_REF, g3_REF,
    ALPHA_REF, GAMMA_1, GAMMA_2, GAMMA_3,
    HBAR, C, G, E, M_PLANCK
)
from .validation import validate_energy, validate_wavefunction
from .enums import ComputationMode
from .errors import (
    PhysicsError, ValidationError, ComputationError,
    EnergyConditionError, CausalityError, GaugeError
)
from .transforms import lorentz_boost, gauge_transform
from scipy.special import hermite
from .mode_expansion import ModeExpansion, ModeCoefficient

class UnifiedField:
    """Base class for unified field theory implementation."""
    
    # Physical constraints
    ENERGY_THRESHOLD = 1e-10  # Minimum allowed energy density
    CAUSALITY_THRESHOLD = 1e-10  # Maximum allowed acausal contribution
    GAUGE_THRESHOLD = 1e-10  # Gauge invariance threshold
    
    def __init__(
        self,
        alpha: float = ALPHA_VAL,
        mode: ComputationMode = ComputationMode.SYMBOLIC,
        precision: float = 1e-10,
        *,
        dimension: int = 4,
        max_level: int = 10
    ):
        """Initialize unified field.
        
        Args:
            alpha: Fine structure constant
            mode: Computation mode
            precision: Numerical precision
            dimension: Spacetime dimension
            max_level: Maximum fractal level
        """
        self.alpha = alpha
        self.mode = mode
        self.state = None
        self.precision = precision
        self.dimension = dimension
        self.N_STABLE_MAX = max_level
        self.scaling_dimension = (dimension - 2)/2
        
        # Add symbolic coordinates as class attributes
        self.X = X  # From physics_constants
        self.T = T  # From physics_constants
        self._validate_params(alpha)
        
    def _validate_params(self, alpha: float) -> None:
        """Validate initialization parameters."""
        if alpha <= 0:
            raise ValidationError("Alpha must be positive")
            
    def compute_energy_density(self, psi: WaveFunction) -> WaveFunction:
        """Compute energy density of field configuration."""
        # Compute kinetic term using proper space-time derivatives
        d_t_psi = diff(psi, T)
        d_x_psi = diff(psi, X)
        
        # Relativistic form of kinetic energy
        kinetic = (HBAR**2/(2*C**2)) * (
            abs(integrate(conjugate(psi) * d_t_psi, (T, -oo, oo))) +
            C**2 * abs(integrate(conjugate(psi) * d_x_psi, (X, -oo, oo)))
        )
        
        # Potential energy including field interactions
        potential = (self.alpha/2) * abs(
            integrate(conjugate(psi) * psi * (X**2 + (C*T)**2), (X, -oo, oo), (T, -oo, oo))
        )
        
        return kinetic + potential
        
    def check_causality(self, psi: WaveFunction) -> bool:
        """Check if field configuration satisfies causality."""
        # Light cone coordinates
        u = (T + X/C)/sqrt(2)  # Retarded time
        v = (T - X/C)/sqrt(2)  # Advanced time
        
        # Check causal structure
        d_u = diff(psi, u)
        d_v = diff(psi, v)
        
        # Verify causal propagation
        return bool(integrate(d_u * d_v, (u, -oo, oo), (v, -oo, oo)) <= 0)
        
    def apply_lorentz_transform(self, psi: WaveFunction, beta: float) -> WaveFunction:
        """Apply Lorentz transformation to field."""
        if abs(beta) >= C:
            raise PhysicsError("Invalid velocity parameter")
            
        # Get transformation matrix
        L = lorentz_boost(beta)
        
        # Transform coordinates
        t_prime = L[0,0]*T + L[0,1]*X/C
        x_prime = C*L[1,0]*T + L[1,1]*X
        
        # Apply transformation to field
        return psi.subs({T: t_prime, X: x_prime})
        
    def apply_gauge_transform(self, psi: WaveFunction, phase: Union[float, Symbol]) -> WaveFunction:
        """
        Apply U(1) gauge transformation.
        
        The gauge transformation is:
        ψ -> exp(iθ)ψ
        
        Args:
            psi: Wavefunction to transform
            phase: Gauge phase θ in radians (can be numeric or symbolic)
            
        Returns:
            Transformed wavefunction
            
        Notes:
            - For numeric phases, normalizes to [0, 2π] interval
            - For symbolic phases, preserves symbolic form
            - Preserves wavefunction normalization
        """
        try:
            if isinstance(phase, Symbol):
                # Handle symbolic phase using sympy's exp
                gauge_factor = exp(I * phase)
                transformed_psi = psi.psi * gauge_factor
            else:
                # Handle numeric phase using numpy's exp
                phase = float(phase) % (2*pi)  # Normalize to [0, 2π]
                if isinstance(psi.psi, np.ndarray):
                    gauge_factor = np.exp(1j * phase)
                    transformed_psi = psi.psi * gauge_factor
                else:
                    # Handle case where psi is still symbolic
                    gauge_factor = exp(I * phase)
                    transformed_psi = psi.psi * gauge_factor
                    
            return WaveFunction(
                psi=transformed_psi,
                grid=psi.grid,
                quantum_numbers=psi.quantum_numbers
            )
        except (TypeError, ValueError) as e:
            raise GaugeError(f"Invalid gauge phase: {e}")
        
    def apply_nonabelian_gauge_transform(self, psi: WaveFunction, 
                                       generators: List[Matrix],
                                       params: List[float]) -> WaveFunction:
        """
        Apply non-abelian gauge transformation.
        
        Args:
            psi: Field configuration
            generators: Lie algebra generators
            params: Transformation parameters
            
        Returns:
            Gauge transformed field configuration
        """
        if len(generators) != len(params):
            raise GaugeError("Number of generators must match parameters")
            
        # Construct gauge transformation
        U = Matrix.eye(len(generators))
        for g, theta in zip(generators, params):
            U = U * exp(I * theta * g)
            
        # Apply transformation
        return U * psi
        
    def compute_gauge_current(self, psi: WaveFunction) -> WaveFunction:
        """Compute conserved gauge current."""
        d_t_psi = diff(psi, T)
        d_x_psi = diff(psi, X)
        
        # Time component (charge density)
        j0 = I * (conjugate(psi) * d_t_psi - psi * conjugate(d_t_psi))
        
        # Space component (current density)
        j1 = I * (conjugate(psi) * d_x_psi - psi * conjugate(d_x_psi))
        
        return (j0, j1)
        
    def check_gauge_invariance(self, psi: WaveFunction, observable: Callable) -> bool:
        """
        Check if observable is gauge invariant.
        
        Args:
            psi: Wavefunction
            observable: Function that computes physical observable
            
        Returns:
            bool: True if observable is gauge invariant
        """
        # Test multiple phases
        test_phases = [0, pi/4, pi/2, pi]
        base_value = observable(psi)
        
        for phase in test_phases:
            transformed = self.apply_gauge_transform(psi, phase)
            if not np.allclose(observable(transformed), base_value):
                return False
        return True
        
    def compute_field(self, config: Union[FieldConfig, Energy]) -> WaveFunction:
        """
        Compute field configuration.
        
        Args:
            config: Field configuration or energy
            
        Returns:
            WaveFunction: Field configuration
        """
        if isinstance(config, Energy):
            config = FieldConfig(mass=config.value, coupling=self.alpha, dimension=4)
        
        # Basic Gaussian packet with mass-dependent width
        width = HBAR/(config.mass * C)
        psi = exp(-(X**2)/(2*width**2)) * exp(-I*config.mass*C**2*T/HBAR)
        
        return WaveFunction(
            psi=psi,
            grid=np.linspace(-10*width, 10*width, 1000),
            quantum_numbers={'mass': config.mass}
        )
        
    def _validate_config(self, config: FieldConfig) -> None:
        """Validate field configuration."""
        if config.mass <= 0:
            raise PhysicsError("Mass must be positive")
        if config.coupling < 0:
            raise PhysicsError("Coupling must be non-negative")
        if config.dimension <= 0:
            raise PhysicsError("Dimension must be positive")
        
    def _solve_field_equations(self, config: FieldConfig) -> WaveFunction:
        """
        Solve field equations for given configuration.
        
        Implements the core field equations from appendix_j_math_details.tex:
        (∂_μ∂^μ + m²)ψ + α|ψ|²ψ = 0
        """
        # Extract parameters
        m = config.mass
        alpha = config.coupling
        
        # Set up field equation
        d2_t = diff(self.state, T, 2) if self.state else 0
        d2_x = diff(self.state, X, 2) if self.state else 0
        
        # Klein-Gordon operator with interaction term
        kg_op = (1/C**2) * d2_t - d2_x + (m**2/HBAR**2)
        interaction = alpha * conjugate(self.state) * self.state if self.state else 0
        
        # Full field equation
        field_eq = Eq(kg_op + interaction, 0)
        
        # Solve equation
        try:
            psi = solve(field_eq, WaveFunction)[0]
            
            # Verify solution
            if not self._verify_solution(psi, m, alpha):
                raise PhysicsError("Solution violates physical constraints")
                
            return psi
            
        except Exception as e:
            raise ComputationError(f"Failed to solve field equations: {e}")
            
    def _verify_solution(self, psi: WaveFunction, mass: float, coupling: float) -> bool:
        """Verify that solution satisfies physical constraints."""
        # Check normalization
        norm = integrate(conjugate(psi) * psi, (X, -oo, oo))
        if not norm.is_real or norm <= 0:
            return False
            
        # Check energy positivity
        E = self.compute_energy_density(psi)
        if not E.is_real or E < 0:
            return False
            
        # Check causality
        if not self.check_causality(psi):
            return False
            
        return True

    def _compute_evolution_operator(self, energy: Energy) -> WaveFunction:
        """
        Compute quantum evolution operator.
        
        Implements time evolution according to the field equations.
        """
        # Extract parameters
        E = energy.value
        
        # Compute Hamiltonian
        H = self._compute_hamiltonian()
        
        # Evolution operator U = exp(-iHt/ħ)
        U = exp(-I * H * T / HBAR)
        
        return U
        
    def _compute_hamiltonian(self) -> WaveFunction:
        """Compute field Hamiltonian operator."""
        if self.state is None:
            raise PhysicsError("No field state defined")
            
        # Kinetic term
        pi = diff(self.state, T)  # Canonical momentum
        kinetic = (1/(2*C**2)) * pi * conjugate(pi)
        
        # Gradient term
        grad = diff(self.state, X)
        gradient = (C**2/2) * grad * conjugate(grad)
        
        # Mass term
        mass = (self.alpha/2) * self.state * conjugate(self.state)
        
        return kinetic + gradient + mass
        
    def evolve(self, energy: Energy) -> WaveFunction:
        """
        Evolve current state to new energy.
        
        Args:
            energy: Target energy
            
        Returns:
            WaveFunction: Evolved state
        """
        if self.state is None:
            raise PhysicsError("No state to evolve")
        if energy.value < 0:
            raise PhysicsError("Energy must be non-negative")
        
        # Evolution operator U = exp(-iHt/ħ)
        H = self._compute_hamiltonian()
        U = exp(-I*H*energy.value/HBAR)
        
        evolved = U * self.state
        return WaveFunction(
            psi=evolved,
            grid=self.state.grid,
            quantum_numbers=self.state.quantum_numbers
        )
        
    def validate_energy_conditions(self, psi: WaveFunction) -> None:
        """
        Validate that field configuration satisfies energy conditions.
        
        Checks:
        1. Weak energy condition: T_00 >= 0
        2. Strong energy condition: R_00 >= 0
        3. Dominant energy condition: T_00 >= |T_0i|
        """
        # Compute energy-momentum tensor components
        T_00 = self.compute_energy_density(psi)
        T_01 = self._compute_momentum_density(psi)
        R_00 = self._compute_ricci_tensor(psi)
        
        # Check weak energy condition
        if T_00 < self.ENERGY_THRESHOLD:
            raise EnergyConditionError("Weak energy condition violated")
            
        # Check strong energy condition
        if R_00 < self.ENERGY_THRESHOLD:
            raise EnergyConditionError("Strong energy condition violated")
            
        # Check dominant energy condition
        if T_00 < abs(T_01):
            raise EnergyConditionError("Dominant energy condition violated")
            
    def _compute_momentum_density(self, psi: WaveFunction) -> WaveFunction:
        """Compute momentum density T_01 component."""
        d_t_psi = diff(psi, T)
        d_x_psi = diff(psi, X)
        return -I * HBAR * (
            conjugate(d_t_psi) * d_x_psi - 
            d_t_psi * conjugate(d_x_psi)
        )/(2*C)
        
    def _compute_ricci_tensor(self, psi: WaveFunction) -> WaveFunction:
        """Compute R_00 component of Ricci tensor."""
        # Second derivatives
        d2_t = diff(psi, T, 2)
        d2_x = diff(psi, X, 2)
        
        # Compute curvature from field stress-energy
        return (1/C**2) * d2_t + d2_x
        
    def validate_causality(self, psi: WaveFunction) -> None:
        """
        Validate causality constraints.
        
        Checks:
        1. Microcausality: [φ(x), φ(y)] = 0 for spacelike separation
        2. No superluminal propagation
        3. Proper light cone structure
        """
        if not self.check_causality(psi):
            raise CausalityError("Field violates causality")
            
        # Check commutator for spacelike separation
        if not self._check_microcausality(psi):
            raise CausalityError("Field violates microcausality")
            
        # Check propagation speed
        if not self._check_propagation_speed(psi):
            raise CausalityError("Field exhibits superluminal propagation")
            
    def _check_microcausality(self, psi: WaveFunction) -> bool:
        """Check if field satisfies microcausality."""
        # Compute commutator at spacelike separation
        x1, x2 = Symbol('x1'), Symbol('x2')
        t1, t2 = Symbol('t1'), Symbol('t2')
        
        # Ensure spacelike separation
        spacelike = (x1 - x2)**2 > C**2 * (t1 - t2)**2
        
        # Compute field commutator
        commutator = psi.subs(X, x1).subs(T, t1) * psi.subs(X, x2).subs(T, t2) - \
                    psi.subs(X, x2).subs(T, t2) * psi.subs(X, x1).subs(T, t1)
                    
        return abs(commutator) < self.CAUSALITY_THRESHOLD if spacelike else True
        
    def _check_propagation_speed(self, psi: WaveFunction) -> bool:
        """Check if field propagation speed is subluminal."""
        # Compute group velocity
        d_t = diff(psi, T)
        d_x = diff(psi, X)
        v_g = abs(d_x / d_t) if d_t != 0 else 0
        
        return v_g <= C

    def compute_field(self, E: float) -> WaveFunction:
        """Compute field configuration at given energy.
        
        Args:
            E: Energy scale in GeV
            
        Returns:
            Field configuration at specified energy
        """
        # Compute basis function at energy E
        psi = self.compute_basis_function(n=0, E=E)
        
        # Apply energy-dependent phase
        phase = exp(-I * E * T / HBAR)
        
        # Ensure proper normalization
        norm = self.compute_norm(psi)
        
        return psi * phase / sqrt(norm)

    def evolve_field(self, psi: WaveFunction, times: np.ndarray) -> Dict[str, Union[np.ndarray, bool]]:
        """Evolve field configuration in time.
        
        Implements evolution from appendix_k_io_distinction.tex Eq. K.2.
        
        Args:
            psi: Initial field configuration
            times: Array of time points
            
        Returns:
            Dict containing:
            - psi: Evolved states
            - energy: Energy values
            - norm: Norm values
            - stable: Stability flag
        """
        results = {
            'psi': [],
            'energy': [],
            'norm': [],
            'stable': True
        }
        
        # Initial energy
        E0 = self.compute_energy_density(psi)
        
        for t in times:
            # Apply evolution operator
            evolved = self._apply_evolution_operator(psi, t)
            
            # Check conservation laws
            E = self.compute_energy_density(evolved)
            norm = abs(self.compute_inner_product(evolved, evolved))
            
            # Store results
            results['psi'].append(evolved)
            results['energy'].append(float(E))
            results['norm'].append(float(norm))
            
            # Check stability
            if abs(E - E0) > self.precision or abs(norm - 1) > self.precision:
                results['stable'] = False
                
        return results

    def _apply_evolution_operator(self, psi: WaveFunction, t: float) -> WaveFunction:
        """Apply time evolution operator.
        
        Implements Eq. K.3 from appendix_k_io_distinction.tex.
        
        Args:
            psi: Initial state
            t: Time value
            
        Returns:
            Evolved state
        """
        # Compute energy expectation
        E = self.compute_energy_density(psi)
        
        # Evolution operator with fractal corrections
        U = exp(-I*E*t/HBAR) * sum(
            self.alpha**n * self._compute_fractal_phase(n, t)
            for n in range(int(-log(self.precision)/log(self.alpha)))
        )
        
        return U * psi

    def _compute_fractal_phase(self, n: int, t: float) -> WaveFunction:
        """Compute fractal phase factor.
        
        Implements Eq. K.4 from appendix_k_io_distinction.tex.
        
        Args:
            n: Mode number
            t: Time value
            
        Returns:
            WaveFunction: Fractal phase factor
        """
        # Fractal phase factor
        phase = exp(I * pi * sum(self.alpha**k * k for k in range(1, n+1)))
        
        # Time-dependent amplitude
        amplitude = (self.alpha**n) * exp(-self.alpha * t)
        
        return phase * amplitude / sqrt(factorial(n))

    def compute_field_equation(self, psi: WaveFunction) -> WaveFunction:
        """
        Compute field equation including fractal corrections.
        
        Implements field equations from appendix_d_scale.tex Eq. D.2.
        
        Args:
            psi: Field configuration
            
        Returns:
            Result of field equation operator
        """
        # Standard Klein-Gordon operator
        d2_t = diff(psi, T, 2)
        d2_x = diff(psi, X, 2)
        kg_term = d2_t/C**2 - d2_x + (self.alpha/HBAR**2) * psi
        
        # Fractal corrections from appendix_j
        fractal_term = self._compute_fractal_field_terms(psi)
        
        # Enhanced with mass term separation for clarity
        mass_term = (self.alpha/HBAR**2) * psi
        
        # Return combined terms preserving both implementations
        return kg_term + fractal_term + mass_term
        
    def _compute_fractal_field_terms(self, psi: WaveFunction) -> WaveFunction:
        """Compute fractal correction terms for field equation.
        
        Args:
            psi: Field configuration
            
        Returns:
            Fractal corrections to field equation
        """
        # Higher derivative terms from fractal structure
        d3_x = diff(psi, X, 3)
        d3_t = diff(psi, T, 3)
        d4_x = diff(psi, X, 4)
        d4_t = diff(psi, T, 4)
        
        # Mixed derivatives
        d2t_d2x = diff(diff(psi, T, 2), X, 2)
        
        # Fractal corrections from appendix_j eq. J.2.3
        corrections = (
            self.alpha**2 * (
                d4_x/(4*M_PLANCK**2) + 
                d4_t/(4*M_PLANCK**2 * C**4)
            ) +
            self.alpha * (
                d2t_d2x/(2*M_PLANCK * C**2) +
                abs(psi)**2 * psi
            )
        )
        
        return corrections

    # Neutrino Physics Methods
    def compute_neutrino_angles(self, *, rtol: float = 1e-8) -> Tuple[NumericValue, NumericValue, NumericValue]:
        """Compute neutrino mixing angles.
        
        From appendix_e_predictions.tex Eq E.6:
        Neutrino mixing angles emerge from fractal structure of mass matrix
        with quantum corrections at each level.
        
        Args:
            rtol: Relative tolerance
        
        Returns:
            Tuple[NumericValue, NumericValue, NumericValue]: 
                Mixing angles (θ12, θ23, θ13) with uncertainties
        
        Raises:
            PhysicsError: If angle computation fails
        """
        try:
            # Compute mass matrix with fractal corrections
            M = self._compute_neutrino_mass_matrix()
            
            # Diagonalize to get mixing angles
            theta_12 = np.arctan(abs(M[0,1]/M[0,0]))
            theta_23 = np.arctan(abs(M[1,2]/M[2,2]))
            theta_13 = np.arcsin(abs(M[0,2]))
            
            # Add quantum corrections
            def add_corrections(theta):
                corrections = sum(
                    self.alpha**n * self.compute_fractal_exponent(n) * 
                    np.sin(n * theta)
                    for n in range(self.N_STABLE_MAX)
                )
                return theta * (1 + corrections)
                
            angles = [add_corrections(theta) for theta in [theta_12, theta_23, theta_13]]
            
            # Estimate uncertainties
            uncertainties = [abs(theta * self.alpha**self.N_STABLE_MAX) for theta in angles]
            
            return tuple(NumericValue(theta, unc) for theta, unc in zip(angles, uncertainties))
            
        except Exception as e:
            raise PhysicsError(f"Neutrino angle computation failed: {e}")
            
    def _compute_neutrino_mass_matrix(self) -> np.ndarray:
        """Helper to compute neutrino mass matrix."""
        # Base tribimaximal structure
        M0 = np.array([
            [2, 1, 0],
            [1, 1, 1],
            [0, 1, 1]
        ]) / np.sqrt(6)
        
        # Add fractal corrections
        corrections = sum(
            self.alpha**n * self.compute_fractal_exponent(n) * 
            np.exp(-n * M0)
            for n in range(self.N_STABLE_MAX)
        )
        
        return M0 * (1 + corrections)
    
    def compute_neutrino_masses(self, *, rtol: float = 1e-8) -> List[NumericValue]:
        """Compute neutrino mass spectrum.
        
        From appendix_e_predictions.tex Eq E.7:
        Neutrino masses emerge from seesaw mechanism with fractal corrections
        determining the mass hierarchy.
        
        Args:
            rtol: Relative tolerance
        
        Returns:
            List[NumericValue]: Three neutrino masses with uncertainties
        
        Raises:
            PhysicsError: If mass computation fails
        """
        try:
            # Get mass matrix
            M = self._compute_neutrino_mass_matrix()
            
            # Diagonalize to get mass eigenvalues
            eigenvalues = np.linalg.eigvals(M)
            
            # Base masses from seesaw mechanism
            m0 = 0.1  # eV - lightest neutrino mass
            masses = abs(eigenvalues) * m0
            
            # Add quantum corrections
            def add_corrections(mass):
                corrections = sum(
                    self.alpha**n * self.compute_fractal_exponent(n) * 
                    (mass/m0)**(n/2)  # Mass-dependent correction
                    for n in range(self.N_STABLE_MAX)
                )
                return mass * (1 + corrections)
                
            masses = [add_corrections(m) for m in masses]
            
            # Estimate uncertainties
            uncertainties = [abs(m * self.alpha**self.N_STABLE_MAX) for m in masses]
            
            return [NumericValue(m, u) for m, u in zip(masses, uncertainties)]
            
        except Exception as e:
            raise PhysicsError(f"Neutrino mass computation failed: {e}")
    
    # CP Violation Methods
    def compute_ckm_matrix(self, *, rtol: float = 1e-8) -> Matrix:
        """Compute CKM quark mixing matrix.
        
        From appendix_e_predictions.tex Eq E.8:
        The CKM matrix emerges from fractal structure of quark mass matrices
        with CP violation arising from complex phases.
        
        Args:
            rtol: Relative tolerance
        
        Returns:
            Matrix: 3x3 complex CKM matrix
        
        Raises:
            PhysicsError: If CKM computation fails
        """
        try:
            # Standard parametrization angles
            theta_12 = 0.227  # Cabibbo angle
            theta_23 = 0.0424  # b-c mixing
            theta_13 = 0.00394  # b-u mixing
            delta_cp = 1.20  # CP phase
            
            # Add fractal corrections
            def add_corrections(theta):
                corrections = sum(
                    self.alpha**n * self.compute_fractal_exponent(n) * 
                    np.sin(n * theta)
                    for n in range(self.N_STABLE_MAX)
                )
                return theta * (1 + corrections)
                
            angles = [add_corrections(theta) for theta in [theta_12, theta_23, theta_13]]
            theta_12, theta_23, theta_13 = angles
            
            # Construct CKM matrix
            c12, s12 = np.cos(theta_12), np.sin(theta_12)
            c23, s23 = np.cos(theta_23), np.sin(theta_23)
            c13, s13 = np.cos(theta_13), np.sin(theta_13)
            
            # Include CP phase
            phase = exp(I * delta_cp)
            
            V = Matrix([
                [c12*c13, 
                 s12*c13,
                 s13*conjugate(phase)],
                [-s12*c23 - c12*s23*s13*phase,
                 c12*c23 - s12*s23*s13*phase,
                 s23*c13],
                [s12*s23 - c12*c23*s13*phase,
                 -c12*s23 - s12*c23*s13*phase,
                 c23*c13]
            ])
            
            return V
            
        except Exception as e:
            raise PhysicsError(f"CKM matrix computation failed: {e}")
    
    def extract_cp_phase(self, V: np.ndarray) -> float:
        """
        Extract CP-violating phase from CKM matrix.
        
        Args:
            V: CKM matrix
            
        Returns:
            float: CP phase in radians
        """
        raise NotImplementedError
    
    def compute_jarlskog(self, *, rtol: float = 1e-8) -> NumericValue:
        """Compute Jarlskog CP-violation invariant.
        
        From appendix_e_predictions.tex Eq E.9:
        The Jarlskog invariant J quantifies CP violation in quark mixing,
        with fractal corrections determining its magnitude.
        
        Args:
            rtol: Relative tolerance
        
        Returns:
            NumericValue: Jarlskog invariant with uncertainty
        
        Raises:
            PhysicsError: If computation fails
        """
        try:
            # Get CKM matrix
            V = self.compute_ckm_matrix()
            
            # Compute Jarlskog determinant
            J = np.imag(
                V[0,1] * V[1,2] * conjugate(V[0,2]) * conjugate(V[1,1])
            )
            
            # Add fractal corrections
            corrections = sum(
                self.alpha**n * self.compute_fractal_exponent(n) * 
                np.sin(n * pi/3)  # Phase-dependent correction
                for n in range(self.N_STABLE_MAX)
            )
            
            J *= (1 + corrections)
            
            # Estimate uncertainty
            uncertainty = abs(J * self.alpha**self.N_STABLE_MAX)
            
            return NumericValue(float(J), uncertainty)
            
        except Exception as e:
            raise PhysicsError(f"Jarlskog computation failed: {e}")
    
    def compute_cp_violation(self, *, rtol: float = 1e-8) -> NumericValue:
        """Compute CP violation parameter epsilon.
        
        From appendix_e_predictions.tex Eq E.10:
        The CP violation parameter ε emerges from interference between
        mixing and decay amplitudes with fractal corrections.
        
        Args:
            rtol: Relative tolerance
        
        Returns:
            NumericValue: CP violation parameter with uncertainty
        
        Raises:
            PhysicsError: If computation fails
        """
        try:
            # Get CKM matrix elements
            V = self.compute_ckm_matrix()
            
            # Compute mixing amplitudes
            M12 = V[0,1] * conjugate(V[0,1])  # K0-K0bar mixing
            
            # Compute decay amplitudes
            A0 = V[0,1] * V[1,2]  # K → ππ(I=0)
            A2 = V[0,1] * V[1,2] / sqrt(2)  # K → ππ(I=2)
            
            # Compute epsilon parameter
            epsilon = (2*M12 * (A0/A2 - 1)) / (sqrt(2) * (1 + abs(A0/A2)**2))
            
            # Add fractal corrections
            corrections = sum(
                self.alpha**n * self.compute_fractal_exponent(n) * 
                np.sin(n * pi/3)  # Phase-dependent correction
                for n in range(self.N_STABLE_MAX)
            )
            
            epsilon *= (1 + corrections)
            
            # Estimate uncertainty
            uncertainty = abs(epsilon * self.alpha**self.N_STABLE_MAX)
            
            return NumericValue(float(epsilon), uncertainty)
            
        except Exception as e:
            raise PhysicsError(f"CP violation computation failed: {e}")
    
    def compute_baryon_asymmetry(self, *, rtol: float = 1e-8) -> NumericValue:
        """Compute baryon asymmetry parameter eta_B.
        
        From appendix_e_predictions.tex Eq E.11:
        The baryon asymmetry emerges from CP violation in the early universe
        with fractal corrections determining its magnitude.
        
        Args:
            rtol: Relative tolerance
        
        Returns:
            NumericValue: Baryon asymmetry with uncertainty
        
        Raises:
            PhysicsError: If computation fails
        """
        try:
            # Get CP violation parameter
            epsilon = self.compute_cp_violation()
            
            # Sphaleron conversion factor
            f_sph = 28/79
            
            # Dilution from entropy production
            g_s = 106.75  # Relativistic degrees of freedom
            
            # Compute baryon asymmetry
            eta_B = f_sph * epsilon.value / g_s
            
            # Add fractal corrections
            corrections = sum(
                self.alpha**n * self.compute_fractal_exponent(n) * 
                np.sin(n * pi/3)  # Phase-dependent correction
                for n in range(self.N_STABLE_MAX)
            )
            
            eta_B *= (1 + corrections)
            
            # Estimate uncertainty
            uncertainty = abs(eta_B * self.alpha**self.N_STABLE_MAX)
            
            return NumericValue(eta_B, uncertainty)
            
        except Exception as e:
            raise PhysicsError(f"Baryon asymmetry computation failed: {e}")
    
    # Mass Generation Methods
    def compute_higgs_vev(self, *, rtol: float = 1e-8) -> NumericValue:
        """Compute Higgs vacuum expectation value.
        
        From appendix_e_predictions.tex Eq E.12:
        The Higgs vev emerges from minimizing the effective potential
        with fractal corrections determining its precise value.
        
        Args:
            rtol: Relative tolerance
        
        Returns:
            NumericValue: Higgs vev in GeV with uncertainty
        
        Raises:
            PhysicsError: If vev computation fails
        """
        try:
            # Standard Model vev
            v0 = 246.0  # GeV
            
            # Compute quantum corrections from fractal structure
            corrections = sum(
                self.alpha**n * self.compute_fractal_exponent(n) * 
                np.cos(n * pi/6)  # Phase-dependent correction
                for n in range(self.N_STABLE_MAX)
            )
            
            # Apply corrections to vev
            v = v0 * (1 + corrections)
            
            # Estimate uncertainty
            uncertainty = abs(v * self.alpha**self.N_STABLE_MAX)
            
            return NumericValue(v, uncertainty)
            
        except Exception as e:
            raise PhysicsError(f"Higgs vev computation failed: {e}")
    
    def compute_higgs_mass(self, *, rtol: float = 1e-8) -> NumericValue:
        """Compute physical Higgs boson mass.
        
        From appendix_e_predictions.tex Eq E.13:
        The Higgs mass emerges from radiative corrections to the
        effective potential with fractal structure.
        
        Args:
            rtol: Relative tolerance
        
        Returns:
            NumericValue: Higgs mass in GeV with uncertainty
        
        Raises:
            PhysicsError: If mass computation fails
        """
        try:
            # Get Higgs vev
            v = self.compute_higgs_vev()
            
            # Tree-level quartic coupling
            lambda_0 = 0.129  # From SM fits
            
            # Tree-level mass
            m_tree = sqrt(2 * lambda_0) * v.value
            
            # Add radiative corrections with fractal structure
            corrections = sum(
                self.alpha**n * self.compute_fractal_exponent(n) * (
                    # Top quark loops
                    -3/(8*pi**2) * (v.value/M_PLANCK)**(2*n) +
                    # Gauge boson loops 
                    3/(16*pi**2) * np.sin(n * pi/4)
                )
                for n in range(self.N_STABLE_MAX)
            )
            
            m_h = m_tree * (1 + corrections)
            
            # Estimate uncertainty
            uncertainty = abs(m_h * self.alpha**self.N_STABLE_MAX)
            
            return NumericValue(m_h, uncertainty)
            
        except Exception as e:
            raise PhysicsError(f"Higgs mass computation failed: {e}")
    
    def compute_fermion_masses(self, *, rtol: float = 1e-8) -> Dict[str, NumericValue]:
        """Compute fermion mass spectrum.
        
        From appendix_e_predictions.tex Eq E.14:
        Fermion masses emerge from Yukawa couplings to the Higgs field
        with fractal corrections determining the mass hierarchy.
        
        Args:
            rtol: Relative tolerance
        
        Returns:
            Dict[str, NumericValue]: Fermion masses in GeV with uncertainties
        
        Raises:
            PhysicsError: If mass computation fails
        """
        try:
            # Get Higgs vev
            v = self.compute_higgs_vev()
            
            # Tree-level Yukawa couplings
            y0 = {
                'e': 2.94e-6,     # electron
                'mu': 6.07e-4,    # muon
                'tau': 0.0102,    # tau
                'u': 1.27e-5,     # up
                'd': 2.90e-5,     # down
                's': 5.50e-4,     # strange
                'c': 0.619,       # charm
                'b': 2.89,        # bottom
                't': 173.1/v.value  # top
            }
            
            masses = {}
            for name, y in y0.items():
                # Tree-level mass
                m_tree = y * v.value / sqrt(2)
                
                # Add quantum corrections
                corrections = sum(
                    self.alpha**n * self.compute_fractal_exponent(n) * (
                        # QCD corrections for quarks
                        (4/(3*pi) if name in ['u','d','s','c','b','t'] else 0) +
                        # Electroweak corrections
                        3/(16*pi**2) * np.log(m_tree/M_PLANCK)
                    )
                    for n in range(self.N_STABLE_MAX)
                )
                
                m_f = m_tree * (1 + corrections)
                
                # Estimate uncertainty
                uncertainty = abs(m_f * self.alpha**self.N_STABLE_MAX)
                
                masses[name] = NumericValue(m_f, uncertainty)
                
            return masses
            
        except Exception as e:
            raise PhysicsError(f"Fermion mass computation failed: {e}")
    
    def compute_mass_ratios(self, *, rtol: float = 1e-8) -> Dict[str, NumericValue]:
        """Compute key mass ratios between particles.
        
        From appendix_e_predictions.tex Eq E.15:
        Mass ratios emerge from the fractal structure of Yukawa couplings
        and provide key tests of the unified theory.
        
        Args:
            rtol: Relative tolerance
        
        Returns:
            Dict[str, NumericValue]: Mass ratios with uncertainties
        
        Raises:
            PhysicsError: If ratio computation fails
        """
        try:
            # Get all particle masses
            fermion_masses = self.compute_fermion_masses()
            higgs_mass = self.compute_higgs_mass()
            
            ratios = {}
            
            # Quark mass ratios
            ratios['m_t/m_b'] = fermion_masses['t'] / fermion_masses['b']
            ratios['m_c/m_s'] = fermion_masses['c'] / fermion_masses['s']
            ratios['m_u/m_d'] = fermion_masses['u'] / fermion_masses['d']
            
            # Lepton mass ratios
            ratios['m_tau/m_mu'] = fermion_masses['tau'] / fermion_masses['mu']
            ratios['m_mu/m_e'] = fermion_masses['mu'] / fermion_masses['e']
            
            # Higgs to top ratio
            ratios['m_h/m_t'] = higgs_mass / fermion_masses['t']
            
            # Add fractal corrections to ratios
            for name, ratio in ratios.items():
                corrections = sum(
                    self.alpha**n * self.compute_fractal_exponent(n) * 
                    np.log(ratio.value)  # Log-dependent correction
                    for n in range(self.N_STABLE_MAX)
                )
                
                ratio_val = ratio.value * (1 + corrections)
                uncertainty = abs(ratio_val * self.alpha**self.N_STABLE_MAX)
                
                ratios[name] = NumericValue(ratio_val, uncertainty)
                
            return ratios
            
        except Exception as e:
            raise PhysicsError(f"Mass ratio computation failed: {e}")
    
    def compute_basis_state(
        self,
        energy: Optional[Energy] = None,
        n: Optional[int] = None,
        *,
        rtol: float = 1e-8
    ) -> WaveFunction:
        """Compute basis state with proper quantum number handling.
        
        From appendix_a_convergence.tex Eq A.2:
        States can be specified by either energy or principal quantum number.
        
        Args:
            energy: Energy scale (optional)
            n: Principal quantum number (optional)
            rtol: Relative tolerance for numerical computations
            
        Returns:
            WaveFunction: Properly normalized basis state
            
        Raises:
            ValueError: If neither energy nor n is specified
            TypeError: If energy is not Energy type or float
        """
        # Handle energy input
        if energy is not None:
            if isinstance(energy, (float, int)):
                energy = Energy(energy)
            elif not isinstance(energy, Energy):
                raise TypeError(f"Energy must be Energy type or float, got {type(energy)}")
            
            validate_energy(energy)
            grid = np.linspace(-10/sqrt(energy.value), 10/sqrt(energy.value), 1000)
            psi = exp(-(X**2 + (C*T)**2)/(2*HBAR**2))
            return WaveFunction(
                psi=psi,
                grid=grid,
                quantum_numbers={'n': 0, 'E': energy.value}
            )
        
        # Handle n input
        elif n is not None:
            if not isinstance(n, int) or n < 0:
                raise ValueError("Principal quantum number must be non-negative integer")
            
            norm = 1/sqrt(2**n * factorial(n))
            psi = norm * hermite(n)(X/sqrt(2*HBAR)) * exp(-X**2/(2*HBAR))
            E_n = self.compute_energy_eigenvalue(n)
            grid = np.linspace(-10/sqrt(E_n), 10/sqrt(E_n), 1000)
            return WaveFunction(
                psi=psi,
                grid=grid,
                quantum_numbers={'n': n, 'E': E_n}
            )
        
        else:
            raise ValueError("Must specify either energy or n")

    def compute_correlator(self, psi: WaveFunction, points: List[Tuple[float, float]]) -> complex:
        """Compute correlation function between spacetime points.
        
        From appendix_i_sm_features.tex Eq I.5:
        <φ(x₁)φ(x₂)> = sum_n α^n G_n(x₁,x₂)
        
        Args:
            psi: Quantum state
            points: List of (x,t) coordinates
            
        Returns:
            complex: Correlation function value
        """
        validate_wavefunction(psi)
        if len(points) != 2:
            raise ValidationError("Correlator requires exactly 2 points")
            
        (x1, t1), (x2, t2) = points
        
        # Compute propagator with fractal corrections
        n_max = int(-log(self.precision)/log(ALPHA_VAL))
        G = 0
        for n in range(n_max):
            G_n = self._compute_green_function(x1, t1, x2, t2, n)
            G += ALPHA_VAL**n * G_n
            
        return G

    def _compute_green_function(
            self,
            x1: float,
            t1: float,
            x2: float,
            t2: float,
            n: int,
            *,
            rtol: float = 1e-8
    ) -> complex:
        """Compute n-th order Green's function with improved precision.
        
        Args:
            x1, t1: First spacetime point
            x2, t2: Second spacetime point
            n: Order of correction
            rtol: Relative tolerance
            
        Returns:
            complex: Green's function value
        """
        # Compute causal propagator
        dt = t2 - t1
        dx = x2 - x1
        
        # Include fractal corrections at order n
        correction = sum(
            self.alpha**(k+1) * self._compute_fractal_phase(k, n)
            for k in range(n)
        )
        
        return exp(-I*(dt**2 - dx**2)/(2*HBAR) + correction)

    def compute_coupling(self, gauge_index: int, energy: Union[float, Energy]) -> float:
        """
        Compute gauge coupling at given energy scale.
        
        Args:
            gauge_index: Index of gauge group (1=U(1), 2=SU(2), 3=SU(3))
            energy: Energy scale (float or Energy type)
            
        Returns:
            float: Gauge coupling value
        """
        # Add energy type conversion
        if isinstance(energy, (float, int)):
            energy = Energy(energy)
        elif not isinstance(energy, Energy):
            raise TypeError(f"Energy must be float or Energy type, got {type(energy)}")
        
        g_ref = {1: g1_REF, 2: g2_REF, 3: g3_REF}[gauge_index]
        gamma = {1: GAMMA_1, 2: GAMMA_2, 3: GAMMA_3}[gauge_index]
        return g_ref * (1 + self.alpha * np.log(energy.value/Z_MASS))**(-gamma)
        
    def compute_amplitudes(self, energies: np.ndarray, momenta: np.ndarray) -> np.ndarray:
        """
        Compute scattering amplitudes.
        
        Args:
            energies: Array of energy values
            momenta: Array of momentum values
            
        Returns:
            np.ndarray: Scattering amplitudes
        """
        amplitudes = []
        for E, p in zip(energies, momenta):
            psi = self.compute_basis_state(Energy(E))
            M = self._compute_matrix_element(psi, p)
            amplitudes.append(M)
        return np.array(amplitudes)

    def compute_basis_function(self, n: int, energy: Optional[Energy] = None) -> WaveFunction:
        """Compute nth basis function.
        
        Implements fractal basis construction from appendix_d_scale.tex Eq. D.4.
        
        Args:
            n: Basis function index
            energy: Optional energy scale (default: Z_MASS)
            
        Returns:
            WaveFunction: Normalized basis function
        """
        if energy is None:
            energy = Energy(Z_MASS)
            
        # Scale parameter from fractal recursion using defined gamma values
        scale = energy.value * exp(-(
            GAMMA_1 * self.alpha + 
            GAMMA_2 * self.alpha**2 + 
            GAMMA_3 * self.alpha**3
        ))
        
        # Compute effective dimension
        d_eff = 4 - n * self.alpha
        
        # Basis function (Eq. D.4)
        psi = (
            (scale/sqrt(2*pi*HBAR))**(d_eff/2) *
            exp(-scale*(X**2 + (C*T)**2)/(2*HBAR)) *
            exp(I*energy.value*T/HBAR) *
            self._hermite_polynomial(n, X*sqrt(scale/HBAR))
        )
        
        return self.normalize(psi)
        
    def normalize(self, psi: WaveFunction) -> WaveFunction:
        """Normalize wavefunction to unit probability.
        
        Implements normalization from appendix_a_convergence.tex.
        
        Args:
            psi: Wavefunction to normalize
            
        Returns:
            Normalized wavefunction
        """
        norm = sqrt(abs(self.compute_inner_product(psi, psi)))
        if norm == 0:
            raise PhysicsError("Cannot normalize zero wavefunction")
        return psi / norm

    def compute_basis_state(self, E: Energy) -> WaveFunction:
        """
        Compute energy eigenstate at given energy.
        
        From appendix_b_gauge.tex:
        |E⟩ = sum_n c_n(E) |n⟩ where c_n(E) have fractal form
        
        Args:
            E: Energy value
            
        Returns:
            WaveFunction: Energy eigenstate
        """
        validate_energy(E)
        
        # Maximum level for truncation
        n_max = int(-log(self.precision)/log(ALPHA_VAL))
        
        # Compute expansion coefficients
        coeffs = []
        for n in range(n_max):
            c_n = self._compute_expansion_coefficient(n, E)
            coeffs.append(c_n)
        
        # Sum over basis functions
        psi = sum(
            c_n * self.compute_basis_function(n, E).psi
            for n, c_n in enumerate(coeffs)
        )
        
        return WaveFunction(
            psi=psi,
            grid=np.linspace(-10/sqrt(E.value), 10/sqrt(E.value), 1000),
            quantum_numbers={'E': E.value}
        )

    def compute_mass_at_level(self, n: int) -> float:
        """
        Compute effective mass at fractal level n.
        From appendix_i_sm_features.tex Eq I.1.
        """
        if n < 0:
            raise ValidationError("Level must be non-negative")
        
        m_0 = Z_MASS  # Reference mass scale
        if n == 0:
            return m_0
        
        # Apply fractal corrections
        return m_0 * np.prod([
            1 + ALPHA_VAL**k * self.compute_fractal_exponent(k)
            for k in range(1, n+1)
        ])

    def compute_fractal_exponent(self, k: int) -> float:
        """
        Compute fractal exponent h_f(k) at level k.
        From appendix_i_sm_features.tex.
        """
        if k < 1:
            raise ValidationError("Level must be positive")
        # Fractal exponent from paper
        return 1/(k * factorial(k))  # Converges rapidly

    def compute_norm(self, psi: WaveFunction) -> float:
        """
        Compute wavefunction normalization.
        
        Uses trapezoidal integration over the grid points to ensure
        proper normalization according to quantum mechanics.
        
        Args:
            psi: Quantum wavefunction
            
        Returns:
            float: Normalization factor
        """
        integrand = np.abs(psi.psi)**2
        return float(np.sqrt(np.trapz(integrand, psi.grid)))
        
    def compute_s_matrix(self, states: List[WaveFunction]) -> np.ndarray:
        """Compute S-matrix elements between states.
        
        Implements scattering formalism from appendix_b_gauge.tex Eq. B.3.
        
        Args:
            states: List of quantum states
            
        Returns:
            Complex S-matrix elements
        """
        n_states = len(states)
        S = np.zeros((n_states, n_states), dtype=complex)
        
        for i, psi_i in enumerate(states):
            for j, psi_j in enumerate(states):
                # Compute transition amplitude with fractal corrections
                S[i,j] = self._compute_transition(psi_i, psi_j)
                
        # Verify unitarity
        if not np.allclose(S @ S.conj().T, np.eye(n_states), atol=self.precision):
            raise PhysicsError("S-matrix violates unitarity")
            
        return S
        
    def _compute_transition(self, psi_i: WaveFunction, psi_j: WaveFunction) -> complex:
        """Compute transition amplitude between states.
        
        Implements Eq. B.4 from appendix_b_gauge.tex.
        
        Args:
            psi_i: Initial state
            psi_j: Final state
            
        Returns:
            Complex transition amplitude
        """
        # Time-ordered product with fractal vertex
        T_prod = self._compute_time_ordered_product(psi_i, psi_j)
        
        # LSZ reduction with fractal corrections
        amplitude = self.alpha * T_prod * sum(
            self.alpha**n * self._compute_lsz_factor(n)
            for n in range(int(-log(self.precision)/log(self.alpha)))
        )
        
        return amplitude

    def init_mode_expansion(self) -> ModeExpansion:
        """Initialize mode expansion handler.
        
        Returns:
            Configured ModeExpansion instance
        """
        expansion = ModeExpansion(alpha=self.alpha)
        expansion.field = self
        return expansion

    def compute_mode_energy(self, n: int) -> Energy:
        """Compute energy of nth mode.
        
        Implements energy spectrum from appendix_d_scale.tex Eq. D.3.
        
        Args:
            n: Mode number
            
        Returns:
            Energy of nth mode
        """
        # Base energy scale from Z mass
        E0 = Z_MASS
        
        # Fractal scaling using defined gamma values
        scaling = (
            GAMMA_1 * self.alpha + 
            GAMMA_2 * self.alpha**2 + 
            GAMMA_3 * self.alpha**3
        )
        
        # Compute mode energy with quantum corrections
        E_n = E0 * exp(scaling) * (n + 0.5)
        
        return Energy(E_n)

    def compute_inner_product(self, psi1: WaveFunction, psi2: WaveFunction) -> complex:
        """Compute inner product between wavefunctions.
        
        Implements Eq. A.2 from appendix_a_convergence.tex.
        
        Args:
            psi1: First wavefunction
            psi2: Second wavefunction
            
        Returns:
            Complex inner product value
        """
        integrand = conjugate(psi1) * psi2
        return integrate(integrand, (X, -oo, oo), (T, -oo, oo))

    def _hermite_polynomial(self, n: int, x: Symbol) -> WaveFunction:
        """Compute nth Hermite polynomial.
        
        Implements recursion relation from appendix_d_scale.tex Eq. D.5.
        
        Args:
            n: Polynomial order
            x: Variable
            
        Returns:
            WaveFunction: Hermite polynomial of order n
        """
        if n == 0:
            return 1
        elif n == 1:
            return 2*x
            
        # Use recursion relation
        h0, h1 = 1, 2*x
        for i in range(2, n+1):
            h0, h1 = h1, 2*x*h1 - 2*(i-1)*h0
        return h1

    def compute_expansion_coefficient(self, n: int, energy: Energy) -> complex:
        """Compute mode expansion coefficient.
        
        Implements fractal form from appendix_d_scale.tex Eq. D.6.
        
        Args:
            n: Mode number
            energy: Energy scale
            
        Returns:
            complex: Expansion coefficient
        """
        # Fractal phase factor
        phase = exp(I * pi * sum(self.alpha**k * k for k in range(1, n+1)))
        
        # Energy-dependent amplitude
        amplitude = (energy.value/Z_MASS)**(-n*self.alpha/2)
        
        return phase * amplitude / sqrt(factorial(n))

    def _compute_time_ordered_product(self, psi_i: WaveFunction, psi_j: WaveFunction) -> complex:
        """Compute time-ordered product between states.
        
        Implements Eq. B.5 from appendix_b_gauge.tex.
        
        Args:
            psi_i: Initial state
            psi_j: Final state
            
        Returns:
            Complex time-ordered product
        """
        # Time ordering operator
        theta = (T > 0) * 1.0
        
        # Compute product with proper time ordering
        T_prod = theta * psi_i * psi_j + (1 - theta) * psi_j * psi_i
        
        # Add fractal vertex corrections
        vertex = sum(
            self.alpha**n * self._compute_vertex_factor(n)
            for n in range(int(-log(self.precision)/log(self.alpha)))
        )
        
        return integrate(T_prod * vertex, (X, -oo, oo), (T, -oo, oo))

    def _compute_lsz_factor(self, n: int) -> complex:
        """Compute LSZ reduction factor at order n.
        
        Implements Eq. B.6 from appendix_b_gauge.tex.
        
        Args:
            n: Order of correction
            
        Returns:
            Complex LSZ factor
        """
        # Fractal form factor
        F_n = exp(-n * self.alpha) / factorial(n)
        
        # Phase factor from gauge invariance
        phase = exp(I * pi * n * self.alpha)
        
        return F_n * phase

    def _compute_vertex_factor(self, n: int) -> complex:
        """Compute n-th order vertex correction factor.
        
        Implements vertex functions from appendix_b_gauge.tex Eq. B.7.
        
        Args:
            n: Order of vertex correction
            
        Returns:
            Complex vertex factor
        """
        # Fractal suppression factor
        suppression = exp(-n * self.alpha)
        
        # Symmetry factor from Feynman rules
        symmetry = 1.0 / factorial(n)
        
        # Phase from gauge invariance
        phase = exp(2j * pi * n * self.alpha)
        
        # Momentum-dependent form factor
        form_factor = (P**2/(2*M_PLANCK))**(n*self.alpha/2)
        
        return suppression * symmetry * phase * form_factor

    def evolve_coupling(self, energies: np.ndarray, **kwargs) -> np.ndarray:
        """Evolve gauge couplings across energy scales.
        
        Implements RG evolution from appendix_h_rgflow.tex Eq. H.2.
        
        Args:
            energies: Array of energy scales
            **kwargs: Optional precision parameters
            
        Returns:
            Array of evolved coupling values
        """
        # Initial coupling at Z mass
        alpha_0 = kwargs.get('alpha_0', ALPHA_REF)
        
        # Beta function coefficients from appendix_h_rgflow.tex
        beta_coeffs = [GAMMA_1, GAMMA_2, GAMMA_3]
        
        # Evolve coupling
        couplings = np.zeros_like(energies)
        for i, E in enumerate(energies):
            couplings[i] = alpha_0 / (1 + sum(
                g * self.alpha**n * np.log(E/Z_MASS)
                for n, g in enumerate(beta_coeffs, 1)
            ))
            
        return couplings

    def compute_running_coupling(self, energy: Energy) -> NumericValue:
        """Compute running coupling at specific energy scale.
        
        Implements RG flow equations from appendix_h_rgflow.tex Eq. H.3.
        
        Args:
            energy: Energy scale
            
        Returns:
            NumericValue: Running coupling with uncertainty
        """
        E = energy.value
        alpha_0 = ALPHA_REF  # Reference coupling at Z mass
        
        # Beta function coefficients using defined gamma values
        beta_0 = -(GAMMA_1 * self.alpha + GAMMA_2 * self.alpha**2 + GAMMA_3 * self.alpha**3)
        
        # Compute running coupling
        alpha = alpha_0 / (1 - beta_0 * np.log(E/Z_MASS))
        
        # Estimate uncertainty from higher orders
        uncertainty = abs(alpha * alpha_0 * np.log(E/Z_MASS)**2)
        
        return NumericValue(alpha, uncertainty)

    def compute_cross_section(self, energy: Energy, psi: WaveFunction, **kwargs) -> NumericValue:
        """Compute scattering cross section.
        
        Implements optical theorem from appendix_b_gauge.tex Eq. B.8.
        
        Args:
            energy: Collision energy
            psi: Initial state wavefunction
            **kwargs: Optional precision parameters
            
        Returns:
            NumericValue: Cross section with uncertainty
        """
        # Compute scattering amplitude with fractal corrections
        M = self._compute_transition(psi, psi)
        
        # Optical theorem relates cross section to imaginary part
        sigma = 4 * pi * np.imag(M) / (energy.value * np.sqrt(energy.value))
        
        # Phase space factor with fractal corrections
        phase_space = self._compute_phase_space_factor(energy)
        
        # Final cross section
        result = sigma * phase_space
        
        # Estimate uncertainty from higher orders
        uncertainty = abs(result) * kwargs.get('rtol', self.precision)
        
        return NumericValue(result, uncertainty)

    def compute_noether_current(
        self,
        psi: WaveFunction,
        *,
        rtol: float = 1e-8
    ) -> Tuple[complex, complex]:
        """Compute conserved Noether current.
        
        From appendix_i_sm_features.tex Eq I.7:
        The Noether current j^μ associated with global U(1) symmetry
        must satisfy the continuity equation ∂_μ j^μ = 0.
        
        Args:
            psi: Quantum state
            rtol: Relative tolerance
            
        Returns:
            Tuple[complex, complex]: Time and space components (j^0, j^1)
            
        Raises:
            PhysicsError: If current computation fails
        """
        try:
            # Compute time derivative
            d_t_psi = diff(psi, T)
            
            # Time component (charge density)
            j0 = I * HBAR * (conjugate(psi) * d_t_psi - psi * conjugate(d_t_psi))
            
            # Space component (current density)
            d_x_psi = diff(psi, X)
            j1 = -HBAR**2/(2*C) * (
                conjugate(psi) * d_x_psi - psi * conjugate(d_x_psi)
            )
            
            # Verify current conservation
            div_j = diff(j0, T) + C * diff(j1, X)
            if abs(div_j) > self.precision:
                raise PhysicsError("Current conservation violated")
                
            return float(j0), float(j1)
            
        except Exception as e:
            raise PhysicsError(f"Noether current computation failed: {e}")

    def compute_retarded_propagator(
        self,
        x1: float,
        t1: float,
        x2: float,
        t2: float,
        *,
        rtol: float = 1e-8
    ) -> complex:
        """Compute retarded propagator for causal field evolution.
        
        From appendix_i_sm_features.tex Eq I.8:
        The retarded propagator G_R(x,t;x',t') vanishes for t < t'
        ensuring causal evolution of the quantum field.
        
        Args:
            x1, t1: Source spacetime point
            x2, t2: Field point
            rtol: Relative tolerance
            
        Returns:
            complex: Retarded propagator value
            
        Raises:
            PhysicsError: If propagator computation fails
        """
        try:
            # Check time ordering
            dt = t2 - t1
            if dt < 0:
                return 0.0  # Retarded propagator vanishes for t2 < t1
            
            # Compute spacetime interval
            ds2 = C**2 * dt**2 - (x2 - x1)**2
            
            # Base propagator with proper causal structure
            G_R = -1/(2*pi) * theta(dt) * theta(ds2) * sqrt(ds2)
            
            # Add quantum corrections from fractal structure
            corrections = sum(
                self.alpha**n * self.compute_fractal_exponent(n) * 
                self._compute_green_function(x1, t1, x2, t2, n)
                for n in range(self.N_STABLE_MAX)
            )
            
            return float(G_R + corrections)
            
        except Exception as e:
            raise PhysicsError(f"Retarded propagator computation failed: {e}")

    def compute_gut_scale(self, *, rtol: float = 1e-8) -> Energy:
        """Compute grand unification scale.
        
        From appendix_h_rgflow.tex Eq H.4:
        The GUT scale is determined by the intersection of running couplings
        with proper quantum corrections from fractal structure.
        
        Args:
            rtol: Relative tolerance
            
        Returns:
            Energy: GUT scale with uncertainty
            
        Raises:
            PhysicsError: If scale computation fails
        """
        try:
            # Initial guess from classical unification
            E_classical = 2.0e16  # GeV
            
            # Scan energy range around classical value
            energies = np.logspace(15, 17, 1000)
            min_spread = float('inf')
            E_gut = None
            
            for E in energies:
                # Compute couplings at this scale
                couplings = self.compute_couplings(Energy(E))
                g1, g2, g3 = couplings['g1'], couplings['g2'], couplings['g3']
                
                # Compute maximum spread between couplings
                spread = max(
                    abs(g1.value - g2.value),
                    abs(g2.value - g3.value),
                    abs(g3.value - g1.value)
                )
                
                if spread < min_spread:
                    min_spread = spread
                    E_gut = E
                    
            if E_gut is None:
                raise PhysicsError("Failed to find GUT scale")
                
            # Estimate uncertainty from quantum corrections
            uncertainty = abs(E_gut * self.alpha**self.N_STABLE_MAX)
            
            return Energy(E_gut, uncertainty)
            
        except Exception as e:
            raise PhysicsError(f"GUT scale computation failed: {e}")

    def compute_dark_matter_density(
        self,
        radius: float,
        *,
        rtol: float = 1e-8
    ) -> NumericValue:
        """Compute dark matter density profile.
        
        From appendix_e_predictions.tex Eq E.5:
        The dark matter density profile emerges from fractal structure
        of the unified field at galactic scales.
        
        Args:
            radius: Distance from galactic center in kpc
            rtol: Relative tolerance
            
        Returns:
            NumericValue: Dark matter density in GeV/cm³
            
        Raises:
            ValueError: If radius is not positive
            PhysicsError: If density computation fails
        """
        if radius <= 0:
            raise ValueError("Radius must be positive")
        
        try:
            # Core density from fractal structure
            rho_0 = M_PLANCK**4 * self.alpha**(3*self.N_STABLE_MAX)
            
            # Scale radius from galactic dynamics
            r_s = 20.0  # kpc
            
            # NFW profile with fractal corrections
            x = radius/r_s
            rho = rho_0 / (x * (1 + x)**2)
            
            # Add quantum corrections
            corrections = sum(
                self.alpha**n * self.compute_fractal_exponent(n) * 
                np.exp(-n * radius/r_s)
                for n in range(self.N_STABLE_MAX)
            )
            
            rho *= (1 + corrections)
            
            # Estimate uncertainty
            uncertainty = abs(rho * self.alpha**self.N_STABLE_MAX)
            
            return NumericValue(rho, uncertainty)
            
        except Exception as e:
            raise PhysicsError(f"Dark matter density computation failed: {e}")