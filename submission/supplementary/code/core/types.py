"""Type definitions for fractal field theory framework."""

from typing import Any, Dict, List, Union, Optional, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
from sympy import Expr, Symbol
from .errors import ValidationError, PhysicsError
from .physics_constants import X
from math import log
from numpy import floating  # For type checking
from functools import wraps
from enum import Enum, auto

__all__ = [
    'RealValue',
    'ComplexValue',
    'NumericValue',
    'Energy',
    'Momentum',
    'CrossSection',
    'BranchingRatio',
    'ErrorEstimate',
    'FractalMode',
    'BasisConfig'
]

class ComputationMode(Enum):
    """
    Computation mode for quantum field calculations.
    
    From appendix_j_math_details.tex Eq J.30-J.32:
    Computation modes determine:
    1. SYMBOLIC: Exact analytical calculations
    2. NUMERIC: High-precision numerical integration
    3. HYBRID: Mixed symbolic-numeric methods
    """
    SYMBOLIC = auto()  # Exact analytical calculations
    NUMERIC = auto()   # Numerical integration
    HYBRID = auto()    # Mixed methods

@dataclass
class RealValue:
    """
    Real-valued physical quantity with uncertainty.
    
    Attributes:
        value (float): Central value
        uncertainty (Optional[float]): Uncertainty (if known)
        
    Examples:
        >>> x = RealValue(1.0, 0.1)  # Value with uncertainty
        >>> y = RealValue(2.0)  # Value without uncertainty
        >>> z = x + y  # Propagates uncertainties
    """
    value: float
    uncertainty: Optional[float] = None
    
    def __post_init__(self):
        """Validate real value."""
        if not isinstance(self.value, (int, float)):
            raise ValueError("Value must be a real number")
        if not np.isfinite(float(self.value)):
            raise ValueError("Value must be finite")
        if self.uncertainty is not None:
            if not isinstance(self.uncertainty, (int, float)):
                raise ValueError("Uncertainty must be a real number")
            if self.uncertainty < 0:
                raise ValueError("Uncertainty must be non-negative")
            if not np.isfinite(float(self.uncertainty)):
                raise ValueError("Uncertainty must be finite")
    
    def __float__(self) -> float:
        return float(self.value)
    
    def __add__(self, other: 'RealValue') -> 'RealValue':
        value = self.value + other.value
        if self.uncertainty is None or other.uncertainty is None:
            uncertainty = None
        else:
            uncertainty = np.sqrt(self.uncertainty**2 + other.uncertainty**2)
        return RealValue(value, uncertainty)
    
    def __mul__(self, other: 'RealValue') -> 'RealValue':
        value = self.value * other.value
        if self.uncertainty is None or other.uncertainty is None:
            uncertainty = None
        else:
            # Relative uncertainties add in quadrature
            rel_unc = np.sqrt((self.uncertainty/self.value)**2 + 
                            (other.uncertainty/other.value)**2)
            uncertainty = value * rel_unc
        return RealValue(value, uncertainty)

@dataclass
class ComplexValue:
    """
    Complex-valued physical quantity with uncertainty.
    
    Attributes:
        value (complex): Complex value
        uncertainty (Optional[float]): Uncertainty in absolute value
        
    Examples:
        >>> z = ComplexValue(1 + 1j, 0.1)
        >>> abs(z)  # Returns magnitude with uncertainty
        >>> z.phase  # Returns phase in radians
    """
    value: complex
    uncertainty: Optional[float] = None
    
    def __post_init__(self):
        """Validate complex value."""
        if not isinstance(self.value, (complex, float, int)):
            raise ValueError("Value must be complex number")
        if not np.isfinite(float(abs(self.value))):
            raise ValueError("Value must be finite")
        if self.uncertainty is not None:
            if not isinstance(self.uncertainty, (int, float)):
                raise ValueError("Uncertainty must be a real number")
            if self.uncertainty < 0:
                raise ValueError("Uncertainty must be non-negative")
            if not np.isfinite(float(self.uncertainty)):
                raise ValueError("Uncertainty must be finite")
    
    @property
    def magnitude(self) -> float:
        """Get magnitude of complex value."""
        return abs(self.value)
    
    @property 
    def phase(self) -> float:
        """Get phase angle in radians."""
        return np.angle(self.value)
    
    def __abs__(self) -> RealValue:
        """Get magnitude with propagated uncertainty."""
        mag = abs(self.value)
        if self.uncertainty is None:
            return RealValue(mag)
        return RealValue(mag, self.uncertainty)
    
    def conjugate(self) -> 'ComplexValue':
        """Get complex conjugate."""
        return ComplexValue(self.value.conjugate(), self.uncertainty)

@dataclass
class Energy:
    """
    Energy value with units.
    
    From appendix_j_math_details.tex Eq J.30-J.32:
    Energy must preserve:
    1. Positivity: E > 0
    2. Scaling: E → E' = αE under RG flow
    3. Unitarity: Im(E) = 0 for physical states
    """
    value: float
    uncertainty: Optional[float] = None
    units: str = 'GeV'
    
    def __post_init__(self):
        """Validate energy value."""
        if not isinstance(self.value, (int, float)):
            raise TypeError("Energy value must be numeric")
        if self.value <= 0:
            raise ValueError("Energy must be positive")
            
    def __truediv__(self, other: Union[float, int, 'Energy']) -> 'Energy':
        """
        Divide energy values while preserving units.
        
        From appendix_d_scale.tex Eq D.8:
        E/E₀ = exp(−S[ϕ]/ℏ) for RG flow
        """
        if isinstance(other, (float, int)):
            return Energy(
                value=self.value / other,
                uncertainty=self.uncertainty / other if self.uncertainty else None,
                units=self.units
            )
        elif isinstance(other, Energy):
            if other.units != self.units:
                raise ValueError("Cannot divide energies with different units")
            return self.value / other.value
        else:
            raise TypeError(f"Cannot divide Energy by {type(other)}")
            
    def __rtruediv__(self, other: Union[float, int]) -> 'Energy':
        """Right division for Energy."""
        if isinstance(other, (float, int)):
            return Energy(
                value=other / self.value,
                uncertainty=other * self.uncertainty / self.value**2 if self.uncertainty else None,
                units=f'1/{self.units}'
            )
        return NotImplemented

    def __eq__(self, other: Union[float, int, 'Energy']) -> bool:
        """
        Compare energy values.
        
        From appendix_j_math_details.tex Eq J.33:
        E₁ = E₂ iff |E₁ - E₂| < ε where ε = uncertainty
        """
        if isinstance(other, (float, int)):
            return abs(self.value - other) < 1e-10
        elif isinstance(other, Energy):
            if other.units != self.units:
                raise ValueError("Cannot compare energies with different units")
            return abs(self.value - other.value) < 1e-10
        return NotImplemented

    def __lt__(self, other: Union[float, int, 'Energy']) -> bool:
        """
        Compare energy values.
        
        From appendix_j_math_details.tex Eq J.34:
        E₁ < E₂ iff E₁ - E₂ < -ε where ε = uncertainty
        """
        if isinstance(other, (float, int)):
            return self.value < other
        elif isinstance(other, Energy):
            if other.units != self.units:
                raise ValueError("Cannot compare energies with different units")
            return self.value < other.value
        return NotImplemented

    def __gt__(self, other: Union[float, int, 'Energy']) -> bool:
        """
        Compare energy values.
        
        From appendix_j_math_details.tex Eq J.35:
        E₁ > E₂ iff E₁ - E₂ > ε where ε = uncertainty
        """
        if isinstance(other, (float, int)):
            return self.value > other
        elif isinstance(other, Energy):
            if other.units != self.units:
                raise ValueError("Cannot compare energies with different units")
            return self.value > other.value
        return NotImplemented

@dataclass
class Momentum(RealValue):
    """
    A physical momentum value with units and uncertainty propagation.
    
    Represents a momentum value in particle physics calculations, with support for
    unit conversions, uncertainty propagation, and validation of physical constraints.
    
    Attributes:
        value (float): Momentum value in GeV/c
        uncertainty (Optional[float]): Statistical uncertainty in GeV/c
        units (str): Momentum units (default: GeV/c)
        systematics (Dict[str, float]): Systematic uncertainties by source
        
    Methods:
        to_units(new_units: str) -> Momentum:
            Convert to different units (GeV/c <-> TeV/c <-> MeV/c)
        __add__, __mul__: Propagate uncertainties in calculations
        
    Examples:
        >>> # Create momentum with uncertainty
        >>> p = Momentum(100.0, 5.0)  # 100 ± 5 GeV/c
        >>> 
        >>> # Convert to TeV/c
        >>> p_tev = p.to_units("TeV/c")
        >>> print(f"{p_tev.value:.3f} ± {p_tev.uncertainty:.3f} {p_tev.units}")
        0.100 ± 0.005 TeV/c
        >>> 
        >>> # Arithmetic with unit validation
        >>> p1 = Momentum(100.0, 5.0)  # GeV/c
        >>> p2 = p1.to_units("TeV/c")  # TeV/c
        >>> with pytest.raises(ValueError):
        ...     p_sum = p1 + p2  # Cannot add different units
        
    Notes:
        - Momentum values must be non-negative (physical requirement)
        - Uncertainties are propagated using standard error propagation rules
        - Unit conversions preserve relative uncertainties
        - Operations between momenta require matching units
        - Supported units: GeV/c, TeV/c, MeV/c
    """
    units: str = "GeV/c"
    
    def __post_init__(self):
        """Validate momentum value and units."""
        super().__post_init__()
        if self.value < 0:
            raise ValueError("Momentum cannot be negative")
        if self.units not in ["GeV/c", "TeV/c", "MeV/c"]:
            raise ValueError(f"Unsupported momentum units: {self.units}")
    
    def to_units(self, new_units: str) -> 'Momentum':
        """
        Convert to different units.
        
        Supported conversions:
        - GeV/c ↔ TeV/c (factor: 10³)
        - GeV/c ↔ MeV/c (factor: 10⁻³)
        
        Args:
            new_units (str): Target unit (TeV/c, GeV/c, MeV/c)
            
        Returns:
            Momentum: Converted momentum value with propagated uncertainty
            
        Raises:
            ValueError: If conversion not supported
        """
        conversions = {
            ("GeV/c", "TeV/c"): 1e-3,
            ("TeV/c", "GeV/c"): 1e3,
            ("GeV/c", "MeV/c"): 1e3,
            ("MeV/c", "GeV/c"): 1e-3
        }
        
        if (self.units, new_units) in conversions:
            factor = conversions[(self.units, new_units)]
            return Momentum(
                value=self.value * factor,
                uncertainty=self.uncertainty * factor if self.uncertainty else None,
                units=new_units
            )
        raise ValueError(f"Unsupported unit conversion: {self.units} to {new_units}")

    def __add__(self, other: 'Momentum') -> 'Momentum':
        """
        Add two momentum values with uncertainty propagation.
        
        Implements vector addition of momenta, propagating both statistical
        and systematic uncertainties according to standard error propagation rules.
        
        Args:
            other (Momentum): Momentum value to add
            
        Returns:
            Momentum: Sum of momenta with propagated uncertainties
            
        Examples:
            >>> p1 = Momentum(100.0, 5.0)  # 100 ± 5 GeV/c
            >>> p2 = Momentum(50.0, 2.0)   # 50 ± 2 GeV/c
            >>> p_total = p1 + p2
            >>> print(f"{p_total.value:.1f} ± {p_total.uncertainty:.1f} {p_total.units}")
            150.0 ± 5.4 GeV/c
        """
        if not isinstance(other, Momentum):
            raise TypeError("Can only add Momentum objects")
        if self.units != other.units:
            raise ValueError("Cannot add momenta with different units")
            
        value = self.value + other.value
        uncertainty = np.sqrt(
            (self.uncertainty or 0)**2 + 
            (other.uncertainty or 0)**2
        ) if self.uncertainty or other.uncertainty else None
        
        return Momentum(value, uncertainty, units=self.units)

    def __mul__(self, other: Union[float, 'Momentum']) -> 'Momentum':
        """
        Multiply momentum by scalar or another momentum.
        
        Handles both scalar multiplication and momentum-momentum multiplication,
        with proper uncertainty propagation using relative uncertainties.
        
        Args:
            other (Union[float, Momentum]): Scalar or momentum to multiply by
            
        Returns:
            Momentum: Product with propagated uncertainties
            
        Examples:
            >>> p = Momentum(100.0, 5.0)  # 100 ± 5 GeV/c
            >>> p_scaled = p * 2.0
            >>> print(f"{p_scaled.value:.1f} ± {p_scaled.uncertainty:.1f} {p_scaled.units}")
            200.0 ± 10.0 GeV/c
        
        Notes:
            - For scalar multiplication, relative uncertainty is preserved
            - For momentum-momentum multiplication, relative uncertainties add in quadrature
        """
        if isinstance(other, (int, float)):
            value = self.value * other
            uncertainty = self.uncertainty * abs(other) if self.uncertainty else None
            return Momentum(value, uncertainty, units=self.units)
        elif isinstance(other, Momentum):
            if self.units != other.units:
                raise ValueError("Cannot multiply momenta with different units")
            value = self.value * other.value
            if self.uncertainty is None and other.uncertainty is None:
                uncertainty = None
            else:
                rel_unc = np.sqrt(
                    (self.uncertainty/self.value)**2 + 
                    (other.uncertainty/other.value)**2
                )
                uncertainty = value * rel_unc
            return Momentum(value, uncertainty, units=self.units)
        else:
            raise TypeError("Unsupported operand type")

@dataclass
class FieldConfig:
    """Configuration for field theory computations.
    
    From appendix_i_sm_features.tex Eq I.1:
    Field configurations are specified by mass, coupling, and dimension.
    
    Attributes:
        mass: Mass parameter (GeV)
        coupling: Coupling constant
        dimension: Spacetime dimension
        max_level: Maximum fractal level
        precision: Numerical precision
    """
    mass: float
    coupling: float
    dimension: int = 4
    max_level: int = 10
    precision: float = 1e-8
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.mass <= 0:
            raise ValueError("Mass must be positive")
        if self.coupling <= 0:
            raise ValueError("Coupling must be positive")
        if self.dimension <= 0:
            raise ValueError("Dimension must be positive")
        if self.max_level < 1:
            raise ValueError("Max level must be positive")
        if self.precision <= 0:
            raise ValueError("Precision must be positive")
            
    @property
    def scaling_dimension(self) -> float:
        """Compute scaling dimension from spacetime dimension."""
        return (self.dimension - 2)/2
        
    def to_dict(self) -> Dict[str, Union[float, int]]:
        """Convert to dictionary for serialization."""
        return {
            'mass': self.mass,
            'coupling': self.coupling,
            'dimension': self.dimension,
            'max_level': self.max_level,
            'precision': self.precision
        }

@dataclass
class WaveFunction:
    """Class representing a quantum wavefunction."""
    
    def __init__(
        self, 
        psi: np.ndarray,
        grid: np.ndarray,
        mass: float,  # Add mass parameter
        quantum_numbers: Dict[str, float]
    ):
        """
        Initialize wavefunction.
        
        Args:
            psi: Complex wavefunction values
            grid: Spatial grid points
            mass: Mass parameter in GeV
            quantum_numbers: Dictionary of quantum numbers
        """
        self.psi = psi
        self.grid = grid
        self.mass = mass  # Store mass
        self.quantum_numbers = quantum_numbers
        
        # Validate inputs
        if not isinstance(psi, np.ndarray):
            raise TypeError("psi must be a numpy array")
        if not isinstance(grid, np.ndarray):
            raise TypeError("grid must be a numpy array")
        if not isinstance(mass, (int, float)):
            raise TypeError("mass must be a number")
        if not isinstance(quantum_numbers, dict):
            raise TypeError("quantum_numbers must be a dictionary")
        
        # Validate wavefunction properties
        self._validate()
    
    def _validate(self):
        """Validate wavefunction properties."""
        if isinstance(self.psi, np.ndarray):
            if not np.all(np.isfinite(self.psi)):
                raise ValueError("Wavefunction must be finite")
            if self.grid is not None and len(self.grid) != len(self.psi):
                raise ValueError("Grid and wavefunction dimensions must match")
                
    @property
    def norm(self) -> float:
        """Compute wavefunction norm."""
        if isinstance(self.psi, np.ndarray):
            return float(np.sqrt(np.sum(np.abs(self.psi)**2 * np.diff(self.grid)[0])))
        return 1.0  # Symbolic wavefunctions assumed normalized

    def normalize(self) -> None:
        """
        Normalize wavefunction.
        
        Ensures the wavefunction is properly normalized according to:
        ∫|ψ(x)|² dx = 1
        
        Uses trapezoidal integration over the grid points.
        Raises PhysicsError if normalization fails.
        
        See appendix_a_convergence.tex for mathematical details.
        """
        # Compute norm using trapezoidal rule
        integrand = np.abs(self.psi)**2
        norm = np.sqrt(np.trapz(integrand, self.grid))
        
        if norm <= 0:
            raise PhysicsError("Cannot normalize zero wavefunction")
        if not np.isfinite(norm):
            raise PhysicsError("Normalization integral diverges")
            
        self.psi /= norm

    def compute_expectation(self, operator: Union[np.ndarray, Callable]) -> complex:
        """
        Compute quantum expectation value.
        
        Args:
            operator: Matrix or function representing observable
            
        Returns:
            Complex expectation value ⟨ψ|A|ψ⟩
            
        See appendix_k_io_distinction.tex for measurement theory.
        """
        if callable(operator):
            op_matrix = operator(self.grid)
        else:
            op_matrix = operator
            
        # Compute ⟨ψ|A|ψ⟩
        expectation = np.trapz(
            np.conjugate(self.psi) * op_matrix @ self.psi,
            self.grid
        )
        
        return complex(expectation)

    def __mul__(self, other: Union[float, complex, 'WaveFunction']) -> 'WaveFunction':
        """Multiply wavefunctions or scale by number."""
        if isinstance(other, (float, complex, int)):
            return WaveFunction(
                psi=self.psi * other,
                grid=self.grid.copy(),
                quantum_numbers=self.quantum_numbers.copy()
            )
        elif isinstance(other, WaveFunction):
            if not np.array_equal(self.grid, other.grid):
                raise ValidationError("Grid points must match for multiplication")
            return WaveFunction(
                psi=self.psi * other.psi,
                grid=self.grid.copy(),
                quantum_numbers={**self.quantum_numbers, **other.quantum_numbers}
            )
        return NotImplemented

    def __rmul__(self, other: Union[float, complex]) -> 'WaveFunction':
        """Right multiplication."""
        return self.__mul__(other)

    def inner_product(self, other: 'WaveFunction') -> complex:
        """
        Compute inner product with another wavefunction.
        
        Args:
            other: WaveFunction to compute inner product with
            
        Returns:
            Complex inner product value
            
        Raises:
            ValidationError: If grids don't match
        """
        if not np.array_equal(self.grid, other.grid):
            raise ValidationError("Wavefunctions must be defined on same grid")
            
        integrand = np.conjugate(self.psi) * other.psi
        return complex(np.trapz(integrand, self.grid))

    @classmethod
    def from_expression(cls, expr: Expr, grid: Optional[np.ndarray] = None):
        """
        Create WaveFunction from symbolic expression.
        
        Args:
            expr (Expr): Symbolic expression for wavefunction
            grid (Optional[np.ndarray]): Grid points for evaluation
            
        Returns:
            WaveFunction: Initialized wavefunction object
            
        Examples:
            >>> expr = exp(-X**2/(2*HBAR))
            >>> wf = WaveFunction.from_expression(expr)
        """
        if grid is None:
            grid = np.linspace(-10, 10, 100)
        psi = np.array([complex(expr.subs(X, x)) for x in grid])
        return cls(psi=psi, grid=grid, quantum_numbers={'n': 0})

    def validate(self) -> None:
        """Additional validation of wavefunction."""
        if not isinstance(self.grid, np.ndarray):
            raise ValidationError("Grid must be numpy array")
        if not isinstance(self.quantum_numbers, dict):
            raise ValidationError("Quantum numbers must be dictionary")

    def compute_overlap(self, other: 'WaveFunction') -> complex:
        """
        Compute quantum mechanical overlap integral.
        
        Implements overlap calculation from appendix_a_convergence.tex:
        ⟨ψ₁|ψ₂⟩ = ∫ ψ₁*(x) ψ₂(x) dx
        
        Args:
            other: WaveFunction to compute overlap with
            
        Returns:
            Complex overlap integral
            
        Raises:
            ValidationError: If grid points don't match
        """
        if not np.array_equal(self.grid, other.grid):
            raise ValidationError("Grid points must match for overlap calculation")
            
        return complex(np.trapz(
            np.conjugate(self.psi) * other.psi,
            self.grid
        ))

    def evolve(self, hamiltonian: Union[np.ndarray, Callable], time: float) -> 'WaveFunction':
        """
        Time evolve wavefunction under given Hamiltonian.
        
        Implements Schrödinger evolution from appendix_k_io_distinction.tex:
        |ψ(t)⟩ = exp(-iHt/ħ)|ψ(0)⟩
        
        Args:
            hamiltonian: Energy operator (matrix or function)
            time: Evolution time
            
        Returns:
            Evolved wavefunction
            
        See appendix_k_io_distinction.tex for detailed evolution equations.
        """
        if callable(hamiltonian):
            H = hamiltonian(self.grid)
        else:
            H = hamiltonian
            
        # Compute evolution operator exp(-iHt/ħ)
        from scipy.linalg import expm
        from .physics_constants import HBAR
        
        U = expm(-1j * H * time / HBAR)
        
        # Evolve state
        evolved_psi = U @ self.psi
        
        return WaveFunction(
            psi=evolved_psi,
            grid=self.grid.copy(),
            quantum_numbers=self.quantum_numbers.copy()
        )

    def to_momentum_space(self) -> 'WaveFunction':
        """
        Transform wavefunction to momentum space.
        
        Implements Fourier transform from appendix_b_gauge.tex:
        ψ(p) = (1/√(2πħ)) ∫ ψ(x) exp(-ipx/ħ) dx
        
        Returns:
            WaveFunction in momentum representation
            
        See appendix_b_gauge.tex for gauge transformations.
        """
        from numpy.fft import fft, fftfreq
        from .physics_constants import HBAR
        
        # Compute momentum grid
        dp = 2*np.pi*HBAR / (self.grid[-1] - self.grid[0])
        p_grid = fftfreq(len(self.grid), d=dp) * 2*np.pi*HBAR
        
        # Perform FFT with proper normalization
        psi_p = fft(self.psi) / np.sqrt(len(self.grid))
        
        # Update quantum numbers for momentum space
        p_numbers = self.quantum_numbers.copy()
        p_numbers['representation'] = 'momentum'
        
        return WaveFunction(
            psi=psi_p,
            grid=p_grid,
            quantum_numbers=p_numbers
        )
        
    def uncertainty(self, operator: Union[np.ndarray, Callable]) -> float:
        """
        Compute quantum mechanical uncertainty.
        
        Implements uncertainty calculation from appendix_k_io_distinction.tex:
        ΔA = √(⟨A²⟩ - ⟨A⟩²)
        
        Args:
            operator: Observable operator
            
        Returns:
            Standard deviation of observable
            
        See appendix_k_io_distinction.tex for measurement theory.
        """
        # Compute expectation values
        exp_A = self.compute_expectation(operator)
        exp_A2 = self.compute_expectation(lambda x: operator(x) @ operator(x))
        
        # Return standard deviation
        return float(np.sqrt(abs(exp_A2 - exp_A**2)))

    def compute_density_matrix(self) -> np.ndarray:
        """
        Compute quantum density matrix.
        
        Implements density operator from appendix_k_io_distinction.tex:
        ρ = |ψ⟩⟨ψ|
        
        Returns:
            Complex density matrix
            
        See appendix_k_io_distinction.tex for measurement theory.
        """
        # Compute outer product |ψ⟩⟨ψ|
        return np.outer(self.psi, np.conjugate(self.psi))
        
    def compute_entropy(self) -> float:
        """
        Compute von Neumann entropy.
        
        Implements entropy calculation from appendix_g_holographic.tex:
        S = -Tr(ρ ln ρ)
        
        Returns:
            Entropy value (dimensionless)
            
        See appendix_g_holographic.tex for holographic principles.
        """
        # Compute eigenvalues of density matrix
        rho = self.compute_density_matrix()
        eigenvals = np.linalg.eigvalsh(rho)
        
        # Remove negligible eigenvalues
        eigenvals = eigenvals[eigenvals > 1e-15]
        
        # Compute von Neumann entropy
        return float(-np.sum(eigenvals * np.log(eigenvals)))
        
    def compute_correlation_length(self) -> float:
        """
        Compute quantum correlation length.
        
        Implements correlation analysis from appendix_k_io_distinction.tex:
        ξ = √(⟨x²⟩ - ⟨x⟩²)
        
        Returns:
            Correlation length in grid units
            
        See appendix_k_io_distinction.tex for locality principles.
        """
        # Position operator
        x_op = lambda x: np.diag(x)
        
        # Compute correlation length using uncertainty
        return self.uncertainty(x_op)

    @classmethod
    def _sympy_(cls):
        """Convert to SymPy expression for symbolic manipulation."""
        # Return a symbolic variable for the wavefunction
        return Symbol('psi', commutative=False)

    def __truediv__(self, other: Union[float, int, complex, floating]) -> 'WaveFunction':
        """
        Implement division by scalar.
        
        Args:
            other: Scalar value to divide by
            
        Returns:
            WaveFunction: New wavefunction with divided values
            
        Raises:
            TypeError: If other is not a scalar type
            ZeroDivisionError: If dividing by zero
        """
        # Handle all numeric types by converting to float
        try:
            other_float = float(other)
            if other_float == 0:
                raise ZeroDivisionError("Cannot divide wavefunction by zero")
            return WaveFunction(
                psi=self.psi / other_float,
                grid=self.grid,
                quantum_numbers=self.quantum_numbers
            )
        except (TypeError, ValueError):
            return NotImplemented

    def __repr__(self) -> str:
        """String representation."""
        return f"WaveFunction(psi={self.psi}, grid={self.grid}, quantum_numbers={self.quantum_numbers})"

@dataclass
class AnalysisResult:
    """
    Analysis result with uncertainties.
    
    Attributes:
        observable (str): Observable name
        value (RealValue): Measured value
        systematics (Dict): Systematic uncertainties
        metadata (Dict): Additional metadata
        
    Examples:
        >>> result = AnalysisResult(
        ...     observable="mass",
        ...     value=RealValue(125.0, 0.5),
        ...     systematics={'detector': 0.3, 'theory': 0.4},
        ...     metadata={'timestamp': '2024-01-01'}
        ... )
        >>> print(result.total_uncertainty)  # Combined uncertainty
    """
    observable: str
    value: RealValue
    systematics: Dict[str, float]
    metadata: Dict[str, Any]
    
    @property
    def total_uncertainty(self) -> float:
        """Compute total uncertainty."""
        stat = self.value.uncertainty or 0
        syst = np.sqrt(sum(s**2 for s in self.systematics.values()))
        return np.sqrt(stat**2 + syst**2)

@dataclass
class CrossSection(RealValue):
    """
    A physical cross-section value with units and uncertainty propagation.
    
    Represents a cross-section in particle physics calculations, with support for
    unit conversions, uncertainty propagation, and validation of physical constraints.
    
    Attributes:
        value (float): Cross-section value in pb (picobarns)
        uncertainty (Optional[float]): Statistical uncertainty in pb
        units (str): Cross-section units (default: pb)
        systematics (Dict[str, float]): Systematic uncertainties by source
        
    Methods:
        to_units(new_units: str) -> CrossSection:
            Convert to different units (pb <-> fb <-> nb)
        __add__, __mul__: Propagate uncertainties in calculations
        
    Examples:
        >>> # Create cross-section with uncertainty
        >>> sigma = CrossSection(45.0, 2.1)  # 45 ± 2.1 pb
        >>> 
        >>> # Convert to femtobarns
        >>> sigma_fb = sigma.to_units("fb")
        >>> print(f"{sigma_fb.value:.1f} ± {sigma_fb.uncertainty:.1f} {sigma_fb.units}")
        45000.0 ± 2100.0 fb
        
    Notes:
        - Cross-section values must be non-negative
        - Uncertainties are propagated using standard error propagation rules
        - Unit conversions preserve relative uncertainties
    """
    units: str = "pb"
    
    def __post_init__(self):
        """Validate cross-section value."""
        super().__post_init__()
        if self.value < 0:
            raise ValueError("Cross-section cannot be negative")
    
    def to_units(self, new_units: str) -> 'CrossSection':
        """
        Convert to different units.
        
        Supported conversions:
        - pb ↔ fb (factor: 10³)
        - pb ↔ nb (factor: 10⁻³)
        
        Args:
            new_units (str): Target unit (pb, fb, nb)
            
        Returns:
            CrossSection: Converted value with propagated uncertainty
            
        Raises:
            ValueError: If conversion not supported
        """
        conversions = {
            ("pb", "fb"): 1e3,
            ("fb", "pb"): 1e-3,
            ("pb", "nb"): 1e-3,
            ("nb", "pb"): 1e3
        }
        
        if (self.units, new_units) in conversions:
            factor = conversions[(self.units, new_units)]
            return CrossSection(
                value=self.value * factor,
                uncertainty=self.uncertainty * factor if self.uncertainty else None,
                units=new_units
            )
        raise ValueError(f"Unsupported unit conversion: {self.units} to {new_units}")

@dataclass
class BranchingRatio(RealValue):
    """
    A physical branching ratio value with uncertainty propagation.
    
    Represents a decay branching ratio in particle physics, with support for
    uncertainty propagation and validation of physical constraints.
    
    Attributes:
        value (float): Branching ratio value (0 to 1)
        uncertainty (Optional[float]): Statistical uncertainty
        process (str): Description of decay process
        systematics (Dict[str, float]): Systematic uncertainties by source
        
    Methods:
        __add__, __mul__: Propagate uncertainties in calculations
        validate(): Ensure value is in valid range
        
    Examples:
        >>> # Create branching ratio with uncertainty
        >>> br = BranchingRatio(0.0321, 0.0015, process="H->ZZ")
        >>> print(f"BR({br.process}) = {br.value:.4f} ± {br.uncertainty:.4f}")
        BR(H->ZZ) = 0.0321 ± 0.0015
        
    Notes:
        - Values must be between 0 and 1
        - Sum of branching ratios must equal 1
        - Uncertainties are propagated using standard error propagation
    """
    process: str = ""
    
    def __post_init__(self):
        """Validate branching ratio value."""
        super().__post_init__()
        if not 0 <= self.value <= 1:
            raise ValueError("Branching ratio must be between 0 and 1")

@dataclass
class ErrorEstimate:
    """
    Represents an error estimate with statistical and systematic components.
    
    Attributes:
        value (float): Central value
        statistical (float): Statistical uncertainty
        systematic (Dict[str, float]): Systematic uncertainties by source
        
    Examples:
        >>> estimate = ErrorEstimate(100.0, 5.0, {'detector': 2.0, 'theory': 3.0})
        >>> print(estimate.total_uncertainty)  # Combined uncertainty
    """
    value: float
    statistical: float
    systematic: Dict[str, float] = None
    
    def __post_init__(self):
        if self.systematic is None:
            self.systematic = {}
    
    @property
    def total_uncertainty(self) -> float:
        """Calculate total uncertainty combining statistical and systematic."""
        syst_squared = sum(x*x for x in self.systematic.values())
        return np.sqrt(self.statistical**2 + syst_squared)
    
    def __str__(self) -> str:
        return f"{self.value:.3g} ± {self.total_uncertainty:.3g}"

@dataclass
class NumericValue:
    """Represents a numeric value with uncertainty."""
    value: float
    uncertainty: float = 0.0

    def __post_init__(self):
        """Initialize and validate numeric value."""
        if self.uncertainty < 0:
            raise ValueError("Uncertainty must be non-negative")

    @property
    def is_real(self) -> bool:
        """Check if value is real (has zero imaginary part)."""
        return abs(float(self.value.imag)) < 1e-10 if hasattr(self.value, 'imag') else True

    def __float__(self):
        return float(self.value)

    def __lt__(self, other: 'NumericValue') -> bool:
        """Less than comparison using value."""
        return self.value < other.value
        
    def __gt__(self, other: 'NumericValue') -> bool:
        """Greater than comparison using value."""
        return self.value > other.value

    def __post_init__(self):
        """Validate and initialize value."""
        # Convert numpy types to standard Python types
        if isinstance(self.value, (np.number, np.ndarray)):
            if isinstance(self.value, np.ndarray):
                if self.value.size != 1:
                    raise ValueError("Can only convert scalar arrays")
                self.value = float(self.value.item())
            else:
                self.value = float(self.value)
        
        self._validate()
    
    def _validate(self):
        """Validate value and uncertainty."""
        # Validate value
        if isinstance(self.value, (int, float, np.number)):
            if not np.isfinite(float(self.value)):
                raise ValueError("Value must be finite")
        elif isinstance(self.value, complex):
            if not (np.isfinite(self.value.real) and np.isfinite(self.value.imag)):
                raise ValueError("Complex value must be finite")
        else:
            raise TypeError(f"Invalid value type: {type(self.value)}")

        # Validate uncertainty
        if self.uncertainty is not None:
            if not isinstance(self.uncertainty, (int, float)):
                raise TypeError("Uncertainty must be real number")
            if self.uncertainty < 0:
                raise ValueError("Uncertainty must be non-negative")
            if not np.isfinite(float(self.uncertainty)):
                raise ValueError("Uncertainty must be finite")

    @property
    def value(self) -> Union[float, complex]:
        """Get the value with type validation.
        
        Returns:
            Union[float, complex]: The stored value
        """
        if isinstance(self._value, (np.number, np.ndarray)):
            return float(self._value)
        return self._value

    @value.setter 
    def value(self, val: Union[float, complex, np.ndarray, 'NumericValue']) -> None:
        """Set value with type conversion.
        
        Args:
            val: New value to set
        """
        if isinstance(val, NumericValue):
            self._value = val.value
        elif isinstance(val, (np.number, np.ndarray)):
            self._value = float(val)
        elif isinstance(val, (int, float, complex)):
            self._value = val
        else:
            raise TypeError(f"Cannot convert {type(val)} to numeric value")

    @property
    def real(self) -> float:
        """Real part of the value."""
        return float(self.value.real if isinstance(self.value, complex) else self.value)

    @property 
    def imag(self) -> float:
        """Imaginary part of the value."""
        return float(self.value.imag if isinstance(self.value, complex) else 0.0)

    def conjugate(self) -> 'NumericValue':
        """Complex conjugate."""
        if isinstance(self.value, complex):
            return NumericValue(self.value.conjugate(), self.uncertainty)
        return self

    @classmethod
    def ensure_numeric(cls, value: Union[int, float, np.number, 'NumericValue']) -> 'NumericValue':
        """Convert raw numeric types to NumericValue."""
        if isinstance(value, NumericValue):
            return value
        if isinstance(value, (int, float, np.number)):
            return cls(float(value))
        raise TypeError(f"Cannot convert {type(value)} to NumericValue")
    @property
    def magnitude(self) -> float:
        """Get absolute magnitude of value."""
        if isinstance(self.value, complex):
            return abs(self.value)
        return abs(float(self.value))
        
    @property
    def phase(self) -> float:
        """Get phase angle for complex values (in radians)."""
        if isinstance(self.value, complex):
            return np.angle(self.value)
        return 0.0

    def __complex__(self) -> complex:
        """Enable complex number conversion.
        
        Enhances numeric compatibility by allowing:
        complex(numeric_value)
        
        Returns:
            complex: Complex representation of value
        """
        return complex(self.value)

    def __abs__(self) -> float:
        """Enable abs() function."""
        return self.magnitude

    def __add__(self, other: Union[float, complex, 'NumericValue']) -> 'NumericValue':
        """Add with uncertainty propagation."""
        other = self.ensure_numeric(other)
        value = self.value + other.value
        
        # Keep existing uncertainty propagation
        if self.uncertainty is None and other.uncertainty is None:
            return NumericValue(value)
            
        unc1 = self.uncertainty or 0
        unc2 = other.uncertainty or 0
        uncertainty = np.sqrt(unc1**2 + unc2**2)
        
        # Handle complex uncertainty propagation
        if isinstance(value, complex):
            uncertainty = abs(uncertainty)  # Use magnitude for complex values
        
        return NumericValue(value, uncertainty)

    def __mul__(self, other: 'NumericValue') -> 'NumericValue':
        """Multiply values with uncertainty propagation."""
        value = self.value * other.value
        
        if self.uncertainty is None or other.uncertainty is None:
            uncertainty = None
        else:
            # Relative uncertainties add in quadrature for multiplication
            rel_unc = np.sqrt(
                (self.uncertainty/abs(self.value))**2 +
                (other.uncertainty/abs(other.value))**2
            )
            uncertainty = abs(value) * rel_unc
            
        return NumericValue(value, uncertainty)
    
    def __abs__(self) -> float:
        """Get absolute value/magnitude."""
        return abs(self.value)
    
    def __str__(self) -> str:
        """String representation with uncertainty."""
        if self.uncertainty is None:
            return f"{self.value}"
        return f"{self.value} ± {self.uncertainty}"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"NumericValue(value={self.value}, uncertainty={self.uncertainty})"
    
    def __truediv__(self, other: Union[int, float, 'NumericValue']) -> 'NumericValue':
        """Implement division with uncertainty propagation."""
        if isinstance(other, (int, float)):
            value = self.value / other
            uncertainty = self.uncertainty / abs(other) if self.uncertainty is not None else None
            return NumericValue(value, uncertainty)
        elif isinstance(other, NumericValue):
            value = self.value / other.value
            if self.uncertainty is None and other.uncertainty is None:
                return NumericValue(value)
            # Propagate uncertainties using quadrature
            rel_unc1 = (self.uncertainty / abs(self.value)) if self.uncertainty is not None else 0
            rel_unc2 = (other.uncertainty / abs(other.value)) if other.uncertainty is not None else 0
            uncertainty = abs(value) * np.sqrt(rel_unc1**2 + rel_unc2**2)
            return NumericValue(value, uncertainty)
        return NotImplemented
    
    def __sub__(self, other: 'NumericValue') -> 'NumericValue':
        """Subtract two values with uncertainty propagation."""
        if not isinstance(other, NumericValue):
            other = NumericValue(float(other))
        value = self.value - other.value
        if self.uncertainty is None and other.uncertainty is None:
            return NumericValue(value)
        unc1 = self.uncertainty or 0
        unc2 = other.uncertainty or 0
        uncertainty = np.sqrt(unc1**2 + unc2**2)
        return NumericValue(value, uncertainty)
    
    def __array__(self) -> np.ndarray:
        """Convert to numpy array (for numpy compatibility)."""
        return np.array(self.value)

    @classmethod
    def from_complex(cls, value: complex, uncertainty: Optional[float] = None) -> 'NumericValue':
        """Create NumericValue from complex number."""
        if not isinstance(value, complex):
            raise TypeError("Value must be complex")
        return cls(value, uncertainty)

    def to_complex(self) -> complex:
        """Convert value to complex number."""
        if isinstance(self.value, complex):
            return self.value
        return complex(self.value)

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for computation.
        
        Enhances scientific computation compatibility.
        
        Returns:
            np.ndarray: Array containing value
        """
        return np.array(self.value)

    @classmethod
    def from_measurement(cls, value: float, instrument_error: float) -> 'NumericValue':
        """Create from physical measurement with instrument error.
        
        Enhances experimental data handling.
        
        Args:
            value: Measured value
            instrument_error: Instrument uncertainty
            
        Returns:
            NumericValue with propagated uncertainty
        """
        return cls(value, abs(instrument_error))

    def __pow__(self, other: Union[int, float, 'NumericValue']) -> 'NumericValue':
        """Implement power operation with uncertainty propagation."""
        if isinstance(other, (int, float)):
            value = self.value ** other
            # Error propagation for power
            uncertainty = abs(other * self.value**(other-1) * self.uncertainty)
            return NumericValue(value, uncertainty)
        elif isinstance(other, NumericValue):
            value = self.value ** other.value
            # Error propagation for power with uncertain exponent
            uncertainty = abs(value * (
                (other.value/self.value * self.uncertainty)**2 +
                (log(self.value) * other.uncertainty)**2
            )**0.5)
            return NumericValue(value, uncertainty)
        return NotImplemented

def ensure_numeric_value(value: Any) -> 'NumericValue':
    """Ensure a value is wrapped in NumericValue."""
    if hasattr(value, 'value'):
        return ensure_numeric_value(value.value)

@dataclass
class FractalMode:
    """
    Fractal basis mode function.
    
    From appendix_b_basis.tex Eq B.4:
    The mode functions include both oscillatory and
    damping terms to ensure proper convergence:
    ψₙ(x) = α^(n/2) exp(-x²/2) exp(ikₙx)
    
    Attributes:
        psi: Complex wavefunction values
        grid: Spatial grid points
        n: Mode number
        alpha: Fractal scaling parameter
    """
    psi: np.ndarray
    grid: np.ndarray
    n: int
    alpha: float
    
    def __post_init__(self):
        """Validate fractal mode."""
        if not isinstance(self.psi, np.ndarray):
            raise TypeError("psi must be numpy array")
        if not isinstance(self.grid, np.ndarray):
            raise TypeError("grid must be numpy array")
        if len(self.grid) != len(self.psi):
            raise ValueError("Grid and wavefunction dimensions must match")
        if not np.allclose(self.grid[[0,-1]], [-3, 3]):  # SACRED: grid range
            raise ValueError("Grid must span [-3,3]")
        if not 0 < self.alpha < 1:
            raise ValueError("Scaling parameter must be between 0 and 1")

@dataclass
class BasisConfig:
    """
    Configuration for fractal basis computations.
    
    From appendix_b_basis.tex Eq B.5-B.7:
    Basis configuration parameters determine:
    1. Dimension: Spacetime dimension d
    2. Precision: Numerical computation accuracy
    3. Max level: Fractal recursion depth
    
    Attributes:
        dimension: Spacetime dimension (default: 4)
        precision: Numerical precision (default: 1e-10)
        max_level: Maximum fractal level (default: 10)
    """
    dimension: int = 4
    precision: float = 1e-10
    max_level: int = 10
    
    def __post_init__(self):
        """Validate basis configuration."""
        if self.dimension <= 0:
            raise ValueError("Dimension must be positive")
        if self.precision <= 0:
            raise ValueError("Precision must be positive")
        if self.max_level < 1:
            raise ValueError("Max level must be at least 1")
            
    @property
    def scaling_dimension(self) -> float:
        """Compute scaling dimension from spacetime dimension."""
        return (self.dimension - 2)/2

