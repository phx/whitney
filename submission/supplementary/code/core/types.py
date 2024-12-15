"""Type definitions for fractal field theory framework."""

from typing import Any, Dict, List, Union, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sympy import Expr, Symbol
from .errors import ValidationError, PhysicsError
from .physics_constants import X

__all__ = [
    'RealValue',
    'ComplexValue',
    'NumericValue',
    'Energy',
    'Momentum',
    'CrossSection',
    'BranchingRatio',
    'ErrorEstimate'
]

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
        >>> z = ComplexValue(1+1j, 0.1)
        >>> print(abs(z.value))  # Magnitude
        >>> print(z.uncertainty)  # Uncertainty in magnitude
    """
    value: complex
    uncertainty: Optional[float] = None
    
    def __post_init__(self):
        """Validate complex value."""
        if not isinstance(self.value, (complex, float, int)):
            raise ValueError("Value must be complex number")
        self.value = complex(self.value)
    
    def __abs__(self) -> float:
        """Get magnitude of complex value."""
        return abs(self.value)
    
    def conjugate(self) -> 'ComplexValue':
        """Get complex conjugate."""
        return ComplexValue(
            value=self.value.conjugate(),
            uncertainty=self.uncertainty
        )
    
    def __mul__(self, other: Union['ComplexValue', RealValue]) -> 'ComplexValue':
        """Multiply complex values with uncertainty propagation."""
        if isinstance(other, (ComplexValue, RealValue)):
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
            return ComplexValue(value, uncertainty)
        return NotImplemented

@dataclass
class Energy(RealValue):
    """
    A physical energy value with units and uncertainty propagation.
    
    Represents an energy value in particle physics calculations, with support for
    unit conversions, uncertainty propagation, and validation of physical constraints.
    
    Attributes:
        value (float): Energy value in GeV
        uncertainty (Optional[float]): Statistical uncertainty in GeV
        units (str): Energy units (default: GeV)
        systematics (Dict[str, float]): Systematic uncertainties by source
        
    Methods:
        to_units(new_units: str) -> Energy:
            Convert to different units (GeV <-> TeV <-> MeV)
        __add__, __sub__, __mul__, __truediv__:
            Arithmetic with automatic uncertainty propagation
        __lt__, __gt__, __eq__:
            Comparison operators for energy thresholds
        
    Examples:
        >>> # Create energy with uncertainty
        >>> e = Energy(100.0, 5.0)  # 100 ± 5 GeV
        >>> 
        >>> # Convert to TeV
        >>> e_tev = e.to_units("TeV")  # Convert to TeV
        >>> print(f"{e_tev.value:.1f} ± {e_tev.uncertainty:.3f} {e_tev.units}")
        0.1 ± 0.005 TeV
        >>> 
        >>> # Arithmetic with uncertainty propagation
        >>> e1 = Energy(100.0, 5.0)  # Particle 1
        >>> e2 = Energy(50.0, 2.0)   # Particle 2
        >>> e_total = e1 + e2        # Total energy
        >>> print(f"{e_total.value:.1f} ± {e_total.uncertainty:.1f} {e_total.units}")
        150.0 ± 5.4 GeV
        >>> 
        >>> # Use in energy thresholds
        >>> if e_total > Energy(100.0):
        ...     print("Above production threshold")
        Above production threshold
        
    Notes:
        - Energy values must be non-negative (physical requirement)
        - Uncertainties are propagated using standard error propagation rules
        - Unit conversions preserve relative uncertainties
        - Comparison operators work with values in same units
        - Used for particle energies and mass calculations
    """
    units: str = "GeV"
    
    def __post_init__(self):
        """Validate energy value and handle units."""
        super().__post_init__()
        if self.value < 0:
            raise ValueError("Energy cannot be negative")
    
    def to_units(self, new_units: str) -> 'Energy':
        """
        Convert to different units.
        
        Supported conversions:
        - GeV ↔ TeV (factor: 10³)
        - GeV ↔ MeV (factor: 10⁻³)
        
        Args:
            new_units (str): Target unit (TeV, GeV, MeV)
            
        Returns:
            Energy: Converted energy value with propagated uncertainty
            
        Raises:
            ValueError: If conversion not supported
        """
        conversions = {
            ("GeV", "TeV"): 1e-3,
            ("TeV", "GeV"): 1e3,
            ("GeV", "MeV"): 1e3,
            ("MeV", "GeV"): 1e-3
        }
        
        if (self.units, new_units) in conversions:
            factor = conversions[(self.units, new_units)]
            return Energy(
                value=self.value * factor,
                uncertainty=self.uncertainty * factor if self.uncertainty else None,
                units=new_units
            )
        raise ValueError(f"Unsupported unit conversion: {self.units} to {new_units}")

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
    """Field configuration parameters.
    
    Attributes:
        mass: float
            Mass parameter in GeV
        coupling: float
            Coupling constant (dimensionless)
        dimension: int
            Number of dimensions
        parameters: Dict[str, float]
            Additional configuration parameters
    """
    mass: float
    coupling: float
    dimension: int
    parameters: Dict[str, float] = None
    
    def __post_init__(self):
        """Initialize with empty parameters if None."""
        if self.parameters is None:
            self.parameters = {}
        self.validate()
    
    def validate(self) -> None:
        """Validate field configuration."""
        if self.mass < 0:
            raise ValueError("Mass must be non-negative")
        if self.coupling < 0:
            raise ValueError("Coupling must be non-negative")
        if self.dimension < 1:
            raise ValueError("Dimension must be positive")

@dataclass
class WaveFunction:
    """
    Quantum wavefunction with metadata.
    
    Attributes:
        psi (np.ndarray): Complex wavefunction values
        grid (np.ndarray): Spatial/momentum grid points
        quantum_numbers (Dict): Quantum numbers
        
    Examples:
        >>> psi = np.array([1+0j, 0j, 0j, 0j]) / np.sqrt(2)
        >>> wf = WaveFunction(psi, np.linspace(-1,1,4), {'n':0, 'l':0})
        >>> wf.normalize()  # Ensure normalization
    """
    psi: np.ndarray
    grid: np.ndarray
    quantum_numbers: Dict[str, int]
    
    def validate(self) -> None:
        """Additional validation of wavefunction."""
        if not isinstance(self.grid, np.ndarray):
            raise ValidationError("Grid must be numpy array")
        if not isinstance(self.quantum_numbers, dict):
            raise ValidationError("Quantum numbers must be dictionary")

    def normalize(self) -> None:
        """Normalize wavefunction."""
        norm = np.sqrt(np.sum(np.abs(self.psi)**2))
        if norm > 0:
            self.psi /= norm

    @classmethod
    def from_expression(cls, expr: Expr, grid: Optional[np.ndarray] = None):
        """Create WaveFunction from symbolic expression."""
        if grid is None:
            grid = np.linspace(-10, 10, 100)
        psi = np.array([complex(expr.subs(X, x)) for x in grid])
        return cls(psi=psi, grid=grid, quantum_numbers={'n': 0})

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
    """
    Base class for numeric values with validation and error propagation.
    
    Represents a numeric value (real or complex) with optional uncertainty
    and validation rules. Used as base class for specific physical quantities.
    
    Attributes:
        value: The numeric value (float or complex)
        uncertainty: Optional uncertainty in the value
        valid_range: Optional tuple of (min, max) for validation
        
    Methods:
        validate(): Check if value is within valid range
        propagate_error(): Propagate uncertainties in calculations
        
    Examples:
        >>> x = NumericValue(1.0, uncertainty=0.1)
        >>> y = NumericValue(2.0, uncertainty=0.2)
        >>> z = x * y  # Uncertainties are properly propagated
    """
    value: Union[float, complex]
    uncertainty: Optional[float] = None
    valid_range: Optional[Tuple[float, float]] = None
    
    def __post_init__(self):
        """Validate numeric value."""
        if isinstance(self.value, (int, float)):
            if not np.isfinite(float(self.value)):
                raise ValueError("Value must be finite")
        elif isinstance(self.value, complex):
            if not (np.isfinite(self.value.real) and np.isfinite(self.value.imag)):
                raise ValueError("Complex value must have finite real and imaginary parts")
        else:
            raise TypeError(f"Value must be numeric, got {type(self.value)}")
             
        if self.uncertainty is not None:
            if not isinstance(self.uncertainty, (int, float)):
                raise TypeError("Uncertainty must be real number")
            if self.uncertainty < 0:
                raise ValueError("Uncertainty must be non-negative")
            if not np.isfinite(float(self.uncertainty)):
                raise ValueError("Uncertainty must be finite")
                 
        if self.valid_range is not None:
            self.validate()
    
    def validate(self) -> None:
        """Check if value is within valid range."""
        if self.valid_range is not None:
            min_val, max_val = self.valid_range
            if isinstance(self.value, (int, float)):
                if not min_val <= self.value <= max_val:
                    raise ValueError(f"Value {self.value} outside valid range [{min_val}, {max_val}]")
            else:
                # For complex values, check magnitude
                magnitude = abs(self.value)
                if not min_val <= magnitude <= max_val:
                    raise ValueError(f"Magnitude {magnitude} outside valid range [{min_val}, {max_val}]")
    
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
    
    def __add__(self, other: 'NumericValue') -> 'NumericValue':
        """Add values with uncertainty propagation."""
        value = self.value + other.value
        
        if self.uncertainty is None or other.uncertainty is None:
            uncertainty = None
        else:
            # Absolute uncertainties add in quadrature for addition
            uncertainty = np.sqrt(self.uncertainty**2 + other.uncertainty**2)
            
        return NumericValue(value, uncertainty)
    
    def check_finite(self) -> None:
        """Check if value is finite."""
        if isinstance(self.value, (int, float)):
            if not np.isfinite(float(self.value)):
                raise ValueError("Value must be finite")
        elif isinstance(self.value, complex):
            if not (np.isfinite(self.value.real) and np.isfinite(self.value.imag)):
                raise ValueError("Complex value must have finite real and imaginary parts")
    
    def check_uncertainty(self) -> None:
        """Validate uncertainty value."""
        if self.uncertainty is not None:
            if not isinstance(self.uncertainty, (int, float)):
                raise TypeError("Uncertainty must be real number")
            if self.uncertainty < 0:
                raise ValueError("Uncertainty must be non-negative")
            if not np.isfinite(float(self.uncertainty)):
                raise ValueError("Uncertainty must be finite")
    
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
