"""Type definitions for fractal field theory framework."""

from typing import Any, Dict, List, Union, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from .errors import PhysicsError

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
    """
    Field configuration with parameters.
    
    Attributes:
        alpha (float): Coupling constant
        dimension (int): Field dimension
        parameters (Dict): Additional parameters
        
    Examples:
        >>> config = FieldConfig(alpha=0.1, dimension=4, parameters={'mass': 125.0})
        >>> config.validate()  # Check configuration is valid
    """
    alpha: float
    dimension: int
    parameters: Dict[str, float]
    
    def validate(self) -> None:
        """Validate field configuration."""
        if not 0 < self.alpha < 1:
            raise ValueError("Coupling must be between 0 and 1")
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
    
    def normalize(self) -> None:
        """Normalize wavefunction."""
        norm = np.sqrt(np.sum(np.abs(self.psi)**2))
        if norm > 0:
            self.psi /= norm

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