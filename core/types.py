from functools import wraps
import numpy as np
from typing import Union, Any, Optional
from dataclasses import dataclass

def ensure_numeric_value(value: Any) -> 'NumericValue':
    """Ensure a value is wrapped in NumericValue.
    
    Implements graceful fallback for unwrapped values:
    - float -> NumericValue(value)
    - np.number -> NumericValue(float(value))
    - np.ndarray (size 1) -> NumericValue(value.item())
    
    Args:
        value: Value to convert
        
    Returns:
        NumericValue: Wrapped numeric value
        
    Raises:
        TypeError: If value cannot be converted
    """
    if hasattr(value, 'value'):
        return ensure_numeric_value(value.value)
        
    if isinstance(value, NumericValue):
        return value
        
    if isinstance(value, (float, int, np.number)):
        return NumericValue(float(value))
        
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return NumericValue(float(value.item()))
        raise TypeError(f"Cannot convert array of size {value.size} to NumericValue")
        
    raise TypeError(f"Cannot convert {type(value)} to NumericValue")

def numeric_property(func):
    """Decorator to handle numeric property access."""
    @property
    @wraps(func)
    def wrapper(self):
        result = func(self)
        return ensure_numeric_value(result)
    return wrapper 

@dataclass
class NumericValue:
    """Represents a numeric value with optional uncertainty."""
    
    value: float
    uncertainty: Optional[float] = None
    
    def __post_init__(self):
        """Validate and convert inputs."""
        if isinstance(self.value, (np.number, np.ndarray)):
            self.value = float(self.value)
        if not np.isfinite(self.value):
            raise ValueError("Value must be finite")
            
    @numeric_property
    def magnitude(self):
        """Get the absolute value."""
        return abs(self.value)
        
    @numeric_property
    def nominal(self):
        """Get the nominal value."""
        return self.value
        
    def __add__(self, other):
        other = ensure_numeric_value(other)
        return NumericValue(self.value + other.value)
        
    def __mul__(self, other):
        other = ensure_numeric_value(other)
        return NumericValue(self.value * other.value)
    
    def __sub__(self, other):
        """Subtract another value."""
        other = ensure_numeric_value(other)
        return NumericValue(self.value - other.value)
    
    def __truediv__(self, other):
        """Divide by another value."""
        other = ensure_numeric_value(other)
        if other.value == 0:
            raise ValueError("Division by zero")
        return NumericValue(self.value / other.value)
    
    def __pow__(self, other):
        """Raise to a power."""
        other = ensure_numeric_value(other)
        return NumericValue(self.value ** other.value)
    
    def __neg__(self):
        """Negate the value."""
        return NumericValue(-self.value)
    
    def __abs__(self):
        """Get absolute value."""
        return NumericValue(abs(self.value))
    
    def __eq__(self, other):
        """Compare for equality."""
        other = ensure_numeric_value(other)
        return np.isclose(self.value, other.value)
    
    def __str__(self):
        """String representation."""
        if self.uncertainty is None:
            return f"{self.value}"
        return f"{self.value} ± {self.uncertainty}"
    
    def __repr__(self):
        """Detailed representation."""
        if self.uncertainty is None:
            return f"NumericValue(value={self.value})"
        return f"NumericValue(value={self.value}, uncertainty={self.uncertainty})"
    
    @property
    def relative_uncertainty(self):
        """Get relative uncertainty."""
        if self.uncertainty is None or self.value == 0:
            return None
        return abs(self.uncertainty / self.value)
    
    def __radd__(self, other):
        """Handle right-hand addition."""
        return self.__add__(other)
        
    def __rmul__(self, other):
        """Handle right-hand multiplication."""
        return self.__mul__(other)
        
    def __rsub__(self, other):
        """Handle right-hand subtraction."""
        other = ensure_numeric_value(other)
        return NumericValue(other.value - self.value)
        
    def __rtruediv__(self, other):
        """Handle right-hand division."""
        other = ensure_numeric_value(other)
        if self.value == 0:
            raise ValueError("Division by zero")
        return NumericValue(other.value / self.value)
    
    @property
    def value(self) -> Union[float, complex]:
        """Get the underlying numeric value."""
        return self._value

    @value.setter 
    def value(self, val: Union[float, complex]) -> None:
        """Set numeric value with validation."""
        if not isinstance(val, (int, float, complex)):
            raise TypeError(f"Value must be numeric, got {type(val)}")
        self._value = complex(val) if isinstance(val, complex) else float(val)

    def __add__(self, other: Union['NumericValue', float]) -> 'NumericValue':
        """Add two values with proper uncertainty propagation."""
        other = ensure_numeric_value(other)
        value = self.value + other.value
        
        # Propagate uncertainties if both present
        if self.uncertainty is not None and other.uncertainty is not None:
            uncertainty = np.sqrt(self.uncertainty**2 + other.uncertainty**2)
        else:
            uncertainty = None
        
        return NumericValue(value, uncertainty)

    def __mul__(self, other: Union['NumericValue', float]) -> 'NumericValue':
        """Multiply with proper uncertainty propagation."""
        other = ensure_numeric_value(other)
        value = self.value * other.value
        
        # Relative uncertainties add in quadrature for multiplication
        if self.uncertainty is not None and other.uncertainty is not None:
            rel_unc = np.sqrt(
                (self.uncertainty/abs(self.value))**2 + 
                (other.uncertainty/abs(other.value))**2
            )
            uncertainty = abs(value) * rel_unc
        else:
            uncertainty = None
        
        return NumericValue(value, uncertainty)