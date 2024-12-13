# API Documentation

## Core Modules

### field.py

The `field.py` module implements the unified field theory framework.

#### UnifiedField

Main class for field theory calculations.

```python
class UnifiedField:
    def __init__(self, alpha: float):
        """Initialize unified field with coupling constant."""
        
    def evolve(self, energy_points: np.ndarray) -> np.ndarray:
        """Evolve field through energy scales."""
        
    def compute_observables(self) -> Dict[str, RealValue]:
        """Compute physical observables."""
```

### basis.py

The `basis.py` module provides fractal basis functions.

#### FractalBasis

Class implementing fractal basis functions.

```python
class FractalBasis:
    def __init__(self, alpha: float, max_level: int):
        """Initialize fractal basis with coupling and max recursion level."""
        
    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate basis functions at given points."""
        
    def transform(self, coefficients: np.ndarray) -> np.ndarray:
        """Transform from coefficient to coordinate space."""
```

### detector.py 

The `detector.py` module simulates detector response.

#### Detector

Class for detector simulation and efficiency calculations.

```python
class Detector:
    def __init__(self, resolution: Dict[str, float],
                 acceptance: Dict[str, Union[float, Tuple[float, float]]]):
        """Initialize detector with resolution and acceptance parameters."""
        
    def simulate_measurement(self, value: Union[Energy, Momentum]) -> Dict[str, RealValue]:
        """Simulate detector measurement with resolution effects."""
        
    def compute_efficiency(self, pt: RealValue, eta: RealValue) -> RealValue:
        """Compute detection efficiency for given kinematics."""
```

### utils.py

The `utils.py` module provides core utility functions.

#### Error Propagation

```python
def propagate_errors(values: List[RealValue],
                    uncertainties: List[RealValue],
                    correlations: Optional[np.ndarray] = None) -> RealValue:
    """Propagate uncertainties through calculations."""
```

#### Performance Profiling

```python
@profile_computation
def heavy_calculation(data: np.ndarray) -> np.ndarray:
    """Example of profiled computation."""
```

## Type System

### Energy

A type representing physical energy values with units and uncertainty propagation.

```python
class Energy(RealValue):
    """
    Energy value with units and uncertainty.
    
    Attributes:
        value (float): Energy value in GeV
        uncertainty (Optional[float]): Uncertainty in GeV
        units (str): Energy units (default: GeV)
        
    Methods:
        to_units(new_units: str) -> Energy:
            Convert to different units (GeV <-> TeV <-> MeV)
        __add__, __mul__: Propagate uncertainties in calculations
        
    Examples:
        >>> e = Energy(100.0, 5.0)  # 100 ± 5 GeV
        >>> e_tev = e.to_units("TeV")  # Convert to TeV
        >>> print(f"{e_tev.value:.1f} ± {e_tev.uncertainty:.3f} {e_tev.units}")
        0.1 ± 0.005 TeV
        
        >>> # Uncertainty propagation
        >>> e1 = Energy(100.0, 5.0)
        >>> e2 = Energy(50.0, 2.0)
        >>> e_sum = e1 + e2
        >>> print(f"{e_sum.value:.1f} ± {e_sum.uncertainty:.1f} {e_sum.units}")
        150.0 ± 5.4 GeV
    
    Notes:
        - Energy values must be non-negative
        - Uncertainties are propagated using standard error propagation rules
        - Unit conversions preserve relative uncertainties
    """
```

The Energy type is used throughout the framework for handling energy values in particle physics calculations. It ensures proper unit handling and uncertainty propagation.

Supported unit conversions:
- GeV ↔ TeV (factor: 10³)
- GeV ↔ MeV (factor: 10⁻³)

Example usage in physics calculations:
```python
# Create energy values
beam_energy = Energy(13000.0, 100.0)  # LHC beam energy
mass = Energy(125.09, 0.24)  # Higgs mass

# Convert to TeV for plotting
beam_tev = beam_energy.to_units("TeV")

# Uncertainty propagation in calculations
total_energy = beam_energy + mass
print(f"Total energy: {total_energy.value:.1f} ± {total_energy.uncertainty:.1f} {total_energy.units}")
```

### Momentum

A type representing physical momentum values with units and uncertainty propagation in particle physics calculations.

```python
class Momentum(RealValue):
    """
    Momentum value with units and uncertainty.
    
    Attributes:
        value (float): Momentum value in GeV/c
        uncertainty (Optional[float]): Uncertainty in GeV/c
        units (str): Momentum units (default: GeV/c)
        
    Methods:
        to_units(new_units: str) -> Momentum:
            Convert to different units (GeV/c <-> TeV/c <-> MeV/c)
        __add__, __mul__: Propagate uncertainties in calculations
        __lt__, __gt__, __eq__: Compare momentum values
        
    Examples:
        >>> p = Momentum(50.0, 2.0)  # 50 ± 2 GeV/c
        >>> p_tev = p.to_units("TeV/c")  # Convert to TeV/c
        >>> print(f"{p_tev.value:.3f} ± {p_tev.uncertainty:.4f} {p_tev.units}")
        0.050 ± 0.0020 TeV/c
        
        >>> # Uncertainty propagation
        >>> p1 = Momentum(100.0, 5.0)  # Transverse momentum
        >>> p2 = Momentum(50.0, 2.0)   # Longitudinal momentum
        >>> p_total = p1 + p2
        >>> print(f"{p_total.value:.1f} ± {p_total.uncertainty:.1f} {p_total.units}")
        150.0 ± 5.4 GeV/c
    
    Notes:
        - Momentum values must be non-negative (physical requirement)
        - Uncertainties follow standard error propagation rules
        - Unit conversions preserve relative uncertainties
        - Used for particle kinematics and phase space calculations
    """
```

The Momentum type is essential for handling particle momenta in high-energy physics calculations. It ensures proper unit handling and uncertainty propagation while enforcing physical constraints.

Supported unit conversions:
- GeV/c ↔ TeV/c (factor: 10³)
- GeV/c ↔ MeV/c (factor: 10⁻³)

Example usage in physics calculations:
```python
# Create momentum values
pt = Momentum(45.0, 1.2)   # Transverse momentum
pz = Momentum(120.0, 3.0)  # Longitudinal momentum

# Convert to TeV/c for high-energy analysis
pt_tev = pt.to_units("TeV/c")

# Check kinematic threshold
if pt > Momentum(20.0):  # Trigger threshold
    # Calculate total momentum
    p_total = pt + pz
    print(f"Total momentum: {p_total.value:.1f} ± {p_total.uncertainty:.1f} {p_total.units}")
```

### FieldConfig

A type representing the configuration parameters for fractal field theory calculations, with built-in validation.

```python
class FieldConfig:
    """
    Field configuration with parameters.
    
    Attributes:
        alpha (float): Coupling constant (0 < α < 1)
        dimension (int): Field dimension (> 0)
        parameters (Dict[str, float]): Additional parameters
        
    Methods:
        validate() -> None: Validate configuration parameters
        to_dict() -> Dict: Convert to dictionary format
        from_dict(data: Dict) -> FieldConfig: Create from dictionary
        
    Examples:
        >>> config = FieldConfig(alpha=0.1, dimension=4, parameters={'mass': 125.0})
        >>> config.validate()  # Check configuration is valid
        
        >>> # Loading from dictionary
        >>> data = {
        ...     'alpha': 0.118,
        ...     'dimension': 4,
        ...     'parameters': {
        ...         'mass': 125.09,
        ...         'width': 4.2,
        ...         'coupling_scale': 1000.0
        ...     }
        ... }
        >>> config = FieldConfig.from_dict(data)
        
    Notes:
        - The coupling constant α must be in (0,1) for convergence
        - Dimension must be positive integer (typically 4 for spacetime)
        - Parameters can include any additional physics parameters
        - Used for configuring field theory calculations
    """
    def validate(self) -> None:
        """
        Validate field configuration.
        
        Checks:
        - Coupling constant in valid range
        - Dimension is positive
        
        Raises:
            ValueError: If configuration invalid
        """
```

The FieldConfig type is central to setting up and validating field theory calculations in the framework. It ensures that all parameters are physically meaningful and mathematically consistent.

Key validation checks:
- 0 < α < 1 (coupling constant range)
- dimension > 0 (spacetime dimension)
- All parameters are finite real numbers

Example usage in field theory calculations:
```python
# Create standard configuration
config = FieldConfig(
    alpha=0.118,      # Strong coupling constant at Z mass
    dimension=4,      # 4D spacetime
    parameters={
        'mass': 125.09,         # Higgs mass in GeV
        'width': 4.2,           # Decay width in GeV
        'coupling_scale': 1000.0 # Renormalization scale in GeV
    }
)

# Validate configuration
try:
    config.validate()
except ValueError as e:
    print(f"Invalid configuration: {e}")

# Use in field calculations
field = UnifiedField(config)
observables = field.compute_observables()
```

## Error Handling

+ The framework uses a hierarchical error system to handle various physics-related errors and exceptions.
+ 
+ ```
+ PhysicsError
+ ├── ValidationError
+ │   ├── ConfigurationError
+ │   └── ParameterError
+ ├── StabilityError
+ │   ├── ConvergenceError
+ │   └── PrecisionError
+ └── ComputationError
+     ├── NumericalError
+     └── ResourceError
+ ```

### PhysicsError

- Base exception class for all physics-related errors.
+ Base exception class for all physics-related errors in the framework. Provides common functionality for error handling and reporting.

```python
class PhysicsError(Exception):
    """
    Base class for physics-related errors.
    
+   Attributes:
+       message (str): Error description
+       context (Dict): Additional error context
+       
+   Methods:
+       with_context(**kwargs) -> PhysicsError: Add context to error
+       
    Examples:
        >>> try:
        ...     process_physics_data(invalid_data)
        ... except PhysicsError as e:
-       ...     print(f"Physics error: {e}")
+       ...     print(f"Physics error: {e}")
+       ...     print(f"Context: {e.context}")
+       
+       >>> # Adding context to errors
+       >>> raise PhysicsError("Invalid value").with_context(
+       ...     value=-1.0,
+       ...     allowed_range=(0, float('inf'))
+       ... )
    """
```

### ValidationError

- Exception raised when physics parameters fail validation.
+ Exception raised when physics parameters or configurations fail validation checks. Used to catch invalid inputs early.

```python
class ValidationError(PhysicsError):
    """
    Error raised for invalid physics parameters.
    
+   Used for:
+   - Parameter range validation
+   - Configuration validation
+   - Input data validation
+   - Physics constraint validation
+   
    Examples:
        >>> try:
        ...     validate_momentum(p=-1.0)  # Invalid negative momentum
        ... except ValidationError as e:
-       ...     print(f"Validation failed: {e}")
+       ...     print(f"Validation failed: {e}")
+       ...     if isinstance(e, ParameterError):
+       ...         print(f"Invalid parameter: {e.param_name}")
+       ...         print(f"Allowed range: {e.valid_range}")
    """
```

### StabilityError

- Exception raised for numerical stability issues.
+ Exception raised for numerical stability and convergence issues in calculations. Helps identify problematic computations.

```python
class StabilityError(PhysicsError):
    """
    Error raised for numerical instabilities.
    
+   Used for:
+   - Convergence failures
+   - Precision loss
+   - Numerical overflow/underflow
+   - Stability threshold violations
+   
    Examples:
        >>> try:
        ...     result = compute_with_stability_check(unstable_value)
        ... except StabilityError as e:
-       ...     print(f"Stability check failed: {e}")
+       ...     print(f"Stability check failed: {e}")
+       ...     if isinstance(e, ConvergenceError):
+       ...         print(f"Failed to converge after {e.iterations} iterations")
+       ...         print(f"Current error: {e.current_error}")
+       ...         print(f"Target error: {e.target_error}")
    """
```

+ ### Error Handling Best Practices
+ 
+ 1. Always catch specific errors before general ones:
+ ```python
+ try:
+     result = compute_physics_observable(data)
+ except ValidationError as e:
+     # Handle invalid inputs
+     log.error(f"Invalid input: {e}")
+ except StabilityError as e:
+     # Handle numerical issues
+     log.error(f"Computation unstable: {e}")
+ except PhysicsError as e:
+     # Handle other physics errors
+     log.error(f"Physics error: {e}")
+ ```
+ 
+ 2. Add context when raising errors:
+ ```python
+ if value < 0:
+     raise ValidationError("Value must be positive").with_context(
+         value=value,
+         function="compute_cross_section",
+         allowed_range=(0, float('inf'))
+     )
+ ```
+ 
+ 3. Use error hierarchies for granular handling:
+ ```python
+ try:
+     result = complex_calculation()
+ except ConvergenceError as e:
+     # Handle convergence specifically
+     retry_with_different_parameters(e.context)
+ except StabilityError as e:
+     # Handle other stability issues
+     use_fallback_method()
+ ```