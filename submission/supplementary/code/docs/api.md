# API Documentation

## Core Modules

### basis.py

#### FractalBasis

```python
class FractalBasis:
    """Implements recursive fractal basis functions."""
    
    def __init__(self, alpha: float = ALPHA_VAL, mode: ComputationMode = ComputationMode.MIXED):
        """
        Initialize fractal basis.
        
        Args:
            alpha: Scaling parameter (default: ALPHA_VAL)
            mode: Computation mode (default: MIXED)
        """
```

**Constants:**
- `E0`: Z boson mass (91.2 GeV)
- `E0_MIN`: Minimum valid energy scale (1.0 GeV)
- `E0_MAX`: Maximum valid energy scale (1000.0 GeV)
- `N_STABLE_MAX`: Maximum n for direct normalization (50)
- `LOG_NORM_THRESHOLD`: Switch to log-space normalization threshold (100)

**Methods:**
- `compute(n: int, E: Energy) -> WaveFunction`: Compute nth basis function
- `normalize(psi: Expr) -> Expr`: Normalize wavefunction
- `check_orthogonality(n1: int, n2: int) -> float`: Check basis orthogonality
- `coupling(gauge_index: int, E: Energy) -> float`: Compute gauge coupling
- `calculate_fractal_dimension(E: float) -> float`: Calculate fractal dimension

### field.py

#### UnifiedField

```python
class UnifiedField:
    """Main framework class implementing fractal field theory."""
    
    def __init__(self, alpha: float = ALPHA_VAL, mode: ComputationMode = ComputationMode.MIXED):
        """
        Initialize unified field.
        
        Args:
            alpha: Scaling parameter
            mode: Computation mode
        """
```

**Constants:**
- `CACHE_SIZE`: Maximum cache size (1024)
- `WINDOW_SIZE`: Evolution window size (100)
- `OVERLAP_SIZE`: Window overlap size (10)
- `MASS_SQUARED_MIN/MAX`: Mass term bounds (0.0, 1e6 GeV²)
- `LAMBDA_MIN/MAX`: Coupling bounds (0.0, 10.0)
- `FIELD_AMPLITUDE_MAX`: Maximum field amplitude (1e3 GeV)

**Methods:**
- `compute_field_equation(psi: FieldConfig) -> FieldConfig`
- `compute_energy_density(psi: FieldConfig) -> RealValue`
- `evolve_field(psi_initial: FieldConfig, t_range: np.ndarray) -> Dict[str, np.ndarray]`
- `analyze_field_configuration(psi: FieldConfig) -> AnalysisResult`

### compute.py

#### ThreadSafeCache

```python
class ThreadSafeCache:
    """Thread-safe computation cache with size limit."""
    
    def __init__(self, maxsize: int = 128):
        """Initialize cache with maximum size."""
```

**Methods:**
- `get(key: Any) -> Any`: Get cached value
- `set(key: Any, value: Any) -> None`: Set cache value
- `clear() -> None`: Clear cache

#### StabilityConfig

```python
class StabilityConfig:
    """Configuration for numerical stability checks."""
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """Initialize with custom thresholds."""
```

**Default Thresholds:**
- `underflow`: 1e-10
- `overflow`: 1e10
- `relative_error`: 1e-6
- `condition_number`: 1e8

**Functions:**
- `memoize_computation(maxsize: int = 128, typed: bool = False) -> Callable`
- `benchmark_computation(func: Callable) -> Callable`
- `check_computation_stability(values: np.ndarray) -> bool`

### types.py

**Type Definitions:**
```python
Energy = NewType('Energy', float)  # GeV
FieldConfig = NewType('FieldConfig', Expr)
WaveFunction = NewType('WaveFunction', Expr)
AnalysisResult = Dict[str, Union[RealValue, ComplexValue, Expr]]
ErrorEstimate = Dict[str, RealValue]
```

### errors.py

**Exception Hierarchy:**
```python
FractalTheoryError
├── ComputationError
│   ├── StabilityError
│   └── BoundsError
├── ValidationError
├── PhysicsError
└── ConfigurationError
```

### modes.py

```python
class ComputationMode(Enum):
    """Computation mode enumeration."""
    SYMBOLIC = auto()  # Full symbolic computation
    NUMERIC = auto()   # Pure numerical computation
    MIXED = auto()     # Cached symbolic-numeric hybrid
```

### constants.py

**Physical Constants:**
```python
ALPHA_VAL = 0.0072973525693  # Fine structure constant
E_REF = 91.1876  # Z boson mass (GeV)
HBAR = 6.582119569e-25  # GeV⋅s
C = 299792458  # Speed of light (m/s)
```

## Utility Functions

### validation.py

```python
def validate_energy(E: Energy, min_val: float = 0.0) -> None
def validate_field_config(psi: FieldConfig) -> None
def validate_parameters(params: Dict[str, Any], bounds: Optional[Dict[str, tuple]] = None) -> None
```

### utils.py

```python
def evaluate_expr(expr: Expr, subs: Optional[dict] = None, precision: int = 53) -> Union[RealValue, ComplexValue]
def cached_evaluation(expr: Expr, *args) -> RealValue
def get_memory_usage() -> float
```

## Type Hints

The framework uses strict type hints throughout:
- `RealValue`: Union[float, np.float64]
- `ComplexValue`: Union[complex, np.complex128]
- `Array`: TypeVar('Array', np.ndarray, List[float])

## Error Handling

All operations that can fail should be wrapped in try-except blocks:
```python
try:
    result = field.compute_energy_density(psi)
except ValidationError as e:
    # Handle input validation error
except ComputationError as e:
    # Handle computation error
except PhysicsError as e:
    # Handle physics constraint violation
except StabilityError as e:
    # Handle numerical stability issue