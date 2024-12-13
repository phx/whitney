# Fractal Field Theory Framework User Guide

## Overview

This framework implements numerical computations for fractal field theory, providing tools for:
- Computing fractal basis functions
- Evolving field configurations
- Analyzing physical observables
- Computing gauge coupling evolution

## Installation

```bash
# Install from source
git clone https://github.com/fractalfield/fractal-field-theory
cd fractal-field-theory
pip install -e .

# Install with optional dependencies
pip install -e .[dev,docs,ml,parallel]
```

## Quick Start

```python
from core.field import UnifiedField
from core.modes import ComputationMode

# Initialize the framework
field = UnifiedField(alpha=0.1, mode=ComputationMode.MIXED)

# Compute basis function
psi = field.compute_basis_function(n=1, E=100.0)  # n=1 state at 100 GeV

# Compute energy density
energy = field.compute_energy_density(psi)

# Analyze field configuration
analysis = field.analyze_field_configuration(psi)
```

## Core Components

### 1. Basis Functions (core.basis.FractalBasis)

The FractalBasis class implements the fundamental fractal basis:

```python
from core.basis import FractalBasis

basis = FractalBasis(alpha=0.1)

# Compute basis function
psi = basis.compute(n=2, E=91.2)  # n=2 state at Z-boson mass

# Check orthogonality
overlap = basis.check_orthogonality(n1=1, n2=2)

# Calculate fractal dimension
dim = basis.calculate_fractal_dimension(E=100.0)
```

Key methods:
- `compute(n, E)`: Compute nth basis function at energy E
- `normalize(psi)`: Normalize wavefunction
- `check_orthogonality(n1, n2)`: Check basis orthogonality
- `coupling(gauge_index, E)`: Compute gauge coupling evolution

### 2. Field Evolution (core.field.UnifiedField)

The UnifiedField class handles field dynamics:

```python
from core.field import UnifiedField
import numpy as np

field = UnifiedField()

# Create initial configuration
psi_initial = field.compute_basis_function(n=0)

# Set up time evolution
t = np.linspace(0, 10, 1000)

# Evolve field
evolution = field.evolve_field(
    psi_initial=psi_initial,
    t_range=t,
    yield_every=10
)

# Access results
times = evolution['time']
energies = evolution['energy']
field_values = evolution['field_values']
```

Key methods:
- `compute_field_equation(psi)`: Compute field evolution equation
- `compute_energy_density(psi)`: Compute energy density
- `analyze_field_configuration(psi)`: Analyze field properties

### 3. Computation Modes

Three computation modes are available:

```python
from core.modes import ComputationMode

# Symbolic mode (most precise, slowest)
field_symbolic = UnifiedField(mode=ComputationMode.SYMBOLIC)

# Numeric mode (fastest, least flexible)
field_numeric = UnifiedField(mode=ComputationMode.NUMERIC)

# Mixed mode (balanced, uses caching)
field_mixed = UnifiedField(mode=ComputationMode.MIXED)
```

### 4. Error Handling

The framework uses custom error types for precise error handling:

```python
from core.errors import (
    ValidationError,    # Input validation errors
    ComputationError,  # Numerical computation errors
    PhysicsError,      # Physics constraint violations
    StabilityError     # Numerical stability issues
)

try:
    result = field.compute_energy_density(psi)
except ValidationError as e:
    print(f"Invalid input: {e}")
except StabilityError as e:
    print(f"Numerical instability: {e}")
```

## Advanced Features

### 1. Memory Management

The framework implements memory-efficient computations:

```python
# Use sliding window evolution for large computations
evolution = field.evolve_field(
    psi_initial=psi,
    t_range=large_time_array,
    yield_every=10  # Store every 10th step
)
```

### 2. Performance Optimization

```python
from core.compute import benchmark_computation

# Benchmark computations
@benchmark_computation
def my_calculation():
    return field.compute_energy_density(psi)

result = my_calculation()
print(f"Execution time: {result['execution_time']} s")
print(f"Memory usage: {result['memory_delta']} MB")
```

### 3. Stability Checks

```python
from core.compute import check_computation_stability

# Configure stability thresholds
thresholds = {
    'underflow': 1e-10,
    'overflow': 1e10,
    'relative_error': 1e-6,
    'condition_number': 1e8
}

is_stable = check_computation_stability(result, thresholds=thresholds)
```

## Best Practices

1. **Memory Efficiency**
   - Use generators for large datasets
   - Implement cleanup in destructors
   - Release memory explicitly when possible

2. **Numerical Stability**
   - Monitor condition numbers
   - Use stable algorithms
   - Implement error checking

3. **Performance**
   - Choose appropriate computation mode
   - Use caching for expensive calculations
   - Implement parallel processing for large computations

## Common Issues and Solutions

1. **Memory Issues**
   ```python
   # Use sliding window evolution
   field.evolve_field(psi, t_range, yield_every=10)
   ```

2. **Numerical Instability**
   ```python
   # Use log-space normalization for large n
   basis.compute(n=large_n, E=E)
   ```

3. **Performance Bottlenecks**
   ```python
   # Cache expensive computations
   @memoize_computation(maxsize=1024)
   def expensive_calculation(x):
       return field.compute_field_equation(x)
   ```

## API Reference

See [API Documentation](api.md) for detailed reference.

## Contributing

See [Contributing Guidelines](contributing.md) for information on contributing to the project. 