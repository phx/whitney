# User Guide

## Installation

### Requirements
- Python >= 3.8.0, < 3.11.0
- CUDA toolkit >= 11.0 (optional, for GPU acceleration)
- 32GB RAM recommended for large computations

### Quick Start
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install package
pip install -r requirements.txt
pip install .

# Verify installation
python -m pytest tests/
```

## Basic Usage

### 1. Initialize Framework
```python
from core.field import UnifiedField
from core.constants import ALPHA_VAL

# Create field instance
field = UnifiedField(alpha=ALPHA_VAL)
```

### 2. Compute Physical Quantities
```python
# Compute gauge couplings
E = 1000.0  # GeV
g1 = field.compute_coupling(1, E)  # U(1) coupling
g2 = field.compute_coupling(2, E)  # SU(2) coupling
g3 = field.compute_coupling(3, E)  # SU(3) coupling

# Compute observables
result = field.compute_observable('sin2_theta_W')
print(f"sin²θW = {result['value']} ± {result['total_uncertainty']}")
```

### 3. Field Evolution
```python
# Create initial state
psi_0 = field.compute_basis_function(n=0, E=100.0)

# Evolve system
t_points = np.linspace(0, 10, 100)
evolution = field.evolve_field(psi_0, t_points)

# Analyze results
energies = [field.compute_energy_density(psi) 
           for psi in evolution['field_values']]
```

## Advanced Features

### 1. High-Precision Computation
```python
from core.modes import ComputationMode

# Use symbolic computation for high precision
field = UnifiedField(mode=ComputationMode.SYMBOLIC)

# Enable extended precision
import mpmath
mpmath.mp.dps = 50  # 50 decimal places
```

### 2. GPU Acceleration
```python
# Enable GPU computation
import os
os.environ['FRACTAL_FIELD_USE_GPU'] = '1'
os.environ['FRACTAL_FIELD_GPU_MEMORY'] = '4G'

# Use GPU-optimized algorithms
field = UnifiedField(mode=ComputationMode.NUMERIC)
```

### 3. Parallel Processing
```python
# Enable parallel processing
os.environ['FRACTAL_FIELD_NUM_THREADS'] = '8'

# Compute with multiple processes
from multiprocessing import Pool
with Pool(8) as p:
    results = p.map(field.compute_coupling, range(1, 4))
```

## Troubleshooting

### Common Issues

1. **Memory Errors**
```python
# Reduce memory usage
os.environ['FRACTAL_FIELD_MEMORY_EFFICIENT'] = '1'
```

2. **Numerical Instability**
```python
# Enable stability checks
from core.stability import stability_check

@stability_check(threshold=1e-10)
def my_computation():
    # Your code here
    pass
```

3. **Performance Issues**
```python
# Profile computation
from core.utils import profile_computation

@profile_computation
def slow_function():
    # Your code here
    pass
```

### Error Messages

1. **ValidationError**
   - Check input ranges
   - Verify physical constraints
   - Ensure proper units

2. **StabilityError**
   - Reduce step size
   - Use higher precision
   - Check for singular points

3. **ComputationError**
   - Verify input parameters
   - Check resource availability
   - Enable debug logging

## Best Practices

### 1. Resource Management
```python
# Use context managers
with field.computation_context():
    result = field.compute_complex_observable()
```

### 2. Error Handling
```python
from core.errors import PhysicsError

try:
    result = field.compute_observable('complex_quantity')
except PhysicsError as e:
    logger.error(f"Physics constraint violated: {e}")
```

### 3. Validation
```python
# Validate results
from core.validation import validate_predictions

validate_predictions(predictions, experimental_data)
```

## Examples

### 1. Coupling Unification
```python
# Compute couplings at GUT scale
E_gut = 2.0e16  # GeV
couplings = [field.compute_coupling(i, E_gut) for i in range(1, 4)]
print(f"Unified coupling: {np.mean(couplings):.6f}")
```

### 2. B-Physics Analysis
```python
# Compute B-physics observables
BR = field.compute_branching_ratio('Bs_to_mumu')
print(f"BR(Bs→μμ) = ({BR['value']:.2e} ± {BR['error']:.2e})")
```

### 3. Field Configuration
```python
# Create and analyze field configuration
psi = field.compute_basis_function(n=0)
energy = field.compute_energy_density(psi)
charges = field.compute_conserved_charges(psi)
``` 