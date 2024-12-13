# Validation Guide

## Test Suite Overview

### Unit Tests
```bash
# Core physics tests
tests/test_field.py      # Field equations and coupling evolution
tests/test_basis.py      # Fractal basis functions
tests/test_integration.py # Framework integration

# Experimental validation
tests/validation/test_cross_sections.py    # Cross-section predictions
tests/validation/test_branching_ratios.py  # B-physics observables
tests/validation/test_weak_mixing.py       # Electroweak precision tests
```

### Validation Datasets
```bash
# Reference data location
data/validation/
  ├── lep/               # LEP electroweak data
  │   ├── z_peak.csv    # Z-pole measurements
  │   └── high_e.csv    # High-energy running
  ├── lhc/              # LHC measurements
  │   ├── atlas/        # ATLAS collaboration
  │   └── cms/          # CMS collaboration
  └── b_physics/        # B-physics data
      ├── lhcb/         # LHCb measurements
      └── belle2/       # Belle II results
```

### Performance Benchmarks
```bash
# Run full benchmark suite
python benchmarks/run_all.py

# Key metrics:
- Numerical precision: < 10⁻¹⁵ relative error
- Computation speed: < 100ms per coupling
- Memory usage: < 4GB for standard runs
```

## Validation Procedure

1. Environment Setup
```bash
# Create clean test environment
python -m venv test_env
source test_env/bin/activate
pip install -r requirements/test.txt
```

2. Run Test Suite
```bash
# Full validation
pytest --runslow --runexperimental

# Quick validation
pytest -m "not slow"
```

3. Verify Results
```bash
# Generate validation report
python scripts/validate_results.py --report

# Key validation criteria:
- All tests pass
- Coverage > 95%
- No performance regressions
```

## Common Issues

### Numerical Stability
- Use `ComputationMode.SYMBOLIC` for precision tests
- Verify against analytic results where possible
- Check condition numbers in matrix operations

### Performance
- Profile memory usage with `memory_profiler`
- Use `pytest-benchmark` for performance tests
- Enable caching for repeated computations 