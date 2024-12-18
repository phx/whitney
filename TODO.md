# Core Framework Priorities

## 1. Type System (Critical)
- [ ] Fix NumericValue property access for 'value' attribute
  - [ ] Add value property to float/numpy types via wrapper
  - [x] Ensure proper type conversion in compute_coupling
  - [ ] Fix NoneType returns in field equations
- [ ] Implement type conversion system
  - [x] Add ensure_numeric_value() wrapper function
  - [x] Add numeric_property decorator
  - [x] Add type conversion utilities for numpy types
  - [ ] Add graceful fallback for unwrapped values
- [ ] Fix symbolic expression handling
  - [ ] Add proper Float conversion
  - [ ] Handle complex number cases
  - [ ] Fix matrix dimension mismatches

## 2. UnifiedField API (High)
- [ ] Add missing compute_* methods:
  - [ ] compute_commutator
  - [ ] compute_noether_current
  - [ ] compute_retarded_propagator
  - [ ] compute_couplings
  - [ ] compute_gut_scale
  - [ ] compute_dark_matter_density
- [ ] Standardize method signatures to accept rtol parameter
- [ ] Fix argument handling in compute_basis_function and compute_basis_state
- [ ] Fix initialization arguments
  - [ ] Add dimension parameter
  - [ ] Add max_level parameter
  - [ ] Validate input ranges

## 3. Test Framework (Medium)
- [ ] Replace function-scoped fixtures with context managers
  - [ ] Fix separation fixture in microcausality tests
  - [ ] Fix distance fixture in cluster_decomposition tests
- [ ] Add HealthCheck suppression where appropriate
- [ ] Optimize slow tests exceeding deadline
- [ ] Fix hypothesis test configurations
- [ ] Fix test timeouts in Lorentz invariance tests
- [ ] Add proper error handling for invalid inputs

## 4. Coverage Goals (High)
- [ ] Increase core/field.py coverage (currently 42%)
- [ ] Increase core/types.py coverage (currently 58%)
- [ ] Increase core/utils.py coverage (currently 59%)
- [ ] Reach overall 80% coverage target (currently 61.47%)
- [ ] Focus on field.py lines 86-89, 94, 128-129 first
- [ ] Add tests for error handling paths

## 5. Documentation (Medium)
- [ ] Update testing_guide.md with new patterns
- [ ] Document type system enhancements
- [ ] Add examples for UnifiedField API
- [ ] Update scientific.md with theoretical foundations
- [ ] Add section on symbolic computation
- [ ] Document type conversion rules

## 6. Performance (Low)
- [ ] Fix ImmutableDenseNDimArray expand issue
- [ ] Optimize phase space integration
- [ ] Resolve memory usage in field evolution
- [ ] Improve numerical stability in gauge transformations
- [ ] Fix ZeroDivisionError in profiling
- [ ] Add bounds checking for numerical stability

## Critical Test Coverage Path

1. Fix Correlation Function Tests (High Priority)
- [ ] Add missing imports in test_correlations.py
- [ ] Fix Symbol vs float type mismatch in correlator
- [ ] Add proper validation for quantum numbers
- [ ] Ensure WaveFunction normalization

2. Fix Källén-Lehmann Tests (Critical)
- [ ] Fix spectral function computation
- [ ] Add proper integration limits
- [ ] Validate positive definiteness
- [ ] Add normalization check

3. Ward Identity Tests (Critical)
- [ ] Fix current conservation tests
- [ ] Add proper gauge current computation
- [ ] Fix differential operator application
- [ ] Add charge conservation validation

4. Test Infrastructure (High Priority)
- [ ] Add proper test fixtures for field states
- [ ] Add quantum number validation
- [ ] Add correlation function helpers
- [ ] Add gauge transformation utilities

5. Critical Test Failures (High Priority)

1. UnifiedField Initialization Issues
- [ ] Fix UnifiedField.__init__() to accept dimension and max_level
- [ ] Add N_STABLE_MAX attribute to UnifiedField
- [ ] Fix compute_basis_state() to accept energy and n kwargs

2. Type System Issues
- [ ] Fix NumericValue handling for numpy arrays
- [ ] Fix FieldConfig initialization requirements
- [ ] Fix WaveFunction normalization

3. Function-scoped Fixture Issues
- [ ] Convert test fixtures to context managers:
  - [ ] quantum_state
  - [ ] rg_flow_context
  - [ ] holographic_context
  - [ ] fractal_context
  - [ ] gauge_context
  - [ ] scale_context

4. Computation Issues
- [ ] Fix symbolic to float conversions
- [ ] Fix time-ordered product computation
- [ ] Fix gauge transformation computations

1. Symbolic Expression Handling
- [ ] Fix float conversion in compute_correlator
- [ ] Fix integral evaluation in compute_energy_density
- [ ] Add proper handling for sympy expressions

2. Method Arguments
- [ ] Update compute_basis_state to accept energy kwarg
- [ ] Add rtol parameter to all compute_* methods
- [ ] Fix method signatures to match tests

3. Missing Methods
- [ ] Implement compute_boundary_state
- [ ] Implement compute_fractal_basis
- [ ] Implement compute_gauge_connection
- [ ] Add compute_rg_correction

# References
- See appendix_k_io_distinction.tex for measurement theory
- See appendix_b_gauge.tex for gauge transformations
- See appendix_g_holographic.tex for entropy calculations
- See appendix_a_convergence.tex for basis expansions
- See appendix_j_math_details.tex for symbolic computation
- See appendix_l_simplification.tex for optimization
