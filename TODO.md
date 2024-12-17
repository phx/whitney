# Core Framework Priorities

## 1. Type System (Critical)
- [ ] Fix NumericValue property access for 'value' attribute
- [ ] Add proper type conversion for numpy types (float64, ndarray)
- [ ] Implement valid_range support in NumericValue.__init__
- [ ] Fix WaveFunction normalization and validation
- [ ] Ensure proper type handling in field computations

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

## 3. Test Framework (Medium)
- [ ] Replace function-scoped fixtures with context managers
- [ ] Add HealthCheck suppression where appropriate
- [ ] Optimize slow tests exceeding deadline
- [ ] Fix hypothesis test configurations

## 4. Coverage Goals (High)
- [ ] Increase core/field.py coverage (currently 42%)
- [ ] Increase core/types.py coverage (currently 58%)
- [ ] Increase core/utils.py coverage (currently 59%)
- [ ] Reach overall 80% coverage target (currently 61.47%)

## 5. Documentation (Medium)
- [ ] Update testing_guide.md with new patterns
- [ ] Document type system enhancements
- [ ] Add examples for UnifiedField API
- [ ] Update scientific.md with theoretical foundations

## 6. Performance (Low)
- [ ] Fix ImmutableDenseNDimArray expand issue
- [ ] Optimize phase space integration
- [ ] Resolve memory usage in field evolution
- [ ] Improve numerical stability in gauge transformations

# References
- See appendix_k_io_distinction.tex for measurement theory
- See appendix_b_gauge.tex for gauge transformations
- See appendix_g_holographic.tex for entropy calculations
- See appendix_a_convergence.tex for basis expansions
