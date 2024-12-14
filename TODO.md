# Project TODO

## Critical Path (Paper Submission Blockers)

### 14.30 Core Theory Implementation
- [ ] Fix Critical Theoretical Components
  - [ ] Fix Import Dependencies (BUG-009)
    - [ ] Move common constants to physics_constants.py
      - [x] Add E symbol to physics_constants.py
      - [x] Update imports in conftest.py
      - [x] Move transformations to separate module
      - [x] Add missing error classes
    - [x] Update import hierarchy in field.py and basis.py
    - [ ] Fix circular dependencies in test fixtures
    - [ ] Refactor imports to resolve circular dependencies (BUG-018)
  - [ ] Fix Test Runner (BUG-021)
    - [ ] Fix Python Path Setup
    - [ ] Fix Test Import Structure
    - [ ] Add Import Validation
  - [ ] Fix Coordinate System (BUG-008, BUG-012)
    - [ ] Consolidate symbol definitions in constants.py
    - [ ] Update field calculations to use consistent coordinates
    - [ ] Add coordinate system tests
  - [ ] Fix UnifiedField Implementation (BUG-010)
    - [ ] Implement UnifiedField class
    - [ ] Add validation and constraints
    - [ ] Fix failing tests
    - [ ] Fix Version Control (BUG-004)

### 14.31 Validation Framework
- [ ] Complete Validation System
  - [ ] Fix Version Control (BUG-004)
  - [x] Add Property Tests (BUG-017)
  - [ ] Add missing test coverage (BUG-002)
  - [x] Generate coverage reports

## Implementation Status
- [x] Fix linter errors
  - [x] Add target_precision to generate_weights.py
  - [x] Add learning_rate to generate_weights.py 
  - [x] Import GUT_SCALE in test_predictions.py
- [ ] Check UnifiedField implementation
  - [x] compute_field() and tests ✓
  - [x] evolve_field() and tests ✓
  - [x] compute_field_equation() and tests ✓
  - [ ] For each remaining method:
    1. Implement method
    2. Write tests
    3. Verify coverage
    4. Move to next method

## Test Coverage & Validation
- [x] Add basic theorem tests
- [ ] Add scale invariance tests
  - [ ] Test scaling at different energy levels
  - [ ] Verify fractal dimension consistency
- [ ] Add gauge coupling tests
  - [ ] Test running coupling convergence
  - [ ] Verify unification point stability
- [ ] Add fermion mass tests
  - [ ] Test mass ratio predictions
  - [ ] Verify generational structure 
- [ ] Verify existing UnifiedField methods
  - [ ] compute_field()
  - [ ] evolve_field() 
  - [ ] compute_field_equation()
- [ ] Fix any failing tests in:
  - [ ] test_physics.py
  - [ ] test_consistency.py
  - [ ] test_integration.py
- [ ] Ensure test coverage meets 80% threshold

## Standard Model Features Implementation
- [ ] Neutrino Physics
  - [ ] compute_neutrino_angles()
  - [ ] compute_neutrino_masses()
  - [ ] Test coverage and validation
- [ ] CP Violation
  - [ ] compute_ckm_matrix()
  - [ ] extract_cp_phase()
  - [ ] compute_jarlskog()
  - [ ] compute_cp_violation()
  - [ ] compute_baryon_asymmetry()
  - [ ] Test coverage and validation
- [ ] Mass Generation
  - [ ] compute_higgs_vev()
  - [ ] compute_higgs_mass()
  - [ ] compute_fermion_masses()
  - [ ] compute_mass_ratios()
  - [ ] Test coverage and validation
