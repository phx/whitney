#!/usr/bin/env bash

# tests/test_field.py::test_field_initialization PASSED                                                                                                    [  2%]
# tests/test_field.py::test_energy_density FAILED                                                                                                          [  5%]
# tests/test_field.py::test_causality FAILED                                                                                                               [  8%]
# tests/test_field.py::test_field_evolution FAILED                                                                                                         [ 11%]
# tests/test_field.py::TestUnifiedFieldCore::test_field_initialization PASSED                                                                              [ 14%]
# tests/test_field.py::TestUnifiedFieldCore::test_invalid_initialization FAILED                                                                            [ 17%]
# tests/test_field.py::TestUnifiedFieldCore::test_sympy_conversion FAILED                                                                                  [ 20%]
# tests/test_field.py::TestUnifiedFieldCore::test_dark_matter_density FAILED                                                                               [ 22%]
# tests/test_field.py::TestUnifiedFieldCore::test_lorentz_invariance FAILED                                                                                [ 25%]
# tests/test_field.py::TestUnifiedFieldCore::test_causality FAILED                                                                                         [ 28%]
# tests/test_field.py::TestUnifiedFieldCore::test_energy_conservation FAILED                                                                               [ 31%]
# tests/test_field.py::TestUnifiedFieldCore::test_field_equations[1.0] FAILED                                                                              [ 34%]
# tests/test_field.py::TestUnifiedFieldCore::test_field_equations[10.0] FAILED                                                                             [ 37%]
# tests/test_field.py::TestUnifiedFieldCore::test_field_equations[100.0] FAILED                                                                            [ 40%]
# tests/test_field.py::TestQuantumFieldTheory::test_neutrino_angles FAILED                                                                                 [ 42%]
# tests/test_field.py::TestQuantumFieldTheory::test_fermion_masses FAILED                                                                                  [ 45%]
# tests/test_field.py::TestQuantumFieldTheory::test_coupling_evolution FAILED                                                                              [ 48%]
# tests/test_field.py::TestQuantumFieldTheory::test_gut_scale FAILED                                                                                       [ 51%]
# tests/test_field.py::TestQuantumFieldTheory::test_vertex_corrections FAILED                                                                              [ 54%]
# tests/test_field.py::TestQuantumFieldTheory::test_lsz_reduction FAILED                                                                                   [ 57%]
# tests/test_field.py::TestQuantumFieldTheory::test_correlator FAILED                                                                                      [ 60%]
# tests/test_field.py::TestQuantumFieldTheory::test_scattering_amplitudes FAILED                                                                           [ 62%]
# tests/test_field.py::TestQuantumFieldTheory::test_gauge_couplings[1] FAILED                                                                              [ 65%]
# tests/test_field.py::TestQuantumFieldTheory::test_gauge_couplings[2] FAILED                                                                              [ 68%]
# tests/test_field.py::TestQuantumFieldTheory::test_gauge_couplings[3] FAILED                                                                              [ 71%]
# tests/test_field.py::TestTheoreticalPredictions::test_fractal_dimension PASSED                                                                           [ 74%]
# tests/test_field.py::TestTheoreticalPredictions::test_mass_hierarchy PASSED                                                                              [ 77%]
# tests/test_field.py::TestTheoreticalPredictions::test_coupling_unification PASSED                                                                        [ 80%]
# tests/test_field.py::TestTheoreticalPredictions::test_holographic_entropy PASSED                                                                         [ 82%]
# tests/test_field.py::TestTheoreticalPredictions::test_quantum_corrections PASSED                                                                         [ 85%]
# tests/test_field.py::TestTheoreticalPredictions::test_beta_function PASSED                                                                               [ 88%]
# tests/test_field.py::TestTheoreticalPredictions::test_ward_identity PASSED                                                                               [ 91%]
# tests/test_field.py::TestTheoreticalPredictions::test_unitarity_bounds PASSED                                                                            [ 94%]
# tests/test_field.py::TestTheoreticalPredictions::test_fractal_recursion PASSED                                                                           [ 97%]
# tests/test_field.py::TestTheoreticalPredictions::test_dark_matter_profile PASSED

# tests/test_physics.py::TestPhysicsCalculations::test_coupling_evolution FAILED                                                                           [  9%]
# tests/test_physics.py::TestPhysicsCalculations::test_cross_sections FAILED                                                                               [ 18%]
# tests/test_physics.py::TestTheorems::test_unitarity FAILED                                                                                               [ 27%]
# tests/test_physics.py::TestNeutrinoPhysics::test_neutrino_mixing_angles PASSED                                                                           [ 36%]
# tests/test_physics.py::TestNeutrinoPhysics::test_neutrino_mass_hierarchy PASSED                                                                          [ 45%]
# tests/test_physics.py::TestNeutrinoPhysics::test_neutrino_oscillations PASSED                                                                            [ 54%]
# tests/test_physics.py::TestCPViolation::test_ckm_matrix PASSED                                                                                           [ 63%]
# tests/test_physics.py::TestCPViolation::test_jarlskog_invariant PASSED                                                                                   [ 72%]
# tests/test_physics.py::TestCPViolation::test_baryon_asymmetry PASSED                                                                                     [ 81%]
# tests/test_physics.py::TestMassGeneration::test_higgs_mechanism FAILED                                                                                   [ 90%]
# tests/test_physics.py::TestMassGeneration::test_fermion_masses PASSED                                                                          [100%]

run_previously_passing_tests() {
  pytest tests/test_field.py::TestTheoreticalPredictions -q --tb=no
  pytest tests/test_physics.py::TestNeutrinoPhysics tests/test_physics.py::TestCPViolation tests/testphysics.py::TestMassGeneration -q --tb=no
}

run_test_file() {
  pytest tests/"$1" ${@:2}
}

run_test_group() {
  pytest tests/test_physics.py::TestPhysicsCalculations ${@}
}

run_current_test() {
  pytest tests/test_field.py::TestTheoreticalPredictions::test_coupling_unification -vv --tb=long --showlocals
}

if [[ ("$1" = "-g") || ("$1" = "--group") ]]; then
  clear
  run_test_group -vv --tb=long --showlocals
  exit $?
elif [[ ("$1" = "-p") || ("$1" = "--previous") ]]; then
  if run_previously_passing_tests 2>&1 | grep FAILED; then
    clear
    run_previously_passing_tests -vv --tb=long --showlocals
    echo "Previously Passing Tests: FAILED"
    exit 1
  else
    echo "Previously Passing Tests: STILL PASSING"
    exit 0
  fi
elif [[ ("$1" = "-f") || ("$1" = "--file") ]]; then
  run_test_file "$2" ${@:3}
  exit $?
elif [[ ("$1" = "-a") || ("$1" = "--all") ]]; then
  clear
  pytest tests/ -q --tb=no
  exit $?
elif [[ -z "$1" ]]; then
  clear
  run_current_test
  exit $?
fi



