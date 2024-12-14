+ # Testing Guide
+ 
+ ## Running Tests
+ 
+ ### Basic Test Commands
+ 
+ 1. Run specific test file with verbose output:
+ ```bash
+ pytest -v test_theorems.py
+ ```
+ 
+ 2. Run with detailed error messages:
+ ```bash
+ pytest -v test_theorems.py -vv --tb=short
+ ```
+ 
+ 3. Check coverage:
+ ```bash
+ pytest --cov=core tests/ -v
+ ```
+ 
+ 4. Run only marked tests:
+ ```bash
+ pytest -v -m "theory" test_theorems.py
+ ```
+ 
+ 5. Stop on first failure:
+ ```bash
+ pytest -v -x test_theorems.py
+ ```
+ 
+ ### Test Organization
+ 
+ While `run_tests.py` provides additional functionality for:
+ - CI/CD pipeline integration
+ - Benchmark organization
+ - Test suite management
+ - Report generation
+ 
+ Direct pytest commands are recommended for development and debugging.
+ 
+ ## Test Categories
+ 
+ - `test_theorems.py`: Mathematical and physical theorems
+ - `test_consistency.py`: Framework consistency checks
+ - `test_integration.py`: Integration tests
+ - `test_physics.py`: Physics predictions
+ 
+ ## Example Test Cases
+ 
+ ### Testing Physical Theorems
+ ```python
+ def test_energy_conservation():
+     """Example: Test energy conservation in field evolution."""
+     field = UnifiedField()
+     psi = field.compute_field(E=100.0)  # Initial state
+     
+     # Evolve system
+     times = np.linspace(0, 10, 100)
+     results = field.evolve_field(psi, times)
+     
+     # Check energy conservation
+     E0 = results['energy'][0]
+     assert np.allclose(results['energy'], E0, rtol=1e-8)
+ ```
+ 
+ ### Testing Framework Consistency
+ ```python
+ def test_fractal_scaling():
+     """Example: Test fractal scaling relations."""
+     field = UnifiedField()
+     
+     # Test at different scales
+     E1, E2 = 100.0, 1000.0
+     ratio = E2/E1
+     
+     # Verify scaling
+     psi1 = field.compute_field(E1)
+     psi2 = field.compute_field(E2)
+     scaled = field.scale_transform(psi1, ratio)
+     assert field.check_equivalence(scaled, psi2)
+ ```
+ 
+ ### Testing Physical Predictions
+ ```python
+ def test_coupling_unification():
+     """Example: Test gauge coupling unification."""
+     field = UnifiedField()
+     
+     # Compute couplings at GUT scale
+     couplings = field.compute_couplings(GUT_SCALE)
+     alpha_gut = couplings['alpha_gut']
+     
+     # Check unification
+     assert abs(alpha_gut - 0.0376) < 0.0002
+ ```
+ 
+ ## Writing Good Tests
+ 
+ ### Test Structure
+ 
+ Follow the AAA pattern:
+ ```python
+ def test_something():
+     # Arrange
+     field = UnifiedField()
+     psi = field.compute_field(E=100.0)
+     
+     # Act
+     result = field.compute_observable(psi)
+     
+     # Assert
+     assert abs(result - expected) < tolerance
+ ```
+ 
+ ### Best Practices
+ 
+ 1. Test Names Should Be Descriptive
+ ```python
+ # Good
+ def test_energy_conservation_during_evolution()
+ 
+ # Bad
+ def test_energy()
+ ```
+ 
+ 2. One Assertion Concept Per Test
+ ```python
+ # Good
+ def test_coupling_unification_at_gut_scale():
+     """Test single concept: coupling convergence."""
+     field = UnifiedField()
+     couplings = field.compute_couplings(GUT_SCALE)
+     assert all(abs(c1 - c2) < 1e-3 for c1, c2 in combinations(couplings.values(), 2))
+ ```
+ 
+ 3. Use Appropriate Tolerances
+ ```python
+ # Physics comparisons need appropriate tolerances
+ assert abs(computed - theoretical) < 1e-8  # For conserved quantities
+ assert abs(ratio - predicted) < 0.1        # For experimental comparisons
+ ```
+ 
+ 4. Test Edge Cases
+ ```python
+ def test_field_behavior_at_singularities():
+     """Test behavior near coordinate singularities."""
+     field = UnifiedField()
+     
+     # Test at r = 0
+     assert not np.isnan(field.compute_field_strength(r=0))
+     
+     # Test at horizon
+     assert field.is_regular_at_horizon()
+ ```
+ 
+ 5. Use Fixtures for Common Setup
+ ```python
+ @pytest.fixture
+ def standard_field():
+     """Provide standard field configuration for tests."""
+     return UnifiedField(alpha=ALPHA_VAL)
+ 
+ def test_something(standard_field):
+     result = standard_field.compute_something()
+     assert result.is_valid()
+ ```
+ 
+ ### Physics-Specific Guidelines
+ 
+ 1. Conservation Laws
+ - Always test energy conservation
+ - Verify charge conservation
+ - Check angular momentum conservation
+ 
+ 2. Symmetries
+ - Test gauge invariance
+ - Verify Lorentz invariance
+ - Check discrete symmetries (C, P, T)
+ 
+ 3. Physical Constraints
+ - Causality (no FTL signaling)
+ - Unitarity (probability conservation)
+ - Positive energy conditions
+ 
+ ### Advanced Physics Test Examples
+ 
+ 1. Testing Fractal Structure
+ ```python
+ def test_fractal_dimension_convergence():
+     """Test convergence of fractal dimension to 4D."""
+     field = UnifiedField()
+     
+     # Compute dimensions at different scales
+     scales = np.logspace(2, 16, 10)  # GeV
+     dims = [field.compute_dimension(E) for E in scales]
+     
+     # Should converge to 4 (spacetime dimension)
+     assert abs(dims[-1] - 4.0) < 1e-10
+     
+     # Check convergence rate
+     diffs = np.diff(dims)
+     assert all(abs(d1) > abs(d2) for d1, d2 in zip(diffs, diffs[1:]))
+ ```
+ 
+ 2. Testing Holographic Principle
+ ```python
+ def test_holographic_entropy_bound():
+     """Test satisfaction of holographic entropy bound."""
+     field = UnifiedField()
+     
+     # Test volumes and areas
+     L = np.array([1.0, 2.0, 4.0])  # Planck units
+     areas = 6 * L**2  # Surface area of cubes
+     
+     for i, l in enumerate(L):
+         # Compute entropy in volume
+         S = field.compute_entropy(l)
+         # Check bound
+         assert S <= areas[i]/4, f"Holographic bound violated at L={l}"
+ ```
+ 
+ 3. Testing Quantum-Classical Transition
+ ```python
+ def test_decoherence_emergence():
+     """Test emergence of classical behavior."""
+     field = UnifiedField()
+     
+     # Prepare quantum superposition
+     psi = field.create_superposition([0, 1], [1/np.sqrt(2), 1/np.sqrt(2)])
+     
+     # Evolve with environment coupling
+     times = np.linspace(0, 10, 100)
+     rho_t = field.evolve_with_environment(psi, times)
+     
+     # Check decoherence
+     coherence = [field.compute_coherence(rho) for rho in rho_t]
+     assert all(c <= coherence[i] for i, c in enumerate(coherence[1:]))
+     assert coherence[-1] < 1e-8
+ ```
+ 
+ 4. Testing Gauge Coupling Unification
+ ```python
+ def test_detailed_coupling_unification():
+     """Test detailed aspects of coupling unification."""
+     field = UnifiedField()
+     
+     # Track coupling evolution
+     E_range = np.logspace(2, 16, 100)  # GeV
+     couplings = {E: field.compute_couplings(E) for E in E_range}
+     
+     # 1. Check convergence rate
+     diffs = []
+     for E in E_range:
+         c = couplings[E]
+         diff = max(abs(c[i] - c[j]) for i in range(3) for j in range(i))
+         diffs.append(diff)
+     
+     # Should decrease monotonically
+     assert all(d1 > d2 for d1, d2 in zip(diffs, diffs[1:]))
+     
+     # 2. Check final unification
+     final_couplings = couplings[E_range[-1]]
+     assert all(abs(final_couplings[i] - final_couplings[j]) < 1e-3
+               for i in range(3) for j in range(i))
+ ```
+ 
+ 5. Testing Dark Matter Properties
+ ```python
+ def test_dark_matter_predictions():
+     """Test dark matter predictions from fractal structure."""
+     field = UnifiedField()
+     
+     # Test relic density
+     omega_dm = field.compute_dark_matter_density()
+     assert abs(omega_dm - 0.258) < 0.008  # Planck 2018 value
+     
+     # Test mass prediction
+     m_dm = field.compute_dark_matter_mass()
+     theoretical = field.compute_theoretical_dm_mass()
+     assert abs(m_dm - theoretical) < 0.1  # GeV
+     
+     # Test galactic rotation curves
+     r = np.logspace(0, 2, 20)  # kpc
+     v_rot = field.compute_rotation_curve(r)
+     # Should be flat at large r
+     outer_v = v_rot[len(r)//2:]
+     assert np.std(outer_v)/np.mean(outer_v) < 0.1
+ ```
+ 
+ 6. Testing Gravitational Wave Spectrum
+ ```python
+ def test_gravitational_wave_predictions():
+     """Test gravitational wave spectrum from fractal structure."""
+     field = UnifiedField()
+     
+     # Test frequency range
+     f = np.logspace(-3, 3, 100)  # Hz
+     Omega_gw = field.compute_gw_spectrum(f)
+     
+     # Check fractal structure
+     for i in range(len(f)-1):
+         ratio = Omega_gw[i+1]/Omega_gw[i]
+         predicted = field.compute_fractal_ratio()
+         assert abs(ratio - predicted) < 0.1
+     
+     # Test LIGO band prediction
+     ligo_freq = 100.0  # Hz
+     Omega_ligo = field.compute_gw_density(ligo_freq)
+     assert Omega_ligo < 1e-7  # Current bounds
+ ```
+ 
+ 7. Testing Quantum Gravity Effects
+ ```python
+ def test_quantum_gravity_regime():
+     """Test behavior near Planck scale."""
+     field = UnifiedField()
+     
+     # Test dimensional reduction
+     E = np.logspace(16, 19, 10)  # GeV
+     dims = [field.compute_effective_dimension(e) for e in E]
+     
+     # Should decrease toward 2 at high energy
+     assert all(d1 > d2 for d1, d2 in zip(dims, dims[1:]))
+     assert 2.0 <= dims[-1] <= 2.1
+     
+     # Test discreteness emergence
+     l_min = field.compute_minimum_length()
+     assert abs(l_min - field.compute_planck_length()) < 1e-10
+ ```
+ 
+ 8. Testing Neutrino Mixing
+ ```python
+ def test_neutrino_mixing_predictions():
+     """Test neutrino mixing angles from fractal structure."""
+     field = UnifiedField()
+     
+     # Get mixing angles
+     theta_12, theta_23, theta_13 = field.compute_neutrino_angles()
+     
+     # Check against experimental values
+     assert abs(np.sin(theta_12)**2 - 0.307) < 0.013  # Solar angle
+     assert abs(np.sin(theta_23)**2 - 0.545) < 0.020  # Atmospheric angle
+     assert abs(np.sin(theta_13)**2 - 0.022) < 0.001  # Reactor angle
+     
+     # Test fractal origin
+     predicted = field.compute_theoretical_angles()
+     assert all(abs(t1 - t2) < 0.1 for t1, t2 in zip([theta_12, theta_23, theta_13], predicted))
+ ```
+ 
+ 9. Testing CP Violation
+ ```python
+ def test_cp_violation_mechanism():
+     """Test CP violation emergence from fractal structure."""
+     field = UnifiedField()
+     
+     # Compute Jarlskog invariant
+     J = field.compute_jarlskog()
+     assert abs(J - 3.2e-5) < 0.3e-5  # Experimental value
+     
+     # Test CKM matrix properties
+     V = field.compute_ckm_matrix()
+     # Unitarity
+     assert np.allclose(V @ V.conj().T, np.eye(3), rtol=1e-10)
+     # CP phase
+     delta = field.extract_cp_phase(V)
+     assert abs(delta - 1.36) < 0.04  # radians
+ ```
+ 
+ 10. Testing Mass Generation
+ ```python
+ def test_mass_generation_mechanism():
+     """Test fractal mass generation mechanism."""
+     field = UnifiedField()
+     
+     # Test Higgs mechanism
+     v = field.compute_higgs_vev()
+     assert abs(v - 246.0) < 0.1  # GeV
+     
+     # Test fermion masses
+     masses = field.compute_fermion_masses()
+     # Check top quark mass
+     assert abs(masses['top'] - 173.0) < 0.4  # GeV
+     # Check electron mass
+     assert abs(masses['electron'] - 0.511) < 0.001  # MeV
+     
+     # Verify fractal hierarchy
+     ratios = field.compute_mass_ratios()
+     assert abs(ratios['mu_tau'] - 0.0595) < 0.0001
+ ```
+ 
+ 11. Testing Baryon Asymmetry
+ ```python
+ def test_baryon_asymmetry_generation():
+     """Test baryon asymmetry generation from fractal dynamics."""
+     field = UnifiedField()
+     
+     # Test CP violation strength
+     epsilon = field.compute_cp_violation()
+     assert abs(epsilon - 1e-6) < 1e-7  # Required for observed asymmetry
+     
+     # Test out-of-equilibrium condition
+     gamma = field.compute_interaction_rate(T=1e15)  # GeV
+     H = field.compute_hubble_rate(T=1e15)
+     assert gamma < H  # Necessary for asymmetry generation
+     
+     # Test final asymmetry
+     eta_B = field.compute_baryon_asymmetry()
+     assert abs(eta_B - 6.1e-10) < 0.3e-10  # Matches observation
+ ```
+ 
+ 12. Testing Inflation Predictions
+ ```python
+ def test_inflation_predictions():
+     """Test inflationary predictions from fractal structure."""
+     field = UnifiedField()
+     
+     # Test spectral index
+     ns = field.compute_spectral_index()
+     assert abs(ns - 0.9649) < 0.0042  # Planck 2018
+     
+     # Test tensor-to-scalar ratio
+     r = field.compute_tensor_ratio()
+     assert r < 0.056  # Current bound
+     
+     # Test fractal structure in fluctuations
+     k = np.logspace(-4, 2, 100)  # Mpc^-1
+     P_k = field.compute_power_spectrum(k)
+     
+     # Check scaling behavior
+     ratios = P_k[1:] / P_k[:-1]
+     predicted = field.compute_fractal_scaling()
+     assert all(abs(r - predicted) < 0.1 for r in ratios)
+ ```
+ 
+ 13. Testing Black Hole Thermodynamics
+ ```python
+ def test_black_hole_thermodynamics():
+     """Test black hole thermodynamics in fractal framework."""
+     field = UnifiedField()
+     
+     # Test Hawking temperature scaling
+     M = np.logspace(0, 4, 10)  # Solar masses
+     T = field.compute_hawking_temperature(M)
+     
+     # Should follow fractal scaling
+     ratios = T[1:] / T[:-1]
+     predicted = field.compute_fractal_temperature_ratio()
+     assert all(abs(r - predicted) < 0.1 for r in ratios)
+     
+     # Test entropy-area relation
+     A = field.compute_horizon_area(M)
+     S = field.compute_black_hole_entropy(M)
+     
+     # Should include fractal corrections
+     S_classical = A / 4  # Classical Bekenstein-Hawking
+     corrections = field.compute_fractal_entropy_corrections(M)
+     assert np.allclose(S, S_classical + corrections, rtol=1e-10)
+ ```
+ 
+ ## Coverage Requirements
+ 
+ Aim for minimum 80% coverage across core modules:
+ - field.py
+ - basis.py
+ - modes.py
+ 
+ ## Running Benchmarks
+ 
+ See `benchmarks/run_benchmarks.py` for performance testing. 