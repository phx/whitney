"""Tests for theoretical physics theorems."""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck
from sympy import exp, I, pi, sqrt, integrate, conjugate, oo, Matrix, diff
from core.field import UnifiedField
from core.basis import FractalBasis
from core.types import Energy
from core.errors import PhysicsError, GaugeError
from core.contexts import (
    gauge_phase,
    quantum_state,
    field_config,
    numeric_precision,
    lorentz_boost
)
from core.physics_constants import (
    ALPHA_VAL,
    Z_MASS,
    X, T, C, HBAR,
    g1_REF, g2_REF, g3_REF
)

@pytest.fixture
def field():
    """Create UnifiedField instance for testing."""
    return UnifiedField(alpha=ALPHA_VAL)

@pytest.fixture
def basis():
    """Create FractalBasis instance for testing."""
    return FractalBasis(alpha=ALPHA_VAL)

@pytest.fixture
def constants(request):
    """Get physics constants from conftest."""
    from core.physics_constants import X, T, P, HBAR, C
    return {'X': X, 'T': T, 'P': P, 'HBAR': HBAR, 'C': C}

class TestMathematicalTheorems:
    """Test mathematical theorems and proofs."""
    
    def test_completeness_theorem(self, basis):
        """Prove completeness of fractal basis."""
        with field_config() as config, numeric_precision(rtol=1e-6) as prec:
            with quantum_state(energy=1.0) as (f, _):
                # Expand in basis up to N
                N = 10
                expansion = 0
                for n in range(N):
                    psi_n = basis.compute(n, config, **prec)
                    cn = integrate(psi_n * f, (X, -oo, oo))
                    expansion += cn * psi_n
                
                # Compute L2 norm of difference
                error = integrate((f - expansion)**2, (X, -oo, oo))
                assert float(error) < prec['rtol']

    def test_unitarity(self, field):
        """Prove unitarity of time evolution."""
        with quantum_state(energy=Z_MASS) as (psi, _), numeric_precision(rtol=1e-8) as prec:
            times = np.linspace(0, 10, 100)
            evolution = field.evolve_field(psi, times, **prec)
            norms = evolution['norm']
            assert np.allclose(norms, 1.0, **prec)

    def test_causality(self, field):
        """Prove causality of field propagation."""
        with quantum_state(energy=1.0) as (psi, _), numeric_precision(rtol=1e-10) as prec:
            # Test points outside light cone (space-like separation)
            with field_config(dimension=4) as config:
                # Compute field commutator at space-like separation
                commutator = field.compute_commutator(psi, config, separation='space-like', **prec)
                assert abs(float(commutator)) < prec['rtol']
                
                # Verify causal propagation
                propagator = field.compute_propagator(psi, config, **prec)
                assert field.check_causality(propagator, **prec)

class TestFieldTheorems:
    """Test theoretical physics theorems."""
    
    def test_noether_current(self, field, constants):
        """Test Noether's theorem for U(1) symmetry."""
        with quantum_state(energy=1.0) as (psi, _), numeric_precision(rtol=1e-10) as prec:
            # Compute conserved current with gauge invariance
            with gauge_phase() as phase:
                # Original current
                j0, j1 = field.compute_noether_current(psi, **prec)
                
                # Verify current conservation: ∂_μj^μ = 0
                divergence = field.compute_current_divergence(j0, j1, **prec)
                assert abs(divergence) < prec['rtol']
                
                # Verify gauge covariance
                psi_transformed = field.apply_gauge_transform(psi, phase)
                j0_trans, j1_trans = field.compute_noether_current(psi_transformed, **prec)
                assert field.check_current_covariance(
                    (j0, j1), 
                    (j0_trans, j1_trans),
                    phase,
                    **prec
                )
    
    def test_cpt_theorem(self, field, constants):
        """Test CPT theorem."""
        X, T, C = constants['X'], constants['T'], constants['C']
        
        with quantum_state(energy=1.0) as (psi, _), numeric_precision(rtol=1e-10) as prec:
            # Original state
            psi = exp(-(X**2 + (C*T)**2))
            
            # Apply CPT transformation
            psi_cpt = field.apply_cpt_transform(psi, **prec)
            
            # Compute observables
            E = field.compute_energy_density(psi, **prec)
            E_cpt = field.compute_energy_density(psi_cpt, **prec)
            
            # Should be equal under CPT
            assert abs(E - E_cpt) < prec['rtol']
    
    def test_spin_statistics(self, field):
        """Test spin-statistics theorem."""
        with field_config(dimension=4) as config, numeric_precision(rtol=1e-8) as prec:
            # Create test particles
            bosons = field.create_spin_0_particles(2)
            fermions = field.create_spin_half_particles(2)
            
            # Test exchange properties
            assert field.check_boson_symmetry(bosons, **prec)
            assert field.check_fermion_antisymmetry(fermions, **prec)

    def test_hierarchy_resolution(self):
        """Test hierarchy problem resolution."""
        field = UnifiedField()
        with field_config(dimension=4) as config, numeric_precision(rtol=1e-3) as prec:
            ratio = field.compute_mass_ratio()
            expected = field.compute_theoretical_ratio()
            assert abs(ratio - expected) < 0.1  # 10% accuracy
            assert field.check_hierarchy_consistency(**prec)

    def test_fractal_convergence_rate(self):
        """Test fractal series convergence rate."""
        field = UnifiedField()
        with numeric_precision(rtol=1e-8) as prec:
            # Check convergence is faster than 1/n^2
            terms = [field.compute_term(n) for n in range(1, 10)]
            ratios = [abs(terms[i+1]/terms[i]) for i in range(len(terms)-1)]
            assert all(r < 1/n**2 for n, r in enumerate(ratios, 2))
            
            # Also verify hierarchy is preserved
            assert field.check_hierarchy_consistency(**prec)

    def test_fractal_recursion_relations(self):
        """Test fractal recursion relations."""
        field = UnifiedField()
        basis = FractalBasis()
        
        with field_config(dimension=4) as config, numeric_precision(rtol=1e-8) as prec:
            # Check recursion relation for first few modes
            for n in range(3):
                fn = basis.compute_term(n)
                fn_plus_1 = basis.compute_term(n + 1)
                relation = field.check_recursion_relation(fn, fn_plus_1)
                assert relation, f"Recursion relation failed for n={n}"
                # Additional verification with precision
                assert field.verify_recursion_coefficients(fn, fn_plus_1, **prec)
                assert field.check_fractal_consistency(fn, n, **prec)

    def test_cpt_invariance(self):
        """Test CPT invariance."""
        field = UnifiedField()
        
        with quantum_state(energy=1.0) as (psi, _), numeric_precision(rtol=1e-10) as prec:
            # Test state
            psi = field.ground_state()
            
            # Apply CPT transformation
            psi_cpt = field.apply_cpt(psi)
            
            # Check observables are invariant
            obs = field.compute_observables(psi)
            obs_cpt = field.compute_observables(psi_cpt)
            
            for key in obs:
                # Use precision parameter for comparison
                assert np.isclose(obs[key], obs_cpt[key], rtol=1e-10)
                assert field.verify_cpt_consistency(obs[key], obs_cpt[key], **prec)

    def test_holographic_entropy_scaling(self):
        """Test holographic entropy scaling."""
        field = UnifiedField()
        
        with field_config(dimension=4) as config, numeric_precision(rtol=1e-8) as prec:
            # Test entropy scaling with area
            areas = [1.0, 2.0, 4.0, 8.0]  # Planck units
            entropies = [field.compute_entropy(A) for A in areas]
            
            # Should scale linearly with area
            ratios = [S/A for S, A in zip(entropies, areas)]
            assert np.allclose(ratios, ratios[0], **prec)
            
            # Check coefficient matches holographic bound
            assert all(S <= A/4 for S, A in zip(entropies, areas))
            
            # Additional holographic principle verifications
            assert field.verify_area_law(areas, entropies, **prec)
            assert field.check_holographic_bound(areas, entropies, **prec)
            assert field.verify_fractal_entropy_scaling(areas, entropies, **prec)
    
    def test_rg_fixed_points(self):
        """Test RG fixed points."""
        field = UnifiedField()
        
        with field_config(dimension=4) as config, numeric_precision(rtol=1e-8) as prec:
            # Compute beta functions near fixed points
            E_gut = field.compute_gut_scale()
            beta_funcs = field.compute_beta_functions(E_gut)
            
            # Should vanish at fixed point
            assert all(abs(beta) < prec['rtol'] for beta in beta_funcs.values())
            
            # Additional RG flow verifications
            assert field.verify_rg_consistency(E_gut, **prec)
            assert field.check_fixed_point_stability(beta_funcs, **prec)
            assert field.verify_fractal_rg_structure(E_gut, **prec)
    
    def test_cp_violation(self):
        """Test CP violation mechanism."""
        field = UnifiedField()
        
        with field_config(dimension=4) as config, numeric_precision(rtol=1e-5) as prec:
            # Compute Jarlskog invariant
            J = field.compute_jarlskog()
            
            # Should match experimental value
            assert abs(J - 3.2e-5) < 0.3e-5
            
            # Additional CP violation verifications
            assert field.verify_cp_violation_mechanism(J, **prec)
            assert field.check_baryon_asymmetry_consistency(J, **prec)
            assert field.verify_ckm_unitarity(**prec)
            assert field.check_cp_phase_origin(**prec)

    def test_gravitational_wave_spectrum(self):
        """Test gravitational wave spectrum."""
        field = UnifiedField()
        
        with field_config(dimension=4) as config, numeric_precision(rtol=1e-6) as prec:
            # Test frequencies
            freqs = np.logspace(-3, 3, 10)  # Hz
            
            # Compute spectrum
            Omega_gw = field.compute_gw_spectrum(freqs)
            
            # Check fractal structure
            for i in range(len(freqs)-1):
                ratio = Omega_gw[i+1] / Omega_gw[i]
                # Should follow fractal scaling
                assert abs(ratio - field.compute_fractal_ratio()) < 1e-6
                
            # Additional gravitational wave verifications
            assert field.verify_spectrum_consistency(freqs, Omega_gw, **prec)
            assert field.check_fractal_spectrum_structure(Omega_gw, **prec)
            assert field.verify_energy_scale_dependence(freqs, Omega_gw, **prec)
    
    def test_quantum_measurement(self):
        """Test quantum measurement process."""
        field = UnifiedField()
        
        with field_config(dimension=4) as config, numeric_precision(rtol=1e-8) as prec:
            with quantum_state(energy=1.0) as (psi_base, _):
                # Prepare superposition state
                psi = field.create_superposition([0, 1], [1/np.sqrt(2), 1/np.sqrt(2)])
                
                # Perform measurement
                outcomes, probs = field.compute_measurement_probabilities(psi)
                
                # Check Born rule
                assert np.allclose(probs, [0.5, 0.5], **prec)
                
                # Verify decoherence
                rho_final = field.evolve_with_environment(psi)
                coherence = field.compute_coherence(rho_final)
                assert coherence < prec['rtol']
                
                # Additional quantum measurement verifications
                assert field.verify_measurement_consistency(outcomes, probs, **prec)
                assert field.check_decoherence_timescale(rho_final, **prec)
                assert field.verify_quantum_classical_transition(psi, rho_final, **prec)
                assert field.check_measurement_basis_independence(psi, **prec)

    def test_scale_invariance(self):
        """Test scale invariance."""
        field = UnifiedField()
        
        with field_config(dimension=4) as config, numeric_precision(rtol=1e-6) as prec:
            # Test energies
            E1, E2 = 100.0, 1000.0  # GeV
            
            # Compute field at different scales
            psi1 = field.compute_field(E1)
            psi2 = field.compute_field(E2)
            
            # Check scaling relation
            ratio = E2/E1
            scaled = field.scale_transform(psi1, ratio)
            assert field.check_equivalence(scaled, psi2, **prec)
            
            # Additional scale invariance verifications
            assert field.verify_scaling_consistency(psi1, psi2, ratio, **prec)
            assert field.check_fractal_scale_invariance(psi1, E1, **prec)
            assert field.verify_scale_dependent_coupling(E1, E2, **prec)

    def test_gauge_coupling_unification(self):
        """Test gauge coupling unification."""
        field = UnifiedField()
        
        with field_config(dimension=4) as config, numeric_precision(rtol=1e-6) as prec:
            # Test range of energies approaching GUT scale
            E_range = np.logspace(2, 16, 10)  # GeV
            
            # Compute couplings
            alphas = {E: field.compute_couplings(E) for E in E_range}
            
            # Verify convergence at GUT scale
            final_couplings = alphas[E_range[-1]]
            assert all(abs(final_couplings[i] - final_couplings[j]) < prec['rtol']
                      for i in range(3) for j in range(i))
            
            # Check running matches beta functions
            for E1, E2 in zip(E_range[:-1], E_range[1:]):
                beta_predicted = field.compute_beta_functions(E1)
                actual_change = {k: (alphas[E2][k] - alphas[E1][k])/np.log(E2/E1)
                               for k in alphas[E1]}
                assert all(abs(beta_predicted[k] - actual_change[k]) < prec['rtol']
                          for k in beta_predicted)
                
                # Additional unification verifications
                assert field.verify_coupling_convergence(E1, E2, **prec)
                assert field.check_beta_function_consistency(beta_predicted, actual_change, **prec)
                assert field.verify_gut_scale_emergence(E_range, alphas, **prec)
                assert field.check_threshold_corrections(E1, E2, **prec)

    def test_fermion_mass_hierarchy(self):
        """Test fermion mass hierarchy."""
        field = UnifiedField()
        
        # Get mass ratios between generations
        ratios = field.compute_mass_ratios()
        
        # Check electron/muon/tau ratios
        assert abs(ratios['e_mu'] - 0.00484) < 0.0001
        assert abs(ratios['mu_tau'] - 0.0595) < 0.0001
        
        # Check quark mass ratios
        assert abs(ratios['u_c'] - 0.0023) < 0.0001
        assert abs(ratios['c_t'] - 0.0074) < 0.0001
        
        # Verify fractal origin
        alpha = field.alpha
        predicted = field.compute_theoretical_ratios(alpha)
        for key in ratios:
            assert abs(ratios[key] - predicted[key]) < 1e-3
        
        # Check generational structure
        assert field.verify_fractal_generations()

    def test_dark_matter_predictions(self):
        """Test dark matter predictions."""
        field = UnifiedField()
        
        # Test relic density prediction
        omega_dm = field.compute_dark_matter_density()
        assert abs(omega_dm - 0.258) < 0.008  # Planck 2018
        
        # Test mass prediction from fractal structure
        m_dm = field.compute_dark_matter_mass()
        assert abs(m_dm - field.compute_theoretical_dm_mass()) < 0.1  # GeV
        
        # Check coupling to visible matter
        sigma_dm = field.compute_dm_cross_section()
        # Should be just below current bounds
        assert sigma_dm < 1e-45  # cm^2
        
        # Verify fractal origin of dark sector
        alpha_dark = field.compute_dark_coupling()
        assert abs(alpha_dark - field.alpha**(field.compute_dark_level())) < 1e-6
        
        # Test structure formation consistency
        power_spectrum = field.compute_matter_power_spectrum()
        assert field.check_structure_formation(power_spectrum)

    def test_galactic_dark_matter(self):
        """Test galactic dark matter distribution."""
        field = UnifiedField()
        
        # Test radii in kpc
        radii = np.logspace(0, 2, 20)  # 1-100 kpc
        
        # Compute rotation curves
        v_rot = field.compute_rotation_curve(radii)
        
        # Should be flat at large radii
        outer_curve = v_rot[len(radii)//2:]
        assert np.std(outer_curve)/np.mean(outer_curve) < 0.1
        
        # Test density profile
        rho = field.compute_dm_density_profile(radii)
        
        # Should follow fractal scaling
        for i in range(len(radii)-1):
            r_ratio = radii[i+1]/radii[i]
            density_ratio = rho[i+1]/rho[i]
            predicted = field.compute_fractal_density_ratio(r_ratio)
            assert abs(density_ratio - predicted) < 0.1
        
        # Test core-cusp resolution
        inner_slope = field.compute_inner_density_slope()
        assert -1.0 < inner_slope < -0.3  # Neither cuspy nor flat
        
        # Verify consistency with dwarf galaxies
        assert field.check_dwarf_galaxy_consistency()

    def test_dark_matter_substructure(self):
        """Test dark matter substructure predictions from appendix_e."""
        field = UnifiedField()
        
        # Test subhalo mass function
        masses = np.logspace(6, 12, 20)  # Solar masses
        dN_dM = field.compute_subhalo_mass_function(masses)
        
        # Should follow fractal scaling law
        slope = field.compute_mass_function_slope(dN_dM, masses)
        assert abs(slope + 1.9) < 0.1  # Close to N(>M) ∝ M^-1.9
        
        # Test satellite galaxy distribution
        radii = np.logspace(0, 3, 20)  # kpc
        N_sats = field.compute_satellite_distribution(radii)
        
        # Verify fractal hierarchy
        levels = field.compute_substructure_levels(radii)
        for i in range(len(levels)-1):
            ratio = N_sats[i+1]/N_sats[i]
            predicted = field.alpha**(levels[i+1] - levels[i])
            assert abs(ratio - predicted) < 0.1
        
        # Test missing satellite problem resolution
        suppression = field.compute_small_scale_suppression()
        assert field.check_satellite_consistency(suppression)
        
        # Verify connection to quantum effects
        coherence_scale = field.compute_quantum_coherence_scale()
        assert field.verify_quantum_substructure_connection(coherence_scale)

    def test_field_evolution(self):
        """Test field evolution preserves key properties."""
        field = UnifiedField()
        
        # Create test state
        psi0 = field.compute_field(E=100.0)  # 100 GeV
        times = np.linspace(0, 10, 100)  # 10 time steps
        
        # Evolve
        results = field.evolve_field(psi0, times)
        
        # Check norm conservation
        assert np.allclose(results['norm'], 1.0, rtol=1e-8)
        
        # Check energy conservation
        E0 = results['energy'][0]
        assert np.allclose(results['energy'], E0, rtol=1e-8)
        
        # Check fractal structure preservation
        for psi in results['psi'][1:]:
            ratio = field._compute_fractal_correction(psi) / field._compute_fractal_correction(psi0)
            assert abs(ratio - 1.0) < 0.1

    def test_field_equation_properties(self):
        """Test field equation satisfies key physical properties."""
        field = UnifiedField()
        
        # Test state
        psi = field.compute_field(E=100.0)  # 100 GeV
        
        # Compute field equation result
        F_psi = field.compute_field_equation(psi)
        
        # 1. Check Klein-Gordon limit at low energy
        kg_term = diff(psi, T, 2)/C**2 - diff(psi, X, 2) + (field.alpha/HBAR**2) * psi
        assert abs(F_psi - kg_term) < field.alpha**2  # Fractal terms are higher order
        
        # 2. Verify energy conservation
        E1 = field.compute_energy(psi)
        evolved = field.evolve_field(psi, np.array([0, 1.0]))
        E2 = evolved['energy'][-1]
        assert abs(E1 - E2) < 1e-10
        
        # 3. Check fractal corrections
        fractal_terms = field._compute_fractal_field_terms(psi)
        # Should scale as alpha^2
        assert abs(fractal_terms) < field.alpha**2 * abs(psi)

    @settings(suppress_health_check=[
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow
    ])
    @given(st.floats(min_value=0.1, max_value=10.0))
    def test_energy_conservation(self, time):
        """Test energy conservation in field evolution."""
        field = UnifiedField()
        psi = field.compute_basis_state(n=0)
        
        # Evolve system
        times = np.linspace(0, time, 100)
        results = field.evolve_field(psi, times)
        
        # Check energy conservation
        E0 = results['energy'][0]
        assert np.allclose(results['energy'], E0, rtol=1e-8)