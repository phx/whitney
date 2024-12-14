"""Tests for theoretical physics theorems."""

import pytest
import numpy as np
from sympy import exp, I, pi, Symbol, integrate, oo, diff
from core.field import UnifiedField
from core.basis import FractalBasis
from core.types import Energy
from core.errors import PhysicsError
from core.physics_constants import (
    ALPHA_VAL,
    Z_MASS,
    ALPHA_REF
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
        """
        Prove completeness of fractal basis.
        Theorem 1 from paper: The fractal basis forms a complete set.
        """
        x = Symbol('x')
        
        # Test function (Gaussian)
        f = exp(-x**2)
        
        # Expand in basis up to N
        N = 10
        expansion = 0
        for n in range(N):
            psi_n = basis.compute(n)
            cn = integrate(psi_n * f, (x, -oo, oo))
            expansion += cn * psi_n
        
        # Compute L2 norm of difference
        error = integrate((f - expansion)**2, (x, -oo, oo))
        assert float(error) < 1e-6

    def test_unitarity(self, field):
        """
        Prove unitarity of time evolution.
        Theorem 2: Time evolution preserves probability.
        """
        psi = field.compute_basis_function(n=0, E=Z_MASS)
        
        # Evolve for various times
        times = np.linspace(0, 10, 100)
        evolution = field.evolve_field(psi, times)
        
        # Check norm conservation
        norms = evolution['norm']
        assert np.allclose(norms, 1.0, rtol=1e-8)

    def test_causality(self, field):
        """
        Prove causality of field propagation.
        Theorem 3: Field propagation respects light cone structure.
        """
        psi = field.compute_basis_function(n=0)
        
        # Test points outside light cone
        t, x = 1.0, 2.0  # Space-like separation
        
        # Compute commutator of fields
        commutator = field.compute_field_equation(psi.subs('t', t).subs('x', x))
        assert abs(float(commutator)) < 1e-10

class TestFieldTheorems:
    """Test fundamental field theory theorems."""
    
    def test_noether_current(self, field, constants):
        """Test Noether's theorem for U(1) symmetry."""
        X, T, C, HBAR = constants['X'], constants['T'], constants['C'], constants['HBAR']
        psi = exp(-(X**2 + (C*T)**2)/(2*HBAR**2))
        
        # Compute conserved current
        j0, j1 = field.compute_noether_current(psi)
        
        # Verify current conservation: ∂_μj^μ = 0
        d_t_j0 = field.compute_time_derivative(j0)
        d_x_j1 = field.compute_space_derivative(j1)
        
        assert abs(d_t_j0 + d_x_j1) < 1e-10
    
    def test_cpt_theorem(self, field, constants):
        """Test CPT theorem."""
        X, T, C = constants['X'], constants['T'], constants['C']
        
        # Original state
        psi = exp(-(X**2 + (C*T)**2))
        
        # Apply CPT transformation
        psi_cpt = field.apply_cpt_transform(psi)
        
        # Compute observables
        E = field.compute_energy_density(psi)
        E_cpt = field.compute_energy_density(psi_cpt)
        
        # Should be equal under CPT
        assert abs(E - E_cpt) < 1e-10
    
    def test_spin_statistics(self, field):
        """Test spin-statistics theorem."""
        # Create test particles
        bosons = field.create_spin_0_particles(2)
        fermions = field.create_spin_half_particles(2)
        
        # Test exchange properties
        assert field.check_boson_symmetry(bosons)
        assert field.check_fermion_antisymmetry(fermions)

    def test_hierarchy_resolution():
        """Test natural resolution of hierarchy problem."""
        field = UnifiedField()
        ratio = field.compute_mass_ratio()
        expected = field.compute_theoretical_ratio()
        assert abs(ratio - expected) < 0.1  # 10% accuracy
        
    @pytest.mark.theory
    def test_fractal_convergence_rate():
        """Test the convergence rate of fractal series."""
        basis = FractalBasis()
        
        # Check convergence is faster than 1/n^2
        terms = [basis.compute_term(n) for n in range(1, 10)]
        ratios = [abs(terms[i+1]/terms[i]) for i in range(len(terms)-1)]
        assert all(r < 1/n**2 for n, r in enumerate(ratios, 2))
        
        # Also verify hierarchy is preserved
        field = UnifiedField()
        assert field.check_hierarchy_consistency()

    def test_fractal_recursion_relations():
        """Test the fractal recursion relations from appendix_a."""
        field = UnifiedField()
        basis = FractalBasis()
        
        # Check recursion relation for first few modes
        for n in range(3):
            fn = basis.compute_term(n)
            fn_plus_1 = basis.compute_term(n + 1)
            relation = field.check_recursion_relation(fn, fn_plus_1)
            assert relation, f"Recursion relation failed for n={n}"

    def test_cpt_invariance():
        """Test CPT invariance of the fractal structure."""
        field = UnifiedField()
        
        # Test state
        psi = field.ground_state()
        
        # Apply CPT transformation
        psi_cpt = field.apply_cpt(psi)
        
        # Check observables are invariant
        obs = field.compute_observables(psi)
        obs_cpt = field.compute_observables(psi_cpt)
        
        for key in obs:
            assert np.isclose(obs[key], obs_cpt[key], rtol=1e-10)

    def test_holographic_entropy_scaling():
        """Test holographic entropy scaling from appendix_g."""
        field = UnifiedField()
        
        # Test entropy scaling with area
        areas = [1.0, 2.0, 4.0, 8.0]  # Planck units
        entropies = [field.compute_entropy(A) for A in areas]
        
        # Should scale linearly with area
        ratios = [S/A for S, A in zip(entropies, areas)]
        assert np.allclose(ratios, ratios[0], rtol=1e-8)
        
        # Check coefficient matches holographic bound
        assert all(S <= A/4 for S, A in zip(entropies, areas))
    
    def test_rg_fixed_points():
        """Test RG flow fixed points from appendix_h."""
        field = UnifiedField()
        
        # Compute beta functions near fixed points
        E_gut = field.compute_gut_scale()
        beta_funcs = field.compute_beta_functions(E_gut)
        
        # Should vanish at fixed point
        assert all(abs(beta) < 1e-8 for beta in beta_funcs.values())
    
    def test_cp_violation():
        """Test CP violation mechanism from appendix_i."""
        field = UnifiedField()
        
        # Compute Jarlskog invariant
        J = field.compute_jarlskog()
        
        # Should match experimental value
        assert abs(J - 3.2e-5) < 0.3e-5

    def test_gravitational_wave_spectrum():
        """Test gravitational wave spectrum from appendix_c."""
        field = UnifiedField()
        
        # Test frequencies
        freqs = np.logspace(-3, 3, 10)  # Hz
        
        # Compute spectrum
        Omega_gw = field.compute_gw_spectrum(freqs)
        
        # Check fractal structure
        for i in range(len(freqs)-1):
            ratio = Omega_gw[i+1] / Omega_gw[i]
            # Should follow fractal scaling
            assert abs(ratio - field.compute_fractal_ratio()) < 1e-6
    
    def test_quantum_measurement():
        """Test quantum measurement emergence from appendix_i."""
        field = UnifiedField()
        
        # Prepare superposition state
        psi = field.create_superposition([0, 1], [1/np.sqrt(2), 1/np.sqrt(2)])
        
        # Perform measurement
        outcomes, probs = field.compute_measurement_probabilities(psi)
        
        # Check Born rule
        assert np.allclose(probs, [0.5, 0.5], rtol=1e-8)
        
        # Verify decoherence
        rho_final = field.evolve_with_environment(psi)
        coherence = field.compute_coherence(rho_final)
        assert coherence < 1e-8

    def test_scale_invariance():
        """Test scale invariance properties from appendix_d."""
        field = UnifiedField()
        
        # Test energies
        E1, E2 = 100.0, 1000.0  # GeV
        
        # Compute field at different scales
        psi1 = field.compute_field(E1)
        psi2 = field.compute_field(E2)
        
        # Check scaling relation
        ratio = E2/E1
        scaled = field.scale_transform(psi1, ratio)
        assert field.check_equivalence(scaled, psi2)

    def test_gauge_coupling_unification():
        """Test gauge coupling unification properties from appendix_h."""
        field = UnifiedField()
        
        # Test range of energies approaching GUT scale
        E_range = np.logspace(2, 16, 10)  # GeV
        
        # Compute couplings
        alphas = {E: field.compute_couplings(E) for E in E_range}
        
        # Verify convergence at GUT scale
        final_couplings = alphas[E_range[-1]]
        assert all(abs(final_couplings[i] - final_couplings[j]) < 1e-3 
                  for i in range(3) for j in range(i))
        
        # Check running matches beta functions
        for E1, E2 in zip(E_range[:-1], E_range[1:]):
            beta_predicted = field.compute_beta_functions(E1)
            actual_change = {k: (alphas[E2][k] - alphas[E1][k])/np.log(E2/E1)
                           for k in alphas[E1]}
            assert all(abs(beta_predicted[k] - actual_change[k]) < 1e-6 
                      for k in beta_predicted)

    def test_fermion_mass_hierarchy():
        """Test fermion mass hierarchy from appendix_i."""
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

    def test_dark_matter_predictions():
        """Test dark matter predictions from appendix_e."""
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

    def test_galactic_dark_matter():
        """Test galactic dark matter distribution from appendix_e."""
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

    def test_dark_matter_substructure():
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

    def test_field_evolution():
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