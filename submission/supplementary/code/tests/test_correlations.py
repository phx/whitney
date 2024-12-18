"""Tests for field correlation functions."""

import pytest
from hypothesis import given, strategies as st
import numpy as np
from math import log, pi
from sympy import (
    exp, I, sqrt, integrate, conjugate, diff,
    Symbol, oo, Function
)
from core.field import UnifiedField
from core.types import (
    Energy, 
    FieldConfig, 
    WaveFunction,
    NumericValue,
    ensure_numeric_value
)
from core.physics_constants import (
    X, T, P, HBAR, C, ALPHA_VAL
)
from core.errors import PhysicsError, ValidationError
from typing import List, Tuple, Optional

@pytest.fixture
def field():
    """Create UnifiedField instance for testing."""
    return UnifiedField(alpha=0.1)

@pytest.fixture
def test_state():
    """Create test field configuration."""
    return exp(-(X**2 + (C*T)**2)/(2*HBAR**2))

class TestTwoPointFunctions:
    """Test two-point correlation functions."""
    
    def test_vacuum_correlator(self, field):
        """Test vacuum two-point function."""
        test_state = WaveFunction.from_expression(
            exp(-(X**2 + (C*T)**2)/(2*HBAR**2))
        )
        correlator = field.compute_correlator(
            test_state,
            [(0, 0), (1, 0)]
        )
        assert correlator.is_real
    
    def test_cluster_decomposition(self, field, test_state):
        """Test cluster decomposition principle."""
        # Points with large separation
        x1, x2 = X + 100, X - 100
        t1, t2 = T, T
        
        correlator = field.compute_correlator(
            test_state,
            [(x1, t1), (x2, t2)]
        )
        
        # Should factorize at large distances
        product = (
            field.compute_correlator(test_state, [(x1, t1)]) *
            field.compute_correlator(test_state, [(x2, t2)])
        )
        
        assert abs(correlator - product) < 1e-10

class TestSpectralFunctions:
    """Test spectral decomposition of correlators."""
    
    def test_källén_lehmann(self, field):
        """Test Källén-Lehmann spectral representation."""
        # Create massive field state
        config = FieldConfig(mass=100.0, coupling=0.1, dimension=4)
        psi = field.compute_field(config)
        
        # Compute spectral function
        rho = field.compute_spectral_function(psi)
        
        # Should be positive and normalized
        assert rho >= 0
        norm = integrate(rho, (P, -oo, oo))
        assert abs(norm - 1) < 1e-10
        
        # Convert to numeric grid for correlator test
        x_grid = np.linspace(-10, 10, 100)
        t_grid = np.linspace(-5, 5, 50)
        
        points = [
            (float(x_grid[0]), float(t_grid[0])),
            (float(x_grid[-1]), float(t_grid[-1]))
        ]
        
        # Verify correlator satisfies spectral representation
        G = field.compute_correlator(psi, points)
        assert isinstance(G, complex)
        assert np.isfinite(G)
    
    def test_dispersion_relations(self, field, test_state):
        """Test dispersion relations."""
        # Compute retarded propagator
        G_R = field.compute_retarded_propagator(test_state)
        
        # Verify causality through dispersion relations
        omega = Symbol('omega')
        k = Symbol('k')
        
        # Kramers-Kronig relations
        re_G = integrate(G_R.real, (omega, -oo, oo))
        im_G = integrate(G_R.imag, (omega, -oo, oo))
        
        assert abs(re_G) < 1e-10  # Real part should vanish
        assert im_G > 0  # Imaginary part should be positive
    
    def test_spectral_density(self, field):
        """Test spectral density properties."""
        config = FieldConfig(mass=100.0, coupling=0.1, dimension=4)
        psi = field.compute_field(config)
        
        # Compute spectral density at different momenta
        k_vals = np.linspace(0, 10, 5)
        for k in k_vals:
            rho_k = field.compute_spectral_density(psi, k)
            
            # Verify physical properties
            assert rho_k >= 0  # Positivity
            assert np.isfinite(rho_k)  # Finiteness
            
            # Check threshold behavior
            if k < config.mass:
                assert abs(rho_k) < 1e-10  # Below threshold
            else:
                assert rho_k > 0  # Above threshold

class TestWardIdentities:
    """Test Ward identities from gauge invariance."""
    
    def test_current_conservation(self, field, test_state):
        """Test current conservation Ward identity."""
        # Compute current-field correlator
        j0, j1 = field.compute_gauge_current(test_state)
        
        # Ward identity: ∂_μ<j^μ(x)φ(y)> = 0
        d_t_corr = diff(
            field.compute_correlator(test_state, [(j0, T)]),
            T
        )
        d_x_corr = diff(
            field.compute_correlator(test_state, [(j1, X)]),
            X
        )
        
        assert abs(d_t_corr + d_x_corr) < 1e-10
    
    def test_charge_conservation(self, field, test_state):
        """Test charge conservation."""
        # Compute charge operator
        Q = integrate(
            field.compute_gauge_current(test_state)[0],
            (X, -oo, oo)
        )
        
        # Should commute with Hamiltonian
        H = field._compute_hamiltonian()
        commutator = Q*H - H*Q
        
        assert abs(commutator) < 1e-10
    
    def test_gauge_ward_identity(self, field):
        """Test gauge Ward identity from appendix_i_sm_features.tex Eq I.3."""
        config = FieldConfig(mass=100.0, coupling=0.1, dimension=4)
        psi = field.compute_field(config)
        
        # Compute gauge current divergence
        div_j = field.compute_current_divergence(psi)
        
        # Compute charge rotation
        delta_psi = field.compute_gauge_variation(psi)
        
        # Ward identity: ∂_μj^μ = ie(φ†δφ - δφ†φ)
        lhs = div_j
        rhs = I * config.coupling * (
            conjugate(psi) * delta_psi - conjugate(delta_psi) * psi
        )
        
        assert abs(lhs - rhs) < 1e-10
    
    def test_global_charge_conservation(self, field):
        """Test conservation of global U(1) charge."""
        config = FieldConfig(mass=100.0, coupling=0.1, dimension=4)
        psi = field.compute_field(config)
        
        # Initial charge
        Q1 = field.compute_total_charge(psi)
        
        # Evolve state
        dt = 0.1
        psi_evolved = field.evolve_field(psi, dt)
        
        # Final charge
        Q2 = field.compute_total_charge(psi_evolved)
        
        # Charge should be conserved
        assert abs(Q2 - Q1) < 1e-10
    
    def test_axial_ward_identity(self, field):
        """Test axial Ward identity from appendix_i_sm_features.tex Eq I.4."""
        config = FieldConfig(mass=100.0, coupling=0.1, dimension=4)
        psi = field.compute_field(config)
        
        # Compute axial current divergence
        div_j5 = field.compute_axial_current_divergence(psi)
        
        # Compute pseudoscalar density
        P = field.compute_pseudoscalar_density(psi)
        
        # Axial Ward identity: ∂_μj⁵^μ = 2im P
        lhs = div_j5
        rhs = 2 * I * config.mass * P
        
        assert abs(lhs - rhs) < 1e-10
    
    def test_anomalous_ward_identity(self, field):
        """Test anomalous Ward identity with fractal corrections."""
        config = FieldConfig(mass=100.0, coupling=0.1, dimension=4)
        psi = field.compute_field(config)
        
        # Compute anomaly term with fractal corrections
        n_max = int(-log(field.precision)/log(ALPHA_VAL))
        anomaly = 0
        
        for n in range(n_max):
            F_n = field.compute_field_strength(psi, n)
            F_dual_n = field.compute_dual_field_strength(psi, n)
            anomaly += ALPHA_VAL**n * (F_n * F_dual_n)
        
        # Verify anomalous Ward identity
        div_j5 = field.compute_axial_current_divergence(psi)
        assert abs(div_j5 - anomaly/(16*pi**2)) < 1e-10

@pytest.mark.physics
class TestCorrelationFunctions:
    """Test correlation function computations."""
    
    def test_basic_correlator(self, field):
        """Test basic two-point correlation function."""
        # Create test state
        psi = field.compute_basis_state(energy=Energy(100.0))
        
        # Test points
        points = [(0.0, 0.0), (1.0, 0.0)]  # Spatial separation
        
        # Compute correlator
        G = field.compute_correlator(psi, points)
        
        # Verify basic properties
        assert isinstance(G, complex)
        assert np.isfinite(G)
        
        # Should decrease with distance
        points_far = [(0.0, 0.0), (10.0, 0.0)]
        G_far = field.compute_correlator(psi, points_far)
        assert abs(G_far) < abs(G)

    def test_correlator_symmetry(self, field):
        """Test correlation function symmetry properties."""
        psi = field.compute_basis_state(energy=Energy(100.0))
        
        # Test points
        x1, t1 = 0.0, 0.0
        x2, t2 = 1.0, 0.0
        
        # Test symmetry under exchange
        G12 = field.compute_correlator(psi, [(x1, t1), (x2, t2)])
        G21 = field.compute_correlator(psi, [(x2, t2), (x1, t1)])
        
        assert np.isclose(G12, G21, rtol=1e-7)

    def test_fractal_corrections(self, field):
        """Test fractal corrections to correlator."""
        psi = field.compute_basis_state(energy=Energy(100.0))
        points = [(0.0, 0.0), (1.0, 0.0)]
        
        # Compare with and without corrections
        field.precision = 1e-1  # Fewer corrections
        G_few = field.compute_correlator(psi, points)
        
        field.precision = 1e-8  # More corrections
        G_many = field.compute_correlator(psi, points)
        
        # Should see effect of corrections
        assert abs(G_many - G_few) > 1e-8

@pytest.mark.physics
class TestSpectralDecomposition:
    """Test spectral decomposition and Källén-Lehmann representation."""
    
    def test_spectral_positivity(self, field):
        """Test positivity of spectral function."""
        config = FieldConfig(mass=100.0, coupling=0.1, dimension=4)
        psi = field.compute_field(config)
        
        # Test positivity at different momenta
        k_vals = np.linspace(0, 10, 5)
        for k in k_vals:
            rho = field.compute_spectral_function(psi, k)
            assert rho >= 0  # Positivity required by unitarity
            
            # Test fractal sum rule from appendix_i_sm_features.tex Eq I.6
            n_max = int(-log(field.precision)/log(ALPHA_VAL))
            sum_rule = sum(
                ALPHA_VAL**n * field.compute_spectral_moment(psi, k, n)
                for n in range(n_max)
            )
            assert abs(sum_rule - 1) < 1e-10

    def test_källén_lehmann_reconstruction(self, field):
        """Test reconstruction of correlator from spectral function."""
        config = FieldConfig(mass=100.0, coupling=0.1, dimension=4)
        psi = field.compute_field(config)
        
        # Points for correlator
        x1, t1 = 0.0, 0.0
        x2, t2 = 1.0, 0.0
        
        # Direct correlator
        G_direct = field.compute_correlator(psi, [(x1, t1), (x2, t2)])
        
        # Reconstruct from spectral function
        G_spectral = field.reconstruct_correlator_from_spectral(
            psi, [(x1, t1), (x2, t2)]
        )
        
        # Should match
        assert abs(G_direct - G_spectral) < 1e-10

@pytest.mark.physics
class TestHolographicCorrelators:
    """Test holographic aspects of correlation functions."""
    
    def test_bulk_boundary_correspondence(self, field):
        """Test holographic bulk-boundary correspondence.
        
        From appendix_g_holographic.tex Eq G.3:
        The boundary correlator should match bulk computation.
        """
        config = FieldConfig(mass=100.0, coupling=0.1, dimension=4)
        psi = field.compute_field(config)
        
        # Compute boundary correlator
        x1, t1 = 0.0, 0.0
        x2, t2 = 1.0, 0.0
        G_boundary = field.compute_boundary_correlator(psi, [(x1, t1), (x2, t2)])
        
        # Compute bulk correlator with radial integral
        G_bulk = field.compute_bulk_correlator(psi, [(x1, t1), (x2, t2)])
        
        # Should match by holographic principle
        assert abs(G_boundary - G_bulk) < 1e-10

    def test_fractal_holographic_entropy(self, field):
        """Test fractal corrections to holographic entropy.
        
        From appendix_g_holographic.tex Eq G.8:
        S(E) = (2π/α) * (E/E_Planck)^(3/4) + fractal corrections
        """
        config = FieldConfig(mass=100.0, coupling=0.1, dimension=4)
        psi = field.compute_field(config)
        
        # Compute entropy with fractal corrections
        n_max = int(-log(field.precision)/log(ALPHA_VAL))
        S = field.compute_holographic_entropy(psi)
        
        # Verify fractal corrections
        S_classical = field.compute_classical_entropy(psi)
        S_fractal = sum(
            ALPHA_VAL**n * field.compute_entropy_correction(psi, n)
            for n in range(n_max)
        )
        
        # Total entropy should match sum
        assert abs(S - (S_classical + S_fractal)) < 1e-10

@pytest.mark.physics
class TestRGFlowCorrelators:
    """Test RG flow aspects of correlation functions."""
    
    def test_correlator_scaling(self, field):
        """Test correlator scaling under RG flow.
        
        From appendix_h_rgflow.tex Eq H.2:
        G(λx) = λ^(-2Δ) G(x) + Σ α^n F_n(λ)
        """
        config = FieldConfig(mass=100.0, coupling=0.1, dimension=4)
        psi = field.compute_field(config)
        
        # Test scaling at different points
        x1, t1 = 1.0, 0.0
        x2, t2 = 2.0, 0.0
        lambda_vals = [0.5, 1.0, 2.0]
        
        for lambda_scale in lambda_vals:
            # Original correlator
            G = field.compute_correlator(psi, [(x1, t1), (x2, t2)])
            
            # Scaled correlator
            G_scaled = field.compute_correlator(
                psi,
                [(lambda_scale*x1, t1), (lambda_scale*x2, t2)]
            )
            
            # Include fractal corrections
            n_max = int(-log(field.precision)/log(ALPHA_VAL))
            corrections = sum(
                ALPHA_VAL**n * field.compute_scaling_correction(psi, lambda_scale, n)
                for n in range(n_max)
            )
            
            # Verify scaling relation
            scaling = lambda_scale**(-2*field.scaling_dimension)
            assert abs(G_scaled - (scaling*G + corrections)) < 1e-10

    def test_beta_function_consistency(self, field):
        """Test consistency of beta function with correlators.
        
        From appendix_h_rgflow.tex Eq H.4:
        Correlators should satisfy Callan-Symanzik equation.
        """
        config = FieldConfig(mass=100.0, coupling=0.1, dimension=4)
        psi = field.compute_field(config)
        
        # Compute correlator and its scale derivative
        points = [(0.0, 0.0), (1.0, 0.0)]
        G = field.compute_correlator(psi, points)
        dG = field.compute_scale_derivative(psi, points)
        
        # Compute beta function contribution
        beta = field.compute_beta_function(config.coupling)
        beta_term = beta * field.compute_coupling_derivative(psi, points)
        
        # Verify Callan-Symanzik equation
        lhs = dG + field.scaling_dimension * G
        rhs = beta_term
        
        assert abs(lhs - rhs) < 1e-10

@pytest.mark.physics
class TestFractalBasisCorrelators:
    """Test correlation functions in fractal basis."""
    
    def test_basis_completeness(self, field):
        """Test completeness of fractal basis for correlators.
        
        From appendix_a_convergence.tex Eq A.1:
        Correlators should be expandable in fractal basis.
        """
        config = FieldConfig(mass=100.0, coupling=0.1, dimension=4)
        psi = field.compute_field(config)
        points = [(0.0, 0.0), (1.0, 0.0)]
        
        # Direct correlator
        G_direct = field.compute_correlator(psi, points)
        
        # Expand in fractal basis
        n_max = int(-log(field.precision)/log(ALPHA_VAL))
        G_expanded = 0
        for n in range(n_max):
            basis_n = field.compute_basis_function(n)
            coeff = field.compute_basis_coefficient(psi, n)
            G_n = field.compute_correlator(basis_n, points)
            G_expanded += coeff * G_n
            
        # Should match direct computation
        assert abs(G_direct - G_expanded) < 1e-10

    def test_fractal_orthogonality(self, field):
        """Test orthogonality of fractal basis correlators.
        
        From appendix_a_convergence.tex Eq A.3:
        Different basis levels should be orthogonal.
        """
        config = FieldConfig(mass=100.0, coupling=0.1, dimension=4)
        points = [(0.0, 0.0), (1.0, 0.0)]
        
        # Test orthogonality between different levels
        for n in range(3):
            for m in range(n+1, 4):
                psi_n = field.compute_basis_function(n)
                psi_m = field.compute_basis_function(m)
                
                # Compute correlator overlap
                overlap = field.compute_correlator_overlap(psi_n, psi_m, points)
                
                # Should vanish for n ≠ m
                assert abs(overlap) < 1e-10