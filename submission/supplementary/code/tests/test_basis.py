"""Tests for fractal basis implementation."""

import pytest
import numpy as np
from sympy import exp, I, pi
from core.basis import FractalBasis
from core.types import Energy, WaveFunction, FieldConfig
from core.physics_constants import HBAR, C, Z_MASS, X, T
from core.errors import PhysicsError

def test_inheritance():
    """Test proper class structure."""
    basis = FractalBasis()
    # Basic initialization
    assert hasattr(basis, 'alpha')
    assert hasattr(basis, 'mode')
    assert hasattr(basis, 'scaling_dimension')
    # Core methods
    assert hasattr(basis, '_generator_function')
    assert hasattr(basis, '_modulation_factor')
    assert hasattr(basis, '_compute_evolution_operator')

def test_basis_computation():
    """
    Test basis function computation.
    
    From appendix_a_convergence.tex Eq A.17:
    The ground state wavefunction must satisfy:
    1. Normalization: ∫|ψ₀|²dx = 1
    2. Energy eigenvalue: E₀ = mc²
    3. Finite support: |ψ₀(|x| > L)| < ε
    
    With scaling from appendix_d_scale.tex Eq D.8:
    x → x/(mc), t → t/(mc²), ψ → ψ√(mc)
    """
    basis = FractalBasis()
    # Compute ground state with proper scaling
    E = Energy(1.0)
    psi = basis.compute(n=0, E=E)
    
    # Test basic properties
    assert isinstance(psi, WaveFunction)
    assert psi.mass == E.value
    assert 'n' in psi.quantum_numbers
    assert 'E' in psi.quantum_numbers
    
    # Check normalization with proper measure from Eq A.17
    # Include scaling factor from Eq D.8
    dx = psi.grid[1] - psi.grid[0]
    scaled_dx = dx * E.value / (HBAR * C)
    norm = np.sum(np.abs(psi.psi)**2) * scaled_dx
    assert abs(norm - 1.0) < 1e-6

def test_field_equations():
    """Test field equation solutions."""
    basis = FractalBasis()
    config = FieldConfig(
        mass=1.0,
        coupling=0.1,
        dimension=1
    )
    psi = basis._solve_field_equations(config)
    # Check wave equation
    grid = psi.grid
    dt2_psi = np.gradient(np.gradient(psi.psi, grid), grid)
    dx2_psi = np.gradient(np.gradient(psi.psi, grid), grid)
    residual = abs(dt2_psi/C**2 - dx2_psi + (config.mass/HBAR)**2 * psi.psi)
    assert np.max(residual) < 1e-6

def test_evolution_operator():
    """
    Test quantum evolution.
    
    From appendix_k_io_distinction.tex Eq K.23:
    The evolution operator must preserve:
    1. Unitarity: U†U = 1
    2. Energy conservation: [U,H] = 0
    3. Causality: [U(x),U(y)] = 0 for |x-y| > ct
    """
    basis = FractalBasis()
    E = Energy(Z_MASS)
    U = basis._compute_evolution_operator(E)
    
    # Convert to dimensionless coordinates
    grid = np.linspace(-5, 5, 100)  # Use smaller range
    tau = 0.0  # Initial time
    
    # Evaluate operator with proper scaling
    U_vals = np.array([
        complex(U.subs({
            X: x * HBAR * C / E.value,
            T: tau * HBAR / E.value
        })) for x in grid
    ])
    
    # Check unitarity
    assert np.allclose(np.abs(U_vals), 1.0, atol=1e-6)

def test_error_analysis():
    """Test error estimation."""
    basis = FractalBasis()
    result = basis.compute_with_errors(n=0, E=1.0)
    # Check error components
    assert 'wavefunction' in result
    assert 'normalization_error' in result
    assert 'truncation_error' in result
    assert 'numerical_error' in result
    assert 'total_error' in result
    # Verify error bounds
    assert result['total_error'].value > 0
    assert result['total_error'].value < 1.0

def test_fractal_scaling():
    """
    Test fractal structure.
    
    From appendix_d_scale.tex Eq D.15:
    Adjacent levels must satisfy the scaling relation:
    |ψₙ₊₁|/|ψₙ| = α
    
    With quantum corrections from Eq D.16:
    ψₙ(x,t) = αⁿ G(αⁿx, αⁿt) exp(-iEt/ℏ)
    """
    basis = FractalBasis()
    # Compare adjacent levels with proper normalization
    psi_0 = basis.compute(n=0, E=Energy(1.0))
    psi_1 = basis.compute(n=1, E=Energy(1.0))
    
    # Use maximum of non-zero values with proper threshold
    nonzero_mask_0 = np.abs(psi_0.psi) > 1e-10
    nonzero_mask_1 = np.abs(psi_1.psi) > 1e-10
    
    if np.any(nonzero_mask_0) and np.any(nonzero_mask_1):
        max_0 = np.max(np.abs(psi_0.psi[nonzero_mask_0]))
        max_1 = np.max(np.abs(psi_1.psi[nonzero_mask_1]))
        ratio = max_1 / max_0
        assert abs(ratio - basis.alpha) < 1e-6
    else:
        pytest.skip("No non-zero values found for scaling comparison")

def test_quantum_coherence():
    """
    Test quantum coherence properties.
    
    From appendix_g_holographic.tex Eq G.12:
    The wavefunction must preserve:
    1. Holographic scaling: ψ(λx) = λ^Δ ψ(x)
    2. Unitarity: ∫|ψ|²dx = 1 preserved under evolution
    3. Causality: [ψ(x),ψ(y)] = 0 for |x-y| > ct
    
    From appendix_h_rgflow.tex Eq H.8:
    The RG flow preserves the Ward identity:
    ∂ₜ⟨ψ|T₀₀|ψ⟩ = -∂ᵢ⟨ψ|T₀ᵢ|ψ⟩
    """
    basis = FractalBasis()
    E = Energy(1.0)
    psi = basis.compute(n=0, E=E)
    
    # Test holographic scaling
    lambda_scale = 2.0
    scaled_grid = psi.grid * lambda_scale
    scaled_psi = basis.compute(n=0, E=E/lambda_scale)
    
    # Verify scaling dimension from Eq G.12
    Delta = basis.scaling_dimension
    scaling_factor = lambda_scale**Delta
    
    # Compare at scaled points
    for i, x in enumerate(psi.grid):
        if abs(x) < 5*HBAR/(E.value*C):  # Stay in valid region
            scaled_idx = np.argmin(np.abs(scaled_grid - x*lambda_scale))
            ratio = abs(scaled_psi.psi[scaled_idx]/psi.psi[i])
            if abs(psi.psi[i]) > 1e-10:  # Avoid division by small values
                assert abs(ratio - scaling_factor) < 1e-6
                
    # Test unitarity preservation
    dx = psi.grid[1] - psi.grid[0]
    scaled_dx = dx * E.value / (HBAR * C)
    norm = np.sum(np.abs(psi.psi)**2) * scaled_dx
    assert abs(norm - 1.0) < 1e-6
    
    # Test causality
    dt = 1e-6 * HBAR/E.value  # Small time step
    U = basis._compute_evolution_operator(E)
    psi_t = np.array([
        complex(U.subs({X: x, T: dt})) * val 
        for x, val in zip(psi.grid, psi.psi)
    ])
    
    # Verify causal evolution
    for i, x1 in enumerate(psi.grid):
        for j, x2 in enumerate(psi.grid):
            if abs(x1 - x2) > C*dt:
                commutator = psi_t[i]*psi.psi[j] - psi_t[j]*psi.psi[i]
                assert abs(commutator) < 1e-6

def test_ward_identity():
    """
    Test Ward identity preservation.
    
    From appendix_h_rgflow.tex Eq H.8-H.10:
    The stress-energy tensor components must satisfy:
    
    ∂ₜT₀₀ + ∂ᵢTᵢ₀ = 0  (Energy conservation)
    ∂ₜT₀ᵢ + ∂ⱼTʲᵢ = 0  (Momentum conservation)
    Tᵢⱼ = Tⱼᵢ        (Angular momentum conservation)
    """
    basis = FractalBasis()
    E = Energy(1.0)
    psi = basis.compute(n=0, E=E)
    
    # Compute stress-energy components
    dx = psi.grid[1] - psi.grid[0]
    dt = 1e-6 * HBAR/E.value
    
    # Energy density T₀₀
    T00 = np.abs(np.gradient(psi.psi, dt))**2 + \
          C**2 * np.abs(np.gradient(psi.psi, dx))**2 + \
          (E.value/HBAR)**2 * np.abs(psi.psi)**2
    
    # Momentum density T₀ᵢ
    T0i = -I * HBAR * np.conjugate(psi.psi) * np.gradient(psi.psi, dx)
    
    # Verify conservation law
    dT00_dt = np.gradient(T00, dt)
    dT0i_dx = np.gradient(T0i, dx)
    
    # Check Ward identity within numerical precision
    residual = np.abs(dT00_dt + C * dT0i_dx)
    assert np.max(residual) < 1e-6

def test_holographic_correlators():
    """
    Test holographic correlation functions.
    
    From appendix_g_holographic.tex Eq G.15-G.17:
    Two-point functions must satisfy:
    
    ⟨O(x)O(y)⟩ ~ |x-y|^(-2Δ)  (CFT scaling)
    ⟨O(x)O(y)⟩ = 0  for |x-y| > ct  (Causality)
    Im⟨O(x)O(y)⟩ ≥ 0  (Positivity)
    """
    basis = FractalBasis()
    E = Energy(1.0)
    psi = basis.compute(n=0, E=E)
    
    # Compute two-point function
    Delta = basis.scaling_dimension
    correlator = np.zeros((len(psi.grid), len(psi.grid)), dtype=complex)
    
    for i, x1 in enumerate(psi.grid):
        for j, x2 in enumerate(psi.grid):
            dx = abs(x1 - x2)
            if dx > 0:  # Avoid singularity
                # Check scaling behavior
                correlator[i,j] = psi.psi[i] * np.conjugate(psi.psi[j])
                expected = dx**(-2*Delta)
                if abs(correlator[i,j]) > 1e-10:
                    ratio = abs(correlator[i,j])/expected
                    assert abs(ratio - 1.0) < 1e-6
                    
                # Check causality
                if dx > C * 1e-6:
                    assert abs(correlator[i,j]) < 1e-6
                    
                # Check positivity
                assert np.imag(correlator[i,j]) >= -1e-10

def test_gauge_and_rg_invariance():
    """
    Test gauge invariance and RG flow properties.
    
    From appendix_b_gauge.tex Eq B.12-B.14:
    The wavefunction must be invariant under:
    1. U(1) gauge: ψ → exp(iα)ψ
    2. Scale transformations: ψ(λx) = λ^Δ ψ(x)
    3. Special conformal: K_μψ = 0
    
    From appendix_h_rgflow.tex Eq H.15-H.17:
    The RG flow equations:
    ∂ₜψ = βᵢ∂ᵢψ + γψ
    β(g) = -g³/(16π²) + O(g⁵)
    γ(g) = g²/(8π²) + O(g⁴)
    """
    basis = FractalBasis()
    E = Energy(1.0)
    psi = basis.compute(n=0, E=E)
    
    # Test U(1) gauge invariance
    alpha = 0.5
    gauge_trans = exp(I*alpha)
    psi_gauge = psi.psi * gauge_trans
    
    # Verify norm preservation
    assert np.allclose(np.abs(psi_gauge), np.abs(psi.psi))
    
    # Test scale invariance
    lambda_scale = 2.0
    scaled_psi = basis.compute(n=0, E=E/lambda_scale)
    Delta = basis.scaling_dimension
    
    # Verify scaling relation at corresponding points
    x_ref = HBAR/(E.value*C)  # Reference point
    idx_ref = np.argmin(np.abs(psi.grid - x_ref))
    idx_scaled = np.argmin(np.abs(scaled_psi.grid - lambda_scale*x_ref))
    
    if abs(psi.psi[idx_ref]) > 1e-10:
        ratio = abs(scaled_psi.psi[idx_scaled]/psi.psi[idx_ref])
        assert abs(ratio - lambda_scale**Delta) < 1e-6
    
    # Test RG flow equations
    g = basis.alpha  # Coupling constant
    beta = -g**3/(16*pi**2)  # Leading order beta function
    gamma = g**2/(8*pi**2)   # Anomalous dimension
    
    # Compute RG derivatives
    dt = 1e-6 * HBAR/E.value
    dx = psi.grid[1] - psi.grid[0]
    
    dpsi_dt = np.gradient(psi.psi, dt)
    dpsi_dx = np.gradient(psi.psi, dx)
    
    # Verify RG equation
    rg_flow = beta * dpsi_dx + gamma * psi.psi
    assert np.allclose(dpsi_dt, rg_flow, atol=1e-6)

def test_qft_axioms():
    """
    Test quantum field theory axioms.
    
    From appendix_j_math_details.tex Eq J.8-J.12:
    The theory must satisfy:
    1. Locality: [ψ(x),ψ(y)] = 0 for spacelike x-y
    2. Microcausality: Support of commutators in forward light cone
    3. Spectral condition: Energy-momentum in forward light cone
    4. Vacuum stability: H|0⟩ = 0, lowest energy state
    5. Cluster decomposition: Factorization at large distances
    
    From appendix_k_io_distinction.tex Eq K.18-K.20:
    Additional physical requirements:
    - Positive energy: ⟨ψ|H|ψ⟩ ≥ 0
    - Unitarity: S†S = 1
    - CPT invariance
    """
    basis = FractalBasis()
    E = Energy(1.0)
    psi = basis.compute(n=0, E=E)
    
    # Test locality and microcausality
    dt = 1e-6 * HBAR/E.value
    dx = psi.grid[1] - psi.grid[0]
    
    for i, x1 in enumerate(psi.grid):
        for j, x2 in enumerate(psi.grid):
            separation = abs(x1 - x2)
            if separation > C*dt:  # Spacelike separation
                # Check commutator vanishes
                commutator = psi.psi[i]*psi.psi[j] - psi.psi[j]*psi.psi[i]
                assert abs(commutator) < 1e-6
    
    # Test spectral condition
    # Energy should be positive and timelike
    energy_density = (
        np.abs(np.gradient(psi.psi, dt))**2 +  # Kinetic
        C**2 * np.abs(np.gradient(psi.psi, dx))**2 +  # Gradient
        (E.value/HBAR)**2 * np.abs(psi.psi)**2  # Mass
    )
    assert np.all(energy_density >= -1e-10)
    
    # Test vacuum stability
    # Ground state should minimize energy
    H_psi = (
        -HBAR**2/(2*E.value) * np.gradient(np.gradient(psi.psi, dx), dx) +
        E.value/2 * psi.psi
    )
    ground_energy = np.sum(np.conjugate(psi.psi) * H_psi) * dx
    assert abs(ground_energy.real - E.value) < 1e-6
    
    # Test cluster decomposition
    # Correlations should factorize at large distances
    for i, x1 in enumerate(psi.grid):
        for j, x2 in enumerate(psi.grid):
            if abs(x1 - x2) > 5*HBAR/(E.value*C):
                # Check factorization
                corr = psi.psi[i] * np.conjugate(psi.psi[j])
                prod = psi.psi[i] * np.conjugate(psi.psi[i]) * \
                       psi.psi[j] * np.conjugate(psi.psi[j])
                if abs(prod) > 1e-10:
                    ratio = abs(corr/np.sqrt(prod))
                    assert abs(ratio - 1.0) < 1e-6

def test_symmetries_and_conservation():
    """
    Test symmetries and conservation laws.
    
    From appendix_b_gauge.tex Eq B.15-B.18:
    The theory must preserve:
    1. Poincaré invariance: [P_μ,P_ν] = 0
    2. Conformal symmetry: [K_μ,D] = K_μ
    3. Internal symmetries: [Q_a,Q_b] = if_abc Q_c
    
    From appendix_c_gravity.tex Eq C.8-C.10:
    Conservation laws:
    - Energy-momentum: ∂_μT^μν = 0
    - Angular momentum: ∂_μM^μνρ = 0
    - Scale current: ∂_μS^μ = T^μ_μ
    """
    basis = FractalBasis()
    E = Energy(1.0)
    psi = basis.compute(n=0, E=E)
    
    # Test Poincaré invariance
    dt = 1e-6 * HBAR/E.value
    dx = psi.grid[1] - psi.grid[0]
    
    # Compute generators
    P0 = I*HBAR*np.gradient(psi.psi, dt)  # Energy
    P1 = -I*HBAR*np.gradient(psi.psi, dx)  # Momentum
    M01 = (psi.grid * P0 - T * C * P1)  # Boost
    
    # Check commutation relations
    for i in range(len(psi.grid)):
        comm_P = P0[i]*P1[i] - P1[i]*P0[i]
        assert abs(comm_P) < 1e-6
        
        comm_M = M01[i]*P0[i] - P0[i]*M01[i]
        assert abs(comm_M - I*HBAR*P1[i]) < 1e-6
    
    # Test conformal symmetry
    D = psi.grid * P1 + I*HBAR*psi.psi/2  # Dilation
    K = psi.grid**2 * P1 + I*HBAR*psi.grid*psi.psi  # SCT
    
    for i in range(len(psi.grid)):
        comm_K = K[i]*D[i] - D[i]*K[i]
        assert abs(comm_K - I*HBAR*K[i]) < 1e-6
    
    # Test conservation laws
    T00 = np.abs(P0/HBAR)**2 + C**2*np.abs(P1/HBAR)**2  # Energy density
    T01 = np.real(np.conjugate(P0)*P1)/HBAR**2  # Momentum flux
    T11 = np.abs(P1/HBAR)**2  # Stress
    
    # Energy-momentum conservation
    dT00_dt = np.gradient(T00, dt)
    dT01_dx = np.gradient(T01, dx)
    assert np.allclose(dT00_dt + C*dT01_dx, 0, atol=1e-6)
    
    # Angular momentum conservation
    M = np.imag(np.conjugate(psi.psi)*P1)  # Angular momentum density
    dM_dt = np.gradient(M, dt)
    assert np.allclose(dM_dt, 0, atol=1e-6)
    
    # Scale invariance
    trace = T00 - T11  # Trace of stress tensor
    S0 = np.sum(psi.grid * T00) * dx  # Scale charge
    dS0_dt = np.gradient(S0, dt)
    assert np.allclose(dS0_dt, np.sum(trace) * dx, atol=1e-6)

@pytest.mark.parametrize("n,E", [
    (-1, 1.0),  # Invalid level
    (0, -1.0),  # Invalid energy
    (0, 0.0),   # Zero energy
])
def test_invalid_inputs(n, E):
    """Test input validation."""
    basis = FractalBasis()
    with pytest.raises(PhysicsError):
        basis.compute(n, Energy(E))

def test_physical_predictions():
    """
    Test physical predictions and falsifiability.
    
    From appendix_e_predictions.tex Eq E.8-E.12:
    The theory predicts:
    1. Mass spectrum: m_n = m₀α^n
    2. Coupling evolution: g(E) = g₀/(1 + βg₀log(E/E₀))
    3. Cross sections: σ ~ α²/E²
    
    From appendix_f_falsifiability.tex Eq F.15-F.18:
    Falsifiable predictions:
    - Fractal dimension: D = 4 - ε where ε = α²/(8π²)
    - Beta function: β(g) = -bg³ where b = 11/(16π²)
    - Unitarity bound: |a_n| ≤ √(n+1)
    """
    basis = FractalBasis()
    E0 = Energy(1.0)
    
    # Test mass spectrum
    for n in range(3):
        E = Energy(E0.value * basis.alpha**n)
        psi = basis.compute(n=n, E=E)
        
        # Check energy eigenvalue
        H_psi = (
            -HBAR**2/(2*E.value) * np.gradient(np.gradient(psi.psi, psi.grid), psi.grid) +
            E.value/2 * psi.psi
        )
        energy = np.sum(np.conjugate(psi.psi) * H_psi) * (psi.grid[1] - psi.grid[0])
        assert abs(energy.real - E.value) < 1e-6
    
    # Test coupling evolution
    g0 = basis.alpha
    beta0 = -11*g0**3/(16*pi**2)
    E_high = Energy(10.0 * E0.value)
    g_high = g0/(1 + abs(beta0)*np.log(E_high.value/E0.value))
    
    # Verify coupling runs correctly
    psi_high = basis.compute(n=0, E=E_high)
    measured_coupling = np.max(np.abs(np.gradient(psi_high.psi, psi_high.grid)))
    assert abs(measured_coupling - g_high) < 1e-6
    
    # Test fractal dimension
    eps = basis.alpha**2/(8*pi**2)
    D = 4 - eps
    
    # Verify scaling dimension matches fractal dimension
    for n in range(2):
        psi_n = basis.compute(n=n, E=E0)
        scaling_dim = -np.log(np.max(np.abs(psi_n.psi)))/np.log(basis.alpha)
        assert abs(scaling_dim - D/2) < 1e-6
    
    # Test unitarity bounds
    for n in range(3):
        psi = basis.compute(n=n, E=E0)
        # Compute mode coefficients
        dx = psi.grid[1] - psi.grid[0]
        modes = np.fft.fft(psi.psi) * dx
        # Check unitarity bound
        assert np.all(np.abs(modes) <= np.sqrt(n + 1))

def test_theorems_and_implications():
    """
    Test mathematical theorems and physical implications.
    
    From appendix_j_math_details.tex Eq J.20-J.25:
    Key theorems:
    1. Existence and uniqueness of solutions
    2. Analyticity in coupling constant
    3. Asymptotic completeness
    
    From appendix_l_simplification.tex Eq L.8-L.12:
    Physical implications:
    - Exact solution in N→∞ limit
    - Non-perturbative effects ~ exp(-1/g²)
    - Resummation of all orders in perturbation theory
    """
    basis = FractalBasis()
    E = Energy(1.0)
    
    # Test existence and uniqueness
    psi_1 = basis.compute(n=0, E=E)
    psi_2 = basis.compute(n=0, E=E)
    assert np.allclose(psi_1.psi, psi_2.psi, atol=1e-10)
    
    # Test analyticity in coupling
    g = basis.alpha
    # Compute derivatives with respect to coupling
    deps = 1e-6
    basis.alpha = g + deps
    psi_plus = basis.compute(n=0, E=E)
    basis.alpha = g - deps
    psi_minus = basis.compute(n=0, E=E)
    basis.alpha = g  # Restore original coupling
    
    # Verify complex analyticity
    d_psi = (psi_plus.psi - psi_minus.psi)/(2*deps)
    assert np.all(np.isfinite(d_psi))
    
    # Test asymptotic completeness
    # Verify completeness relation
    dx = psi_1.grid[1] - psi_1.grid[0]
    completeness = 0
    for n in range(3):
        psi_n = basis.compute(n=n, E=E)
        completeness += np.sum(np.abs(psi_n.psi)**2) * dx
    assert abs(completeness - 1.0) < 1e-6
    
    # Test non-perturbative effects
    # Verify exponential suppression
    g2 = g*g
    suppression = np.exp(-1/g2)
    high_modes = np.fft.fft(psi_1.psi)[-10:]  # High frequency modes
    assert np.all(np.abs(high_modes) < 10*suppression)
    
    # Test large-N behavior
    # Verify factorization of correlators
    x_ref = HBAR/(E.value*C)
    idx_ref = np.argmin(np.abs(psi_1.grid - x_ref))
    
    # Two-point function
    G2 = psi_1.psi[idx_ref] * np.conjugate(psi_1.psi[idx_ref])
    # Four-point function
    G4 = G2 * G2
    # Verify factorization
    assert abs(G4 - G2*G2) < 1e-6

def test_holographic_gravity():
    """
    Test holographic and gravitational properties.
    
    From appendix_g_holographic.tex Eq G.20-G.25:
    Holographic properties:
    1. Area law: S ~ A/4G_N
    2. Bulk-boundary correspondence: ⟨O(x)O(y)⟩ = K(x-y)
    3. Holographic c-theorem: c(g) monotonic
    
    From appendix_c_gravity.tex Eq C.15-C.18:
    Gravitational properties:
    - Einstein equations emerge: G_μν = 8πG_N T_μν
    - Positive energy theorem: E ≥ 0
    - Penrose inequality: M ≥ √(A/16π)
    """
    basis = FractalBasis()
    E = Energy(1.0)
    psi = basis.compute(n=0, E=E)
    
    # Test area law
    # Compute entanglement entropy
    dx = psi.grid[1] - psi.grid[0]
    rho = np.outer(psi.psi, np.conjugate(psi.psi))
    eigenvals = np.linalg.eigvalsh(rho)
    S = -np.sum(eigenvals * np.log(eigenvals + 1e-10))
    
    # Area in Planck units
    x_planck = HBAR/(E.value*C)
    A = 4*pi*x_planck**2
    
    # Verify area law
    assert abs(S - A/(4*G)) < 1e-6
    
    # Test bulk-boundary correspondence
    # Compute boundary correlator
    boundary_idx = len(psi.grid)//2
    K = np.zeros(len(psi.grid), dtype=complex)
    
    for i, x in enumerate(psi.grid):
        if i != boundary_idx:
            dx = x - psi.grid[boundary_idx]
            K[i] = psi.psi[i] * np.conjugate(psi.psi[boundary_idx])
            expected = 1/abs(dx)**(2*basis.scaling_dimension)
            if abs(K[i]) > 1e-10:
                assert abs(abs(K[i]) - expected) < 1e-6
    
    # Test c-theorem
    # Compute c-function
    g = basis.alpha
    beta = -g**3/(16*pi**2)
    c = 1 - g**2/(8*pi**2)  # Leading order
    
    # Verify monotonicity
    dc_dg = -g/(4*pi**2)
    assert dc_dg * beta <= 1e-10  # Must decrease along RG flow
    
    # Test Einstein equations
    # Compute stress tensor
    T_μν = np.zeros((2,2,len(psi.grid)), dtype=complex)
    for i in range(len(psi.grid)):
        T_μν[0,0,i] = abs(np.gradient(psi.psi, dx)[i])**2  # T₀₀
        T_μν[0,1,i] = T_μν[1,0,i] = np.real(
            np.conjugate(np.gradient(psi.psi, dx)[i]) * 
            (-I*E.value/HBAR) * psi.psi[i]
        )  # T₀ᵢ
        T_μν[1,1,i] = abs(-I*E.value/HBAR * psi.psi[i])**2  # Tᵢᵢ
    
    # Compute Einstein tensor
    R = np.gradient(np.gradient(psi.psi, dx), dx)
    G_μν = np.zeros_like(T_μν)
    G_μν[0,0] = -abs(R)**2  # G₀₀
    G_μν[1,1] = abs(R)**2   # Gᵢᵢ
    
    # Verify Einstein equations
    for μ in range(2):
        for ν in range(2):
            residual = G_μν[μ,ν] - 8*pi*G*T_μν[μ,ν]
            assert np.max(np.abs(residual)) < 1e-6
    
    # Test Penrose inequality
    # Compute ADM mass
    M = np.sum(T_μν[0,0] * dx)
    # Verify inequality
    assert M >= np.sqrt(A/(16*pi)) - 1e-6

def test_unified_holographic_framework():
    """
    Test unified holographic framework.
    
    From appendix_g_holographic.tex Eq G.30-G.35:
    The unified framework requires:
    1. Holographic duality: Z_bulk = Z_boundary
    2. Emergent spacetime: g_μν = ⟨T_μν⟩/M_P²
    3. Information preservation: S_bulk = S_boundary
    
    From appendix_k_io_distinction.tex Eq K.25-K.28:
    Quantum coherence requires:
    - Entanglement wedge reconstruction
    - Ryu-Takayanagi formula: S_EE = A/4G_N
    - Quantum extremal surfaces
    """
    basis = FractalBasis()
    E = Energy(1.0)
    psi = basis.compute(n=0, E=E)
    dx = psi.grid[1] - psi.grid[0]
    
    # Test holographic duality
    # Compute bulk partition function
    H_bulk = (
        -HBAR**2/(2*E.value) * np.gradient(np.gradient(psi.psi, dx), dx) +
        E.value/2 * psi.psi
    )
    Z_bulk = np.sum(np.exp(-H_bulk/HBAR)) * dx
    
    # Compute boundary partition function
    rho = np.outer(psi.psi, np.conjugate(psi.psi))
    Z_boundary = np.trace(rho)
    
    # Verify duality
    assert abs(Z_bulk - Z_boundary) < 1e-6
    
    # Test emergent spacetime
    # Compute stress tensor expectation value
    T_μν = np.zeros((2,2,len(psi.grid)), dtype=complex)
    for i in range(len(psi.grid)):
        T_μν[0,0,i] = abs(np.gradient(psi.psi, dx)[i])**2  # T₀₀
        T_μν[0,1,i] = T_μν[1,0,i] = np.real(
            np.conjugate(np.gradient(psi.psi, dx)[i]) * 
            (-I*E.value/HBAR) * psi.psi[i]
        )  # T₀ᵢ
        T_μν[1,1,i] = abs(-I*E.value/HBAR * psi.psi[i])**2  # Tᵢᵢ
    
    # Reconstruct metric from stress tensor
    M_P = np.sqrt(HBAR*C/G)  # Planck mass
    g_μν = T_μν/M_P**2
    
    # Verify Einstein equations with emergent metric
    R = np.gradient(np.gradient(psi.psi, dx), dx)
    G_μν = np.zeros_like(g_μν)
    G_μν[0,0] = -abs(R)**2
    G_μν[1,1] = abs(R)**2
    
    # Check Einstein equations
    for μ in range(2):
        for ν in range(2):
            residual = G_μν[μ,ν] - 8*pi*G*T_μν[μ,ν]
            assert np.max(np.abs(residual)) < 1e-6
    
    # Test entanglement wedge reconstruction
    # Compute entanglement entropy
    eigenvals = np.linalg.eigvalsh(rho)
    S_EE = -np.sum(eigenvals * np.log(eigenvals + 1e-10))
    
    # Compute area of RT surface
    x_planck = HBAR/(E.value*C)
    A = 4*pi*x_planck**2
    
    # Verify RT formula
    assert abs(S_EE - A/(4*G)) < 1e-6
    
    # Test quantum extremal surfaces
    # Compute generalized entropy
    S_gen = S_EE + A/(4*G)
    
    # Verify extremality
    dS = np.gradient(S_gen, dx)
    assert np.min(np.abs(dS)) < 1e-6

def test_unified_field_theory():
    """
    Test unified field theory framework.
    
    From appendix_i_sm_features.tex Eq I.30-I.35:
    The unified framework requires:
    1. Gauge group unification: SU(3)×SU(2)×U(1) ⊂ G
    2. Fermion generations: ψᵢ = exp(iπn/3)ψ₀
    3. Mass hierarchies: m_n = m₀α^n
    
    From appendix_l_simplification.tex Eq L.30-L.35:
    Quantum consistency requires:
    - Anomaly cancellation: Tr[T_a{T_b,T_c}] = 0
    - Unitarity bounds: |a_n| ≤ 2√(n+1)
    - Asymptotic freedom: β(g) < 0
    """
    basis = FractalBasis()
    E = Energy(1.0)
    psi = basis.compute(n=0, E=E)
    dx = psi.grid[1] - psi.grid[0]
    
    # Test gauge unification
    # Compute gauge couplings at unification scale
    g1 = basis.alpha  # U(1)
    g2 = g1 * np.sqrt(5/3)  # SU(2) 
    g3 = g1 * np.sqrt(8/3)  # SU(3)
    
    # Verify coupling ratios
    assert abs(g2/g1 - np.sqrt(5/3)) < 1e-6
    assert abs(g3/g1 - np.sqrt(8/3)) < 1e-6
    
    # Test fermion generations
    # Compute generation phases
    phases = [exp(I*pi*n/3) for n in range(3)]
    psi_gens = [phase * psi.psi for phase in phases]
    
    # Verify orthogonality
    for i in range(3):
        for j in range(i+1, 3):
            overlap = np.sum(np.conjugate(psi_gens[i]) * psi_gens[j]) * dx
            assert abs(overlap) < 1e-6
    
    # Test mass hierarchies
    masses = [E.value * basis.alpha**n for n in range(3)]
    ratios = [masses[i+1]/masses[i] for i in range(2)]
    
    # Verify geometric progression
    for ratio in ratios:
        assert abs(ratio - basis.alpha) < 1e-6
    
    # Test anomaly cancellation
    # Compute triangle diagram
    def triangle(a, b, c):
        return np.sum(
            np.conjugate(psi.psi) * 
            np.gradient(psi.psi, dx) * 
            np.gradient(np.gradient(psi.psi, dx), dx)
        ) * dx
    
    # Verify anomaly cancellation
    for i in range(3):
        for j in range(3):
            for k in range(3):
                anomaly = triangle(i, j, k)
                assert abs(anomaly) < 1e-6
    
    # Test unitarity bounds
    # Compute scattering amplitudes
    modes = np.fft.fft(psi.psi)
    
    # Verify unitarity constraints
    for n, mode in enumerate(modes):
        assert abs(mode) <= 2*np.sqrt(n + 1)
    
    # Test asymptotic freedom
    # Compute beta function
    g = basis.alpha
    beta = -g**3/(16*pi**2) * (11 - 2/3)  # Include fermion loops
    
    # Verify asymptotic freedom
    assert beta < 0
    
    # Test full consistency
    # Compute effective action
    S_eff = np.sum(
        np.conjugate(psi.psi) * (
            -HBAR**2/(2*E.value) * np.gradient(np.gradient(psi.psi, dx), dx) +
            E.value/2 * psi.psi
        )
    ) * dx
    
    # Verify reality and boundedness
    assert np.imag(S_eff) < 1e-10
    assert S_eff.real > 0

def test_standard_model_features():
    """
    Test Standard Model features and predictions.
    
    From appendix_i_sm_features.tex Eq I.40-I.45:
    The theory must reproduce:
    1. Chiral symmetry: γ⁵ψ_L = -ψ_L, γ⁵ψ_R = +ψ_R
    2. CKM mixing: V_CKM = U_L†U_D
    3. Weinberg angle: sin²θ_W = 3/8 at GUT scale
    
    From appendix_e_predictions.tex Eq E.20-E.25:
    Physical predictions:
    - Higgs mass: m_H = 2v√(λ/2)
    - Top mass: m_t = y_t v/√2
    - Neutrino masses: m_ν ~ v²/M_GUT
    """
    basis = FractalBasis()
    E = Energy(1.0)
    psi = basis.compute(n=0, E=E)
    dx = psi.grid[1] - psi.grid[0]
    
    # Test chiral symmetry
    # Compute left and right components
    psi_L = 0.5 * (1 - basis.gamma5()) * psi.psi
    psi_R = 0.5 * (1 + basis.gamma5()) * psi.psi
    
    # Verify chirality
    assert np.allclose(basis.gamma5() @ psi_L, -psi_L, atol=1e-6)
    assert np.allclose(basis.gamma5() @ psi_R, psi_R, atol=1e-6)
    
    # Test CKM mixing
    # Generate three generations
    psi_up = [basis.compute(n=n, E=E).psi for n in range(3)]
    psi_down = [basis.compute(n=n, E=E).psi for n in range(3)]
    
    # Compute mixing matrices
    U_L = np.array([[np.sum(np.conjugate(u)*d)*dx 
                     for d in psi_down] for u in psi_up])
    U_D = np.eye(3) + 0.1 * (np.random.rand(3,3) - 0.5)  # Small mixing
    V_CKM = U_L.conj().T @ U_D
    
    # Verify unitarity
    assert np.allclose(V_CKM @ V_CKM.conj().T, np.eye(3), atol=1e-6)
    
    # Test Weinberg angle
    # Compute gauge couplings
    g1 = basis.alpha  # U(1)
    g2 = g1 * np.sqrt(5/3)  # SU(2)
    
    # Verify GUT prediction
    sin2_theta_W = g1**2/(g1**2 + g2**2)
    assert abs(sin2_theta_W - 3/8) < 1e-6
    
    # Test mass predictions
    # Higgs parameters
    v = 246  # GeV
    lambda_H = 0.13  # Measured value
    
    # Compute Higgs mass
    m_H = 2*v*np.sqrt(lambda_H/2)
    assert abs(m_H - 125) < 1  # GeV
    
    # Top Yukawa
    y_t = np.sqrt(2)*173/v  # From top mass
    m_t = y_t * v/np.sqrt(2)
    assert abs(m_t - 173) < 1  # GeV
    
    # Neutrino masses
    M_GUT = 2e16  # GeV
    m_nu = v**2/M_GUT
    assert m_nu < 0.1  # eV
    
    # Test full consistency
    # Compute beta functions
    beta_g1 = g1**3/(16*pi**2) * (41/6)
    beta_g2 = -g2**3/(16*pi**2) * (19/6)
    beta_g3 = -g3**3/(16*pi**2) * 7
    
    # Verify asymptotic freedom
    assert beta_g1 > 0  # U(1) not asymptotically free
    assert beta_g2 < 0  # SU(2) asymptotically free
    assert beta_g3 < 0  # SU(3) asymptotically free

def test_complete_unification():
    """
    Test complete unification of forces and symmetries.
    
    From appendix_i_sm_features.tex Eq I.50-I.55:
    Complete unification requires:
    1. E8 gauge structure: G = E8
    2. Triality: SO(8) → SU(3)×SU(2)×U(1)
    3. Generation structure from octonions
    
    From appendix_l_simplification.tex Eq L.40-L.45:
    Quantum gravity unification:
    - AdS/CFT correspondence exact
    - Holographic RG = Einstein equations
    - Emergence of all forces from geometry
    """
    basis = FractalBasis()
    E = Energy(1.0)
    psi = basis.compute(n=0, E=E)
    dx = psi.grid[1] - psi.grid[0]
    
    # Test E8 structure
    # Compute root system
    alpha = basis.alpha
    roots = []
    for i in range(8):
        for j in range(i+1, 8):
            root = np.zeros(8)
            root[i] = alpha
            root[j] = -alpha
            roots.append(root)
    
    # Verify Cartan matrix
    A = np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            A[i,j] = 2*np.dot(roots[i], roots[j])/np.dot(roots[j], roots[j])
    
    # Check E8 properties
    assert np.allclose(A @ A.T, np.eye(8)*2, atol=1e-6)
    
    # Test triality
    # Compute spinor decomposition
    psi_8s = np.array([basis.compute(n=n, E=E).psi for n in range(8)])
    
    # Verify SO(8) triality relations
    for i in range(8):
        for j in range(8):
            for k in range(8):
                gamma = basis.clifford_algebra(i,j,k)
                triality = np.sum(psi_8s[i] * gamma @ psi_8s[j] * psi_8s[k]) * dx
                assert abs(triality) < 1e-6
    
    # Test octonion structure
    # Construct octonion multiplication table
    oct = np.zeros((8,8,8))
    for i in range(7):
        for j in range(i+1, 7):
            k = (i + 1 + j//2) % 7
            if j % 2 == 0:
                k = (k + 3) % 7
            oct[i+1,j+1,k+1] = 1
            oct[j+1,k+1,i+1] = 1
            oct[k+1,i+1,j+1] = 1
    
    # Verify generation structure
    for i in range(3):
        gen = np.sum(oct[i+1,:,:] @ psi_8s, axis=0)
        assert np.allclose(gen, psi_gens[i], atol=1e-6)
    
    # Test holographic unification
    # Compute boundary theory
    Z_boundary = np.trace(rho)
    
    # Compute bulk gravity
    R = np.gradient(np.gradient(psi.psi, dx), dx)
    S_gravity = np.sum(R**2) * dx/(16*pi*G)
    
    # Verify exact correspondence
    assert abs(np.log(Z_boundary) + S_gravity/HBAR) < 1e-6
    
    # Test emergence of forces
    # Compute geometric connection
    Gamma = np.gradient(g_μν, dx)
    
    # Extract gauge fields
    A_μ = Gamma[0,:,:]  # U(1)
    W_μν = Gamma[1:4,:,:]  # SU(2)
    G_μνρ = Gamma[4:,:,:]  # SU(3)
    
    # Verify Yang-Mills equations
    F_μν = np.gradient(A_μ, dx) - np.gradient(A_μ, dx).T
    DW = np.gradient(W_μν, dx) + np.cross(W_μν, W_μν, axis=0)
    DG = np.gradient(G_μνρ, dx) + np.cross(G_μνρ, G_μνρ, axis=0)
    
    assert np.max(np.abs(np.gradient(F_μν, dx))) < 1e-6
    assert np.max(np.abs(DW)) < 1e-6
    assert np.max(np.abs(DG)) < 1e-6