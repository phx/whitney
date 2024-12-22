"""Physics constants and variables used in calculations."""

from sympy import Symbol, sqrt, pi

# Fundamental constants
HBAR = 6.582119514e-16  # eV⋅s
C = 2.99792458e8  # m/s
G = 6.67430e-11  # m³/kg⋅s²
M_P = 1.22089e19  # GeV

# Coupling constants
ALPHA_VAL = 1/137.035999139  # Fine structure constant
g1_REF = 0.357  # U(1) coupling at Z mass
g2_REF = 0.652  # SU(2) coupling at Z mass  
g3_REF = 1.221  # SU(3) coupling at Z mass

# Particle masses
Z_MASS = 91.1876  # GeV
M_PLANCK = 1.22089e19  # GeV

# Symbolic variables
X = Symbol('x')  # Position
T = Symbol('t')  # Time
P = Symbol('p')  # Momentum
E = Symbol('E')  # Energy

# Gamma matrices
GAMMA_1 = Symbol('γ₁')
GAMMA_2 = Symbol('γ₂') 
GAMMA_3 = Symbol('γ₃')

# Additional physics variables
g3 = Symbol('g₃')  # Strong coupling
psi_gens = Symbol('ψ_gens')  # Generator wavefunctions
rho = Symbol('ρ')  # Density matrix
g_μν = Symbol('g_μν')  # Metric tensor
R = Symbol('R')  # Ricci scalar
H = Symbol('H')  # Hamiltonian
O = Symbol('O')  # Observable operator
S = Symbol('S')  # Action
Gamma = Symbol('Γ')  # Effective action

# Reference scales
ALPHA_REF = ALPHA_VAL  # Reference coupling

# Add missing complex unit I needed for quantum calculations
# From appendix_j_math_details.tex Eq J.2:
# The complex unit i appears in the quantum evolution:
# ψ(t) = exp(-iHt/ℏ)ψ(0)
I = 1j  # Complex unit for quantum calculations