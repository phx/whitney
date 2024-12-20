"""Physical constants and fundamental quantities."""

from sympy import Symbol, I, pi

# Fundamental constants
HBAR = 1.0545718e-34  # Reduced Planck constant (J⋅s)
C = 2.99792458e8      # Speed of light (m/s)
G = 6.67430e-11       # Gravitational constant (m³/kg⁻¹/s⁻²)
M_PLANCK = 1.22089e19    # Planck mass (GeV)

# Energy scales
Z_MASS = 91.1876      # Z boson mass (GeV)
GUT_SCALE = 2.1e16    # Grand unification scale (GeV)

# Coordinates and fields
X = Symbol('x')       # Spatial coordinate
T = Symbol('t')       # Time coordinate
P = Symbol('p')       # Momentum coordinate
E = Symbol('E')       # Energy coordinate/field

# Reference values
ALPHA_VAL = 1/137    # Fine structure constant

# Coupling constants
g1_REF = 0.357  # U(1) coupling
g2_REF = 0.652  # SU(2) coupling
g3_REF = 1.221  # SU(3) coupling

# Other constants
ALPHA_REF = 1/128.9  # Reference alpha value
GAMMA_1 = 0.01       # First gamma factor
GAMMA_2 = 0.02       # Second gamma factor
GAMMA_3 = 0.03       # Third gamma factor