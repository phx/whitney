"""Physical constants and common definitions."""

from sympy import Symbol

# Symbolic variables
X = Symbol('x')  # Spatial coordinate
E = Symbol('E')  # Energy scale

# Physical constants
Z_MASS = 91.2  # Z boson mass in GeV
ALPHA_VAL = 0.1  # Default scaling parameter
ALPHA_REF = 1/137.036  # Fine structure constant

# Coupling constants at Z mass
g1_REF = 0.357  # U(1) coupling
g2_REF = 0.652  # SU(2) coupling  
g3_REF = 1.221  # SU(3) coupling

# Anomalous dimensions
GAMMA_1 = 0.1  # U(1) anomalous dimension
GAMMA_2 = -0.2  # SU(2) anomalous dimension
GAMMA_3 = -0.3  # SU(3) anomalous dimension 