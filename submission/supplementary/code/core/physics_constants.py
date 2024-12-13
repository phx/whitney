"""Physical constants and common definitions."""

from sympy import (
    Symbol, exp, integrate, conjugate, sqrt,
    oo, I, pi, Matrix
)

# Coordinate system
# Space-time coordinates
X = Symbol('x')  # Spatial coordinate
T = Symbol('t')  # Time coordinate
P = Symbol('p')  # Momentum coordinate

# Field theory parameters
ALPHA_VAL = 0.1  # Default scaling parameter
ALPHA_REF = 1/137.036  # Fine structure constant

# Gauge coupling constants at Z mass
g1_REF = 0.357  # U(1) coupling
g2_REF = 0.652  # SU(2) coupling  
g3_REF = 1.221  # SU(3) coupling

# Particle masses (GeV)
Z_MASS = 91.2  # Z boson mass
H_MASS = 125.1  # Higgs mass
W_MASS = 80.4  # W boson mass

# Anomalous dimensions
GAMMA_1 = 0.1  # U(1) anomalous dimension
GAMMA_2 = -0.2  # SU(2) anomalous dimension
GAMMA_3 = -0.3  # SU(3) anomalous dimension

# Coordinate transformations
def lorentz_boost(beta: float) -> Matrix:
    """
    Generate Lorentz boost matrix.
    
    Args:
        beta: Velocity parameter (v/c)
        
    Returns:
        2x2 Lorentz transformation matrix
    """
    gamma = 1/sqrt(1 - beta**2)
    return Matrix([
        [gamma, -gamma*beta],
        [-gamma*beta, gamma]
    ])

def gauge_transform(phase: float) -> exp:
    """
    Generate gauge transformation.
    
    Args:
        phase: Gauge transformation phase
        
    Returns:
        Gauge transformation operator
    """
    return exp(I * phase)

# Physical constants
HBAR = 1.0  # Natural units
C = 1.0  # Speed of light
G = 1.0  # Gravitational constant