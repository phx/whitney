"""Physical constants for fractal field theory."""

from sympy import Symbol, pi
from typing import Tuple

# Figure and plot settings
FIGURE_SIZE = (10, 6)  # Default figure size in inches
PLOT_DPI = 300        # Resolution for saved figures
PLOT_STYLE = {
    'lines.linewidth': 2,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': FIGURE_SIZE
}

# Fundamental constants (all values from PDG 2022)
ALPHA_VAL = 0.0072973525693  # Fine structure constant at M_Z
ALPHA_REF = 1/128.9  # Reference alpha at M_Z

# Physical masses (GeV) stored as (value, uncertainty)
Z_MASS: Tuple[float, float] = (91.1876, 0.0021)  # Z boson mass
W_MASS: Tuple[float, float] = (80.379, 0.012)  # W boson mass

# Electroweak parameters (from PDG 2022)
SIN2_THETA_W = 0.23122  # Weak mixing angle at M_Z

# Reference masses and scales
M_Z = Z_MASS[0]  # Z boson mass in GeV (central value)
M_PLANCK = 1.22e19  # Planck mass in GeV
E_PLANCK = M_PLANCK  # Planck energy scale (GeV)
LAMBDA_QCD = 0.217  # QCD scale (GeV)

# Gauge couplings at reference scale (with uncertainties)
g1_REF: Tuple[float, float] = (0.357, 0.001)  # U(1) coupling
g2_REF: Tuple[float, float] = (0.652, 0.003)  # SU(2) coupling
g3_REF: Tuple[float, float] = (1.221, 0.009)  # SU(3) coupling

# Anomalous dimensions
GAMMA_1 = 0.0017  # U(1) anomalous dimension
GAMMA_2 = 0.0033  # SU(2) anomalous dimension
GAMMA_3 = -0.0072  # SU(3) anomalous dimension

# Symbolic variables
X = Symbol('x')  # Position/momentum variable
E = Symbol('E')  # Energy variable
T = Symbol('t')  # Time variable
PSI = Symbol('ψ')  # Field configuration
PHI = Symbol('φ')  # Auxiliary field

# Numerical constants
PI = 3.14159265359
HBAR = 6.582119e-25  # GeV*s
C_LIGHT = 2.99792458e8  # m/s

# Unit conversion factors
GEV_TO_TEV = 1e-3
GEV_TO_MEV = 1e3
FB_TO_PB = 1e-3
PB_TO_NB = 1e-3