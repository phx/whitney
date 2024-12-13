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
ALPHA_REF = 0.0072973525693  # Reference value at M_Z scale
# Physical masses (GeV) stored as (value, uncertainty)
Z_MASS: Tuple[float, float] = (91.1876, 0.0021)  # Z boson mass
W_MASS: Tuple[float, float] = (80.379, 0.012)  # W boson mass

# Electroweak parameters (from PDG 2022)
SIN2_THETA_W = 0.23122  # Weak mixing angle at M_Z

# Analysis configuration
N_MAX = 5  # Maximum basis function index for truncation
STABILITY_THRESHOLD = 1e-10  # Numerical stability threshold
CONVERGENCE_TOL = 1e-8  # Integration convergence tolerance

# Reference scales (from paper Sec. 2.2)
E_GUT = 2.0e16  # Grand unification scale (GeV)
E_PLANCK = 1.22089e19  # Planck scale (GeV)
LAMBDA_QCD = 0.217  # QCD scale (GeV)

# Symbolic variables
X = Symbol('x')  # Spatial coordinate
T = Symbol('t')  # Time coordinate
E = Symbol('E')  # Energy scale
PSI = Symbol('ψ')  # Field configuration
PHI = Symbol('φ')  # Auxiliary field

# Coupling constants at M_Z scale (from paper Eq. 2.4)
# Gauge couplings stored as (value, uncertainty)
g1_REF: Tuple[float, float] = (0.357, 0.001)  # U(1) coupling
g2_REF: Tuple[float, float] = (0.652, 0.003)  # SU(2) coupling
g3_REF: Tuple[float, float] = (1.221, 0.009)  # SU(3) coupling

# Fractal scaling exponents (from paper Eq. 3.8)
GAMMA_1 = 0.017  # U(1) scaling
GAMMA_2 = 0.023  # SU(2) scaling
GAMMA_3 = -0.007  # SU(3) scaling

# Experimental data points with uncertainties
EXPERIMENTAL_DATA = {
    'sin2_theta_W': (0.23122, 0.00003),  # LEP/SLD
    'Z_mass': (91.1876, 0.0021),         # LEP
    'W_mass': (80.379, 0.012),           # Tevatron/LHC
    'BR_Bs_mumu': (3.09e-9, 0.12e-9),    # LHCb
    'BR_Bd_mumu': (1.06e-10, 0.09e-10),  # LHCb/Belle II
    'Delta_Ms': (17.757, 0.021)          # LHCb
}

# Physical constants (all values from PDG 2022)
# Fundamental constants
h_bar = 6.582119569e-16  # Reduced Planck constant (eV⋅s)
k_B = 8.617333262e-5     # Boltzmann constant (eV/K)
e = 1.602176634e-19      # Elementary charge (C)