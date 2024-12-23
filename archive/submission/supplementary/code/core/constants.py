"""Framework constants and configuration."""

from .physics_constants import (
    X, E, T, P,
    Z_MASS, ALPHA_VAL, ALPHA_REF,
    g1_REF, g2_REF, g3_REF,
    GAMMA_1, GAMMA_2, GAMMA_3
)

# Framework configuration constants
VERSION = "1.0.0"
DEBUG = False
MAX_WORKERS = 4
CACHE_SIZE = 1000
LOG_LEVEL = "INFO"

# Physics scales
GUT_SCALE = 2.1e16  # GeV - Grand unification scale

from dataclasses import dataclass

@dataclass
class Constants:
    G: float = 6.67430e-11  # m^3 kg^-1 s^-2
    hbar: float = 1.054571817e-34  # J·s
    M_PLANCK: float = 2.176434e-8  # kg
    c: float = 299792458  # m/s
    M_Z: float = 91.1876  # GeV
    # Add other necessary constants as required