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