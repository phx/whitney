"""Enumeration types for the framework."""

from enum import Enum, auto

class ComputationMode(Enum):
    """Defines computation modes for framework."""
    SYMBOLIC = auto()  # Fully symbolic computation
    NUMERIC = auto()   # Numerical computation
    MIXED = auto()     # Mixed symbolic/numeric (cached evaluation) 