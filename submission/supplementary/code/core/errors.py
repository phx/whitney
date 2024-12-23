"""Error definitions for fractal field theory framework."""

class FractalTheoryError(Exception):
    """Base class for all framework errors."""
    pass

class PhysicsError(FractalTheoryError):
    """Physics constraint violation."""
    pass

class ValidationError(PhysicsError):
    """Input validation error."""
    pass

class ComputationError(PhysicsError):
    """Error in numerical or symbolic computation."""
    pass

class StabilityError(ComputationError):
    """Numerical stability violation."""
    pass

class PrecisionError(ComputationError):
    """Error in precision requirements."""
    pass

class BoundsError(ComputationError):
    """Value outside acceptable bounds."""
    pass

class ConvergenceError(ComputationError):
    """Error when iteration fails to converge."""
    pass

class EnergyConditionError(PhysicsError):
    """Error when energy conditions are violated."""
    pass

class CausalityError(PhysicsError):
    """Error when causality is violated."""
    pass

class GaugeError(PhysicsError):
    """Error in gauge transformations."""
    pass

class ConfigurationError(FractalTheoryError):
    """Framework configuration error."""
    pass

class VersionError(FractalTheoryError):
    """Error in version handling."""
    pass

"""Custom error types for physics computations."""

class PhysicsError(Exception):
    """Base class for physics-related errors."""
    pass

class ValidationError(PhysicsError):
    """Error for invalid input validation."""
    pass

class ComputationError(PhysicsError):
    """Error for failed computations."""
    pass

class EnergyConditionError(PhysicsError):
    """Error for violated energy conditions."""
    pass

class CausalityError(PhysicsError):
    """Error for causality violations."""
    pass

class GaugeError(PhysicsError):
    """Error for gauge symmetry violations."""
    pass

# Add new error types
class CrossSectionError(PhysicsError):
    """Error for cross section computation failures."""
    pass

class EnergyScaleError(PhysicsError):
    """Error for invalid energy scales."""
    pass

class NumericalStabilityError(PhysicsError):
    """Error for numerical stability violations."""
    pass

class QuantumNumberError(PhysicsError):
    """Error for invalid quantum number configurations."""
    pass

class UnitarityError(PhysicsError):
    """Error for unitarity violations."""
    pass

class HolographicError(PhysicsError):
    """Error for holographic bound violations."""
    pass

class BasisError(Exception):
    """
    Error in fractal basis computations.
    
    From appendix_b_basis.tex Eq B.8:
    Basis errors occur when:
    1. Invalid basis function parameters
    2. Normalization failure
    3. Orthogonality violation
    4. Grid range violation
    """
    pass

class ValidationError(Exception):
    """Error in validation checks."""
    pass

class ComputationError(Exception):
    """Error in numerical computations."""
    pass

class PhysicsError(Exception):
    """Error in physics constraints."""
    pass

class EnergyConditionError(PhysicsError):
    """Violation of energy conditions."""
    pass

class CausalityError(PhysicsError):
    """Violation of causality."""
    pass

class GaugeError(PhysicsError):
    """Error in gauge transformations."""
    pass

class CoherenceError(PhysicsError):
    """
    Error in quantum coherence preservation.
    
    From appendix_j_math_details.tex Eq J.40-J.42:
    Coherence violations occur when:
    1. Phase evolution is non-unitary
    2. Quantum correlations are lost
    3. Decoherence exceeds bounds
    4. Measurement basis is inconsistent
    """
    pass 