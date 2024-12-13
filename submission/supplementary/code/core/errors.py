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