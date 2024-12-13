"""Error definitions for fractal field theory framework."""

class FractalTheoryError(Exception):
    """Base class for all framework errors."""
    pass

class ComputationError(FractalTheoryError):
    """Error in numerical or symbolic computation."""
    pass

class StabilityError(ComputationError):
    """Numerical stability violation."""
    pass

class BoundsError(ComputationError):
    """Value outside acceptable bounds."""
    pass

class ValidationError(FractalTheoryError):
    """Input validation error."""
    pass

class PhysicsError(FractalTheoryError):
    """Physics constraint violation."""
    pass

class ConfigurationError(FractalTheoryError):
    """Framework configuration error."""
    pass

class PrecisionError(FractalTheoryError):
    """Error in precision requirements."""
    pass

class VersionError(FractalTheoryError):
    """Error in version handling."""
    pass 