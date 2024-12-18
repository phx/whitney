from sympy import exp, diff, I
from ..submission.supplementary.code.core.physics_constants import X, T, HBAR, C
from ..submission.supplementary.code.core.types import WaveFunction

class TestTheoreticalPredictions:
    def test_ward_identity(self):
        """Test Ward identity for current conservation."""
        # Create test state with proper quantum numbers
        test_psi = WaveFunction(
            psi=exp(-X**2/(2*HBAR)) * exp(-I*T/HBAR),  # Gaussian packet
            grid=(-10, 10, 100),  # Spatial grid
            quantum_numbers={'n': 0, 'l': 0, 'm': 0}  # Ground state
        )
        
        # Call compute_noether_current with test state
        current = self.field.compute_noether_current(test_psi)
        
        # Verify current conservation with enhanced precision
        div_j = diff(current[0], T) + C * diff(current[1], X)
        assert abs(float(div_j)) < self.field.precision * self.field.alpha