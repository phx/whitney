"""Performance benchmarks and profiling tests."""

import pytest
import numpy as np
import platform
import os
from core.field import UnifiedField
from core.numeric import integrate_phase_space
from core.stability import check_convergence
from core.utils import get_memory_usage
from core.types import NumericValue, Energy, RealValue
from core.errors import ComputationError

# Conditionally import psutil
try:
    import psutil  # type: ignore
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

@pytest.mark.performance
@pytest.mark.benchmark(
    group="field-evolution",
    min_rounds=100,
    max_time=30.0
)
def test_field_evolution_performance(benchmark, standard_field):
    """Benchmark field evolution performance."""
    def run_evolution():
        psi0 = np.zeros(10)
        psi0[0] = 1.0
        times = np.linspace(0, 10, 100)
        return standard_field.evolve_field(psi0, times)
    
    result = benchmark(run_evolution)
    assert result is not None
    assert result['stable']

@pytest.mark.performance
@pytest.mark.benchmark(
    group="integration",
    min_rounds=50,
    max_time=30.0
)
def test_phase_space_integration_performance(benchmark, numeric_precision):
    """Benchmark phase space integration performance."""
    def integrand(p, q):
        return np.exp(-((p-q)/10.0)**2)
    
    def run_integration():
        return integrate_phase_space(
            integrand,
            limits=[(0, 100), (0, 100)],
            **numeric_precision
        )
    
    result = benchmark(run_integration)
    assert np.isfinite(result)

@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.skipif(not PSUTIL_AVAILABLE, 
                    reason="psutil not available for memory monitoring")
def test_memory_usage():
    """Test memory usage during large calculations."""
    initial_memory = get_memory_usage()
    
    # Perform memory-intensive calculation
    field = UnifiedField(dimension=4, max_level=10)
    field.compute_basis_functions(n_max=100)
    
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    
    if initial_memory >= 0 and final_memory >= 0:
        assert memory_increase < 1000  # Less than 1GB increase
    else:
        pytest.skip("Memory monitoring not available")