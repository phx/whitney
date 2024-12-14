"""Performance benchmarking framework."""

import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

from core.field import UnifiedField
from core.basis import FractalBasis
from core.types import Energy
from core.errors import BenchmarkError
from core.physics_constants import ALPHA_VAL

# Optional psutil import with fallback
try:
    import psutil # type: ignore
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    execution_time: float
    memory_usage: float
    throughput: Optional[float] = None
    parameters: Optional[Dict[str, Any]] = None

class PerformanceBenchmark:
    """Framework for performance benchmarking."""
    
    def __init__(self):
        """Initialize benchmark framework."""
        self.field = UnifiedField(alpha=ALPHA_VAL)
        self.basis = FractalBasis(alpha=ALPHA_VAL)
        self.results: List[BenchmarkResult] = []
    
    def measure_execution_time(self, func: Callable, *args, **kwargs) -> float:
        """Measure execution time of a function."""
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        return end_time - start_time
    
    def measure_memory_usage(self, func: Callable, *args, **kwargs) -> float:
        """Measure peak memory usage of a function."""
        if not HAS_PSUTIL:
            return 0.0  # Return 0 if psutil not available
            
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        func(*args, **kwargs)
        
        final_memory = process.memory_info().rss
        return (final_memory - initial_memory) / 1024 / 1024  # Convert to MB
    
    def benchmark_field_evolution(self, n_steps: int = 1000) -> BenchmarkResult:
        """Benchmark field evolution performance."""
        def evolution_task():
            psi = self.field.compute_basis_function(n=0)
            times = np.linspace(0, 10, n_steps)
            self.field.evolve_field(psi, times)
        
        exec_time = self.measure_execution_time(evolution_task)
        memory_usage = self.measure_memory_usage(evolution_task)
        
        return BenchmarkResult(
            name="field_evolution",
            execution_time=exec_time,
            memory_usage=memory_usage,
            parameters={"n_steps": n_steps}
        )
    
    def benchmark_basis_computation(self, n_max: int = 100) -> BenchmarkResult:
        """Benchmark basis function computation."""
        def basis_task():
            for n in range(n_max):
                self.basis.compute(n)
        
        exec_time = self.measure_execution_time(basis_task)
        memory_usage = self.measure_memory_usage(basis_task)
        
        return BenchmarkResult(
            name="basis_computation",
            execution_time=exec_time,
            memory_usage=memory_usage,
            parameters={"n_max": n_max}
        )
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmark tests."""
        try:
            self.results = [
                self.benchmark_field_evolution(),
                self.benchmark_basis_computation()
            ]
            return self.results
        except Exception as e:
            raise BenchmarkError(f"Benchmark failed: {str(e)}")