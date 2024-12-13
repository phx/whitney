"""Benchmarking utilities for fractal field theory."""

import time
import numpy as np
from typing import Dict, Any, List, Callable
from dataclasses import dataclass
from core.basis import FractalBasis
from core.field import UnifiedField
from core.compute import benchmark_computation, get_memory_usage

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    execution_time: float
    memory_usage: float
    throughput: float
    parameters: Dict[str, Any]

class PerformanceBenchmark:
    """Framework performance benchmarking."""
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.basis = FractalBasis()
        self.field = UnifiedField()
        self.results: List[BenchmarkResult] = []
    
    @benchmark_computation
    def benchmark_basis_computation(self, n_max: int = 10) -> Dict[str, float]:
        """Benchmark basis function computation."""
        start_time = time.perf_counter()
        
        for n in range(n_max):
            self.basis.compute(n)
            
        end_time = time.perf_counter()
        throughput = n_max / (end_time - start_time)
        
        return {
            'execution_time': end_time - start_time,
            'throughput': throughput,
            'n_max': n_max
        }
    
    @benchmark_computation
    def benchmark_field_evolution(self, 
                                time_steps: int = 100,
                                space_points: int = 1000) -> Dict[str, float]:
        """Benchmark field evolution computation."""
        x = np.linspace(-10, 10, space_points)
        t = np.linspace(0, 1, time_steps)
        
        start_time = time.perf_counter()
        start_mem = get_memory_usage()
        
        # Initial field configuration
        psi = self.basis.compute(0).subs('x', x[0])
        
        # Evolution
        for t_i in t:
            field_eq = self.field.compute_field_equation(psi)
            energy = self.field.compute_energy_density(psi)
        
        end_time = time.perf_counter()
        end_mem = get_memory_usage()
        
        return {
            'execution_time': end_time - start_time,
            'memory_delta': end_mem - start_mem,
            'throughput': time_steps * space_points / (end_time - start_time)
        }
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmarks and collect results."""
        benchmarks = [
            ('basis_computation', self.benchmark_basis_computation),
            ('field_evolution', self.benchmark_field_evolution)
        ]
        
        for name, func in benchmarks:
            result = func()
            self.results.append(BenchmarkResult(
                name=name,
                execution_time=result['execution_time'],
                memory_usage=result.get('memory_delta', 0.0),
                throughput=result['throughput'],
                parameters=result
            ))
        
        return self.results 