"""Computation utilities for numerical evaluation and optimization."""

from typing import Callable, Any, Optional, Dict, Union
from functools import wraps, lru_cache, partial
import time
import contextlib
import numpy as np
from sympy import Expr, N
from .types import RealValue, ComplexValue
from .errors import ComputationError, StabilityError
import threading
from collections import OrderedDict
import warnings

# Default stability thresholds
DEFAULT_THRESHOLDS = {
    'underflow': 1e-10,
    'overflow': 1e10,
    'relative_error': 1e-6,
    'condition_number': 1e8
}

class ThreadSafeCache:
    """Thread-safe computation cache with size limit."""
    
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.RLock()
    
    def get(self, key: Any) -> Any:
        """Get cached value."""
        with self.lock:
            return self.cache[key]
    
    def set(self, key: Any, value: Any) -> None:
        """Set cache value with LRU eviction."""
        with self.lock:
            if len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)
            self.cache[key] = value
    
    def clear(self) -> None:
        """Clear all cached values."""
        with self.lock:
            self.cache.clear()

class StabilityConfig:
    """Configuration for numerical stability checks."""
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = DEFAULT_THRESHOLDS.copy()
        if thresholds:
            self.thresholds.update(thresholds)
    
    def check_value(self, value: Union[float, np.ndarray]) -> bool:
        """Check if value meets stability criteria."""
        if not np.all(np.isfinite(value)):
            return False
        
        abs_value = np.abs(value)
        nonzero = abs_value != 0
        
        # Check underflow
        if np.any(abs_value[nonzero] < self.thresholds['underflow']):
            return False
        
        # Check overflow
        if np.any(abs_value > self.thresholds['overflow']):
            return False
        
        return True
    
    def check_condition(self, matrix: np.ndarray) -> bool:
        """Check matrix condition number."""
        try:
            cond = np.linalg.cond(matrix)
            return cond < self.thresholds['condition_number']
        except np.linalg.LinAlgError:
            return False

@contextlib.contextmanager
def resource_manager():
    """Context manager for resource cleanup."""
    resources = []
    try:
        yield resources
    finally:
        # Clean up all resources
        for resource in resources:
            try:
                resource.close()
            except Exception as e:
                warnings.warn(f"Failed to clean up resource: {e}")

def memoize_computation(maxsize: int = 128, 
                       typed: bool = False,
                       thread_safe: bool = True) -> Callable:
    """
    Memoization decorator with size limit and type checking.
    
    Args:
        maxsize: Maximum cache size
        typed: Whether to account for argument types
        thread_safe: Whether to use thread-safe cache
    """
    def decorator(func: Callable) -> Callable:
        # Choose cache implementation
        if thread_safe:
            cache = ThreadSafeCache(maxsize=maxsize)
        else:
            # Use standard lru_cache for non-threaded code
            return lru_cache(maxsize=maxsize, typed=typed)(func)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Create cache key
                key = (args, tuple(sorted(kwargs.items())))
                if typed:
                    key += tuple(type(arg) for arg in args)
                    key += tuple(type(v) for _, v in sorted(kwargs.items()))
                
                # Check cache
                try:
                    return cache.get(key)
                except KeyError:
                    result = func(*args, **kwargs)
                    cache.set(key, result)
                    return result
            except Exception as e:
                raise ComputationError(f"Cached computation failed: {e}")
        
        # Add cache clear method
        wrapper.cache_clear = cache.clear
        return wrapper
    return decorator

def benchmark_computation(func: Callable) -> Callable:
    """
    Decorator to benchmark computation time and memory usage.
    
    Uses context manager to ensure proper resource cleanup.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Dict[str, Any]:
        with resource_manager() as resources:
            start_time = time.perf_counter()
            start_memory = get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                resources.append(partial(clear_computation_cache, result))
                
                end_time = time.perf_counter()
                end_memory = get_memory_usage()
                
                return {
                    'result': result,
                    'execution_time': end_time - start_time,
                    'memory_delta': end_memory - start_memory
                }
            except Exception as e:
                raise ComputationError(f"Benchmarked computation failed: {e}")
    return wrapper

def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil # type: ignore
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        # Fallback to a simpler memory estimate if psutil not available
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def check_computation_stability(values: np.ndarray,
                               thresholds: Optional[Dict[str, float]] = None) -> bool:
    """
    Check numerical stability of computation.
    
    Args:
        values: Array of computed values
        thresholds: Optional custom stability thresholds
        
    Returns:
        bool: True if computation is stable
    """
    config = StabilityConfig(thresholds)
    return config.check_value(values)

def clear_computation_cache(result: Any) -> None:
    """Clear any cached computations."""
    if hasattr(result, 'cache_clear'):
        result.cache_clear()
    elif isinstance(result, dict):
        for value in result.values():
            clear_computation_cache(value) 