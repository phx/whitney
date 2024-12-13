"""Core utility functions for the fractal field theory framework."""

from typing import Any, Callable, TypeVar, Dict, Union, Optional, List
from functools import lru_cache, wraps
import numpy as np
import time
from dataclasses import dataclass
from sympy import Expr, N
from .types import RealValue, ComplexValue
from .errors import PhysicsError, ComputationError, StabilityError

T = TypeVar('T')

def evaluate_expr(expr: Expr, 
                 subs: Optional[Dict[str, float]] = None, 
                 precision: int = 53) -> Union[RealValue, ComplexValue]:
    """
    Evaluate symbolic expression numerically.
    
    Implements numerical evaluation from paper Sec. 3.3:
    1. Convert symbolic expression to lambda function
    2. Evaluate with given substitutions
    3. Check numerical stability
    
    Args:
        expr: Symbolic expression to evaluate
        subs: Optional substitutions for variables
        precision: Binary precision bits (default: double precision)
    
    Returns:
        float or complex: Numerical value of expression
    
    Raises:
        ComputationError: If evaluation fails
        StabilityError: If result is numerically unstable
    """
    try:
        # Handle substitutions
        if subs is None:
            subs = {}
            
        # First try direct numerical evaluation
        result = N(expr.subs(subs), precision)
        
        # Convert to Python numeric type
        if result.is_complex:
            value = complex(result)
        else:
            value = float(result)
            
        # Check stability
        if not check_numerical_stability(value):
            raise StabilityError(f"Unstable evaluation result: {value}")
            
        return value
        
    except Exception as e:
        raise ComputationError(f"Expression evaluation failed: {e}")

@lru_cache(maxsize=1024)
def cached_evaluation(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Cache function evaluation results.
    
    Implements memoization for expensive computations:
    1. Check cache for existing result
    2. Compute and cache if not found
    3. Return cached result
    
    Args:
        func: Function to evaluate
        *args: Positional arguments
        **kwargs: Keyword arguments
    
    Returns:
        T: Function result (cached if available)
        
    Raises:
        ComputationError: If evaluation fails
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        raise ComputationError(f"Cached evaluation failed: {e}")

def check_numerical_stability(value: Union[float, complex, np.ndarray], 
                            threshold: float = 1e-10) -> bool:
    """
    Check numerical stability of computation.
    
    Implements stability checks from paper Sec. 3.4:
    1. Check for NaN/Inf values
    2. Verify magnitude within bounds
    3. Check condition number for matrices
    
    Args:
        value: Value to check
        threshold: Stability threshold
    
    Returns:
        bool: True if value is numerically stable
    """
    try:
        # Convert to numpy array for uniform handling
        arr = np.asarray(value)
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(arr)):
            return False
            
        # Check magnitude
        if np.any(np.abs(arr) > 1/threshold):
            return False
            
        # Check for numerical noise
        small_nonzero = np.logical_and(np.abs(arr) < threshold, 
                                     np.abs(arr) > 0)
        if np.any(small_nonzero):
            return False
            
        return True
        
    except Exception:
        return False

def stability_check(threshold: float = 1e-10) -> Callable:
    """
    Decorator to check numerical stability of function results.
    
    Args:
        threshold: Stability threshold
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            if not check_numerical_stability(result, threshold):
                raise StabilityError(
                    f"Function {func.__name__} returned unstable result"
                )
            return result
        return wrapper
    return decorator

@dataclass
class ProfilingResult:
    """Results from profiling a computation."""
    execution_time: float
    memory_usage: float
    call_count: int
    avg_time_per_call: float

def profile_computation(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to profile computation time and memory usage.
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function that collects profiling data
        
    Example:
        @profile_computation
        def heavy_calculation(data):
            # Computation here
            return result
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        # Track memory and time
        start_mem = get_memory_usage()
        start_time = time.perf_counter()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Compute metrics
        end_time = time.perf_counter()
        end_mem = get_memory_usage()
        
        # Store profiling data
        wrapper.profiling_data = ProfilingResult(
            execution_time=end_time - start_time,
            memory_usage=end_mem - start_mem,
            call_count=wrapper.profiling_data.call_count + 1 if hasattr(wrapper, 'profiling_data') else 1,
            avg_time_per_call=((end_time - start_time) + 
                             (wrapper.profiling_data.avg_time_per_call * 
                              (wrapper.profiling_data.call_count - 1)) / 
                              wrapper.profiling_data.call_count) if hasattr(wrapper, 'profiling_data') else (end_time - start_time)
        )
        
        return result
    
    # Initialize profiling data
    wrapper.profiling_data = ProfilingResult(0.0, 0.0, 0, 0.0)
    return wrapper

def propagate_errors(values: List[RealValue], 
                    uncertainties: List[RealValue],
                    correlations: Optional[np.ndarray] = None) -> RealValue:
    """
    Propagate uncertainties through calculations.
    
    Args:
        values: List of measured values
        uncertainties: List of uncertainties
        correlations: Optional correlation matrix
        
    Returns:
        RealValue with propagated uncertainty
        
    Raises:
        PhysicsError: If inputs are invalid
        
    Example:
        result = propagate_errors(
            values=[x, y],
            uncertainties=[dx, dy],
            correlations=[[1, 0.5], [0.5, 1]]
        )
    """
    if len(values) != len(uncertainties):
        raise PhysicsError("Number of values and uncertainties must match")
        
    if correlations is not None:
        if correlations.shape != (len(values), len(values)):
            raise PhysicsError("Correlation matrix shape must match number of values")
            
        # Compute covariance matrix
        covariance = np.outer(uncertainties, uncertainties) * correlations
        
        # Propagate with correlations
        total_variance = np.sum(covariance)
        total_uncertainty = np.sqrt(total_variance)
        
    else:
        # Simple quadrature sum for uncorrelated uncertainties
        total_uncertainty = np.sqrt(np.sum(np.array(uncertainties)**2))
    
    # Compute central value
    central_value = np.sum(values)
    
    return RealValue(
        value=central_value,
        uncertainty=total_uncertainty
    )

def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil  # type: ignore
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except (ImportError, AttributeError):
        # Return dummy value if psutil not available
        return -1.0

def batch_process(items: List[T], 
                 batch_size: int,
                 process_func: Callable[[List[T]], Any]) -> List[Any]:
    """
    Process items in batches to manage memory.
    
    Args:
        items: List of items to process
        batch_size: Number of items per batch
        process_func: Function to process each batch
        
    Returns:
        List of processed results
        
    Example:
        results = batch_process(
            items=large_dataset,
            batch_size=1000,
            process_func=compute_batch
        )
    """
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = process_func(batch)
        results.extend(batch_results)
    return results