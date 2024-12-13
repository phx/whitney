"""Numerical computation utilities for fractal field theory."""

from typing import List, Tuple, Union, Optional
import numpy as np
from scipy import integrate, optimize
from .errors import ComputationError, StabilityError
from .types import NumericValue

def solve_field_equation(hamiltonian: np.ndarray,
                        initial_state: np.ndarray,
                        t_range: np.ndarray,
                        dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve quantum field equation using split-operator method.
    
    Implements numerical solution from paper Sec. 3.7:
    1. Split kinetic and potential terms
    2. Apply exponential operators
    3. Combine using Strang splitting
    
    Args:
        hamiltonian: System Hamiltonian matrix
        initial_state: Initial state vector
        t_range: Time points for evolution
        dt: Time step size
        
    Returns:
        Tuple containing:
        - Time points
        - Evolved state vectors
        
    Raises:
        ComputationError: If evolution fails
        StabilityError: If solution becomes unstable
    """
    try:
        # Validate inputs
        if not np.all(np.isfinite(hamiltonian)):
            raise ValueError("Hamiltonian contains invalid values")
            
        if not np.all(np.isfinite(initial_state)):
            raise ValueError("Initial state contains invalid values")
            
        # Split Hamiltonian
        kinetic = hamiltonian - np.diag(np.diag(hamiltonian))
        potential = np.diag(np.diag(hamiltonian))
        
        # Prepare evolution operators
        exp_k = np.exp(-1j * kinetic * dt/2)
        exp_v = np.exp(-1j * potential * dt)
        
        # Initialize results
        n_steps = len(t_range)
        states = np.zeros((n_steps, len(initial_state)), dtype=complex)
        states[0] = initial_state
        
        # Time evolution
        for i in range(1, n_steps):
            # Split-operator step
            psi = states[i-1]
            psi = exp_k @ psi
            psi = exp_v @ psi
            psi = exp_k @ psi
            
            # Normalize
            psi /= np.sqrt(np.sum(np.abs(psi)**2))
            
            # Check stability
            if not np.all(np.isfinite(psi)):
                raise StabilityError("Evolution became unstable")
                
            states[i] = psi
            
        return t_range, states
        
    except Exception as e:
        raise ComputationError(f"Field equation solution failed: {e}")

def compute_correlation_function(field_values: np.ndarray,
                               r_points: np.ndarray,
                               periodic: bool = True) -> np.ndarray:
    """
    Compute two-point correlation function.
    
    Implements correlation calculation from paper Sec. 4.4:
    G(r) = <ψ(x)ψ(x+r)>
    
    Args:
        field_values: Field configuration values
        r_points: Separation points
        periodic: Use periodic boundary conditions
        
    Returns:
        Correlation function values
        
    Raises:
        ComputationError: If calculation fails
    """
    try:
        n_points = len(field_values)
        correlations = np.zeros_like(r_points, dtype=complex)
        
        for i, r in enumerate(r_points):
            # Convert r to index shift
            shift = int(round(r * n_points))
            
            if periodic:
                # Use periodic boundary conditions
                shifted = np.roll(field_values, shift)
            else:
                # Use zero padding
                if shift >= 0:
                    shifted = np.pad(field_values[:-shift], (shift, 0))
                else:
                    shifted = np.pad(field_values[-shift:], (0, -shift))
                    
            # Compute correlation
            correlations[i] = np.mean(field_values.conj() * shifted)
            
        return correlations
        
    except Exception as e:
        raise ComputationError(f"Correlation function calculation failed: {e}")

def compute_spectral_function(green_function: np.ndarray,
                            omega: np.ndarray,
                            eta: float = 1e-3) -> np.ndarray:
    """
    Compute spectral function from Green's function.
    
    Implements spectral function calculation from paper Sec. 4.5:
    A(ω) = -1/π * Im[G(ω + iη)]
    
    Args:
        green_function: Green's function values
        omega: Frequency points
        eta: Small imaginary part
        
    Returns:
        Spectral function values
        
    Raises:
        ComputationError: If calculation fails
    """
    try:
        # Add small imaginary part
        z = omega + 1j * eta
        
        # Compute Fourier transform
        g_omega = np.fft.fft(green_function)
        freq = np.fft.fftfreq(len(green_function))
        
        # Interpolate to requested frequencies
        g_z = np.interp(omega, freq, g_omega)
        
        # Extract spectral function
        spectral = -1/np.pi * np.imag(g_z)
        
        return spectral
        
    except Exception as e:
        raise ComputationError(f"Spectral function calculation failed: {e}")