"""Detector simulation and response modeling."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Union, Tuple

from .types import Energy, Momentum, RealValue
from .errors import PhysicsError
from .utils import propagate_errors

@dataclass
class Detector:
    """
    Detector simulation with resolution and acceptance modeling.
    
    Attributes:
        resolution (Dict[str, float]): Resolution parameters
            - 'energy': Energy resolution (ΔE/E)
            - 'position': Position resolution (meters)
        acceptance (Dict[str, Union[float, Tuple[float, float]]]): Acceptance cuts
            - 'eta': (min, max) pseudorapidity range
            - 'pt': Minimum transverse momentum (GeV)
    """
    resolution: Dict[str, float]
    acceptance: Dict[str, Union[float, Tuple[float, float]]]
    
    def __post_init__(self):
        """Validate detector parameters."""
        self._validate_resolution()
        self._validate_acceptance()
    
    def _validate_resolution(self):
        """Validate resolution parameters."""
        required = {'energy', 'position'}
        if not all(k in self.resolution for k in required):
            raise PhysicsError(f"Missing required resolution parameters: {required}")
        
        if not all(isinstance(v, (int, float)) for v in self.resolution.values()):
            raise PhysicsError("Resolution values must be numeric")
        
        if not all(v > 0 for v in self.resolution.values()):
            raise PhysicsError("Resolution values must be positive")
    
    def _validate_acceptance(self):
        """Validate acceptance parameters."""
        required = {'eta', 'pt'}
        if not all(k in self.acceptance for k in required):
            raise PhysicsError(f"Missing required acceptance parameters: {required}")
        
        eta = self.acceptance['eta']
        if not (isinstance(eta, (list, tuple)) and len(eta) == 2):
            raise PhysicsError("eta acceptance must be (min, max) tuple")
        
        if not isinstance(self.acceptance['pt'], (int, float)):
            raise PhysicsError("pt threshold must be numeric")
    
    def simulate_measurement(self, value: Union[Energy, Momentum]) -> Dict[str, RealValue]:
        """
        Simulate detector measurement with resolution effects.
        
        Args:
            value: True value to measure (Energy or Momentum)
            
        Returns:
            dict: Measured value and uncertainty
            
        Raises:
            PhysicsError: If value is invalid
        """
        if isinstance(value, Energy):
            resolution = self.resolution['energy']
            uncertainty = resolution * float(value)
            measured = np.random.normal(float(value), uncertainty)
            return {
                'energy': RealValue(measured),
                'uncertainty': RealValue(uncertainty)
            }
        
        elif isinstance(value, Momentum):
            resolution = self.resolution['position']
            uncertainty = resolution * float(value)
            measured = np.random.normal(float(value), uncertainty)
            return {
                'momentum': RealValue(measured),
                'uncertainty': RealValue(uncertainty)
            }
        
        raise PhysicsError(f"Cannot measure value of type {type(value)}")
    
    def compute_efficiency(self, pt: RealValue, eta: RealValue) -> RealValue:
        """
        Compute detection efficiency for given kinematics.
        
        Args:
            pt: Transverse momentum (GeV)
            eta: Pseudorapidity
            
        Returns:
            float: Detection efficiency (0-1)
        """
        # Check if within acceptance
        if not self.check_acceptance(float(pt), float(eta)):
            return RealValue(0.0)
        
        # Model efficiency roll-off near threshold
        pt_eff = 1 / (1 + np.exp(-(float(pt) - self.acceptance['pt'])))
        
        # Model efficiency vs eta
        eta_min, eta_max = self.acceptance['eta']
        eta_val = float(eta)
        eta_eff = (1 - ((eta_val - eta_min)/(eta_max - eta_min))**2)
        
        return RealValue(pt_eff * eta_eff)
    
    def check_acceptance(self, pt: float, eta: float) -> bool:
        """
        Check if kinematics pass acceptance cuts.
        
        Args:
            pt: Transverse momentum (GeV)
            eta: Pseudorapidity
            
        Returns:
            bool: True if passes acceptance
            
        Raises:
            PhysicsError: If inputs are invalid
        """
        if not np.isfinite(pt) or pt < 0:
            raise PhysicsError(f"Invalid pt value: {pt}")
        
        if not np.isfinite(eta):
            raise PhysicsError(f"Invalid eta value: {eta}")
        
        eta_min, eta_max = self.acceptance['eta']
        return (pt >= self.acceptance['pt'] and 
                eta_min <= eta <= eta_max)