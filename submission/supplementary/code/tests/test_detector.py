"""Unit tests for detector implementation."""

import unittest
import numpy as np
from core.detector import (
    AdvancedDetector, SuperconductingSensor,
    QuantumDotDetector, TopologicalSensor
)

class TestDetector(unittest.TestCase):
    """Test cases for detector implementation."""
    
    def setUp(self):
        """Initialize test environment."""
        self.detector = AdvancedDetector()
        
    def test_superconducting_response(self):
        """Test superconducting sensor response."""
        sensor = SuperconductingSensor()
        signal = np.sin(np.linspace(0, 10, 1000))
        
        # Test response
        output = sensor.detect(signal)
        
        # Verify physical properties
        self.assertTrue(np.all(np.isfinite(output)))
        self.assertLess(np.std(output), np.std(signal))  # Noise reduction
        
    def test_quantum_dot_response(self):
        """Test quantum dot detector response."""
        sensor = QuantumDotDetector()
        signal = np.random.normal(0, 1, 1000)
        
        # Test response
        output = sensor.detect(signal)
        
        # Verify quantization
        unique_levels = np.unique(output[output != 0])
        self.assertLess(len(unique_levels), len(signal))
        
    def test_topological_protection(self):
        """Test topological protection against noise."""
        sensor = TopologicalSensor()
        signal = np.sin(np.linspace(0, 10, 1000))
        noise = np.random.normal(0, 0.1, 1000)
        
        # Test with and without noise
        clean_output = sensor.detect(signal)
        noisy_output = sensor.detect(signal + noise)
        
        # Verify noise suppression
        clean_std = np.std(clean_output - signal)
        noisy_std = np.std(noisy_output - signal)
        self.assertLess(noisy_std / clean_std, 1.5) 
    
    def test_full_detector_chain(self):
        """Test complete detector measurement chain."""
        # Generate test signal with known properties
        t = np.linspace(0, 1e-6, 1000)  # 1 microsecond
        signal = 1e-3 * np.sin(2*np.pi*1e6*t)  # 1 MHz, 1 mV signal
        
        # Add realistic noise
        noise = np.random.normal(0, 1e-4, len(t))
        noisy_signal = signal + noise
        
        # Process through detector
        cleaned, uncertainty = self.detector.measure(noisy_signal)
        
        # Verify signal recovery
        correlation = np.corrcoef(signal, cleaned)[0,1]
        self.assertGreater(correlation, 0.9)  # Strong correlation with original
        
        # Verify uncertainty estimation
        self.assertGreater(uncertainty, 0)
        self.assertLess(uncertainty, np.std(signal))  # Uncertainty < signal