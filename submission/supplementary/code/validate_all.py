#!/usr/bin/env python3
"""Validate all aspects of the paper submission."""

import os
import sys
from pathlib import Path

import pytest
from core.generate_data import generate_gw_spectrum_data

def main():
    """Run all validation checks."""
    # Add project root to path
    project_root = Path(__file__).parent.absolute()
    sys.path.insert(0, str(project_root))
    
    # Run tests
    return pytest.main([
        "tests",
        "-v", 
        "--cov=core",
        "--cov-report=term-missing"
    ])

def validate_gravitational_waves():
    """Validate gravitational wave predictions."""
    print("Validating gravitational wave predictions...")
    
    # Generate fresh test data
    generate_gw_spectrum_data()
    
    # Run integration tests
    pytest.main([
        "tests/test_integration.py::test_gravitational_wave_coherence",
        "-v"
    ])
    
    # Run prediction tests
    pytest.main([
        "tests/test_predictions.py::test_gravitational_wave_spectrum",
        "-v"
    ])

if __name__ == "__main__":
    sys.exit(main()) 