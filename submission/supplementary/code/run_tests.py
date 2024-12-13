#!/usr/bin/env python3
"""Run test suite with coverage reporting."""

import os
import sys
import pytest

def main():
    """Run all tests with proper configuration."""
    # Add project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Run pytest with configuration
    pytest.main([
        "tests",
        "-v",
        "--cov=core",
        "--cov-report=term-missing"
    ])

if __name__ == "__main__":
    main() 