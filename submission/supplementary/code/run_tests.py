#!/usr/bin/env python3
"""Run test suite with coverage reporting."""

import pytest
import sys
import os

def main():
    """Run test suite with coverage reporting."""
    # Add source directory to path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Run tests with coverage
    args = [
        "--cov=core",
        "--cov-report=html",
        "--cov-report=term-missing",
        "tests/"
    ]
    
    exit_code = pytest.main(args)
    
    # Print coverage report location
    if exit_code == 0:
        print("\nCoverage report generated in htmlcov/index.html")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main()) 