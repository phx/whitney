#!/usr/bin/env python3
"""Validate all aspects of the paper submission."""

import os
import sys
from pathlib import Path

import pytest

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

if __name__ == "__main__":
    sys.exit(main()) 