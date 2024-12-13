#!/usr/bin/env python3
"""Run test suite with coverage reporting."""

import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run test suite with coverage."""
    # Run tests with coverage
    result = subprocess.run([
        'pytest',
        '--cov=core',
        '--cov-report=term-missing',
        '--cov-report=html',
        '--cov-fail-under=80'
    ])
    
    if result.returncode != 0:
        print("Tests failed!")
        sys.exit(1)
    
    # Generate coverage badge
    subprocess.run(['python', 'tests/generate_badge.py'])
    
    print("\nTest suite completed successfully!")
    print("Coverage report available in coverage_html/index.html")

if __name__ == '__main__':
    run_tests() 