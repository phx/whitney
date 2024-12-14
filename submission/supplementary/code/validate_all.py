#!/usr/bin/env python3
"""
Comprehensive validation of the entire framework.
"""

import sys
from pathlib import Path
import pytest
import subprocess
from typing import List, Optional

def setup_python_path() -> None:
    """Set up Python path to include project root and core modules."""
    project_root = Path(__file__).parent.absolute()
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    core_dir = project_root / "core"
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))

def run_tests() -> int:
    """Run all tests with coverage."""
    return pytest.main([
        "tests",
        "-v",
        "--cov=core",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--import-mode=importlib",
        "--pythonpath", ".",
    ])

def run_linters() -> int:
    """Run code quality checks."""
    results = []
    
    # Run mypy type checking
    results.append(subprocess.run(
        ["mypy", "core", "tests", "--strict"],
        capture_output=True
    ).returncode)
    
    # Run flake8
    results.append(subprocess.run(
        ["flake8", "core", "tests", "--max-line-length=100"],
        capture_output=True
    ).returncode)
    
    # Run pylint
    results.append(subprocess.run(
        ["pylint", "core", "tests", "--rcfile=.pylintrc"],
        capture_output=True
    ).returncode)
    
    return max(results)  # Return worst result

def validate_documentation() -> int:
    """Build and validate documentation."""
    return subprocess.run(
        ["sphinx-build", "-b", "html", "docs", "docs/_build/html"],
        capture_output=True
    ).returncode

def main() -> int:
    """Run all validation checks."""
    setup_python_path()
    
    print("Running tests...")
    test_result = run_tests()
    
    print("\nRunning linters...")
    lint_result = run_linters()
    
    print("\nValidating documentation...")
    doc_result = validate_documentation()
    
    # Return worst result
    return max(test_result, lint_result, doc_result)

if __name__ == "__main__":
    sys.exit(main()) 