#!/usr/bin/env python3
"""Run test suite with coverage reporting."""

import sys
from pathlib import Path
import pytest

def setup_python_path() -> None:
    """Set up Python path to include project root and core modules."""
    project_root = Path(__file__).parent.absolute()
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    core_dir = project_root / "core"
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))

def main() -> int:
    """Run all tests with coverage reporting."""
    # Set up paths
    setup_python_path()
    
    # Run pytest with coverage
    return pytest.main([
        "tests",
        "-v",
        "--cov=core",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--import-mode=importlib",
        "--pythonpath", ".",
    ])

if __name__ == "__main__":
    sys.exit(main()) 