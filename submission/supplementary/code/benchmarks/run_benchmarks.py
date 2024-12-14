#!/usr/bin/env python3
"""Run performance benchmarks."""

import sys
from pathlib import Path
import pytest

def setup_python_path():
    """Set up Python path to include project root and core modules."""
    # Get absolute path to project root (parent of benchmarks directory)
    project_root = Path(__file__).parent.parent.absolute()
    
    # Add project root to Python path if not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Add core module directory
    core_dir = project_root / "core"
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))

def main():
    """Run all benchmarks with proper configuration."""
    # Set up paths
    setup_python_path()
    
    # Import pytest only after path is set up
    import pytest
    
    # Run pytest with benchmark configuration
    return pytest.main([
        "benchmarks",
        "-v",
        "--benchmark-only",
        "--benchmark-group-by=func",
        "--benchmark-min-rounds=100",
        "--benchmark-warmup=on",
        "--import-mode=importlib",
        "--pythonpath", ".",
    ])

if __name__ == "__main__":
    sys.exit(main()) 