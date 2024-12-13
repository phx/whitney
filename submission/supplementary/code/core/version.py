"""Version information for the fractal field theory framework."""

import pkg_resources
import datetime

# Version information
VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_PATCH = 0
VERSION_SUFFIX = 'alpha'

# Build version string
VERSION = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}"
if VERSION_SUFFIX:
    VERSION += f"-{VERSION_SUFFIX}"

# Build timestamp
BUILD_TIMESTAMP = datetime.datetime.now().isoformat()

def get_version():
    """Get current version string."""
    return VERSION

def get_build_info():
    """Get build information."""
    return {
        'version': VERSION,
        'build_timestamp': BUILD_TIMESTAMP,
        'python_version': pkg_resources.get_distribution('python').version,
        'numpy_version': pkg_resources.get_distribution('numpy').version,
        'scipy_version': pkg_resources.get_distribution('scipy').version,
        'sympy_version': pkg_resources.get_distribution('sympy').version
    }

def check_version_compatibility(min_version):
    """
    Check if current version meets minimum requirement.
    
    Args:
        min_version (str): Minimum required version (e.g. "0.1.0")
        
    Returns:
        bool: True if current version is compatible
        
    Raises:
        ValueError: If version string is invalid
    """
    def parse_version(v):
        try:
            parts = v.split('-')[0].split('.')
            return tuple(map(int, parts))
        except:
            raise ValueError(f"Invalid version format: {v}")
    
    current = parse_version(VERSION)
    required = parse_version(min_version)
    
    return current >= required 