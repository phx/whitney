"""Version information for the fractal field theory framework."""

import pkg_resources
import datetime
import re
from packaging import version
from typing import Optional
from .errors import VersionError

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

VERSION_PATTERN = r'^\d+\.\d+\.\d+$'

def get_version() -> str:
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

def parse_version_string(version_str: str) -> version.Version:
    """
    Parse version string into Version object.
    
    Args:
        version_str: Version string in X.Y.Z format
        
    Returns:
        Version object
        
    Raises:
        VersionError: If version string is invalid
    """
    if not re.match(VERSION_PATTERN, version_str):
        raise VersionError(f"Invalid version format: {version_str}")
    return version.parse(version_str)

def check_compatibility(
    v1: str,
    v2: str,
    *,
    min_version: Optional[str] = None
) -> bool:
    """
    Check version compatibility.
    
    Args:
        v1: First version string
        v2: Second version string
        min_version: Optional minimum required version
        
    Returns:
        bool: True if versions are compatible
        
    Raises:
        VersionError: If any version string is invalid
    """
    try:
        ver1 = parse_version_string(v1)
        ver2 = parse_version_string(v2)
        
        if min_version:
            min_ver = parse_version_string(min_version)
            if ver1 < min_ver or ver2 < min_ver:
                return False
        
        # Major version must match
        return ver1.major == ver2.major
        
    except Exception as e:
        raise VersionError(f"Version comparison failed: {e}")