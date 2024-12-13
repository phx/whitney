"""Tests for version handling and compatibility."""

import pytest
import re
from packaging import version
from core.version import (
    get_version,
    check_compatibility,
    parse_version_string,
    VERSION_PATTERN
)
from core.errors import VersionError

@pytest.mark.version
class TestVersionHandling:
    """Test version string handling."""
    
    def test_version_format(self):
        """Test version string format."""
        version_str = get_version()
        assert re.match(VERSION_PATTERN, version_str)
        
        # Should be in format X.Y.Z
        major, minor, patch = version_str.split('.')
        assert all(x.isdigit() for x in [major, minor, patch])
    
    def test_version_parsing(self):
        """Test version string parsing."""
        test_versions = [
            "1.0.0",
            "2.1.3",
            "0.9.5"
        ]
        
        for ver_str in test_versions:
            result = parse_version_string(ver_str)
            assert isinstance(result, version.Version)
            assert str(result) == ver_str
    
    def test_invalid_version(self):
        """Test handling of invalid version strings."""
        invalid_versions = [
            "1.0",  # Missing patch
            "1.a.0",  # Non-numeric
            "1.0.0-alpha",  # No pre-release support
            ""  # Empty string
        ]
        
        for ver_str in invalid_versions:
            with pytest.raises(VersionError):
                parse_version_string(ver_str)

@pytest.mark.version
class TestCompatibility:
    """Test version compatibility checking."""
    
    def test_basic_compatibility(self):
        """Test basic version compatibility."""
        assert check_compatibility("1.0.0", "1.0.0")
        assert check_compatibility("1.0.0", "1.0.1")
        assert check_compatibility("1.0.0", "1.1.0")
        assert not check_compatibility("1.0.0", "2.0.0")
    
    def test_minimum_version(self):
        """Test minimum version requirements."""
        assert check_compatibility("1.0.0", "1.2.0", min_version="1.0.0")
        assert not check_compatibility("0.9.0", "1.0.0", min_version="1.0.0")
    
    @pytest.mark.parametrize('v1,v2,expected', [
        ("1.0.0", "1.0.1", True),
        ("1.0.0", "1.1.0", True),
        ("1.0.0", "2.0.0", False),
        ("2.0.0", "1.0.0", False),
        ("1.2.3", "1.2.3", True),
    ])
    def test_version_combinations(self, v1, v2, expected):
        """Test various version combinations."""
        assert check_compatibility(v1, v2) == expected
    
    def test_invalid_comparison(self):
        """Test error handling for invalid comparisons."""
        with pytest.raises(VersionError):
            check_compatibility("invalid", "1.0.0")
        
        with pytest.raises(VersionError):
            check_compatibility("1.0.0", "invalid") 