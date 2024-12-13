"""Tests for version control and compatibility."""

import pytest
from pathlib import Path
from core.version import Version, VersionManager
from core.errors import VersionError

@pytest.fixture
def version_manager(tmp_path):
    """Create test version manager with temporary files."""
    # Create test version.json
    version_json = tmp_path / "version.json"
    version_json.write_text("""{
        "core": "1.0.0",
        "field": "1.0.0-beta+001",
        "test": "2.1.3"
    }""")
    
    # Create test compatibility.json
    compat_json = tmp_path / "compatibility.json"
    compat_json.write_text("""{
        "field": ["core"],
        "test": ["core", "field"]
    }""")
    
    return VersionManager(root_dir=tmp_path)

class TestVersion:
    """Test Version class functionality."""
    
    def test_version_parsing(self):
        """Test parsing version strings."""
        v1 = Version.from_string("1.2.3")
        assert v1.major == 1
        assert v1.minor == 2
        assert v1.patch == 3
        assert v1.pre_release is None
        assert v1.build is None
        
        v2 = Version.from_string("2.0.0-alpha+001")
        assert v2.major == 2
        assert v2.minor == 0
        assert v2.patch == 0
        assert v2.pre_release == "alpha"
        assert v2.build == "001"
    
    def test_invalid_version(self):
        """Test handling of invalid version strings."""
        with pytest.raises(VersionError):
            Version.from_string("invalid")
        with pytest.raises(VersionError):
            Version.from_string("1.2")
        with pytest.raises(VersionError):
            Version.from_string("1.2.3.4")
    
    def test_version_comparison(self):
        """Test version comparison operations."""
        v1 = Version.from_string("1.0.0")
        v2 = Version.from_string("2.0.0")
        v3 = Version.from_string("2.1.0")
        
        assert v1 < v2
        assert v2 < v3
        assert not v3 < v2
        assert v1 == Version.from_string("1.0.0")

class TestVersionManager:
    """Test VersionManager functionality."""
    
    def test_load_versions(self, version_manager):
        """Test loading version information."""
        assert "core" in version_manager.versions
        assert "field" in version_manager.versions
        assert version_manager.versions["core"].major == 1
    
    def test_validate_component(self, version_manager):
        """Test component version validation."""
        # Valid version
        assert version_manager.validate_component("core", "1.0.0")
        
        # Invalid major version
        with pytest.raises(VersionError):
            version_manager.validate_component("core", "2.0.0")
        
        # Invalid minor version
        with pytest.raises(VersionError):
            version_manager.validate_component("test", "2.0.0")
    
    def test_check_compatibility(self, version_manager):
        """Test compatibility checking."""
        assert version_manager.check_compatibility("field")
        
        # Test with missing dependency
        with pytest.raises(VersionError):
            version_manager.check_compatibility("missing")
    
    def test_missing_files(self, tmp_path):
        """Test handling of missing configuration files."""
        with pytest.raises(VersionError):
            VersionManager(root_dir=tmp_path)
    
    def test_invalid_json(self, tmp_path):
        """Test handling of invalid JSON files."""
        # Create invalid version.json
        version_json = tmp_path / "version.json"
        version_json.write_text("invalid json")
        
        compat_json = tmp_path / "compatibility.json"
        compat_json.write_text("{}")
        
        with pytest.raises(VersionError):
            VersionManager(root_dir=tmp_path) 