"""Version control and compatibility management."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import os
from pathlib import Path
from .errors import VersionError, ValidationError

@dataclass
class Version:
    """Version information for code components."""
    major: int
    minor: int
    patch: int
    pre_release: Optional[str] = None
    build: Optional[str] = None

    def __str__(self) -> str:
        """Convert to string format."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre_release:
            version += f"-{self.pre_release}"
        if self.build:
            version += f"+{self.build}"
        return version

    @classmethod
    def from_string(cls, version_str: str) -> "Version":
        """Create Version from string."""
        try:
            # Parse version string (e.g. "1.2.3-alpha+001")
            version_parts = version_str.split("+")
            version_base = version_parts[0]
            build = version_parts[1] if len(version_parts) > 1 else None
            
            base_parts = version_base.split("-")
            version_nums = base_parts[0]
            pre_release = base_parts[1] if len(base_parts) > 1 else None
            
            major, minor, patch = map(int, version_nums.split("."))
            
            return cls(
                major=major,
                minor=minor,
                patch=patch,
                pre_release=pre_release,
                build=build
            )
        except (ValueError, IndexError) as e:
            raise VersionError(f"Invalid version string: {e}")

    def __lt__(self, other: "Version") -> bool:
        """Compare versions."""
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __eq__(self, other: "Version") -> bool:
        """Check version equality."""
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

class VersionManager:
    """Manage version compatibility and validation."""
    
    VERSION_FILE = "version.json"
    COMPATIBILITY_FILE = "compatibility.json"
    
    def __init__(self, root_dir: Optional[Path] = None):
        """Initialize version manager."""
        self.root_dir = root_dir or Path(__file__).parent.parent
        self.versions = self._load_versions()
        self.compatibility = self._load_compatibility()
        
    def _load_versions(self) -> Dict[str, Version]:
        """Load component versions from version.json."""
        version_path = self.root_dir / self.VERSION_FILE
        if not version_path.exists():
            raise VersionError(f"Version file not found: {version_path}")
            
        with open(version_path) as f:
            version_data = json.load(f)
            
        return {
            component: Version.from_string(ver_str)
            for component, ver_str in version_data.items()
        }
        
    def _load_compatibility(self) -> Dict[str, List[str]]:
        """Load compatibility requirements."""
        compat_path = self.root_dir / self.COMPATIBILITY_FILE
        if not compat_path.exists():
            raise VersionError(f"Compatibility file not found: {compat_path}")
            
        with open(compat_path) as f:
            return json.load(f)
            
    def validate_component(self, component: str, version: str) -> bool:
        """
        Validate component version against compatibility requirements.
        
        Args:
            component: Component name
            version: Version string to validate
            
        Returns:
            bool: True if version is compatible
            
        Raises:
            VersionError: If version is incompatible
        """
        if component not in self.versions:
            raise VersionError(f"Unknown component: {component}")
            
        try:
            current = Version.from_string(version)
            required = self.versions[component]
            
            # Check major version compatibility
            if current.major != required.major:
                raise VersionError(
                    f"Incompatible major version for {component}. "
                    f"Required: {required}, Found: {current}"
                )
                
            # Check minimum minor version
            if current.minor < required.minor:
                raise VersionError(
                    f"Incompatible minor version for {component}. "
                    f"Minimum required: {required}, Found: {current}"
                )
                
            return True
            
        except ValueError as e:
            raise VersionError(f"Invalid version format: {e}")
            
    def check_compatibility(self, component: str) -> bool:
        """
        Check compatibility requirements for component.
        
        Args:
            component: Component to check
            
        Returns:
            bool: True if all dependencies are compatible
        """
        if component not in self.compatibility:
            raise VersionError(f"No compatibility info for: {component}")
            
        for dep in self.compatibility[component]:
            if not self.validate_component(dep, str(self.versions[dep])):
                return False
                
        return True