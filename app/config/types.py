"""Shared types and classes for configuration system.

This module contains shared types used across the configuration persistence system
to avoid circular imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Callable, Dict
from datetime import datetime
from enum import Enum


class ChangeType(Enum):
    """Types of configuration changes."""
    ADDED = "added"
    MODIFIED = "modified"
    REMOVED = "removed"


@dataclass(slots=True)
class ConfigVersion:
    """Configuration schema version with semantic versioning."""
    major: int = 1
    minor: int = 0
    patch: int = 0

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other: 'ConfigVersion') -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __eq__(self, other: 'ConfigVersion') -> bool:
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    @classmethod
    def from_string(cls, version_str: str) -> 'ConfigVersion':
        """Create ConfigVersion from string like '1.2.3'."""
        try:
            parts = version_str.split('.')
            return cls(int(parts[0]), int(parts[1]), int(parts[2]))
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid version string '{version_str}': {e}")


@dataclass(slots=True)
class Change:
    """Represents a single configuration change."""
    key: str
    change_type: ChangeType
    old_value: Any = None
    new_value: Any = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'key': self.key,
            'change_type': self.change_type.value,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'timestamp': self.timestamp.isoformat()
        }


class PersistenceError(Exception):
    """Base exception for configuration persistence errors."""
    pass


class AtomicWriteError(PersistenceError):
    """Exception raised when atomic write operations fail."""
    pass


class LockError(PersistenceError):
    """Exception raised when file locking fails."""
    pass


class MigrationError(PersistenceError):
    """Exception raised during configuration migration."""
    pass


class BackupError(PersistenceError):
    """Exception raised during backup operations."""
    pass


__all__ = [
    'ConfigVersion', 'Change', 'ChangeType',
    'PersistenceError', 'AtomicWriteError', 'LockError',
    'MigrationError', 'BackupError'
]