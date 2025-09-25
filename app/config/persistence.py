"""Configuration Persistence Manager with atomic operations, versioning, and backup management.

This module provides a comprehensive configuration persistence system that ensures:
- Atomic write operations for data integrity
- Automatic versioning and schema migration
- Backup management with rotation and compression
- Concurrent access control with file locking
- Data integrity validation and security
- Change tracking and diff generation

Performance targets:
- Save operation: <100ms
- Backup creation: <50ms
- Migration execution: <200ms
- Lock acquisition: <10ms

Safety features:
- Never lose user configuration
- Always maintain valid state
- Automatic recovery from corruption
- Transaction log for debugging
"""
from __future__ import annotations

import json
import os
import time
import tempfile
import shutil
import hashlib
import threading
import gzip
try:
    import fcntl
except ImportError:
    fcntl = None  # Not available on Windows
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union, Callable, Protocol
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager

from .settings import Config
from .types import (
    ConfigVersion, Change, ChangeType,
    PersistenceError, AtomicWriteError, LockError,
    MigrationError, BackupError
)


@dataclass(slots=True)
class BackupInfo:
    """Information about a configuration backup."""
    id: str
    path: Path
    timestamp: datetime
    version: ConfigVersion
    compressed: bool = False
    checksum: str = ""
    size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'path': str(self.path),
            'timestamp': self.timestamp.isoformat(),
            'version': str(self.version),
            'compressed': self.compressed,
            'checksum': self.checksum,
            'size_bytes': self.size_bytes
        }


class AtomicFileWriter:
    """Provides atomic file write operations to prevent data corruption."""

    def __init__(self, temp_dir: Optional[Path] = None):
        self.temp_dir = temp_dir or Path(tempfile.gettempdir())
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def write_atomic(self, target_path: Path, data: Dict[str, Any],
                    validate_func: Optional[Callable[[Dict], bool]] = None) -> bool:
        """Write data to file atomically.

        Args:
            target_path: Final path for the file
            data: Data to write
            validate_func: Optional validation function

        Returns:
            bool: True if write was successful

        Raises:
            AtomicWriteError: If atomic write fails
        """
        start_time = time.time()
        temp_path = None

        try:
            # Create temporary file in same directory as target for atomic rename
            target_dir = target_path.parent
            target_dir.mkdir(parents=True, exist_ok=True)

            # Create temporary file with similar name
            temp_path = target_dir / f".{target_path.name}.tmp.{int(time.time() * 1000000)}"

            # Write to temporary file
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Validate if function provided
            if validate_func:
                with open(temp_path, 'r', encoding='utf-8') as f:
                    test_data = json.load(f)
                if not validate_func(test_data):
                    raise AtomicWriteError("Data validation failed after write")

            # Atomic rename (only operation that's guaranteed atomic on most filesystems)
            if os.name == 'nt':  # Windows
                # On Windows, need to remove target first
                if target_path.exists():
                    target_path.unlink()

            temp_path.rename(target_path)

            duration_ms = (time.time() - start_time) * 1000
            logging.debug(f"Atomic write completed in {duration_ms:.1f}ms")

            return True

        except Exception as e:
            # Cleanup temp file on failure
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

            raise AtomicWriteError(f"Atomic write failed: {e}") from e


class ConfigSerializer:
    """Handles configuration serialization/deserialization with validation."""

    @staticmethod
    def serialize(config: Config) -> Dict[str, Any]:
        """Serialize Config object to dictionary."""
        try:
            data = config.to_dict()

            # Remove internal metadata that shouldn't be persisted
            data.pop('_has_secure_api_key', None)
            data.pop('_environment_config', None)

            # Add metadata
            data['_schema_version'] = "1.0.0"
            data['_saved_at'] = datetime.now().isoformat()

            return data

        except Exception as e:
            raise PersistenceError(f"Failed to serialize config: {e}") from e

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> Config:
        """Deserialize dictionary to Config object."""
        try:
            # Remove metadata before creating Config
            clean_data = data.copy()
            clean_data.pop('_schema_version', None)
            clean_data.pop('_saved_at', None)

            # Extract extra keys not in Config schema
            config_fields = set(Config.__annotations__.keys())
            extra = {k: v for k, v in clean_data.items()
                    if k not in config_fields and not k.startswith('_')}

            # Filter to only known Config fields
            config_data = {k: v for k, v in clean_data.items()
                          if k in config_fields and not k.startswith('_')}

            return Config(**config_data, extra=extra)

        except Exception as e:
            raise PersistenceError(f"Failed to deserialize config: {e}") from e

    @staticmethod
    def validate_schema(data: Dict[str, Any]) -> bool:
        """Validate configuration data against schema."""
        try:
            # Basic structure validation
            if not isinstance(data, dict):
                return False

            # Check for required fields (at minimum, should have some config fields)
            config_fields = set(Config.__annotations__.keys())
            data_fields = set(data.keys())

            # Should have at least some overlap with config fields
            if not (data_fields & config_fields):
                return False

            # Try to deserialize to validate
            ConfigSerializer.deserialize(data)
            return True

        except Exception:
            return False


class ConfigLock:
    """Provides file locking for concurrent access control."""

    def __init__(self, lock_file: Path, timeout: float = 10.0):
        self.lock_file = lock_file
        self.timeout = timeout
        self._lock_fd = None
        self._thread_lock = threading.RLock()

    @contextmanager
    def acquire_write_lock(self):
        """Acquire exclusive write lock."""
        acquired = False
        start_time = time.time()

        try:
            with self._thread_lock:
                # Create lock file if it doesn't exist
                self.lock_file.parent.mkdir(parents=True, exist_ok=True)
                self.lock_file.touch()

                self._lock_fd = open(self.lock_file, 'w')

                # Try to acquire lock with timeout
                while time.time() - start_time < self.timeout:
                    try:
                        if os.name == 'nt':  # Windows
                            import msvcrt
                            msvcrt.locking(self._lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
                        else:  # Unix-like
                            if fcntl:
                                fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            else:
                                # Fallback for systems without fcntl
                                pass

                        acquired = True
                        break
                    except (OSError, IOError):
                        time.sleep(0.01)  # 10ms

                if not acquired:
                    raise LockError(f"Failed to acquire write lock within {self.timeout}s")

                yield

        finally:
            if self._lock_fd:
                try:
                    if os.name == 'nt':
                        import msvcrt
                        msvcrt.locking(self._lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        if fcntl:
                            fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_UN)
                    self._lock_fd.close()
                except Exception:
                    pass
                finally:
                    self._lock_fd = None

    @contextmanager
    def acquire_read_lock(self):
        """Acquire shared read lock."""
        acquired = False
        start_time = time.time()

        try:
            with self._thread_lock:
                self.lock_file.parent.mkdir(parents=True, exist_ok=True)
                self.lock_file.touch()

                self._lock_fd = open(self.lock_file, 'r')

                while time.time() - start_time < self.timeout:
                    try:
                        if os.name == 'nt':  # Windows - use exclusive for simplicity
                            import msvcrt
                            msvcrt.locking(self._lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
                        else:  # Unix-like
                            if fcntl:
                                fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                            else:
                                # Fallback for systems without fcntl
                                pass

                        acquired = True
                        break
                    except (OSError, IOError):
                        time.sleep(0.01)

                if not acquired:
                    raise LockError(f"Failed to acquire read lock within {self.timeout}s")

                yield

        finally:
            if self._lock_fd:
                try:
                    if os.name == 'nt':
                        import msvcrt
                        msvcrt.locking(self._lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        if fcntl:
                            fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_UN)
                    self._lock_fd.close()
                except Exception:
                    pass
                finally:
                    self._lock_fd = None


class ConfigIntegrity:
    """Handles data integrity validation and security."""

    @staticmethod
    def calculate_checksum(data: Dict[str, Any]) -> str:
        """Calculate SHA-256 checksum of configuration data."""
        try:
            # Serialize data in deterministic way for consistent checksum
            serialized = json.dumps(data, sort_keys=True, separators=(',', ':'))
            return hashlib.sha256(serialized.encode('utf-8')).hexdigest()
        except Exception as e:
            raise PersistenceError(f"Failed to calculate checksum: {e}") from e

    @staticmethod
    def verify_integrity(config_path: Path) -> bool:
        """Verify configuration file integrity."""
        try:
            if not config_path.exists():
                return False

            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Basic validation
            return ConfigSerializer.validate_schema(data)

        except Exception:
            return False

    @staticmethod
    def encrypt_sensitive(data: Dict[str, Any], sensitive_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Encrypt sensitive configuration values."""
        if sensitive_keys is None:
            sensitive_keys = ['gemini_api_key', 'api_key', 'password', 'secret']

        result = data.copy()

        for key in sensitive_keys:
            if key in result and result[key]:
                # For now, just clear sensitive data - encryption would require key management
                result[key] = ""
                logging.debug(f"Cleared sensitive key '{key}' for persistence")

        return result

    @staticmethod
    def audit_change(user: str, changes: List[Change], audit_file: Optional[Path] = None) -> None:
        """Record configuration changes for audit trail."""
        if audit_file is None:
            return

        try:
            audit_entry = {
                'timestamp': datetime.now().isoformat(),
                'user': user,
                'changes': [change.to_dict() for change in changes]
            }

            # Append to audit log
            audit_file.parent.mkdir(parents=True, exist_ok=True)

            with open(audit_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(audit_entry) + '\n')

        except Exception as e:
            logging.error(f"Failed to write audit log: {e}")


class ConfigDiff:
    """Handles configuration change detection and merging."""

    @staticmethod
    def diff(old: Config, new: Config) -> List[Change]:
        """Generate list of changes between two configurations."""
        changes = []

        old_dict = old.to_dict()
        new_dict = new.to_dict()

        all_keys = set(old_dict.keys()) | set(new_dict.keys())

        for key in all_keys:
            old_val = old_dict.get(key)
            new_val = new_dict.get(key)

            if key not in old_dict:
                changes.append(Change(key, ChangeType.ADDED, None, new_val))
            elif key not in new_dict:
                changes.append(Change(key, ChangeType.REMOVED, old_val, None))
            elif old_val != new_val:
                changes.append(Change(key, ChangeType.MODIFIED, old_val, new_val))

        return changes

    @staticmethod
    def apply_diff(base: Config, changes: List[Change]) -> Config:
        """Apply changes to base configuration."""
        base_dict = base.to_dict()

        for change in changes:
            if change.change_type == ChangeType.ADDED or change.change_type == ChangeType.MODIFIED:
                base_dict[change.key] = change.new_value
            elif change.change_type == ChangeType.REMOVED:
                base_dict.pop(change.key, None)

        return ConfigSerializer.deserialize(base_dict)

    @staticmethod
    def generate_changelog(changes: List[Change]) -> str:
        """Generate human-readable changelog from changes."""
        if not changes:
            return "No changes"

        changelog = ["Configuration Changes:"]

        for change in changes:
            if change.change_type == ChangeType.ADDED:
                changelog.append(f"  + Added {change.key}: {change.new_value}")
            elif change.change_type == ChangeType.MODIFIED:
                changelog.append(f"  * Modified {change.key}: {change.old_value} → {change.new_value}")
            elif change.change_type == ChangeType.REMOVED:
                changelog.append(f"  - Removed {change.key}: {change.old_value}")

        return "\n".join(changelog)


class BackupManager:
    """Manages configuration backups with rotation and compression."""

    def __init__(self, backup_dir: Path, max_backups: int = 10):
        self.backup_dir = backup_dir
        self.max_backups = max_backups
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(self, config: Config, version: ConfigVersion) -> BackupInfo:
        """Create a backup of the current configuration."""
        start_time = time.time()

        try:
            timestamp = datetime.now()
            backup_id = f"config_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            backup_path = self.backup_dir / f"{backup_id}.json"

            # Serialize configuration
            config_data = ConfigSerializer.serialize(config)

            # Write backup
            writer = AtomicFileWriter()
            writer.write_atomic(backup_path, config_data)

            # Calculate size and checksum
            size_bytes = backup_path.stat().st_size
            checksum = ConfigIntegrity.calculate_checksum(config_data)

            # Create backup info
            backup_info = BackupInfo(
                id=backup_id,
                path=backup_path,
                timestamp=timestamp,
                version=version,
                compressed=False,
                checksum=checksum,
                size_bytes=size_bytes
            )

            # Compress if backup is large (>10KB)
            if size_bytes > 10240:
                backup_info = self._compress_backup(backup_info)

            # Save backup metadata
            self._save_backup_metadata(backup_info)

            duration_ms = (time.time() - start_time) * 1000
            logging.debug(f"Backup created in {duration_ms:.1f}ms: {backup_id}")

            return backup_info

        except Exception as e:
            raise BackupError(f"Failed to create backup: {e}") from e

    def list_backups(self) -> List[BackupInfo]:
        """List all available backups."""
        try:
            metadata_file = self.backup_dir / "backup_metadata.json"
            if not metadata_file.exists():
                return []

            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            backups = []
            for backup_data in metadata.get('backups', []):
                backup_info = BackupInfo(
                    id=backup_data['id'],
                    path=Path(backup_data['path']),
                    timestamp=datetime.fromisoformat(backup_data['timestamp']),
                    version=ConfigVersion.from_string(backup_data['version']),
                    compressed=backup_data.get('compressed', False),
                    checksum=backup_data.get('checksum', ''),
                    size_bytes=backup_data.get('size_bytes', 0)
                )
                backups.append(backup_info)

            # Sort by timestamp (newest first)
            return sorted(backups, key=lambda b: b.timestamp, reverse=True)

        except Exception as e:
            logging.error(f"Failed to list backups: {e}")
            return []

    def restore_backup(self, backup_id: str) -> Config:
        """Restore configuration from backup."""
        try:
            backups = self.list_backups()
            backup_info = next((b for b in backups if b.id == backup_id), None)

            if not backup_info:
                raise BackupError(f"Backup '{backup_id}' not found")

            if not backup_info.path.exists():
                raise BackupError(f"Backup file not found: {backup_info.path}")

            # Load backup data
            if backup_info.compressed:
                with gzip.open(backup_info.path, 'rt', encoding='utf-8') as f:
                    config_data = json.load(f)
            else:
                with open(backup_info.path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

            # Verify integrity
            checksum = ConfigIntegrity.calculate_checksum(config_data)
            if backup_info.checksum and checksum != backup_info.checksum:
                raise BackupError(f"Backup integrity check failed for '{backup_id}'")

            # Deserialize and return
            return ConfigSerializer.deserialize(config_data)

        except Exception as e:
            raise BackupError(f"Failed to restore backup '{backup_id}': {e}") from e

    def cleanup_old_backups(self, keep_count: int = None) -> None:
        """Remove old backups, keeping only the specified number."""
        if keep_count is None:
            keep_count = self.max_backups

        try:
            backups = self.list_backups()

            if len(backups) <= keep_count:
                return

            # Remove oldest backups
            to_remove = backups[keep_count:]

            for backup in to_remove:
                try:
                    if backup.path.exists():
                        backup.path.unlink()
                    logging.debug(f"Removed old backup: {backup.id}")
                except Exception as e:
                    logging.error(f"Failed to remove backup {backup.id}: {e}")

            # Update metadata
            remaining_backups = backups[:keep_count]
            metadata = {
                'backups': [backup.to_dict() for backup in remaining_backups],
                'last_cleanup': datetime.now().isoformat()
            }

            metadata_file = self.backup_dir / "backup_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logging.error(f"Failed to cleanup old backups: {e}")

    def _compress_backup(self, backup_info: BackupInfo) -> BackupInfo:
        """Compress backup file to save space."""
        try:
            compressed_path = backup_info.path.with_suffix('.json.gz')

            with open(backup_info.path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Remove original
            backup_info.path.unlink()

            # Update backup info
            backup_info.path = compressed_path
            backup_info.compressed = True
            backup_info.size_bytes = compressed_path.stat().st_size

            return backup_info

        except Exception as e:
            logging.error(f"Failed to compress backup: {e}")
            return backup_info

    def _save_backup_metadata(self, backup_info: BackupInfo) -> None:
        """Save backup metadata to index file."""
        try:
            metadata_file = self.backup_dir / "backup_metadata.json"

            # Load existing metadata
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                metadata = {'backups': []}

            # Add new backup
            metadata['backups'].append(backup_info.to_dict())

            # Sort by timestamp (newest first)
            metadata['backups'].sort(key=lambda b: b['timestamp'], reverse=True)

            # Save updated metadata
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logging.error(f"Failed to save backup metadata: {e}")


class ConfigPersistenceManager:
    """Main configuration persistence manager with atomic operations and versioning."""

    def __init__(self, config_path: Path, backup_dir: Optional[Path] = None,
                 audit_file: Optional[Path] = None):
        self.config_path = config_path
        self.backup_dir = backup_dir or (config_path.parent / "config_backups")
        self.audit_file = audit_file
        self.current_version = ConfigVersion(1, 0, 0)

        # Initialize components
        self.writer = AtomicFileWriter()
        self.backup_manager = BackupManager(self.backup_dir)
        self.lock = ConfigLock(config_path.with_suffix('.lock'))

        # Ensure directories exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: Config, user: str = "system",
                   create_backup: bool = True) -> bool:
        """Save configuration with atomic operations and optional backup.

        Args:
            config: Configuration to save
            user: User making the change (for audit trail)
            create_backup: Whether to create backup before saving

        Returns:
            bool: True if save was successful

        Raises:
            PersistenceError: If save operation fails
        """
        start_time = time.time()

        try:
            with self.lock.acquire_write_lock():
                # Load current config for diff if it exists
                old_config = None
                if self.config_path.exists():
                    try:
                        old_config = self.load_config()
                    except Exception as e:
                        logging.warning(f"Failed to load current config for diff: {e}")

                # Create backup if requested and current config exists
                if create_backup and old_config:
                    try:
                        backup_info = self.backup_manager.create_backup(old_config, self.current_version)
                        logging.info(f"Created backup: {backup_info.id}")
                    except BackupError as e:
                        logging.error(f"Backup creation failed: {e}")
                        # Continue with save even if backup fails

                # Serialize configuration
                config_data = ConfigSerializer.serialize(config)

                # Add metadata
                config_data['_schema_version'] = str(self.current_version)
                config_data['_checksum'] = ConfigIntegrity.calculate_checksum(config_data)

                # Atomic write
                success = self.writer.write_atomic(
                    self.config_path,
                    config_data,
                    ConfigSerializer.validate_schema
                )

                if success and old_config:
                    # Generate and audit changes
                    changes = ConfigDiff.diff(old_config, config)
                    if changes and self.audit_file:
                        ConfigIntegrity.audit_change(user, changes, self.audit_file)

                    if changes:
                        changelog = ConfigDiff.generate_changelog(changes)
                        logging.info(f"Configuration updated:\n{changelog}")

                # Cleanup old backups
                self.backup_manager.cleanup_old_backups()

                duration_ms = (time.time() - start_time) * 1000
                logging.debug(f"Config save completed in {duration_ms:.1f}ms")

                return success

        except Exception as e:
            raise PersistenceError(f"Failed to save configuration: {e}") from e

    def load_config(self) -> Config:
        """Load configuration with integrity checking.

        Returns:
            Config: Loaded configuration

        Raises:
            PersistenceError: If load operation fails
        """
        try:
            with self.lock.acquire_read_lock():
                if not self.config_path.exists():
                    raise PersistenceError(f"Configuration file not found: {self.config_path}")

                # Verify integrity
                if not ConfigIntegrity.verify_integrity(self.config_path):
                    raise PersistenceError("Configuration file integrity check failed")

                # Load and deserialize
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                # Verify checksum if present
                stored_checksum = config_data.pop('_checksum', None)
                if stored_checksum:
                    calculated_checksum = ConfigIntegrity.calculate_checksum(config_data)
                    if stored_checksum != calculated_checksum:
                        logging.warning("Configuration checksum mismatch - possible corruption")

                # Check schema version for migration
                schema_version = config_data.get('_schema_version', '1.0.0')
                if schema_version != str(self.current_version):
                    logging.info(f"Schema version mismatch: {schema_version} vs {self.current_version}")

                    # Perform automatic migration
                    try:
                        from .migrations.migrator import get_migrator
                        migrator = get_migrator()
                        if migrator.can_migrate(schema_version, str(self.current_version)):
                            logging.info(f"Starting automatic migration from {schema_version} to {self.current_version}")

                            # Create backup before migration
                            backup_config = ConfigSerializer.deserialize(config_data.copy())
                            backup_info = self.backup_manager.create_backup(
                                backup_config,
                                ConfigVersion.from_string(schema_version)
                            )
                            logging.info(f"Pre-migration backup created: {backup_info.id}")

                            # Perform migration
                            config_data = migrator.migrate(config_data, schema_version, str(self.current_version))

                            # Save migrated configuration immediately
                            migrated_config = ConfigSerializer.deserialize(config_data)
                            self.save_config(migrated_config, "migration_system", create_backup=False)

                            logging.info(f"Configuration successfully migrated to version {self.current_version}")
                        else:
                            logging.warning(f"No migration path available from {schema_version} to {self.current_version}")
                    except Exception as e:
                        logging.error(f"Migration failed: {e}")
                        # Continue with loading the old version - may cause compatibility issues
                        logging.warning("Continuing with potentially incompatible configuration")

                return ConfigSerializer.deserialize(config_data)

        except Exception as e:
            raise PersistenceError(f"Failed to load configuration: {e}") from e

    def get_backup_info(self) -> List[BackupInfo]:
        """Get information about available backups."""
        return self.backup_manager.list_backups()

    def restore_from_backup(self, backup_id: str, user: str = "system") -> bool:
        """Restore configuration from backup.

        Args:
            backup_id: ID of backup to restore
            user: User performing restore

        Returns:
            bool: True if restore was successful
        """
        try:
            # Load backup
            restored_config = self.backup_manager.restore_backup(backup_id)

            # Save as current configuration
            success = self.save_config(restored_config, user, create_backup=True)

            if success:
                logging.info(f"Configuration restored from backup: {backup_id}")

            return success

        except Exception as e:
            logging.error(f"Failed to restore from backup '{backup_id}': {e}")
            return False

    def recover_from_corruption(self) -> Optional[Config]:
        """Attempt to recover from configuration corruption using latest backup.

        Returns:
            Config or None: Recovered configuration if successful
        """
        try:
            backups = self.backup_manager.list_backups()

            if not backups:
                logging.error("No backups available for recovery")
                return None

            # Try to restore from most recent backup
            for backup in backups[:3]:  # Try up to 3 most recent backups
                try:
                    config = self.backup_manager.restore_backup(backup.id)
                    logging.info(f"Successfully recovered from backup: {backup.id}")
                    return config
                except BackupError as e:
                    logging.warning(f"Failed to recover from backup {backup.id}: {e}")
                    continue

            logging.error("Failed to recover from any available backup")
            return None

        except Exception as e:
            logging.error(f"Recovery attempt failed: {e}")
            return None

    def get_schema_version(self) -> ConfigVersion:
        """Get current schema version."""
        return self.current_version

    def set_schema_version(self, version: ConfigVersion) -> None:
        """Set current schema version."""
        self.current_version = version
        logging.info(f"Schema version updated to {version}")

    def validate_config_file(self) -> bool:
        """Validate the current configuration file integrity.

        Returns:
            bool: True if configuration is valid
        """
        try:
            if not self.config_path.exists():
                return False

            return ConfigIntegrity.verify_integrity(self.config_path)
        except Exception as e:
            logging.error(f"Configuration validation failed: {e}")
            return False

    def get_config_info(self) -> Dict[str, Any]:
        """Get detailed information about the current configuration.

        Returns:
            Dict: Configuration metadata and status
        """
        try:
            info = {
                'config_path': str(self.config_path),
                'backup_dir': str(self.backup_dir),
                'schema_version': str(self.current_version),
                'file_exists': self.config_path.exists(),
                'is_valid': self.validate_config_file(),
                'backups_available': len(self.backup_manager.list_backups()),
                'last_modified': None,
                'file_size': 0
            }

            if self.config_path.exists():
                stat = self.config_path.stat()
                info['last_modified'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
                info['file_size'] = stat.st_size

            return info

        except Exception as e:
            logging.error(f"Failed to get config info: {e}")
            return {'error': str(e)}

    def force_migration(self, target_version: str, user: str = "system") -> bool:
        """Force migration to a specific version.

        Args:
            target_version: Target schema version
            user: User performing the migration

        Returns:
            bool: True if migration was successful
        """
        try:
            if not self.config_path.exists():
                raise PersistenceError("Configuration file does not exist")

            # Load current config
            current_config = self.load_config()
            current_data = ConfigSerializer.serialize(current_config)
            current_version = current_data.get('_schema_version', '1.0.0')

            if current_version == target_version:
                logging.info(f"Configuration is already at version {target_version}")
                return True

            # Perform migration
            from .migrations.migrator import get_migrator
            migrator = get_migrator()
            if not migrator.can_migrate(current_version, target_version):
                raise MigrationError(f"No migration path from {current_version} to {target_version}")

            # Create backup
            backup_info = self.backup_manager.create_backup(
                current_config,
                ConfigVersion.from_string(current_version)
            )
            logging.info(f"Pre-migration backup created: {backup_info.id}")

            # Migrate
            migrated_data = migrator.migrate(current_data, current_version, target_version)
            migrated_config = ConfigSerializer.deserialize(migrated_data)

            # Save migrated configuration
            success = self.save_config(migrated_config, user, create_backup=False)

            if success:
                logging.info(f"Force migration completed: {current_version} -> {target_version}")
            else:
                raise PersistenceError("Failed to save migrated configuration")

            return success

        except Exception as e:
            logging.error(f"Force migration failed: {e}")
            return False


__all__ = [
    'ConfigPersistenceManager', 'BackupManager', 'ConfigVersion', 'Change', 'BackupInfo',
    'AtomicFileWriter', 'ConfigSerializer', 'ConfigLock', 'ConfigIntegrity',
    'ConfigDiff', 'PersistenceError', 'AtomicWriteError', 'LockError',
    'MigrationError', 'BackupError', 'ChangeType'
]