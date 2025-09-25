"""Comprehensive tests for the Configuration Persistence system.

Tests cover:
- Atomic file operations
- Configuration serialization/deserialization
- Version migration system
- Backup management with rotation and compression
- Concurrent access control
- Data integrity validation
- Change tracking and diff generation
"""
import pytest
import json
import tempfile
import threading
import time
import os
import gzip
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from app.config.settings import Config
from app.config.persistence import (
    ConfigPersistenceManager, AtomicFileWriter, ConfigSerializer,
    ConfigLock, ConfigIntegrity, ConfigDiff, BackupManager, BackupInfo
)
from app.config.types import (
    ConfigVersion, Change, ChangeType,
    PersistenceError, AtomicWriteError, LockError, MigrationError, BackupError
)
from app.config.migrations.migrator import ConfigMigrator, Migration


class TestAtomicFileWriter:
    """Test atomic file write operations."""

    def test_atomic_write_success(self, tmp_path):
        """Test successful atomic write operation."""
        writer = AtomicFileWriter(tmp_path)
        target_file = tmp_path / "test_config.json"
        test_data = {"key": "value", "number": 42}

        success = writer.write_atomic(target_file, test_data)

        assert success
        assert target_file.exists()

        with open(target_file, 'r') as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data

    def test_atomic_write_with_validation(self, tmp_path):
        """Test atomic write with validation function."""
        writer = AtomicFileWriter(tmp_path)
        target_file = tmp_path / "validated_config.json"
        test_data = {"required_field": "value"}

        def validate_data(data):
            return "required_field" in data

        success = writer.write_atomic(target_file, test_data, validate_data)
        assert success

        # Test validation failure
        invalid_data = {"wrong_field": "value"}
        with pytest.raises(AtomicWriteError):
            writer.write_atomic(target_file, invalid_data, validate_data)

    def test_atomic_write_permission_error(self, tmp_path):
        """Test atomic write handles permission errors."""
        writer = AtomicFileWriter(tmp_path)
        target_file = tmp_path / "readonly_config.json"
        test_data = {"key": "value"}

        # Make directory read-only (on systems that support it)
        if os.name != 'nt':  # Skip on Windows
            os.chmod(tmp_path, 0o444)
            try:
                with pytest.raises(AtomicWriteError):
                    writer.write_atomic(target_file, test_data)
            finally:
                os.chmod(tmp_path, 0o755)  # Restore permissions

    def test_atomic_write_performance(self, tmp_path):
        """Test atomic write performance meets requirements (<100ms)."""
        writer = AtomicFileWriter(tmp_path)
        target_file = tmp_path / "performance_test.json"
        test_data = {"large_data": "x" * 10000, "numbers": list(range(1000))}

        start_time = time.time()
        success = writer.write_atomic(target_file, test_data)
        duration_ms = (time.time() - start_time) * 1000

        assert success
        assert duration_ms < 100  # Should complete in <100ms


class TestConfigSerializer:
    """Test configuration serialization and validation."""

    def test_serialize_config(self):
        """Test Config object serialization."""
        config = Config()
        config.debug = True
        config.target_fps = 60

        data = ConfigSerializer.serialize(config)

        assert isinstance(data, dict)
        assert data["debug"] == True
        assert data["target_fps"] == 60
        assert "_schema_version" in data
        assert "_saved_at" in data
        assert "_has_secure_api_key" not in data

    def test_deserialize_config(self):
        """Test dictionary deserialization to Config."""
        data = {
            "debug": True,
            "target_fps": 60,
            "extra_field": "extra_value",
            "_schema_version": "1.0.0"
        }

        config = ConfigSerializer.deserialize(data)

        assert isinstance(config, Config)
        assert config.debug == True
        assert config.target_fps == 60
        assert config.extra.get("extra_field") == "extra_value"

    def test_validate_schema_valid(self):
        """Test schema validation with valid data."""
        valid_data = {
            "debug": True,
            "target_fps": 60,
            "gemini_api_key": "test_key"
        }

        assert ConfigSerializer.validate_schema(valid_data)

    def test_validate_schema_invalid(self):
        """Test schema validation with invalid data."""
        # Empty dict
        assert not ConfigSerializer.validate_schema({})

        # Non-dict
        assert not ConfigSerializer.validate_schema("not a dict")

        # Dict with no recognizable config fields
        assert not ConfigSerializer.validate_schema({"random": "data"})


class TestConfigVersion:
    """Test configuration version handling."""

    def test_version_creation(self):
        """Test ConfigVersion creation and string representation."""
        version = ConfigVersion(1, 2, 3)
        assert str(version) == "1.2.3"

    def test_version_comparison(self):
        """Test version comparison operations."""
        v1 = ConfigVersion(1, 0, 0)
        v2 = ConfigVersion(1, 1, 0)
        v3 = ConfigVersion(2, 0, 0)

        assert v1 < v2
        assert v2 < v3
        assert v1 != v2
        assert v1 == ConfigVersion(1, 0, 0)

    def test_version_from_string(self):
        """Test version creation from string."""
        version = ConfigVersion.from_string("1.2.3")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

        with pytest.raises(ValueError):
            ConfigVersion.from_string("invalid")


class TestConfigLock:
    """Test concurrent access control."""

    def test_write_lock_exclusive(self, tmp_path):
        """Test exclusive write lock prevents concurrent access."""
        lock_file = tmp_path / "test.lock"
        lock1 = ConfigLock(lock_file, timeout=0.1)
        lock2 = ConfigLock(lock_file, timeout=0.1)

        acquired_count = 0

        def try_acquire():
            nonlocal acquired_count
            try:
                with lock2.acquire_write_lock():
                    acquired_count += 1
                    time.sleep(0.05)
            except LockError:
                pass

        # First lock should succeed
        with lock1.acquire_write_lock():
            # Start second thread that should fail to acquire
            thread = threading.Thread(target=try_acquire)
            thread.start()
            time.sleep(0.05)  # Let thread try to acquire
            thread.join()

        # Only first lock should have succeeded
        assert acquired_count == 0

    def test_lock_timeout(self, tmp_path):
        """Test lock timeout behavior."""
        lock_file = tmp_path / "timeout_test.lock"
        lock = ConfigLock(lock_file, timeout=0.01)  # Very short timeout

        # This should work
        with lock.acquire_write_lock():
            pass

        # Test timeout with concurrent access
        def hold_lock():
            with lock.acquire_write_lock():
                time.sleep(0.1)

        thread = threading.Thread(target=hold_lock)
        thread.start()
        time.sleep(0.01)  # Let thread acquire lock

        start_time = time.time()
        with pytest.raises(LockError):
            with lock.acquire_write_lock():
                pass

        duration = time.time() - start_time
        assert duration < 0.05  # Should timeout quickly

        thread.join()

    def test_lock_performance(self, tmp_path):
        """Test lock acquisition performance (<10ms)."""
        lock_file = tmp_path / "performance.lock"
        lock = ConfigLock(lock_file)

        start_time = time.time()
        with lock.acquire_write_lock():
            pass
        duration_ms = (time.time() - start_time) * 1000

        assert duration_ms < 10  # Should acquire in <10ms


class TestConfigIntegrity:
    """Test data integrity validation and security."""

    def test_calculate_checksum(self):
        """Test checksum calculation."""
        data1 = {"key": "value", "number": 42}
        data2 = {"key": "value", "number": 42}
        data3 = {"key": "different", "number": 42}

        checksum1 = ConfigIntegrity.calculate_checksum(data1)
        checksum2 = ConfigIntegrity.calculate_checksum(data2)
        checksum3 = ConfigIntegrity.calculate_checksum(data3)

        assert checksum1 == checksum2  # Same data, same checksum
        assert checksum1 != checksum3  # Different data, different checksum
        assert len(checksum1) == 64  # SHA-256 hex length

    def test_verify_integrity_valid(self, tmp_path):
        """Test integrity verification with valid file."""
        config_file = tmp_path / "valid_config.json"
        valid_data = {"debug": True, "target_fps": 60}

        with open(config_file, 'w') as f:
            json.dump(valid_data, f)

        assert ConfigIntegrity.verify_integrity(config_file)

    def test_verify_integrity_invalid(self, tmp_path):
        """Test integrity verification with invalid file."""
        # Non-existent file
        assert not ConfigIntegrity.verify_integrity(tmp_path / "nonexistent.json")

        # Invalid JSON
        invalid_file = tmp_path / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("invalid json content")

        assert not ConfigIntegrity.verify_integrity(invalid_file)

    def test_encrypt_sensitive(self):
        """Test sensitive data encryption/clearing."""
        data = {
            "gemini_api_key": "secret_key",
            "password": "secret_pass",
            "normal_field": "normal_value"
        }

        encrypted = ConfigIntegrity.encrypt_sensitive(data)

        assert encrypted["gemini_api_key"] == ""
        assert encrypted["password"] == ""
        assert encrypted["normal_field"] == "normal_value"

    def test_audit_change(self, tmp_path):
        """Test audit trail recording."""
        audit_file = tmp_path / "audit.log"
        changes = [
            Change("key1", ChangeType.ADDED, None, "value1"),
            Change("key2", ChangeType.MODIFIED, "old", "new")
        ]

        ConfigIntegrity.audit_change("test_user", changes, audit_file)

        assert audit_file.exists()
        with open(audit_file, 'r') as f:
            audit_entry = json.loads(f.readline())

        assert audit_entry["user"] == "test_user"
        assert len(audit_entry["changes"]) == 2
        assert audit_entry["changes"][0]["key"] == "key1"


class TestConfigDiff:
    """Test configuration change detection and merging."""

    def test_diff_no_changes(self):
        """Test diff with identical configurations."""
        config1 = Config()
        config2 = Config()

        changes = ConfigDiff.diff(config1, config2)
        assert len(changes) == 0

    def test_diff_with_changes(self):
        """Test diff with configuration changes."""
        config1 = Config()
        config1.debug = False
        config1.target_fps = 30

        config2 = Config()
        config2.debug = True
        config2.target_fps = 60
        config2.extra = {"new_field": "new_value"}

        changes = ConfigDiff.diff(config1, config2)

        # Find specific changes
        debug_change = next((c for c in changes if c.key == "debug"), None)
        fps_change = next((c for c in changes if c.key == "target_fps"), None)

        assert debug_change is not None
        assert debug_change.change_type == ChangeType.MODIFIED
        assert debug_change.old_value == False
        assert debug_change.new_value == True

        assert fps_change is not None
        assert fps_change.change_type == ChangeType.MODIFIED

    def test_apply_diff(self):
        """Test applying changes to configuration."""
        base_config = Config()
        base_config.debug = False

        changes = [
            Change("debug", ChangeType.MODIFIED, False, True),
            Change("new_field", ChangeType.ADDED, None, "new_value")
        ]

        modified_config = ConfigDiff.apply_diff(base_config, changes)

        assert modified_config.debug == True
        assert modified_config.extra.get("new_field") == "new_value"

    def test_generate_changelog(self):
        """Test changelog generation."""
        changes = [
            Change("field1", ChangeType.ADDED, None, "value1"),
            Change("field2", ChangeType.MODIFIED, "old", "new"),
            Change("field3", ChangeType.REMOVED, "removed_value", None)
        ]

        changelog = ConfigDiff.generate_changelog(changes)

        assert "Added field1: value1" in changelog
        assert "Modified field2: old → new" in changelog
        assert "Removed field3: removed_value" in changelog


class TestBackupManager:
    """Test backup management with rotation and compression."""

    def test_create_backup(self, tmp_path):
        """Test backup creation."""
        backup_dir = tmp_path / "backups"
        manager = BackupManager(backup_dir)

        config = Config()
        config.debug = True
        version = ConfigVersion(1, 0, 0)

        start_time = time.time()
        backup_info = manager.create_backup(config, version)
        duration_ms = (time.time() - start_time) * 1000

        assert backup_info is not None
        assert backup_info.path.exists()
        assert backup_info.version == version
        assert backup_info.checksum != ""
        assert duration_ms < 50  # Should complete in <50ms

    def test_list_backups(self, tmp_path):
        """Test backup listing."""
        backup_dir = tmp_path / "backups"
        manager = BackupManager(backup_dir)

        # Create multiple backups
        config = Config()
        version = ConfigVersion(1, 0, 0)

        backup1 = manager.create_backup(config, version)
        time.sleep(0.01)  # Ensure different timestamps
        backup2 = manager.create_backup(config, version)

        backups = manager.list_backups()

        assert len(backups) == 2
        assert backups[0].timestamp > backups[1].timestamp  # Sorted by newest first

    def test_restore_backup(self, tmp_path):
        """Test backup restoration."""
        backup_dir = tmp_path / "backups"
        manager = BackupManager(backup_dir)

        # Create config with specific values
        original_config = Config()
        original_config.debug = True
        original_config.target_fps = 60

        version = ConfigVersion(1, 0, 0)
        backup_info = manager.create_backup(original_config, version)

        # Restore backup
        restored_config = manager.restore_backup(backup_info.id)

        assert restored_config.debug == True
        assert restored_config.target_fps == 60

    def test_backup_compression(self, tmp_path):
        """Test backup compression for large files."""
        backup_dir = tmp_path / "backups"
        manager = BackupManager(backup_dir)

        # Create config with large data to trigger compression
        config = Config()
        config.extra = {"large_data": "x" * 20000}  # >10KB to trigger compression

        version = ConfigVersion(1, 0, 0)
        backup_info = manager.create_backup(config, version)

        assert backup_info.compressed
        assert backup_info.path.suffix == ".gz"
        assert backup_info.path.exists()

        # Should still be restorable
        restored_config = manager.restore_backup(backup_info.id)
        assert restored_config.extra["large_data"] == "x" * 20000

    def test_cleanup_old_backups(self, tmp_path):
        """Test backup rotation and cleanup."""
        backup_dir = tmp_path / "backups"
        manager = BackupManager(backup_dir, max_backups=3)

        config = Config()
        version = ConfigVersion(1, 0, 0)

        # Create 5 backups
        backup_infos = []
        for i in range(5):
            backup_info = manager.create_backup(config, version)
            backup_infos.append(backup_info)
            time.sleep(0.01)

        # Cleanup should keep only 3 newest
        manager.cleanup_old_backups()

        remaining_backups = manager.list_backups()
        assert len(remaining_backups) == 3

        # Oldest 2 should be removed
        assert not backup_infos[0].path.exists()
        assert not backup_infos[1].path.exists()
        assert backup_infos[2].path.exists()


class TestConfigMigrator:
    """Test configuration migration system."""

    def test_migration_registry(self):
        """Test migration registration and path finding."""
        migrator = ConfigMigrator()

        # Should have built-in migrations
        migrations = migrator.get_migrations()
        assert len(migrations) > 0

        # Test migration path finding
        assert migrator.can_migrate("1.0.0", "1.1.0")
        assert migrator.can_migrate("1.0.0", "1.2.0")

    def test_perform_migration(self):
        """Test actual migration execution."""
        migrator = ConfigMigrator()

        # Test data that would need migration
        old_data = {
            "debug": True,
            "target_fps": 60,
            "_schema_version": "1.0.0"
        }

        # Migrate to 1.1.0
        migrated_data = migrator.migrate(old_data, "1.0.0", "1.1.0")

        assert migrated_data["_schema_version"] == "1.1.0"
        assert "workflow_performance_monitoring" in migrated_data
        assert "cache_analysis_results" in migrated_data
        assert migrated_data["workflow_performance_monitoring"] == True

    def test_migration_error_handling(self):
        """Test migration error handling."""
        migrator = ConfigMigrator()

        # Test invalid version
        with pytest.raises(ValueError):
            migrator.migrate({}, "invalid", "1.1.0")

        # Test no migration path (should raise MigrationError)
        with pytest.raises(MigrationError):
            migrator.migrate({}, "1.0.0", "99.0.0")


class TestConfigPersistenceManager:
    """Test main persistence manager integration."""

    def test_save_and_load_config(self, tmp_path):
        """Test complete save and load cycle."""
        config_file = tmp_path / "test_config.json"
        manager = ConfigPersistenceManager(config_file)

        # Create test config
        config = Config()
        config.debug = True
        config.target_fps = 60

        # Save config
        success = manager.save_config(config)
        assert success
        assert config_file.exists()

        # Load config
        loaded_config = manager.load_config()
        assert loaded_config.debug == True
        assert loaded_config.target_fps == 60

    def test_config_with_backup(self, tmp_path):
        """Test config save with automatic backup."""
        config_file = tmp_path / "test_config.json"
        manager = ConfigPersistenceManager(config_file)

        # Save initial config
        config1 = Config()
        config1.debug = False
        manager.save_config(config1)

        # Save modified config (should create backup)
        config2 = Config()
        config2.debug = True
        manager.save_config(config2, create_backup=True)

        # Check backup was created
        backups = manager.get_backup_info()
        assert len(backups) >= 1

    def test_automatic_migration(self, tmp_path):
        """Test automatic migration during load."""
        config_file = tmp_path / "migration_test.json"

        # Create old format config
        old_data = {
            "debug": True,
            "target_fps": 60,
            "_schema_version": "1.0.0"
        }

        with open(config_file, 'w') as f:
            json.dump(old_data, f)

        # Load should trigger migration
        manager = ConfigPersistenceManager(config_file)
        manager.current_version = ConfigVersion(1, 1, 0)  # Newer version

        loaded_config = manager.load_config()

        # Should have migrated and saved
        assert hasattr(loaded_config, 'debug')
        assert loaded_config.debug == True

        # Check file was updated with new version
        with open(config_file, 'r') as f:
            updated_data = json.load(f)
        assert updated_data.get("_schema_version") == "1.1.0"

    def test_corruption_recovery(self, tmp_path):
        """Test automatic recovery from corruption."""
        config_file = tmp_path / "corrupt_config.json"
        manager = ConfigPersistenceManager(config_file)

        # Create a good backup first
        good_config = Config()
        good_config.debug = True
        manager.save_config(good_config)

        # Corrupt the main config file
        with open(config_file, 'w') as f:
            f.write("corrupted json content")

        # Recovery should work
        recovered_config = manager.recover_from_corruption()
        assert recovered_config is not None
        assert recovered_config.debug == True

    def test_force_migration(self, tmp_path):
        """Test forced migration to specific version."""
        config_file = tmp_path / "force_migration_test.json"
        manager = ConfigPersistenceManager(config_file)

        # Create config with old version
        config = Config()
        config.debug = True
        manager.save_config(config)

        # Force migrate to newer version
        success = manager.force_migration("1.2.0")
        assert success

        # Verify migration occurred
        info = manager.get_config_info()
        assert info["schema_version"] == "1.2.0"

    def test_concurrent_access(self, tmp_path):
        """Test concurrent access to config file."""
        config_file = tmp_path / "concurrent_test.json"
        manager1 = ConfigPersistenceManager(config_file)
        manager2 = ConfigPersistenceManager(config_file)

        results = []

        def save_config(manager, value):
            try:
                config = Config()
                config.debug = value
                success = manager.save_config(config)
                results.append(success)
            except Exception as e:
                results.append(False)

        # Try concurrent saves
        thread1 = threading.Thread(target=save_config, args=(manager1, True))
        thread2 = threading.Thread(target=save_config, args=(manager2, False))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        # At least one should succeed
        assert any(results)

    def test_performance_requirements(self, tmp_path):
        """Test performance meets requirements."""
        config_file = tmp_path / "performance_test.json"
        manager = ConfigPersistenceManager(config_file)

        config = Config()
        config.debug = True

        # Test save performance (<100ms)
        start_time = time.time()
        success = manager.save_config(config)
        save_duration_ms = (time.time() - start_time) * 1000

        assert success
        assert save_duration_ms < 100

        # Test load performance
        start_time = time.time()
        loaded_config = manager.load_config()
        load_duration_ms = (time.time() - start_time) * 1000

        assert loaded_config is not None
        assert load_duration_ms < 100

    def test_get_config_info(self, tmp_path):
        """Test configuration information retrieval."""
        config_file = tmp_path / "info_test.json"
        manager = ConfigPersistenceManager(config_file)

        # Save config first
        config = Config()
        manager.save_config(config)

        info = manager.get_config_info()

        assert info["file_exists"] == True
        assert info["is_valid"] == True
        assert info["schema_version"] == "1.0.0"
        assert info["file_size"] > 0
        assert "last_modified" in info


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])