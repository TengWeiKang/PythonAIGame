"""Demonstration of the Configuration Persistence System.

This script shows how to use the comprehensive configuration persistence features:
- Atomic save/load operations
- Automatic versioning and migration
- Backup management with rotation
- Change tracking and diff generation
- Concurrent access control
- Data integrity validation
"""
import os
import time
import logging
from pathlib import Path

from app.config.settings import Config, load_config, save_config
from app.config.persistence import (
    ConfigPersistenceManager, ConfigVersion, ConfigDiff,
    BackupManager, Change, ChangeType
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_basic_persistence():
    """Demonstrate basic persistence operations."""
    print("\n=== Basic Persistence Demo ===")

    # Create persistence manager
    config_path = Path("demo_config.json")
    persistence_manager = ConfigPersistenceManager(config_path)

    # Create a sample configuration
    config = Config()
    config.debug = True
    config.target_fps = 60
    config.camera_width = 1920
    config.camera_height = 1080
    config.gemini_api_key = "demo_key_123"

    print(f"Saving configuration to {config_path}")
    success = persistence_manager.save_config(config, user="demo_user")
    print(f"Save successful: {success}")

    # Load configuration back
    print("Loading configuration...")
    loaded_config = persistence_manager.load_config()
    print(f"Loaded config - Debug: {loaded_config.debug}, FPS: {loaded_config.target_fps}")

    # Get configuration info
    info = persistence_manager.get_config_info()
    print(f"Config info: {info}")

    return persistence_manager


def demo_versioning_and_migration():
    """Demonstrate version migration system."""
    print("\n=== Versioning and Migration Demo ===")

    config_path = Path("migration_demo_config.json")

    # Create old format configuration (version 1.0.0)
    old_config_data = {
        "debug": True,
        "target_fps": 30,
        "camera_width": 1280,
        "camera_height": 720,
        "_schema_version": "1.0.0"
    }

    # Save old format
    with open(config_path, 'w') as f:
        import json
        json.dump(old_config_data, f, indent=2)

    print(f"Created old format config (v1.0.0)")

    # Create persistence manager with newer version
    persistence_manager = ConfigPersistenceManager(config_path)
    persistence_manager.set_schema_version(ConfigVersion(1, 2, 0))  # Newer version

    print("Loading config (should trigger automatic migration)...")

    # Load should trigger migration
    try:
        migrated_config = persistence_manager.load_config()
        print(f"Migration successful! New version: {persistence_manager.get_schema_version()}")

        # Check if new fields were added during migration
        print(f"Has performance monitoring: {hasattr(migrated_config, 'workflow_performance_monitoring')}")

    except Exception as e:
        print(f"Migration failed: {e}")

    # Demonstrate force migration
    print("\nForcing migration to specific version...")
    success = persistence_manager.force_migration("1.2.0", user="migration_demo")
    print(f"Force migration successful: {success}")

    return persistence_manager


def demo_backup_management():
    """Demonstrate backup management features."""
    print("\n=== Backup Management Demo ===")

    config_path = Path("backup_demo_config.json")
    backup_dir = Path("demo_backups")

    persistence_manager = ConfigPersistenceManager(config_path, backup_dir)

    # Create initial configuration
    config = Config()
    config.debug = False
    config.target_fps = 30

    print("Saving initial configuration...")
    persistence_manager.save_config(config, user="backup_demo")

    # Make several changes to create backups
    for i in range(5):
        config.target_fps = 30 + (i * 10)
        config.camera_width = 1280 + (i * 160)

        print(f"Updating config (iteration {i+1}): FPS={config.target_fps}, Width={config.camera_width}")
        persistence_manager.save_config(config, user="backup_demo", create_backup=True)
        time.sleep(0.1)  # Small delay for different timestamps

    # List available backups
    backups = persistence_manager.get_backup_info()
    print(f"\nAvailable backups: {len(backups)}")

    for backup in backups[:3]:  # Show first 3
        print(f"  - {backup.id}: {backup.timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
              f"(v{backup.version}, {backup.size_bytes} bytes)")

    # Restore from backup
    if backups:
        backup_to_restore = backups[2].id  # Restore 3rd backup
        print(f"\nRestoring from backup: {backup_to_restore}")

        success = persistence_manager.restore_from_backup(backup_to_restore, user="backup_demo")
        print(f"Restore successful: {success}")

        # Verify restoration
        restored_config = persistence_manager.load_config()
        print(f"Restored config: FPS={restored_config.target_fps}, Width={restored_config.camera_width}")

    return persistence_manager


def demo_change_tracking():
    """Demonstrate configuration change tracking."""
    print("\n=== Change Tracking Demo ===")

    # Create two different configurations
    config1 = Config()
    config1.debug = False
    config1.target_fps = 30
    config1.camera_width = 1280
    config1.gemini_model = "gemini-1.5-flash"

    config2 = Config()
    config2.debug = True  # Changed
    config2.target_fps = 60  # Changed
    config2.camera_width = 1280  # Same
    config2.camera_height = 1080  # New field
    config2.gemini_model = "gemini-1.5-pro"  # Changed
    # Remove gemini_api_key (if it was in config1)

    # Generate diff
    print("Generating configuration diff...")
    changes = ConfigDiff.diff(config1, config2)

    print(f"Found {len(changes)} changes:")
    for change in changes:
        if change.change_type == ChangeType.ADDED:
            print(f"  + Added {change.key}: {change.new_value}")
        elif change.change_type == ChangeType.MODIFIED:
            print(f"  * Modified {change.key}: {change.old_value} → {change.new_value}")
        elif change.change_type == ChangeType.REMOVED:
            print(f"  - Removed {change.key}: {change.old_value}")

    # Generate changelog
    changelog = ConfigDiff.generate_changelog(changes)
    print(f"\nGenerated changelog:\n{changelog}")

    # Apply changes to create config3
    print("\nApplying changes to create new configuration...")
    config3 = ConfigDiff.apply_diff(config1, changes)
    print(f"Applied config: Debug={config3.debug}, FPS={config3.target_fps}")

    return changes


def demo_concurrent_access():
    """Demonstrate concurrent access control."""
    print("\n=== Concurrent Access Demo ===")

    import threading
    import concurrent.futures

    config_path = Path("concurrent_demo_config.json")

    def worker_save_config(worker_id, iterations=5):
        """Worker function that saves configurations."""
        persistence_manager = ConfigPersistenceManager(config_path)
        results = []

        for i in range(iterations):
            try:
                config = Config()
                config.debug = worker_id % 2 == 0  # Alternate true/false
                config.target_fps = 30 + (worker_id * 10) + i
                config.extra = {f"worker_{worker_id}": f"iteration_{i}"}

                success = persistence_manager.save_config(
                    config,
                    user=f"worker_{worker_id}",
                    create_backup=(i == 0)  # Create backup only on first iteration
                )
                results.append(success)

                if success:
                    print(f"Worker {worker_id}: Saved config (iteration {i+1})")
                else:
                    print(f"Worker {worker_id}: Failed to save config (iteration {i+1})")

                time.sleep(0.01)  # Small delay

            except Exception as e:
                print(f"Worker {worker_id}: Error - {e}")
                results.append(False)

        return results

    print("Starting concurrent access test with 4 workers...")

    # Run multiple workers concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker_save_config, i, 3) for i in range(4)]

        # Wait for all workers to complete
        results = []
        for future in concurrent.futures.as_completed(futures):
            worker_results = future.result()
            results.extend(worker_results)

    success_count = sum(results)
    total_attempts = len(results)

    print(f"\nConcurrent access test completed:")
    print(f"  Total attempts: {total_attempts}")
    print(f"  Successful saves: {success_count}")
    print(f"  Success rate: {success_count/total_attempts*100:.1f}%")

    # Load final configuration
    persistence_manager = ConfigPersistenceManager(config_path)
    final_config = persistence_manager.load_config()
    print(f"Final config: Debug={final_config.debug}, FPS={final_config.target_fps}")


def demo_data_integrity():
    """Demonstrate data integrity features."""
    print("\n=== Data Integrity Demo ===")

    config_path = Path("integrity_demo_config.json")
    persistence_manager = ConfigPersistenceManager(config_path)

    # Create configuration with sensitive data
    config = Config()
    config.debug = True
    config.gemini_api_key = "super_secret_api_key_123"
    config.target_fps = 60

    print("Saving configuration with sensitive data...")
    persistence_manager.save_config(config, user="integrity_demo")

    # Check file integrity
    print("Verifying file integrity...")
    is_valid = persistence_manager.validate_config_file()
    print(f"Configuration file is valid: {is_valid}")

    # Load and check if sensitive data is handled properly
    loaded_config = persistence_manager.load_config()
    print(f"Loaded API key: {'*' * len(loaded_config.gemini_api_key) if loaded_config.gemini_api_key else '[empty]'}")

    # Demonstrate corruption recovery
    print("\nSimulating file corruption...")

    # Corrupt the config file
    with open(config_path, 'w') as f:
        f.write("corrupted json data {invalid")

    print("Attempting to load corrupted configuration...")
    try:
        corrupted_config = persistence_manager.load_config()
        print("Unexpected: corrupted config loaded successfully")
    except Exception as e:
        print(f"Expected: loading corrupted config failed - {e}")

    # Attempt recovery
    print("Attempting automatic recovery...")
    recovered_config = persistence_manager.recover_from_corruption()

    if recovered_config:
        print(f"Recovery successful! Recovered config: Debug={recovered_config.debug}")
    else:
        print("Recovery failed - no valid backups available")


def main():
    """Run all persistence system demonstrations."""
    print("Configuration Persistence System Demonstration")
    print("=" * 50)

    try:
        # Run demonstrations
        demo_basic_persistence()
        demo_versioning_and_migration()
        demo_backup_management()
        demo_change_tracking()
        demo_concurrent_access()
        demo_data_integrity()

        print("\n" + "=" * 50)
        print("Demonstration completed successfully!")

        # Cleanup demo files
        cleanup_demo_files()
        print("Demo files cleaned up.")

    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


def cleanup_demo_files():
    """Clean up demonstration files."""
    demo_files = [
        "demo_config.json",
        "migration_demo_config.json",
        "backup_demo_config.json",
        "concurrent_demo_config.json",
        "integrity_demo_config.json"
    ]

    demo_dirs = [
        "demo_backups",
        "config_backups"
    ]

    import shutil

    # Remove demo files
    for file_path in demo_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

    # Remove demo directories
    for dir_path in demo_dirs:
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
        except Exception:
            pass

    # Remove lock files
    for file_path in demo_files:
        lock_file = file_path + ".lock"
        try:
            if os.path.exists(lock_file):
                os.remove(lock_file)
        except Exception:
            pass


if __name__ == "__main__":
    main()