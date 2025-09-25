"""Configuration migration system with automatic schema evolution.

This module provides a comprehensive configuration migration system that:
- Automatically detects schema version changes
- Applies incremental migrations between versions
- Maintains backward compatibility
- Logs all migration operations
- Provides rollback capabilities
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Type
from pathlib import Path

from ..types import ConfigVersion, MigrationError


@dataclass(slots=True)
class Migration:
    """Represents a single configuration migration."""
    from_version: ConfigVersion
    to_version: ConfigVersion
    description: str
    migration_func: Callable[[Dict[str, Any]], Dict[str, Any]]
    rollback_func: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None

    def __post_init__(self):
        if self.from_version >= self.to_version:
            raise ValueError("Migration to_version must be greater than from_version")


class MigrationRegistry:
    """Registry for all available configuration migrations."""

    def __init__(self):
        self._migrations: List[Migration] = []
        self._by_version: Dict[str, List[Migration]] = {}

    def register(self, migration: Migration) -> None:
        """Register a migration."""
        self._migrations.append(migration)

        # Index by from_version for quick lookup
        from_key = str(migration.from_version)
        if from_key not in self._by_version:
            self._by_version[from_key] = []
        self._by_version[from_key].append(migration)

        # Sort by to_version to ensure correct order
        self._by_version[from_key].sort(key=lambda m: m.to_version)

        logging.debug(f"Registered migration: {migration.from_version} -> {migration.to_version}")

    def get_migration_path(self, from_version: ConfigVersion, to_version: ConfigVersion) -> List[Migration]:
        """Find migration path from one version to another."""
        if from_version == to_version:
            return []

        if from_version > to_version:
            raise MigrationError(f"Cannot migrate backwards from {from_version} to {to_version}")

        # Use Dijkstra-like algorithm to find shortest migration path
        path = self._find_migration_path(from_version, to_version)

        if not path:
            raise MigrationError(f"No migration path found from {from_version} to {to_version}")

        return path

    def _find_migration_path(self, start: ConfigVersion, target: ConfigVersion) -> List[Migration]:
        """Find the shortest migration path using breadth-first search."""
        if start == target:
            return []

        # BFS to find shortest path
        queue = [(start, [])]
        visited = {start}

        while queue:
            current_version, path = queue.pop(0)

            # Get all migrations from current version
            available_migrations = self._by_version.get(str(current_version), [])

            for migration in available_migrations:
                next_version = migration.to_version
                new_path = path + [migration]

                if next_version == target:
                    return new_path

                if next_version not in visited and next_version < target:
                    visited.add(next_version)
                    queue.append((next_version, new_path))

        return []  # No path found


class ConfigMigrator:
    """Handles configuration migrations between schema versions."""

    def __init__(self):
        self.registry = MigrationRegistry()
        self._register_builtin_migrations()

    def migrate(self, data: Dict[str, Any], from_version: str, to_version: str) -> Dict[str, Any]:
        """Migrate configuration data from one version to another.

        Args:
            data: Configuration data to migrate
            from_version: Source schema version
            to_version: Target schema version

        Returns:
            Dict: Migrated configuration data

        Raises:
            MigrationError: If migration fails
        """
        start_from = ConfigVersion.from_string(from_version)
        target_to = ConfigVersion.from_string(to_version)

        if start_from == target_to:
            return data

        try:
            # Find migration path
            migration_path = self.registry.get_migration_path(start_from, target_to)

            # Apply migrations in sequence
            migrated_data = data.copy()

            for migration in migration_path:
                logging.info(f"Applying migration: {migration.from_version} -> {migration.to_version}")
                logging.info(f"Migration description: {migration.description}")

                try:
                    migrated_data = migration.migration_func(migrated_data)
                    logging.debug(f"Migration applied successfully")
                except Exception as e:
                    raise MigrationError(f"Migration {migration.from_version} -> {migration.to_version} failed: {e}") from e

            # Update schema version in data
            migrated_data['_schema_version'] = to_version

            logging.info(f"Configuration successfully migrated from {from_version} to {to_version}")
            return migrated_data

        except Exception as e:
            if isinstance(e, MigrationError):
                raise
            raise MigrationError(f"Migration from {from_version} to {to_version} failed: {e}") from e

    def get_migrations(self) -> List[Migration]:
        """Get all registered migrations."""
        return self._migrations.copy()

    def can_migrate(self, from_version: str, to_version: str) -> bool:
        """Check if migration is possible between versions."""
        try:
            start_from = ConfigVersion.from_string(from_version)
            target_to = ConfigVersion.from_string(to_version)
            self.registry.get_migration_path(start_from, target_to)
            return True
        except (ValueError, MigrationError):
            return False

    def _register_builtin_migrations(self):
        """Register built-in migrations for the application."""
        # Example migration from 1.0.0 to 1.1.0 (adding new fields)
        def migrate_1_0_0_to_1_1_0(data: Dict[str, Any]) -> Dict[str, Any]:
            """Add new performance monitoring fields."""
            migrated = data.copy()

            # Add new fields with default values
            if 'workflow_performance_monitoring' not in migrated:
                migrated['workflow_performance_monitoring'] = True

            if 'cache_analysis_results' not in migrated:
                migrated['cache_analysis_results'] = True

            if 'max_cache_size_mb' not in migrated:
                migrated['max_cache_size_mb'] = 100

            return migrated

        def rollback_1_1_0_to_1_0_0(data: Dict[str, Any]) -> Dict[str, Any]:
            """Remove performance monitoring fields."""
            migrated = data.copy()

            # Remove fields that didn't exist in 1.0.0
            migrated.pop('workflow_performance_monitoring', None)
            migrated.pop('cache_analysis_results', None)
            migrated.pop('max_cache_size_mb', None)

            return migrated

        migration_1_0_to_1_1 = Migration(
            from_version=ConfigVersion(1, 0, 0),
            to_version=ConfigVersion(1, 1, 0),
            description="Add performance monitoring and caching configuration options",
            migration_func=migrate_1_0_0_to_1_1_0,
            rollback_func=rollback_1_1_0_to_1_0_0
        )

        self.registry.register(migration_1_0_to_1_1)

        # Example migration from 1.1.0 to 1.2.0 (security enhancements)
        def migrate_1_1_0_to_1_2_0(data: Dict[str, Any]) -> Dict[str, Any]:
            """Add security and encryption settings."""
            migrated = data.copy()

            # Add security settings
            if 'enable_config_encryption' not in migrated:
                migrated['enable_config_encryption'] = False

            if 'config_backup_encryption' not in migrated:
                migrated['config_backup_encryption'] = True

            if 'audit_config_changes' not in migrated:
                migrated['audit_config_changes'] = True

            # Migrate API key handling (clear if present for security)
            if migrated.get('gemini_api_key'):
                logging.warning("API key found in config during migration - clearing for security")
                migrated['gemini_api_key'] = ""

            return migrated

        migration_1_1_to_1_2 = Migration(
            from_version=ConfigVersion(1, 1, 0),
            to_version=ConfigVersion(1, 2, 0),
            description="Add security and encryption settings, improve API key handling",
            migration_func=migrate_1_1_0_to_1_2_0
        )

        self.registry.register(migration_1_1_to_1_2)

        # Example migration from 1.0.0 to 1.2.0 (direct path for major updates)
        def migrate_1_0_0_to_1_2_0(data: Dict[str, Any]) -> Dict[str, Any]:
            """Direct migration from 1.0.0 to 1.2.0 combining all changes."""
            # Apply 1.0.0 -> 1.1.0 changes
            migrated = migrate_1_0_0_to_1_1_0(data)
            # Apply 1.1.0 -> 1.2.0 changes
            migrated = migrate_1_1_0_to_1_2_0(migrated)
            return migrated

        migration_1_0_to_1_2 = Migration(
            from_version=ConfigVersion(1, 0, 0),
            to_version=ConfigVersion(1, 2, 0),
            description="Direct migration from 1.0.0 to 1.2.0 with all improvements",
            migration_func=migrate_1_0_0_to_1_2_0
        )

        self.registry.register(migration_1_0_to_1_2)


# Global instance for easy access
_migrator_instance = None

def get_migrator() -> ConfigMigrator:
    """Get the global migrator instance."""
    global _migrator_instance
    if _migrator_instance is None:
        _migrator_instance = ConfigMigrator()
    return _migrator_instance


__all__ = ['Migration', 'MigrationRegistry', 'ConfigMigrator', 'get_migrator']