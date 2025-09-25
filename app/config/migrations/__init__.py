"""Configuration migration system.

Handles automatic migration of configuration files between schema versions.
"""

from .migrator import ConfigMigrator, Migration, MigrationRegistry

__all__ = ['ConfigMigrator', 'Migration', 'MigrationRegistry']