"""Enhanced configuration management with validation, monitoring, and hot-reloading.

This module provides a comprehensive configuration management system that integrates
with the base service architecture to provide:
- Real-time configuration validation
- Configuration change monitoring and hot-reloading
- Configuration versioning and rollback
- Security auditing and compliance checking
- Performance impact analysis
"""
from __future__ import annotations

import json
import logging
import threading
import time
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from ..core.base_service import BaseService
from ..core.exceptions import ConfigError, ValidationError, ServiceError
from ..core.performance import performance_timer
from .settings import Config, load_config, save_config
from .validation import SettingsValidator, ValidationResult
from ..utils.validation import InputValidator, SecurityValidator


@dataclass
class ConfigChange:
    """Represents a configuration change."""
    timestamp: datetime
    field_name: str
    old_value: Any
    new_value: Any
    user_id: Optional[str] = None
    source: str = "manual"  # manual, api, file, environment


@dataclass
class ConfigVersion:
    """Represents a configuration version."""
    version: int
    timestamp: datetime
    config_data: Dict[str, Any]
    changes: List[ConfigChange]
    is_valid: bool
    validation_errors: List[str] = field(default_factory=list)


class ConfigurationFileHandler(FileSystemEventHandler):
    """Handles configuration file changes for hot-reloading."""

    def __init__(self, config_manager: 'EnhancedConfigManager'):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__ + ".FileHandler")

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if file_path.name == "config.json":
            self.logger.info(f"Configuration file modified: {file_path}")
            self.config_manager._handle_file_change(str(file_path))


class EnhancedConfigManager(BaseService):
    """Enhanced configuration manager with validation, monitoring, and hot-reloading.

    Features:
    - Real-time configuration validation
    - Hot-reloading of configuration changes
    - Configuration versioning and history
    - Security auditing and compliance
    - Performance monitoring for config changes
    - Automatic backup and rollback capabilities
    """

    def __init__(self,
                 config_path: str = "config.json",
                 enable_hot_reload: bool = True,
                 enable_versioning: bool = True,
                 max_versions: int = 10,
                 auto_backup: bool = True,
                 **kwargs):
        """Initialize enhanced configuration manager.

        Args:
            config_path: Path to configuration file
            enable_hot_reload: Whether to enable hot-reloading
            enable_versioning: Whether to enable versioning
            max_versions: Maximum number of versions to keep
            auto_backup: Whether to automatically backup configs
            **kwargs: Additional base service arguments
        """
        super().__init__(service_name="ConfigManager", **kwargs)

        self._config_path = Path(config_path)
        self._enable_hot_reload = enable_hot_reload
        self._enable_versioning = enable_versioning
        self._max_versions = max_versions
        self._auto_backup = auto_backup

        # Configuration state
        self._current_config: Optional[Config] = None
        self._config_lock = threading.RLock()

        # Versioning and history
        self._versions: List[ConfigVersion] = []
        self._current_version = 0
        self._changes_log: List[ConfigChange] = []

        # Hot-reloading
        self._file_observer: Optional[Observer] = None
        self._file_handler: Optional[ConfigurationFileHandler] = None
        self._last_file_mtime = 0.0

        # Change notifications
        self._change_listeners: Set[Callable[[ConfigChange], None]] = set()
        self._validation_listeners: Set[Callable[[Dict[str, ValidationResult]], None]] = set()

        # Security and audit
        self._security_violations: List[Dict[str, Any]] = []
        self._compliance_checks_enabled = True

        # Performance tracking
        self._load_times: List[float] = []
        self._validation_times: List[float] = []

    def _initialize(self) -> None:
        """Initialize configuration manager."""
        self.logger.info("Initializing enhanced configuration manager")

        # Create backup directory
        self._backup_dir = self._config_path.parent / "config_backups"
        self._backup_dir.mkdir(exist_ok=True)

        # Load initial configuration
        self._load_initial_config()

        # Set up file monitoring if enabled
        if self._enable_hot_reload:
            self._setup_file_monitoring()

        self.logger.info("Enhanced configuration manager initialized")

    def _shutdown(self) -> None:
        """Shutdown configuration manager."""
        self.logger.info("Shutting down configuration manager")

        # Stop file monitoring
        if self._file_observer:
            self._file_observer.stop()
            self._file_observer.join()

        # Save final backup
        if self._auto_backup and self._current_config:
            self._create_backup("shutdown")

        self.logger.info("Configuration manager shutdown completed")

    def _health_check(self) -> bool:
        """Perform health check."""
        base_healthy = super()._health_check()

        # Check if configuration is loaded and valid
        config_healthy = self._current_config is not None

        # Check if file monitoring is working (if enabled)
        monitoring_healthy = True
        if self._enable_hot_reload:
            monitoring_healthy = (self._file_observer is not None and
                                self._file_observer.is_alive())

        return base_healthy and config_healthy and monitoring_healthy

    def _get_health_details(self) -> Dict[str, Any]:
        """Get detailed health information."""
        details = super()._get_health_details()
        details.update({
            'config_loaded': self._current_config is not None,
            'config_path': str(self._config_path),
            'hot_reload_enabled': self._enable_hot_reload,
            'monitoring_active': (self._file_observer is not None and
                                self._file_observer.is_alive()) if self._enable_hot_reload else False,
            'current_version': self._current_version,
            'total_versions': len(self._versions),
            'total_changes': len(self._changes_log),
            'security_violations': len(self._security_violations)
        })
        return details

    @performance_timer("config_load")
    def load_configuration(self, reload: bool = False) -> Config:
        """Load configuration with validation and caching.

        Args:
            reload: Whether to reload from file even if already loaded

        Returns:
            Loaded and validated configuration

        Raises:
            ConfigError: If configuration cannot be loaded
        """
        with self.operation_context("load_configuration"):
            with self._config_lock:
                if self._current_config and not reload:
                    return self._current_config

                start_time = time.time()

                try:
                    # Load configuration from file
                    config = load_config(str(self._config_path))

                    # Validate configuration
                    validation_results = self._validate_configuration(config.to_dict())

                    # Check for validation errors
                    errors = [r.error_message for r in validation_results.values()
                            if not r.is_valid and r.error_message]

                    if errors:
                        self.logger.warning(f"Configuration validation issues: {errors}")

                    # Apply auto-corrections
                    corrected_config = self._apply_corrections(config.to_dict(), validation_results)
                    if corrected_config != config.to_dict():
                        self.logger.info("Applied automatic configuration corrections")
                        config = Config(**{k: corrected_config[k] for k in Config.__annotations__
                                         if k in corrected_config})

                    # Security audit
                    if self._compliance_checks_enabled:
                        self._perform_security_audit(config)

                    # Store configuration
                    self._current_config = config

                    # Track loading time
                    load_time = time.time() - start_time
                    self._load_times.append(load_time)

                    # Create version if versioning is enabled
                    if self._enable_versioning:
                        self._create_version(corrected_config, [])

                    # Notify validation listeners
                    for listener in self._validation_listeners:
                        try:
                            listener(validation_results)
                        except Exception as e:
                            self.logger.error(f"Error in validation listener: {e}")

                    self.logger.info(f"Configuration loaded successfully in {load_time:.3f}s")
                    return config

                except Exception as e:
                    self.logger.error(f"Failed to load configuration: {e}")
                    raise ConfigError(f"Failed to load configuration: {e}") from e

    def save_configuration(self, config: Optional[Config] = None) -> None:
        """Save configuration with backup and validation.

        Args:
            config: Configuration to save (uses current if None)

        Raises:
            ConfigError: If configuration cannot be saved
        """
        with self.operation_context("save_configuration"):
            with self._config_lock:
                config = config or self._current_config
                if not config:
                    raise ConfigError("No configuration to save")

                try:
                    # Validate before saving
                    validation_results = self._validate_configuration(config.to_dict())
                    critical_errors = [r.error_message for r in validation_results.values()
                                     if not r.is_valid and 'critical' in (r.error_message or '').lower()]

                    if critical_errors:
                        raise ConfigError(f"Cannot save configuration with critical errors: {critical_errors}")

                    # Create backup before saving
                    if self._auto_backup:
                        self._create_backup("pre_save")

                    # Save configuration
                    save_config(config, str(self._config_path))

                    self.logger.info("Configuration saved successfully")

                except Exception as e:
                    self.logger.error(f"Failed to save configuration: {e}")
                    raise ConfigError(f"Failed to save configuration: {e}") from e

    def update_setting(self,
                      key: str,
                      value: Any,
                      user_id: Optional[str] = None,
                      source: str = "manual") -> bool:
        """Update a configuration setting with validation.

        Args:
            key: Setting key to update
            value: New value
            user_id: ID of user making the change
            source: Source of the change

        Returns:
            True if update was successful

        Raises:
            ValidationError: If new value is invalid
            ConfigError: If update fails
        """
        with self.operation_context("update_setting"):
            with self._config_lock:
                if not self._current_config:
                    raise ConfigError("No configuration loaded")

                try:
                    # Get current value
                    old_value = getattr(self._current_config, key, None)

                    # Validate new value
                    temp_config = self._current_config.to_dict()
                    temp_config[key] = value

                    validation_results = self._validate_configuration(temp_config)
                    if key in validation_results and not validation_results[key].is_valid:
                        raise ValidationError(f"Invalid value for {key}: {validation_results[key].error_message}")

                    # Apply correction if available
                    if key in validation_results and validation_results[key].corrected_value is not None:
                        value = validation_results[key].corrected_value
                        self.logger.info(f"Auto-corrected {key} value to {value}")

                    # Update configuration
                    setattr(self._current_config, key, value)

                    # Record change
                    change = ConfigChange(
                        timestamp=datetime.now(),
                        field_name=key,
                        old_value=old_value,
                        new_value=value,
                        user_id=user_id,
                        source=source
                    )

                    self._changes_log.append(change)

                    # Create new version if versioning is enabled
                    if self._enable_versioning:
                        self._create_version(self._current_config.to_dict(), [change])

                    # Notify change listeners
                    for listener in self._change_listeners:
                        try:
                            listener(change)
                        except Exception as e:
                            self.logger.error(f"Error in change listener: {e}")

                    self.logger.info(f"Updated setting {key}: {old_value} -> {value}")
                    return True

                except Exception as e:
                    self.logger.error(f"Failed to update setting {key}: {e}")
                    raise

    def get_current_config(self) -> Optional[Config]:
        """Get current configuration.

        Returns:
            Current configuration or None if not loaded
        """
        with self._config_lock:
            return self._current_config

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a configuration setting value.

        Args:
            key: Setting key
            default: Default value if not found

        Returns:
            Setting value or default
        """
        with self._config_lock:
            if not self._current_config:
                return default

            return getattr(self._current_config, key, default)

    def register_change_listener(self, callback: Callable[[ConfigChange], None]) -> None:
        """Register a callback for configuration changes.

        Args:
            callback: Function to call when configuration changes
        """
        self._change_listeners.add(callback)

    def unregister_change_listener(self, callback: Callable[[ConfigChange], None]) -> None:
        """Unregister a configuration change callback.

        Args:
            callback: Function to remove from listeners
        """
        self._change_listeners.discard(callback)

    def register_validation_listener(self, callback: Callable[[Dict[str, ValidationResult]], None]) -> None:
        """Register a callback for validation results.

        Args:
            callback: Function to call when validation is performed
        """
        self._validation_listeners.add(callback)

    def unregister_validation_listener(self, callback: Callable[[Dict[str, ValidationResult]], None]) -> None:
        """Unregister a validation callback.

        Args:
            callback: Function to remove from listeners
        """
        self._validation_listeners.discard(callback)

    def get_version_history(self) -> List[ConfigVersion]:
        """Get configuration version history.

        Returns:
            List of configuration versions
        """
        return self._versions.copy()

    def rollback_to_version(self, version: int) -> bool:
        """Rollback to a specific configuration version.

        Args:
            version: Version number to rollback to

        Returns:
            True if rollback was successful

        Raises:
            ConfigError: If rollback fails
        """
        with self.operation_context("rollback_version"):
            with self._config_lock:
                try:
                    # Find the version
                    target_version = None
                    for v in self._versions:
                        if v.version == version:
                            target_version = v
                            break

                    if not target_version:
                        raise ConfigError(f"Version {version} not found")

                    if not target_version.is_valid:
                        raise ConfigError(f"Version {version} is invalid and cannot be restored")

                    # Create backup before rollback
                    if self._auto_backup:
                        self._create_backup(f"pre_rollback_to_{version}")

                    # Restore configuration
                    self._current_config = Config(**{k: target_version.config_data[k]
                                                   for k in Config.__annotations__
                                                   if k in target_version.config_data})

                    # Save to file
                    save_config(self._current_config, str(self._config_path))

                    # Record rollback as a change
                    change = ConfigChange(
                        timestamp=datetime.now(),
                        field_name="__rollback__",
                        old_value=self._current_version,
                        new_value=version,
                        source="rollback"
                    )

                    self._changes_log.append(change)
                    self._current_version = version

                    # Notify listeners
                    for listener in self._change_listeners:
                        try:
                            listener(change)
                        except Exception as e:
                            self.logger.error(f"Error in change listener: {e}")

                    self.logger.info(f"Rolled back to configuration version {version}")
                    return True

                except Exception as e:
                    self.logger.error(f"Failed to rollback to version {version}: {e}")
                    raise ConfigError(f"Failed to rollback to version {version}: {e}") from e

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get configuration performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        avg_load_time = sum(self._load_times) / len(self._load_times) if self._load_times else 0
        avg_validation_time = sum(self._validation_times) / len(self._validation_times) if self._validation_times else 0

        return {
            'total_loads': len(self._load_times),
            'average_load_time_ms': avg_load_time * 1000,
            'total_validations': len(self._validation_times),
            'average_validation_time_ms': avg_validation_time * 1000,
            'total_changes': len(self._changes_log),
            'current_version': self._current_version,
            'total_versions': len(self._versions),
            'security_violations': len(self._security_violations)
        }

    def export_configuration(self, include_history: bool = False) -> Dict[str, Any]:
        """Export configuration for backup or migration.

        Args:
            include_history: Whether to include version history

        Returns:
            Dictionary with configuration data
        """
        with self._config_lock:
            export_data = {
                'current_config': self._current_config.to_dict() if self._current_config else None,
                'current_version': self._current_version,
                'export_timestamp': datetime.now().isoformat(),
                'manager_version': '1.0'
            }

            if include_history:
                export_data['version_history'] = [
                    {
                        'version': v.version,
                        'timestamp': v.timestamp.isoformat(),
                        'config_data': v.config_data,
                        'is_valid': v.is_valid,
                        'validation_errors': v.validation_errors
                    }
                    for v in self._versions
                ]

                export_data['changes_log'] = [
                    {
                        'timestamp': c.timestamp.isoformat(),
                        'field_name': c.field_name,
                        'old_value': str(c.old_value),
                        'new_value': str(c.new_value),
                        'user_id': c.user_id,
                        'source': c.source
                    }
                    for c in self._changes_log
                ]

            return export_data

    def _load_initial_config(self) -> None:
        """Load initial configuration."""
        try:
            self.load_configuration()
        except Exception as e:
            self.logger.warning(f"Failed to load initial configuration: {e}")
            # Create default configuration
            self._current_config = Config()
            if self._enable_versioning:
                self._create_version(self._current_config.to_dict(), [])

    def _setup_file_monitoring(self) -> None:
        """Set up file monitoring for hot-reloading."""
        try:
            self._file_handler = ConfigurationFileHandler(self)
            self._file_observer = Observer()
            self._file_observer.schedule(
                self._file_handler,
                str(self._config_path.parent),
                recursive=False
            )
            self._file_observer.start()

            # Record initial file modification time
            if self._config_path.exists():
                self._last_file_mtime = self._config_path.stat().st_mtime

            self.logger.info("File monitoring enabled for hot-reloading")

        except Exception as e:
            self.logger.warning(f"Failed to set up file monitoring: {e}")
            self._enable_hot_reload = False

    def _handle_file_change(self, file_path: str) -> None:
        """Handle configuration file changes."""
        try:
            # Check if file actually changed (avoid duplicate events)
            if self._config_path.exists():
                current_mtime = self._config_path.stat().st_mtime
                if current_mtime <= self._last_file_mtime:
                    return
                self._last_file_mtime = current_mtime

            # Brief delay to ensure file write is complete
            time.sleep(0.1)

            self.logger.info("Configuration file changed, reloading...")
            old_config = self._current_config.to_dict() if self._current_config else {}

            # Reload configuration
            new_config = self.load_configuration(reload=True)

            # Detect changes
            changes = self._detect_changes(old_config, new_config.to_dict())

            # Record changes
            for change in changes:
                self._changes_log.append(change)

                # Notify listeners
                for listener in self._change_listeners:
                    try:
                        listener(change)
                    except Exception as e:
                        self.logger.error(f"Error in change listener: {e}")

            if changes:
                self.logger.info(f"Hot-reloaded configuration with {len(changes)} changes")

        except Exception as e:
            self.logger.error(f"Error handling file change: {e}")

    def _validate_configuration(self, config_dict: Dict[str, Any]) -> Dict[str, ValidationResult]:
        """Validate configuration with timing.

        Args:
            config_dict: Configuration dictionary to validate

        Returns:
            Dictionary of validation results
        """
        start_time = time.time()

        try:
            # Use existing validator
            results = SettingsValidator.validate_all_settings(config_dict)

            # Add custom security validations
            self._add_security_validations(config_dict, results)

            validation_time = time.time() - start_time
            self._validation_times.append(validation_time)

            return results

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return {}

    def _add_security_validations(self, config_dict: Dict[str, Any], results: Dict[str, ValidationResult]) -> None:
        """Add security-specific validations."""
        # Check for potential security issues
        security_checks = [
            ('gemini_api_key', lambda v: SecurityValidator.validate_api_key_format(v) if v else True),
            ('data_dir', lambda v: SecurityValidator.validate_safe_path(v)),
            ('models_dir', lambda v: SecurityValidator.validate_safe_path(v)),
        ]

        for key, validator in security_checks:
            if key in config_dict:
                try:
                    is_secure = validator(config_dict[key])
                    if not is_secure:
                        results[key] = ValidationResult(
                            is_valid=False,
                            error_message=f"Security validation failed for {key}"
                        )
                except Exception as e:
                    self.logger.warning(f"Security validation error for {key}: {e}")

    def _apply_corrections(self, config_dict: Dict[str, Any], validation_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Apply automatic corrections to configuration.

        Args:
            config_dict: Original configuration dictionary
            validation_results: Validation results with corrections

        Returns:
            Corrected configuration dictionary
        """
        corrected = config_dict.copy()

        for key, result in validation_results.items():
            if not result.is_valid and result.corrected_value is not None:
                corrected[key] = result.corrected_value
                self.logger.info(f"Auto-corrected {key}: {config_dict.get(key)} -> {result.corrected_value}")

        return corrected

    def _create_version(self, config_data: Dict[str, Any], changes: List[ConfigChange]) -> None:
        """Create a new configuration version.

        Args:
            config_data: Configuration data for this version
            changes: List of changes that led to this version
        """
        if not self._enable_versioning:
            return

        # Validate the configuration
        validation_results = self._validate_configuration(config_data)
        errors = [r.error_message for r in validation_results.values()
                 if not r.is_valid and r.error_message]

        version = ConfigVersion(
            version=self._current_version + 1,
            timestamp=datetime.now(),
            config_data=config_data.copy(),
            changes=changes.copy(),
            is_valid=len(errors) == 0,
            validation_errors=errors
        )

        self._versions.append(version)
        self._current_version = version.version

        # Maintain version limit
        if len(self._versions) > self._max_versions:
            removed_versions = self._versions[:-self._max_versions]
            self._versions = self._versions[-self._max_versions:]
            self.logger.debug(f"Removed {len(removed_versions)} old versions")

    def _create_backup(self, reason: str) -> None:
        """Create a configuration backup.

        Args:
            reason: Reason for creating the backup
        """
        if not self._auto_backup or not self._config_path.exists():
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"config_backup_{timestamp}_{reason}.json"
            backup_path = self._backup_dir / backup_name

            shutil.copy2(self._config_path, backup_path)

            # Clean up old backups (keep last 20)
            backups = sorted(self._backup_dir.glob("config_backup_*.json"),
                           key=lambda p: p.stat().st_mtime)

            if len(backups) > 20:
                for old_backup in backups[:-20]:
                    old_backup.unlink()

            self.logger.debug(f"Created configuration backup: {backup_name}")

        except Exception as e:
            self.logger.warning(f"Failed to create configuration backup: {e}")

    def _detect_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> List[ConfigChange]:
        """Detect changes between two configurations.

        Args:
            old_config: Previous configuration
            new_config: New configuration

        Returns:
            List of detected changes
        """
        changes = []
        all_keys = set(old_config.keys()) | set(new_config.keys())

        for key in all_keys:
            old_value = old_config.get(key)
            new_value = new_config.get(key)

            if old_value != new_value:
                changes.append(ConfigChange(
                    timestamp=datetime.now(),
                    field_name=key,
                    old_value=old_value,
                    new_value=new_value,
                    source="file_change"
                ))

        return changes

    def _perform_security_audit(self, config: Config) -> None:
        """Perform security audit on configuration.

        Args:
            config: Configuration to audit
        """
        violations = []

        # Check for insecure settings
        if config.debug and not config.get('allow_debug_in_production', False):
            violations.append({
                'type': 'debug_enabled',
                'severity': 'medium',
                'message': 'Debug mode is enabled',
                'timestamp': datetime.now().isoformat()
            })

        # Check for weak API key patterns
        if config.gemini_api_key and len(config.gemini_api_key) < 30:
            violations.append({
                'type': 'weak_api_key',
                'severity': 'high',
                'message': 'API key appears to be weak or invalid',
                'timestamp': datetime.now().isoformat()
            })

        # Check for insecure paths
        dangerous_paths = ['/tmp', 'C:\\Temp', '/var/tmp']
        for path_attr in ['data_dir', 'models_dir', 'results_export_dir']:
            path_value = getattr(config, path_attr, '')
            if any(dangerous in path_value for dangerous in dangerous_paths):
                violations.append({
                    'type': 'insecure_path',
                    'severity': 'medium',
                    'message': f'Potentially insecure path for {path_attr}: {path_value}',
                    'timestamp': datetime.now().isoformat()
                })

        # Store violations
        self._security_violations.extend(violations)

        # Log security issues
        for violation in violations:
            if violation['severity'] == 'high':
                self.logger.warning(f"Security violation: {violation['message']}")
            else:
                self.logger.info(f"Security notice: {violation['message']}")


# Global configuration manager instance
_config_manager: Optional[EnhancedConfigManager] = None


def get_config_manager() -> EnhancedConfigManager:
    """Get the global configuration manager instance.

    Returns:
        Global configuration manager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = EnhancedConfigManager()
        _config_manager.initialize()
    return _config_manager


def initialize_config_manager(**kwargs) -> EnhancedConfigManager:
    """Initialize the global configuration manager with custom settings.

    Args:
        **kwargs: Configuration manager initialization arguments

    Returns:
        Initialized configuration manager
    """
    global _config_manager
    if _config_manager:
        _config_manager.shutdown()

    _config_manager = EnhancedConfigManager(**kwargs)
    _config_manager.initialize()
    return _config_manager