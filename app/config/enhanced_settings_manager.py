"""Enhanced Settings Manager with Atomic Application Support.

This module extends the base SettingsManager with transactional capabilities,
providing atomic updates with full rollback support.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set, List, Callable, TYPE_CHECKING
from contextlib import contextmanager
import difflib
import hashlib

from .settings import Config
from .settings_manager import SettingsManager
from .atomic_applier import (
    create_transactional_applier
)
from .validation import ValidationEngine, ValidationLevel, ValidationResult

if TYPE_CHECKING:
    from ..services.webcam_service import WebcamService
    from ..services.gemini_service import AsyncGeminiService
    from ..services.detection_service import DetectionService

logger = logging.getLogger(__name__)


class ConfigChangeTracker:
    """Tracks configuration changes and determines affected categories."""

    def __init__(self):
        self.category_mappings = self._initialize_category_mappings()

    def _initialize_category_mappings(self) -> Dict[str, str]:
        """Map configuration fields to categories."""
        return {
            # Webcam settings
            'last_webcam_index': 'camera',
            'camera_width': 'camera',
            'camera_height': 'camera',
            'camera_fps': 'camera',
            'camera_brightness': 'camera',
            'camera_contrast': 'camera',
            'camera_saturation': 'camera',
            'camera_auto_exposure': 'camera',
            'camera_auto_focus': 'camera',

            # Detection settings
            'detection_confidence_threshold': 'detection',
            'detection_iou_threshold': 'detection',
            'detection_model': 'detection',

            # AI settings
            'gemini_api_key': 'ai',
            'gemini_model': 'ai',
            'gemini_temperature': 'ai',
            'gemini_max_tokens': 'ai',
            'gemini_timeout': 'ai',
            'enable_ai_analysis': 'ai',
            'chatbot_persona': 'ai',
            'enable_rate_limiting': 'ai',
            'requests_per_minute': 'ai',
            'context_window_size': 'ai',

            # UI settings
            'app_theme': 'appearance',
            'remember_window_state': 'appearance',
            'window_width': 'appearance',
            'window_height': 'appearance',
            'window_x': 'appearance',
            'window_y': 'appearance',

            # Performance settings
            'enable_gpu': 'performance',
            'gpu_device_index': 'performance',
            'thread_count': 'performance',
            'buffer_size': 'performance',
            'enable_caching': 'performance',
            'cache_size_mb': 'performance',

            # Export settings
            'export_quality': 'export',
            'results_export_dir': 'export',

            # Debug settings
            'debug_mode': 'debug',
            'log_level': 'debug',
            'enable_profiling': 'debug'
        }

    def get_changed_categories(self, old_config: Config, new_config: Config) -> Set[str]:
        """Determine which configuration categories have changed."""
        changed_categories = set()

        # Convert configs to dictionaries for comparison
        old_dict = old_config.to_dict() if old_config else {}
        new_dict = new_config.to_dict() if new_config else {}

        # Find changed fields
        for field, new_value in new_dict.items():
            old_value = old_dict.get(field)
            if old_value != new_value:
                # Map field to category
                category = self.category_mappings.get(field, 'other')
                changed_categories.add(category)

        return changed_categories

    def get_detailed_changes(self, old_config: Config, new_config: Config) -> Dict[str, Any]:
        """Get detailed changes between configurations."""
        changes = {}

        old_dict = old_config.to_dict() if old_config else {}
        new_dict = new_config.to_dict() if new_config else {}

        for field, new_value in new_dict.items():
            old_value = old_dict.get(field)
            if old_value != new_value:
                changes[field] = {
                    'old': old_value,
                    'new': new_value,
                    'category': self.category_mappings.get(field, 'other')
                }

        return changes


class EnhancedSettingsManager(SettingsManager):
    """Enhanced settings manager with atomic application capabilities."""

    def __init__(self, config_path: str = "config.json"):
        super().__init__(config_path)

        # Initialize atomic applier
        self.atomic_applier = create_transactional_applier()

        # Initialize validation engine
        self.validation_engine = ValidationEngine()

        # Initialize change tracker
        self.change_tracker = ConfigChangeTracker()

        # Configuration history for advanced rollback
        self.config_history: List[Dict[str, Any]] = []
        self.max_history_size = 20

        # Track application state
        self.last_successful_config: Optional[Config] = None
        self.application_start_time = datetime.now()

        # Performance metrics
        self.apply_metrics = {
            'total_applications': 0,
            'successful_applications': 0,
            'failed_applications': 0,
            'total_rollbacks': 0,
            'average_apply_time_ms': 0.0
        }

        # Initialize with current config as last successful
        self.last_successful_config = self.config

    def register_service(self, name: str, service: Any) -> None:
        """Register a service for both standard and atomic updates."""
        super().register_service(name, service)
        self.atomic_applier.register_service(name, service)
        logger.debug(f"Service '{name}' registered with atomic applier")

    def save_settings(self, config: Config, atomic: bool = True) -> bool:
        """Save configuration with optional atomic application.

        Args:
            config: Configuration object to save
            atomic: Whether to use atomic application (default True)

        Returns:
            bool: True if save and application were successful
        """
        with self._lock:
            start_time = datetime.now()

            try:
                # Validate configuration
                validation_result = self._validate_with_engine(config)

                if not validation_result.is_valid:
                    logger.error(f"Configuration validation failed: {validation_result.errors}")
                    self._update_metrics(success=False, duration_ms=0)
                    return False

                # Add to history before applying
                self._add_to_history(config)

                # Determine changed categories
                changed_categories = self.change_tracker.get_changed_categories(
                    self._config, config
                )

                logger.info(f"Applying configuration changes to categories: {changed_categories}")

                # Apply settings
                if atomic:
                    success = self._apply_atomically(config, changed_categories)
                else:
                    success = self._apply_standard(config)

                if success:
                    # Save to file after successful application
                    if self._save_to_file(config):
                        self._config = config
                        self.last_successful_config = config

                        # Update metrics
                        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                        self._update_metrics(success=True, duration_ms=duration_ms)

                        logger.info(f"Configuration saved and applied successfully in {duration_ms:.2f}ms")
                        return True
                    else:
                        # Rollback if file save failed
                        self._rollback_to_last_successful()
                        self._update_metrics(success=False, duration_ms=0)
                        return False
                else:
                    self._update_metrics(success=False, duration_ms=0)
                    return False

            except Exception as e:
                logger.error(f"Failed to save settings: {e}")
                self._update_metrics(success=False, duration_ms=0)
                return False

    def _apply_atomically(self, config: Config, changed_categories: Set[str]) -> bool:
        """Apply settings using atomic applier with rollback capability."""
        try:
            return self.atomic_applier.apply_settings(config, changed_categories)
        except Exception as e:
            logger.error(f"Atomic application failed: {e}")
            return False

    def _apply_standard(self, config: Config) -> bool:
        """Apply settings using standard method (fallback)."""
        try:
            self._apply_settings_to_services(config)
            self._notify_change_callbacks(config)
            return True
        except Exception as e:
            logger.error(f"Standard application failed: {e}")
            return False

    def _validate_with_engine(self, config: Config) -> ValidationResult:
        """Validate configuration using the validation engine."""
        try:
            # Convert config to dict for validation
            config_dict = config.to_dict()

            # Perform validation at STRICT level
            return self.validation_engine.validate_config(
                config_dict,
                level=ValidationLevel.STRICT
            )

        except Exception as e:
            logger.error(f"Validation engine error: {e}")
            # Return invalid result on error
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation engine error: {str(e)}"],
                warnings=[],
                level=ValidationLevel.STRICT
            )

    def _save_to_file(self, config: Config) -> bool:
        """Save configuration to file with atomic write."""
        try:
            # Create backup first
            backup_path = self._create_backup()

            # Write to temporary file
            temp_path = self.config_path.with_suffix('.tmp')
            config_dict = config.to_dict()

            # Add metadata
            config_dict['_metadata'] = {
                'saved_at': datetime.now().isoformat(),
                'version': '2.0',
                'checksum': self._calculate_checksum(config_dict)
            }

            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            # Atomic replace
            temp_path.replace(self.config_path)

            logger.debug(f"Configuration written to {self.config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save configuration to file: {e}")
            return False

    def _calculate_checksum(self, config_dict: Dict[str, Any]) -> str:
        """Calculate checksum for configuration integrity."""
        # Remove metadata before calculating checksum
        config_copy = {k: v for k, v in config_dict.items() if not k.startswith('_')}
        config_str = json.dumps(config_copy, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def _add_to_history(self, config: Config):
        """Add configuration to history for rollback capability."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'config': config.to_dict(),
            'checksum': self._calculate_checksum(config.to_dict())
        }

        self.config_history.append(entry)

        # Trim history if needed
        if len(self.config_history) > self.max_history_size:
            self.config_history = self.config_history[-self.max_history_size:]

    def _rollback_to_last_successful(self):
        """Rollback to last successful configuration."""
        if self.last_successful_config:
            logger.warning("Rolling back to last successful configuration")
            try:
                self._config = self.last_successful_config
                self._apply_standard(self.last_successful_config)
                self.apply_metrics['total_rollbacks'] += 1
            except Exception as e:
                logger.error(f"Failed to rollback: {e}")

    def _update_metrics(self, success: bool, duration_ms: float):
        """Update performance metrics."""
        self.apply_metrics['total_applications'] += 1

        if success:
            self.apply_metrics['successful_applications'] += 1
        else:
            self.apply_metrics['failed_applications'] += 1

        # Update average apply time (exponential moving average)
        alpha = 0.3  # Smoothing factor
        if self.apply_metrics['average_apply_time_ms'] == 0:
            self.apply_metrics['average_apply_time_ms'] = duration_ms
        else:
            self.apply_metrics['average_apply_time_ms'] = (
                alpha * duration_ms +
                (1 - alpha) * self.apply_metrics['average_apply_time_ms']
            )

    @contextmanager
    def atomic_update(self, auto_save: bool = True):
        """Context manager for atomic configuration updates with automatic save."""
        with self._lock:
            original_config = self._config.copy() if hasattr(self._config, 'copy') else self._config

            try:
                yield self._config

                # Auto-save if requested and context exits normally
                if auto_save:
                    self.save_settings(self._config, atomic=True)

            except Exception as e:
                # Rollback on error
                logger.error(f"Error in atomic update: {e}")
                self._config = original_config
                self._apply_standard(original_config)
                raise

    def get_config_diff(self, other_config: Optional[Config] = None) -> Dict[str, Any]:
        """Get differences between current and another configuration."""
        if other_config is None:
            other_config = self.last_successful_config

        if other_config is None:
            return {}

        return self.change_tracker.get_detailed_changes(self._config, other_config)

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for configuration management."""
        uptime_seconds = (datetime.now() - self.application_start_time).total_seconds()

        return {
            **self.apply_metrics,
            'uptime_seconds': uptime_seconds,
            'success_rate': (
                self.apply_metrics['successful_applications'] /
                max(1, self.apply_metrics['total_applications'])
            ),
            'history_size': len(self.config_history),
            'services_registered': len(self._services)
        }

    def rollback_to_checkpoint(self, timestamp: Optional[datetime] = None) -> bool:
        """Rollback configuration to a specific checkpoint in history.

        Args:
            timestamp: Target timestamp to rollback to (None for last checkpoint)

        Returns:
            bool: True if rollback was successful
        """
        if not self.config_history:
            logger.warning("No configuration history available for rollback")
            return False

        try:
            if timestamp:
                # Find closest checkpoint
                target_entry = None
                for entry in reversed(self.config_history):
                    entry_time = datetime.fromisoformat(entry['timestamp'])
                    if entry_time <= timestamp:
                        target_entry = entry
                        break
            else:
                # Use last checkpoint
                target_entry = self.config_history[-1]

            if target_entry:
                # Verify checksum
                expected_checksum = target_entry['checksum']
                actual_checksum = self._calculate_checksum(target_entry['config'])

                if expected_checksum != actual_checksum:
                    logger.error("Checkpoint integrity check failed")
                    return False

                # Create config from checkpoint
                restored_config = self._create_config_from_dict(target_entry['config'])

                # Apply atomically
                return self.save_settings(restored_config, atomic=True)
            else:
                logger.warning("No suitable checkpoint found")
                return False

        except Exception as e:
            logger.error(f"Failed to rollback to checkpoint: {e}")
            return False

    def export_diagnostic_info(self, export_path: str) -> bool:
        """Export diagnostic information for troubleshooting.

        Args:
            export_path: Path to export diagnostic data

        Returns:
            bool: True if export was successful
        """
        try:
            diagnostic_data = {
                'timestamp': datetime.now().isoformat(),
                'current_config': self.config.to_dict(),
                'metrics': self.get_metrics(),
                'history_count': len(self.config_history),
                'services': list(self._services.keys()),
                'last_successful_config': (
                    self.last_successful_config.to_dict()
                    if self.last_successful_config else None
                ),
                'validation_rules': self.validation_engine.get_rule_summary()
            }

            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(diagnostic_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Diagnostic info exported to {export_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export diagnostic info: {e}")
            return False


# Global instance management
_enhanced_manager: Optional[EnhancedSettingsManager] = None


def get_enhanced_settings_manager(config_path: str = "config.json") -> EnhancedSettingsManager:
    """Get global enhanced settings manager instance."""
    global _enhanced_manager
    if _enhanced_manager is None:
        _enhanced_manager = EnhancedSettingsManager(config_path)
    return _enhanced_manager


__all__ = [
    'EnhancedSettingsManager',
    'ConfigChangeTracker',
    'get_enhanced_settings_manager'
]