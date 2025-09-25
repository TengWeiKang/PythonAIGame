"""Enhanced settings manager with real-time application and comprehensive error handling.

Provides centralized management of application settings with:
- Atomic save operations with backup/restore
- Real-time application to all services  
- Configuration validation and migration
- Type-safe configuration with proper error handling
"""
from __future__ import annotations

import json
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Callable, List, TYPE_CHECKING, Set
import logging
from dataclasses import asdict
from contextlib import contextmanager
import copy

from .settings import Config, load_config, save_config
from .defaults import DEFAULT_CONFIG
from .atomic_applier import (
    TransactionalSettingsApplier,
    create_transactional_applier,
    UpdateType,
    ServicePriority
)
from .validation import SettingsValidator, ValidationResult

if TYPE_CHECKING:
    from ..services.webcam_service import WebcamService
    from ..services.gemini_service import AsyncGeminiService

logger = logging.getLogger(__name__)


class SettingsManager:
    """Manages application settings with real-time updates and validation."""

    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.backup_dir = Path("config_backups")
        self._config: Optional[Config] = None
        self._services: Dict[str, Any] = {}
        self._change_callbacks: List[Callable[[Config], None]] = []
        self._lock = threading.RLock()

        # Transactional settings applier for atomic updates
        self._transactional_applier = create_transactional_applier()

        # Settings validator
        self._validator = SettingsValidator()

        # Track changed categories for optimized updates
        self._changed_categories: Set[str] = set()

        # Previous config for differential tracking
        self._previous_config: Optional[Config] = None

        # Ensure backup directory exists
        self.backup_dir.mkdir(exist_ok=True)

        # Load initial configuration
        self._load_configuration()
    
    @property
    def config(self) -> Config:
        """Get current configuration."""
        if self._config is None:
            self._load_configuration()
        return self._config
    
    def register_service(self, name: str, service: Any) -> None:
        """Register a service to receive settings updates."""
        with self._lock:
            self._services[name] = service
            # Also register with transactional applier
            self._transactional_applier.register_service(name, service)
            logger.debug(f"Registered service: {name}")
    
    def add_change_callback(self, callback: Callable[[Config], None]) -> None:
        """Add callback to be called when settings change."""
        with self._lock:
            self._change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: Callable[[Config], None]) -> None:
        """Remove change callback."""
        with self._lock:
            if callback in self._change_callbacks:
                self._change_callbacks.remove(callback)
    
    def _load_configuration(self) -> None:
        """Load configuration from file with migration support."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Migrate configuration if needed
                migrated_data = self._migrate_configuration(data)
                
                # Validate and load
                self._config = self._create_config_from_dict(migrated_data)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                # Create default configuration
                self._config = Config()
                self.save_settings(self._config)
                logger.info("Created default configuration")
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Fall back to defaults
            self._config = Config()
    
    def _migrate_configuration(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate configuration to current version."""
        # Remove deprecated settings based on requirements
        deprecated_settings = {
            # Update-related settings to remove
            'check_for_updates',
            'update_check_interval_days',
            'backup_settings_on_change',
            # Image processing settings to remove
            'enable_noise_reduction',
            'enable_contrast_enhancement',
            # Export format to be hardcoded
            'export_format',
        }
        
        migrated = {k: v for k, v in data.items() if k not in deprecated_settings}
        
        # Force export quality to 100%
        migrated['export_quality'] = 100
        
        # Ensure all required settings exist with defaults
        for key, default_value in DEFAULT_CONFIG.items():
            if key not in migrated and key not in deprecated_settings:
                migrated[key] = default_value
        
        return migrated
    
    def _create_config_from_dict(self, data: Dict[str, Any]) -> Config:
        """Create Config object from dictionary with proper handling."""
        # Separate known fields from extra fields
        known_fields = {k: v for k, v in data.items() if k in Config.__annotations__}
        extra_fields = {k: v for k, v in data.items() if k not in Config.__annotations__}
        
        try:
            config = Config(**known_fields, extra=extra_fields)
            return config
        except TypeError as e:
            logger.warning(f"Configuration validation error: {e}")
            # Fall back to defaults for invalid fields
            config = Config(extra=extra_fields)
            return config
    
    def save_settings(self, config: Config, use_atomic: bool = True) -> bool:
        """Save configuration to file with atomic operations and validation.

        Args:
            config: Configuration object to save
            use_atomic: Whether to use atomic transactional updates

        Returns:
            bool: True if save was successful, False otherwise
        """
        with self._lock:
            try:
                # Validate configuration
                if not self._validate_config(config):
                    logger.error("Configuration validation failed")
                    return False

                # Detect changed categories
                changed_categories = self._detect_changed_categories(config)

                # Create backup of current config
                backup_path = None
                if self.config_path.exists():
                    backup_path = self._create_backup()

                # Write to temporary file first (atomic operation)
                temp_path = self.config_path.with_suffix('.tmp')

                config_dict = config.to_dict()

                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)

                # Apply settings to services
                if use_atomic:
                    # Use transactional applier for atomic updates
                    success = self._transactional_applier.apply_settings(config, changed_categories)
                    if not success:
                        logger.error("Failed to apply settings atomically")
                        # Don't save config file if service update failed
                        temp_path.unlink()
                        return False
                else:
                    # Use traditional non-atomic update
                    self._apply_settings_to_services(config)

                # Atomically replace the original file
                if os.name == 'nt':  # Windows
                    if self.config_path.exists():
                        self.config_path.unlink()
                    temp_path.rename(self.config_path)
                else:  # Unix-like systems
                    temp_path.replace(self.config_path)

                # Update internal config
                self._previous_config = copy.deepcopy(self._config)
                self._config = config

                # Notify callbacks
                self._notify_change_callbacks(config)

                logger.info(f"Configuration saved successfully to {self.config_path}")
                return True

            except Exception as e:
                logger.error(f"Failed to save configuration: {e}")

                # Restore from backup if available
                if backup_path and backup_path.exists():
                    try:
                        shutil.copy2(backup_path, self.config_path)
                        logger.info("Configuration restored from backup")
                        # Also restore services to previous state if atomic was used
                        if use_atomic and self._previous_config:
                            self._transactional_applier.apply_settings(self._previous_config, None)
                    except Exception as restore_error:
                        logger.error(f"Failed to restore backup: {restore_error}")

                return False
    
    def load_settings(self) -> Config:
        """Load configuration with migration and validation.
        
        Returns:
            Config: Loaded configuration object
        """
        self._load_configuration()
        return self.config
    
    def apply_settings(self, config: Config, use_atomic: bool = True) -> bool:
        """Apply settings to running application services.

        Args:
            config: Configuration to apply
            use_atomic: Whether to use atomic transactional updates

        Returns:
            bool: True if settings were applied successfully
        """
        with self._lock:
            if use_atomic:
                # Detect changed categories
                changed_categories = self._detect_changed_categories(config)
                # Use transactional applier
                success = self._transactional_applier.apply_settings(config, changed_categories)
                if success:
                    self._notify_change_callbacks(config)
                    self._previous_config = copy.deepcopy(self._config)
                    self._config = config
                return success
            else:
                # Use traditional non-atomic update
                self._apply_settings_to_services(config)
                self._notify_change_callbacks(config)
                self._previous_config = copy.deepcopy(self._config)
                self._config = config
                return True
    
    def _validate_config(self, config: Config) -> bool:
        """Validate configuration object.
        
        Args:
            config: Configuration to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check required string fields are not empty
            if not config.gemini_api_key.strip() and config.enable_ai_analysis:
                logger.warning("Gemini API key required when AI analysis is enabled")
            
            # Check numeric ranges
            if not (0.0 <= config.detection_confidence_threshold <= 1.0):
                logger.error("Detection confidence threshold must be between 0.0 and 1.0")
                return False
            
            if not (0.0 <= config.detection_iou_threshold <= 1.0):
                logger.error("Detection IoU threshold must be between 0.0 and 1.0")
                return False
            
            if not (0.0 <= config.gemini_temperature <= 1.0):
                logger.error("Gemini temperature must be between 0.0 and 1.0")
                return False
            
            if config.camera_width <= 0 or config.camera_height <= 0:
                logger.error("Camera dimensions must be positive")
                return False
            
            if config.camera_fps <= 0 or config.camera_fps > 240:
                logger.error("Camera FPS must be between 1 and 240")
                return False
            
            # Check directory paths exist or can be created
            for path_attr in ['data_dir', 'models_dir', 'results_export_dir']:
                path_value = getattr(config, path_attr)
                if path_value:
                    path_obj = Path(path_value)
                    try:
                        path_obj.mkdir(parents=True, exist_ok=True)
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Cannot access {path_attr} '{path_value}': {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    def _create_backup(self) -> Optional[Path]:
        """Create backup of current configuration.
        
        Returns:
            Path: Path to backup file, or None if failed
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"config_{timestamp}.json"
            
            shutil.copy2(self.config_path, backup_path)
            
            # Clean up old backups (keep last 10)
            backups = sorted(self.backup_dir.glob("config_*.json"))
            if len(backups) > 10:
                for old_backup in backups[:-10]:
                    old_backup.unlink()
            
            logger.debug(f"Configuration backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to create configuration backup: {e}")
            return None
    
    def _apply_settings_to_services(self, config: Config) -> None:
        """Apply configuration to registered services."""
        try:
            # Apply webcam settings
            webcam_service = self._services.get('webcam')
            if webcam_service:
                self._apply_webcam_settings(webcam_service, config)
            
            # Apply Gemini settings
            gemini_service = self._services.get('gemini')
            if gemini_service:
                self._apply_gemini_settings(gemini_service, config)
            
            # Apply UI theme settings
            main_window = self._services.get('main_window')
            if main_window:
                self._apply_theme_settings(main_window, config)
            
            # Apply analysis settings
            detection_service = self._services.get('detection')
            if detection_service:
                self._apply_analysis_settings(detection_service, config)
                
        except Exception as e:
            logger.error(f"Error applying settings to services: {e}")
    
    def _apply_webcam_settings(self, service: 'WebcamService', config: Config) -> None:
        """Apply webcam settings to webcam service."""
        try:
            # Update camera device
            if config.last_webcam_index != service.current_camera_index:
                service.set_camera(config.last_webcam_index)
            
            # Update camera properties
            service.set_resolution(config.camera_width, config.camera_height)
            service.set_fps(config.camera_fps)
            
            # Update camera controls if supported
            if hasattr(service, 'set_brightness'):
                service.set_brightness(config.camera_brightness)
            if hasattr(service, 'set_contrast'):
                service.set_contrast(config.camera_contrast)
            if hasattr(service, 'set_saturation'):
                service.set_saturation(config.camera_saturation)
            
            # Update auto settings
            if hasattr(service, 'set_auto_exposure'):
                service.set_auto_exposure(config.camera_auto_exposure)
            if hasattr(service, 'set_auto_focus'):
                service.set_auto_focus(config.camera_auto_focus)
            
            logger.debug("Webcam settings applied successfully")
            
        except Exception as e:
            logger.error(f"Failed to apply webcam settings: {e}")
    
    def _apply_gemini_settings(self, service: 'AsyncGeminiService', config: Config) -> None:
        """Apply Gemini AI settings to service."""
        try:
            # Update API configuration
            if hasattr(service, 'update_config'):
                service.update_config(
                    api_key=config.gemini_api_key,
                    model=config.gemini_model,
                    temperature=config.gemini_temperature,
                    max_tokens=config.gemini_max_tokens,
                    timeout=config.gemini_timeout,
                    persona=config.chatbot_persona
                )
            
            # Update rate limiting
            if hasattr(service, 'set_rate_limit'):
                service.set_rate_limit(
                    enabled=config.enable_rate_limiting,
                    requests_per_minute=config.requests_per_minute
                )
            
            # Update context settings
            if hasattr(service, 'set_context_window'):
                service.set_context_window(config.context_window_size)
            
            logger.debug("Gemini settings applied successfully")
            
        except Exception as e:
            logger.error(f"Failed to apply Gemini settings: {e}")
    
    def _apply_theme_settings(self, main_window, config: Config) -> None:
        """Apply theme settings to main window."""
        try:
            if hasattr(main_window, 'apply_theme'):
                main_window.apply_theme(config.app_theme)
            
            # Update window state settings
            if hasattr(main_window, 'set_remember_state'):
                main_window.set_remember_state(config.remember_window_state)
            
            logger.debug("Theme settings applied successfully")
            
        except Exception as e:
            logger.error(f"Failed to apply theme settings: {e}")
    
    def _apply_analysis_settings(self, service, config: Config) -> None:
        """Apply analysis settings to detection service."""
        try:
            # Update detection thresholds
            if hasattr(service, 'set_confidence_threshold'):
                service.set_confidence_threshold(config.detection_confidence_threshold)
            
            if hasattr(service, 'set_iou_threshold'):
                service.set_iou_threshold(config.detection_iou_threshold)
            
            # Update ROI settings
            if hasattr(service, 'set_roi'):
                if config.enable_roi:
                    service.set_roi(config.roi_x, config.roi_y, 
                                   config.roi_width, config.roi_height)
                else:
                    service.clear_roi()
            
            logger.debug("Analysis settings applied successfully")
            
        except Exception as e:
            logger.error(f"Failed to apply analysis settings: {e}")
    
    def _detect_changed_categories(self, new_config: Config) -> Set[str]:
        """Detect which setting categories have changed."""
        if not self._config:
            # All categories are "changed" on first load
            return {'camera', 'detection', 'ai', 'appearance', 'paths', 'advanced'}

        changed = set()
        old_dict = self._config.to_dict()
        new_dict = new_config.to_dict()

        # Category mappings
        category_mappings = {
            'camera': [
                'last_webcam_index', 'camera_width', 'camera_height', 'camera_fps',
                'camera_brightness', 'camera_contrast', 'camera_saturation',
                'camera_auto_exposure', 'camera_auto_focus', 'camera_buffer_size',
                'camera_recording_format'
            ],
            'detection': [
                'detection_confidence_threshold', 'detection_iou_threshold',
                'yolo_model', 'enable_roi', 'roi_x', 'roi_y', 'roi_width', 'roi_height',
                'difference_sensitivity', 'use_gpu', 'enable_object_tracking'
            ],
            'ai': [
                'gemini_api_key', 'gemini_model', 'gemini_temperature', 'gemini_timeout',
                'gemini_max_tokens', 'enable_ai_analysis', 'chatbot_persona',
                'response_format', 'enable_rate_limiting', 'requests_per_minute',
                'context_window_size', 'enable_conversation_memory'
            ],
            'appearance': [
                'app_theme', 'language', 'startup_fullscreen', 'remember_window_state',
                'window_width', 'window_height', 'show_tooltips', 'show_status_bar'
            ],
            'paths': [
                'data_dir', 'models_dir', 'results_export_dir', 'reference_image_path',
                'default_data_dir', 'default_models_dir', 'default_results_dir'
            ],
            'advanced': [
                'debug', 'enable_logging', 'log_level', 'performance_mode',
                'max_memory_usage_mb', 'auto_save_config', 'auto_save_interval_minutes'
            ]
        }

        # Check each category for changes
        for category, fields in category_mappings.items():
            for field in fields:
                if old_dict.get(field) != new_dict.get(field):
                    changed.add(category)
                    break

        return changed

    def _notify_change_callbacks(self, config: Config) -> None:
        """Notify all registered callbacks of configuration changes."""
        for callback in self._change_callbacks:
            try:
                callback(config)
            except Exception as e:
                logger.error(f"Error in settings change callback: {e}")
    
    @contextmanager
    def atomic_update(self):
        """Context manager for atomic configuration updates."""
        with self._lock:
            original_config = copy.deepcopy(self._config)
            try:
                yield self._config
                # Apply settings atomically when context exits normally
                success = self.save_settings(self._config, use_atomic=True)
                if not success:
                    # Restore original if save failed
                    self._config = original_config
                    raise RuntimeError("Failed to save configuration atomically")
            except Exception:
                # Restore original config on error
                self._config = original_config
                # Also restore service states
                self._transactional_applier.apply_settings(original_config, None)
                raise
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to default values.
        
        Returns:
            bool: True if reset was successful
        """
        try:
            default_config = Config()
            return self.save_settings(default_config)
        except Exception as e:
            logger.error(f"Failed to reset configuration to defaults: {e}")
            return False
    
    def export_settings(self, export_path: str) -> bool:
        """Export current settings to a file.
        
        Args:
            export_path: Path to export settings to
            
        Returns:
            bool: True if export was successful
        """
        try:
            export_data = {
                'exported_at': datetime.now().isoformat(),
                'version': '1.0',
                'settings': self.config.to_dict()
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Settings exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export settings: {e}")
            return False
    
    def import_settings(self, import_path: str) -> bool:
        """Import settings from a file.
        
        Args:
            import_path: Path to import settings from
            
        Returns:
            bool: True if import was successful
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            settings_data = import_data.get('settings', {})
            migrated_data = self._migrate_configuration(settings_data)
            imported_config = self._create_config_from_dict(migrated_data)
            
            return self.save_settings(imported_config)
            
        except Exception as e:
            logger.error(f"Failed to import settings: {e}")
            return False


# Global settings manager instance
_settings_manager: Optional[SettingsManager] = None


def get_settings_manager(config_path: str = "config.json") -> SettingsManager:
    """Get global settings manager instance."""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager(config_path)
    return _settings_manager


def save_settings(config: Config, use_atomic: bool = True) -> bool:
    """Save configuration using global settings manager.

    Args:
        config: Configuration to save
        use_atomic: Whether to use atomic transactional updates

    Returns:
        bool: True if save was successful
    """
    return get_settings_manager().save_settings(config, use_atomic)


def load_settings() -> Config:
    """Load configuration using global settings manager."""
    return get_settings_manager().load_settings()


def apply_settings(config: Config, use_atomic: bool = True) -> bool:
    """Apply settings using global settings manager.

    Args:
        config: Configuration to apply
        use_atomic: Whether to use atomic transactional updates

    Returns:
        bool: True if settings were applied successfully
    """
    return get_settings_manager().apply_settings(config, use_atomic)


__all__ = [
    'SettingsManager', 
    'get_settings_manager', 
    'save_settings', 
    'load_settings', 
    'apply_settings'
]