"""Configuration dataclass and loading utilities.

Provides a strongly-typed configuration object that can be injected into
services instead of relying on a global module-level dictionary.

Security Features:
- Secure environment variable integration
- API key validation and protection
- Input sanitization for all configuration values
- Secure defaults and fallbacks
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
import json, os, logging
from .defaults import DEFAULT_CONFIG
from .env_config import load_environment_config, EnvironmentConfig, EnvironmentError

@dataclass(slots=True)
class Config:
    # Legacy/core settings
    python_version: str = DEFAULT_CONFIG["python_version"]
    target_fps: int = DEFAULT_CONFIG["target_fps"]
    iou_match_threshold: float = DEFAULT_CONFIG["iou_match_threshold"]
    data_dir: str = DEFAULT_CONFIG["data_dir"]
    models_dir: str = DEFAULT_CONFIG["models_dir"]
    master_dir: str = DEFAULT_CONFIG["master_dir"]
    locales_dir: str = DEFAULT_CONFIG["locales_dir"]
    default_locale: str = DEFAULT_CONFIG["default_locale"]
    results_export_dir: str = DEFAULT_CONFIG["results_export_dir"]
    default_data_dir: str = DEFAULT_CONFIG["default_data_dir"]
    default_models_dir: str = DEFAULT_CONFIG["default_models_dir"]
    default_results_dir: str = DEFAULT_CONFIG["default_results_dir"]
    model_size: str = DEFAULT_CONFIG["model_size"]
    train_epochs: int = DEFAULT_CONFIG["train_epochs"]
    batch_size: int = DEFAULT_CONFIG["batch_size"]
    last_webcam_index: int = DEFAULT_CONFIG["last_webcam_index"]
    camera_width: int = DEFAULT_CONFIG["camera_width"]
    camera_height: int = DEFAULT_CONFIG["camera_height"]
    camera_fps: int = DEFAULT_CONFIG["camera_fps"]
    
    # General Application Settings
    language: str = DEFAULT_CONFIG["language"]
    
    # Enhanced Webcam Settings
    camera_device_name: str = DEFAULT_CONFIG["camera_device_name"]
    
    # Enhanced Image Analysis Settings
    detection_confidence_threshold: float = DEFAULT_CONFIG["detection_confidence_threshold"]
    detection_iou_threshold: float = DEFAULT_CONFIG["detection_iou_threshold"]
    preferred_model: str = DEFAULT_CONFIG["preferred_model"]
    
    # Enhanced Chatbot Settings (secured with environment variables)
    gemini_api_key: str = DEFAULT_CONFIG["gemini_api_key"]
    gemini_model: str = DEFAULT_CONFIG["gemini_model"]
    gemini_timeout: int = DEFAULT_CONFIG["gemini_timeout"]
    gemini_temperature: float = DEFAULT_CONFIG["gemini_temperature"]
    gemini_max_tokens: int = DEFAULT_CONFIG["gemini_max_tokens"]
    chatbot_persona: str = DEFAULT_CONFIG["chatbot_persona"]

    # Security flags
    _has_secure_api_key: bool = False
    _environment_config: Optional[EnvironmentConfig] = None
    
    # Arbitrary extra values retained for forward compatibility
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # merge extra keys at top-level for saving
        extra = d.pop("extra", {})
        d.update(extra)
        return d

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, self.extra.get(key, default))
    
    def get_export_format(self) -> str:
        """Get export format (always PNG for highest quality)."""
        return "PNG"
    
    def get_export_quality(self) -> int:
        """Get export quality (always 100% for best results)."""
        return 100


def load_config(path: str = "config.json", env_file: Optional[str] = None) -> Config:
    """Load configuration from JSON file with comprehensive error handling and secure environment integration.

    Args:
        path: Path to config.json file
        env_file: Path to .env file (optional)

    Returns:
        Config: Loaded and validated configuration

    Security:
        - Environment variables override config file values for sensitive data
        - API keys are never stored in config files if environment variables are available
        - All input is validated and sanitized
    """
    data: Dict[str, Any] = {}
    env_config: Optional[EnvironmentConfig] = None

    # Load environment configuration first (highest priority)
    try:
        env_config = load_environment_config(env_file)
        logging.info("Environment configuration loaded successfully")
    except EnvironmentError as e:
        logging.warning(f"Environment configuration failed: {e}")
    except Exception as e:
        logging.error(f"Unexpected error loading environment config: {e}")
    
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)
                if loaded_data is None:
                    logging.warning(f"Configuration file '{path}' is empty, using defaults")
                    data = {}
                elif not isinstance(loaded_data, dict):
                    logging.error(f"Configuration file '{path}' does not contain a valid JSON object, using defaults")
                    data = {}
                else:
                    data = loaded_data
                    logging.info(f"Successfully loaded configuration from '{path}'")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON configuration file '{path}': {e}. Using defaults.")
            data = {}
        except PermissionError:
            logging.error(f"Permission denied reading configuration file '{path}'. Using defaults.")
            data = {}
        except FileNotFoundError:
            logging.info(f"Configuration file '{path}' not found. Using defaults.")
            data = {}
        except Exception as e:
            logging.error(f"Unexpected error loading configuration file '{path}': {e}. Using defaults.")
            data = {}
    else:
        logging.info(f"Configuration file '{path}' does not exist. Using defaults.")
    
    # Validate and merge with defaults
    try:
        merged = {**DEFAULT_CONFIG, **data}

        # Apply environment variable overrides (secure priority)
        if env_config:
            merged = _apply_environment_overrides(merged, env_config)

        # Validate critical path settings
        _validate_path_settings(merged)

        # Sanitize sensitive configuration values
        merged = _sanitize_config_values(merged)

        # capture unknown keys
        extra = {k: v for k, v in merged.items() if k not in Config.__annotations__}
        if extra:
            logging.info(f"Found extra configuration keys: {list(extra.keys())}")

        cfg = Config(**{k: merged[k] for k in Config.__annotations__ if k not in ('extra', '_has_secure_api_key', '_environment_config')}, extra=extra)

        # Set security metadata
        cfg._has_secure_api_key = env_config is not None and env_config.is_api_key_configured
        cfg._environment_config = env_config

        # Ensure critical directories exist
        _ensure_critical_directories(cfg)

        # Log security status (consolidated with environment config message)
        if cfg._has_secure_api_key:
            logging.debug("Configuration loaded with secure API key from environment")
        # API key status already logged by environment config

        return cfg
    except Exception as e:
        logging.error(f"Failed to create configuration object: {e}. Falling back to pure defaults.")
        # Even with fallback, try to apply environment config if available
        fallback_cfg = Config()
        if env_config:
            try:
                fallback_cfg.gemini_api_key = env_config.gemini_api_key or ""
                fallback_cfg.gemini_model = env_config.gemini_model
                fallback_cfg.gemini_timeout = env_config.gemini_timeout
                fallback_cfg.gemini_temperature = env_config.gemini_temperature
                fallback_cfg.gemini_max_tokens = env_config.gemini_max_tokens
                fallback_cfg.enable_rate_limiting = env_config.gemini_rate_limiting
                fallback_cfg.requests_per_minute = env_config.gemini_requests_per_minute
                fallback_cfg._has_secure_api_key = env_config.is_api_key_configured
                fallback_cfg._environment_config = env_config
                logging.info("Applied environment configuration to fallback config")
            except Exception as env_e:
                logging.error(f"Failed to apply environment config to fallback: {env_e}")
        return fallback_cfg


def save_config(cfg: Config, path: str = "config.json") -> None:
    """Save configuration to JSON file with comprehensive error handling.

    Security Note:
    - API keys from environment variables are never saved to config files
    - Only non-sensitive configuration is persisted
    - Sensitive data is excluded from saved configuration
    """
    try:
        # Create backup of existing config
        backup_path = f"{path}.backup"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as src:
                    with open(backup_path, "w", encoding="utf-8") as dst:
                        dst.write(src.read())
                logging.debug(f"Created backup configuration at '{backup_path}'")
            except Exception as e:
                logging.warning(f"Failed to create configuration backup: {e}")

        # Prepare config dict with security filtering
        config_dict = cfg.to_dict()

        # Remove sensitive data if it came from environment variables
        if cfg._has_secure_api_key:
            config_dict["gemini_api_key"] = ""  # Don't save env-provided API keys
            logging.info("API key excluded from saved config (using environment variable)")

        # Remove internal security metadata
        config_dict.pop("_has_secure_api_key", None)
        config_dict.pop("_environment_config", None)

        # Save new configuration
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        logging.info(f"Configuration saved successfully to '{path}'")
        
        # Remove backup if save was successful
        if os.path.exists(backup_path):
            try:
                os.remove(backup_path)
            except Exception:
                pass  # Keep backup if removal fails
                
    except PermissionError:
        logging.error(f"Permission denied writing configuration file '{path}'")
    except OSError as e:
        logging.error(f"OS error saving configuration file '{path}': {e}")
    except Exception as e:
        logging.error(f"Unexpected error saving configuration file '{path}': {e}")


def _validate_path_settings(config_dict: Dict[str, Any]) -> None:
    """Validate path-related configuration settings."""
    path_keys = ['data_dir', 'models_dir', 'master_dir', 'locales_dir', 'results_export_dir']
    
    for key in path_keys:
        if key in config_dict:
            path_value = config_dict[key]
            if not isinstance(path_value, str):
                logging.warning(f"Path setting '{key}' is not a string: {type(path_value)}. Using default.")
                config_dict[key] = DEFAULT_CONFIG.get(key, "data")
            elif not path_value.strip():
                logging.warning(f"Path setting '{key}' is empty. Using default.")
                config_dict[key] = DEFAULT_CONFIG.get(key, "data")


def _ensure_critical_directories(config: Config) -> None:
    """Ensure critical directories exist."""
    critical_dirs = [
        config.data_dir,
        config.models_dir,
        config.results_export_dir
    ]
    
    for dir_path in critical_dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
            logging.debug(f"Ensured directory exists: {dir_path}")
        except Exception as e:
            logging.error(f"Failed to create directory '{dir_path}': {e}")


def _apply_environment_overrides(config_dict: Dict[str, Any], env_config: EnvironmentConfig) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration.

    Args:
        config_dict: Base configuration dictionary
        env_config: Environment configuration object

    Returns:
        dict: Updated configuration with environment overrides
    """
    # Override API settings with environment values
    if env_config.gemini_api_key:
        config_dict["gemini_api_key"] = env_config.gemini_api_key
        config_dict["enable_ai_analysis"] = True  # Enable AI if API key is provided

    config_dict["gemini_model"] = env_config.gemini_model
    config_dict["gemini_timeout"] = env_config.gemini_timeout
    config_dict["gemini_temperature"] = env_config.gemini_temperature
    config_dict["gemini_max_tokens"] = env_config.gemini_max_tokens
    config_dict["enable_rate_limiting"] = env_config.gemini_rate_limiting
    config_dict["requests_per_minute"] = env_config.gemini_requests_per_minute

    # Override path settings if provided
    if env_config.data_dir:
        config_dict["data_dir"] = env_config.data_dir
        config_dict["default_data_dir"] = env_config.data_dir

    if env_config.models_dir:
        config_dict["models_dir"] = env_config.models_dir
        config_dict["default_models_dir"] = env_config.models_dir

    if env_config.results_export_dir:
        config_dict["results_export_dir"] = env_config.results_export_dir
        config_dict["default_results_dir"] = env_config.results_export_dir

    # Override debug settings
    if env_config.debug_logging:
        config_dict["debug"] = True
        config_dict["log_level"] = "DEBUG"

    logging.debug("Applied environment variable overrides to configuration")
    return config_dict


def _sanitize_config_values(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize configuration values for security.

    Args:
        config_dict: Configuration dictionary to sanitize

    Returns:
        dict: Sanitized configuration dictionary
    """
    from ..utils.validation import InputValidator

    sanitized = config_dict.copy()

    # Sanitize string paths
    path_keys = ['data_dir', 'models_dir', 'master_dir', 'locales_dir', 'results_export_dir',
                 'default_data_dir', 'default_models_dir', 'default_results_dir', 'reference_image_path']

    for key in path_keys:
        if key in sanitized and isinstance(sanitized[key], str) and sanitized[key].strip():
            try:
                # Basic path validation (don't use full sanitize_file_path as it may be too restrictive)
                path_value = sanitized[key].strip()
                if os.path.isabs(path_value):
                    # Check for dangerous absolute paths
                    dangerous_dirs = ['C:\\Windows', 'C:\\Program Files', '/etc', '/usr', '/bin', '/root']
                    for dangerous in dangerous_dirs:
                        if path_value.startswith(dangerous):
                            logging.warning(f"Dangerous path detected for {key}: {path_value}, using default")
                            sanitized[key] = DEFAULT_CONFIG.get(key, "data")
                            break
                    else:
                        sanitized[key] = os.path.normpath(path_value)
                else:
                    sanitized[key] = os.path.normpath(path_value)
            except Exception as e:
                logging.warning(f"Failed to sanitize path {key}: {e}, using default")
                sanitized[key] = DEFAULT_CONFIG.get(key, "data")

    # Sanitize string values that could contain user input
    string_keys = ['chatbot_persona', 'camera_device_name', 'app_theme', 'language']

    for key in string_keys:
        if key in sanitized and isinstance(sanitized[key], str):
            try:
                sanitized[key] = InputValidator.sanitize_string_input(sanitized[key], max_length=5000)
            except Exception as e:
                logging.warning(f"Failed to sanitize string {key}: {e}, using default")
                sanitized[key] = DEFAULT_CONFIG.get(key, "")

    # Validate numeric ranges
    numeric_validations = {
        'gemini_timeout': (5, 300),
        'gemini_temperature': (0.0, 1.0),
        'gemini_max_tokens': (1, 8192),
        'requests_per_minute': (1, 100),
        'img_size': (64, 2048),
        'target_fps': (1, 120),
        'camera_fps': (1, 120),
    }

    for key, (min_val, max_val) in numeric_validations.items():
        if key in sanitized:
            try:
                value = sanitized[key]
                if isinstance(value, (int, float)):
                    if not (min_val <= value <= max_val):
                        logging.warning(f"Value {key}={value} out of range [{min_val}, {max_val}], using default")
                        sanitized[key] = DEFAULT_CONFIG.get(key)
            except Exception as e:
                logging.warning(f"Failed to validate {key}: {e}, using default")
                sanitized[key] = DEFAULT_CONFIG.get(key)

    return sanitized


def validate_config_changes(new_config_dict: Dict[str, Any],
                           current_config: Optional[Config] = None) -> 'ValidationReport':
    """Validate configuration changes using the comprehensive ValidationEngine.

    Args:
        new_config_dict: Dictionary of configuration changes to validate
        current_config: Current Config object for comparison (optional)

    Returns:
        ValidationReport: Comprehensive validation results

    Example:
        >>> from app.config.settings import validate_config_changes
        >>> changes = {'camera_width': 1920, 'detection_confidence_threshold': 0.8}
        >>> report = validate_config_changes(changes)
        >>> if report.is_valid:
        ...     # Apply changes safely
        ...     pass
        >>> else:
        ...     # Show validation errors
        ...     print(report.errors)
    """
    try:
        from ..utils.validation import ValidationEngine

        engine = ValidationEngine()

        # Convert current config to dict if provided
        current_dict = None
        if current_config:
            current_dict = current_config.to_dict()

        return engine.validate_config_changes(new_config_dict, current_dict)

    except ImportError as e:
        # Fallback to basic validation if ValidationEngine not available
        logging.warning(f"ValidationEngine not available, using basic validation: {e}")

        # Create a minimal ValidationReport-like object
        from dataclasses import dataclass
        from typing import Dict, List

        @dataclass
        class FallbackReport:
            is_valid: bool = True
            errors: Dict[str, List[str]] = None
            warnings: Dict[str, List[str]] = None
            auto_corrections: Dict[str, Any] = None
            validation_time_ms: float = 0.0
            stages_completed: List[str] = None

            def __post_init__(self):
                if self.errors is None:
                    self.errors = {}
                if self.warnings is None:
                    self.warnings = {}
                if self.auto_corrections is None:
                    self.auto_corrections = {}
                if self.stages_completed is None:
                    self.stages_completed = ["basic_validation"]

        # Basic validation using existing _sanitize_config_values
        try:
            merged_config = {**DEFAULT_CONFIG, **new_config_dict}
            sanitized = _sanitize_config_values(merged_config)

            report = FallbackReport()

            # Check for any obvious issues
            for key, value in new_config_dict.items():
                if key in DEFAULT_CONFIG:
                    expected_type = type(DEFAULT_CONFIG[key])
                    if not isinstance(value, expected_type):
                        report.errors.setdefault(key, []).append(
                            f"Type mismatch: expected {expected_type.__name__}, got {type(value).__name__}"
                        )
                        report.is_valid = False

            return report

        except Exception as fallback_error:
            logging.error(f"Fallback validation failed: {fallback_error}")
            report = FallbackReport()
            report.is_valid = False
            report.errors["_validation"] = [f"Validation failed: {str(fallback_error)}"]
            return report


def load_and_validate_config(path: str = "config.json",
                           env_file: Optional[str] = None,
                           validate_changes: bool = True) -> tuple[Config, Optional['ValidationReport']]:
    """Load configuration with optional validation.

    Args:
        path: Path to config.json file
        env_file: Path to .env file (optional)
        validate_changes: Whether to validate the loaded configuration

    Returns:
        tuple: (Config object, ValidationReport if validation was performed)
    """
    config = load_config(path, env_file)

    if validate_changes:
        try:
            # Validate the loaded configuration
            config_dict = config.to_dict()
            report = validate_config_changes(config_dict)

            if not report.is_valid:
                logging.warning("Loaded configuration has validation issues:")
                for field, errors in report.errors.items():
                    for error in errors:
                        logging.warning(f"  {field}: {error}")

                # Apply auto-corrections if available
                if report.auto_corrections:
                    logging.info("Applying auto-corrections to loaded configuration:")
                    for field, correction in report.auto_corrections.items():
                        if hasattr(config, field):
                            old_value = getattr(config, field)
                            setattr(config, field, correction)
                            logging.info(f"  {field}: {old_value} -> {correction}")

            return config, report

        except Exception as e:
            logging.error(f"Configuration validation failed: {e}")
            return config, None
    else:
        return config, None


__all__ = ["Config", "load_config", "save_config", "validate_config_changes", "load_and_validate_config"]