"""Configuration dataclass and loading utilities.

Provides a strongly-typed configuration object that can be injected into
services instead of relying on a global module-level dictionary.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict
import json, os, logging
from .defaults import DEFAULT_CONFIG

@dataclass(slots=True)
class Config:
    # Legacy/core settings
    python_version: str = DEFAULT_CONFIG["python_version"]
    img_size: int = DEFAULT_CONFIG["img_size"]
    target_fps: int = DEFAULT_CONFIG["target_fps"]
    iou_match_threshold: float = DEFAULT_CONFIG["iou_match_threshold"]
    master_tolerance_px: int = DEFAULT_CONFIG["master_tolerance_px"]
    angle_tolerance_deg: int = DEFAULT_CONFIG["angle_tolerance_deg"]
    use_gpu: bool = DEFAULT_CONFIG["use_gpu"]
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
    debug: bool = DEFAULT_CONFIG["debug"]
    last_webcam_index: int = DEFAULT_CONFIG["last_webcam_index"]
    preview_max_width: int = DEFAULT_CONFIG["preview_max_width"]
    preview_max_height: int = DEFAULT_CONFIG["preview_max_height"]
    camera_width: int = DEFAULT_CONFIG["camera_width"]
    camera_height: int = DEFAULT_CONFIG["camera_height"]
    camera_fps: int = DEFAULT_CONFIG["camera_fps"]
    
    # General Application Settings
    app_theme: str = DEFAULT_CONFIG["app_theme"]
    language: str = DEFAULT_CONFIG["language"]
    performance_mode: str = DEFAULT_CONFIG["performance_mode"]
    max_memory_usage_mb: int = DEFAULT_CONFIG["max_memory_usage_mb"]
    enable_logging: bool = DEFAULT_CONFIG["enable_logging"]
    log_level: str = DEFAULT_CONFIG["log_level"]
    
    # Enhanced Webcam Settings
    camera_auto_exposure: bool = DEFAULT_CONFIG["camera_auto_exposure"]
    camera_auto_focus: bool = DEFAULT_CONFIG["camera_auto_focus"]
    camera_brightness: int = DEFAULT_CONFIG["camera_brightness"]
    camera_contrast: int = DEFAULT_CONFIG["camera_contrast"]
    camera_saturation: int = DEFAULT_CONFIG["camera_saturation"]
    camera_recording_format: str = DEFAULT_CONFIG["camera_recording_format"]
    camera_buffer_size: int = DEFAULT_CONFIG["camera_buffer_size"]
    camera_preview_enabled: bool = DEFAULT_CONFIG["camera_preview_enabled"]
    camera_device_name: str = DEFAULT_CONFIG["camera_device_name"]
    
    # Enhanced Image Analysis Settings
    detection_confidence_threshold: float = DEFAULT_CONFIG["detection_confidence_threshold"]
    detection_iou_threshold: float = DEFAULT_CONFIG["detection_iou_threshold"]
    roi_x: int = DEFAULT_CONFIG["roi_x"]
    roi_y: int = DEFAULT_CONFIG["roi_y"]
    roi_width: int = DEFAULT_CONFIG["roi_width"]
    roi_height: int = DEFAULT_CONFIG["roi_height"]
    enable_roi: bool = DEFAULT_CONFIG["enable_roi"]
    preferred_model: str = DEFAULT_CONFIG["preferred_model"]
    export_quality: int = DEFAULT_CONFIG["export_quality"]  # Fixed at 100%
    difference_sensitivity: float = DEFAULT_CONFIG["difference_sensitivity"]
    highlight_differences: bool = DEFAULT_CONFIG["highlight_differences"]
    
    # Enhanced Chatbot Settings
    gemini_api_key: str = DEFAULT_CONFIG["gemini_api_key"]
    gemini_model: str = DEFAULT_CONFIG["gemini_model"]
    gemini_timeout: int = DEFAULT_CONFIG["gemini_timeout"]
    gemini_temperature: float = DEFAULT_CONFIG["gemini_temperature"]
    gemini_max_tokens: int = DEFAULT_CONFIG["gemini_max_tokens"]
    enable_ai_analysis: bool = DEFAULT_CONFIG["enable_ai_analysis"]
    chat_history_limit: int = DEFAULT_CONFIG["chat_history_limit"]
    chat_auto_save: bool = DEFAULT_CONFIG["chat_auto_save"]
    response_format: str = DEFAULT_CONFIG["response_format"]
    enable_rate_limiting: bool = DEFAULT_CONFIG["enable_rate_limiting"]
    requests_per_minute: int = DEFAULT_CONFIG["requests_per_minute"]
    context_window_size: int = DEFAULT_CONFIG["context_window_size"]
    enable_conversation_memory: bool = DEFAULT_CONFIG["enable_conversation_memory"]
    chatbot_persona: str = DEFAULT_CONFIG["chatbot_persona"]
    
    # Additional Enhanced Settings
    export_include_metadata: bool = DEFAULT_CONFIG["export_include_metadata"]
    reference_image_path: str = DEFAULT_CONFIG["reference_image_path"]
    analysis_history_days: int = DEFAULT_CONFIG["analysis_history_days"]
    chat_export_format: str = DEFAULT_CONFIG["chat_export_format"]
    
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


def load_config(path: str = "config.json") -> Config:
    """Load configuration from JSON file with comprehensive error handling."""
    data: Dict[str, Any] = {}
    
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
        
        # Validate critical path settings
        _validate_path_settings(merged)
        
        # capture unknown keys
        extra = {k: v for k, v in merged.items() if k not in Config.__annotations__}
        if extra:
            logging.info(f"Found extra configuration keys: {list(extra.keys())}")
        
        cfg = Config(**{k: merged[k] for k in Config.__annotations__ if k != 'extra'}, extra=extra)
        
        # Ensure critical directories exist
        _ensure_critical_directories(cfg)
        
        return cfg
    except Exception as e:
        logging.error(f"Failed to create configuration object: {e}. Falling back to pure defaults.")
        return Config()


def save_config(cfg: Config, path: str = "config.json") -> None:
    """Save configuration to JSON file with comprehensive error handling."""
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
        
        # Save new configuration
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)
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


__all__ = ["Config", "load_config", "save_config"]