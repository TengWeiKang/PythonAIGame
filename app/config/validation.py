"""Settings validation utilities and types."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import re


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    
    is_valid: bool
    error_message: Optional[str] = None
    corrected_value: Optional[Any] = None


class SettingsValidator:
    """Comprehensive settings validation with auto-correction."""
    
    # Theme validation
    VALID_THEMES = ["Dark", "Light", "Auto"]
    
    # Language codes
    VALID_LANGUAGES = ["en", "es", "fr", "de", "zh", "ja", "ko"]
    
    # Performance modes
    VALID_PERFORMANCE_MODES = ["Performance", "Balanced", "Power_Saving"]
    
    # Log levels
    VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]
    
    # Recording formats
    VALID_RECORDING_FORMATS = ["MP4", "AVI", "MOV"]
    
    # Export formats
    VALID_EXPORT_FORMATS = ["PNG", "JPG", "BMP"]
    
    # Response formats
    VALID_RESPONSE_FORMATS = ["Brief", "Detailed", "Technical"]
    
    # Chat export formats
    VALID_CHAT_EXPORT_FORMATS = ["JSON", "TXT", "CSV"]
    
    # Gemini models
    VALID_GEMINI_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]
    
    # Model sizes
    VALID_MODEL_SIZES = [
        "yolo12n", "yolo12s", "yolo12m", "yolo12l", "yolo12x",  # YOLOv12
        "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",  # YOLOv11
        "yolo10n", "yolo10s", "yolo10m", "yolo10l", "yolo10x",  # YOLOv10
    ]
    
    @staticmethod
    def validate_theme(value: str) -> ValidationResult:
        """Validate application theme."""
        if not isinstance(value, str):
            return ValidationResult(False, "Theme must be a string", "Dark")
        
        if value not in SettingsValidator.VALID_THEMES:
            return ValidationResult(
                False, 
                f"Invalid theme '{value}'. Must be one of: {', '.join(SettingsValidator.VALID_THEMES)}", 
                "Dark"
            )
        
        return ValidationResult(True)
    
    @staticmethod
    def validate_language(value: str) -> ValidationResult:
        """Validate language code."""
        if not isinstance(value, str):
            return ValidationResult(False, "Language must be a string", "en")
        
        if value not in SettingsValidator.VALID_LANGUAGES:
            return ValidationResult(
                False,
                f"Invalid language '{value}'. Must be one of: {', '.join(SettingsValidator.VALID_LANGUAGES)}",
                "en"
            )
        
        return ValidationResult(True)
    
    @staticmethod
    def validate_boolean(value: Any, field_name: str = "Value") -> ValidationResult:
        """Validate boolean values."""
        if not isinstance(value, bool):
            if isinstance(value, str):
                if value.lower() in ("true", "1", "yes", "on"):
                    return ValidationResult(True, None, True)
                elif value.lower() in ("false", "0", "no", "off"):
                    return ValidationResult(True, None, False)
                else:
                    return ValidationResult(False, f"{field_name} must be true or false", False)
            elif isinstance(value, (int, float)):
                return ValidationResult(True, None, bool(value))
            else:
                return ValidationResult(False, f"{field_name} must be true or false", False)
        
        return ValidationResult(True)
    
    @staticmethod
    def validate_integer_range(value: Any, min_val: int, max_val: int, field_name: str = "Value") -> ValidationResult:
        """Validate integer within range."""
        if not isinstance(value, (int, float)):
            if isinstance(value, str) and value.isdigit():
                value = int(value)
            else:
                return ValidationResult(False, f"{field_name} must be a number", min_val)
        
        value = int(value)
        
        if value < min_val:
            return ValidationResult(True, f"{field_name} was below minimum, corrected to {min_val}", min_val)
        elif value > max_val:
            return ValidationResult(True, f"{field_name} was above maximum, corrected to {max_val}", max_val)
        
        return ValidationResult(True)
    
    @staticmethod
    def validate_float_range(value: Any, min_val: float, max_val: float, field_name: str = "Value") -> ValidationResult:
        """Validate float within range."""
        try:
            value = float(value)
        except (ValueError, TypeError):
            return ValidationResult(False, f"{field_name} must be a number", min_val)
        
        if value < min_val:
            return ValidationResult(True, f"{field_name} was below minimum, corrected to {min_val}", min_val)
        elif value > max_val:
            return ValidationResult(True, f"{field_name} was above maximum, corrected to {max_val}", max_val)
        
        return ValidationResult(True)
    
    @staticmethod
    def validate_positive_integer(value: Any, field_name: str = "Value") -> ValidationResult:
        """Validate positive integer."""
        return SettingsValidator.validate_integer_range(value, 1, 2**31-1, field_name)
    
    @staticmethod
    def validate_string_choice(value: Any, choices: List[str], field_name: str = "Value") -> ValidationResult:
        """Validate string from list of choices."""
        if not isinstance(value, str):
            return ValidationResult(False, f"{field_name} must be a string", choices[0])
        
        if value not in choices:
            return ValidationResult(
                False,
                f"Invalid {field_name} '{value}'. Must be one of: {', '.join(choices)}",
                choices[0]
            )
        
        return ValidationResult(True)
    
    @staticmethod
    def validate_gemini_api_key(value: str) -> ValidationResult:
        """Validate Gemini API key format."""
        if not isinstance(value, str):
            return ValidationResult(False, "API key must be a string", "")
        
        # Allow empty string (unconfigured)
        if not value:
            return ValidationResult(True)
        
        # Basic format validation (starts with AIza)
        if not value.startswith("AIza") or len(value) < 30:
            return ValidationResult(False, "Invalid API key format. Should start with 'AIza' and be at least 30 characters.", "")
        
        return ValidationResult(True)
    
    @staticmethod
    def validate_camera_index(value: Any) -> ValidationResult:
        """Validate camera index."""
        return SettingsValidator.validate_integer_range(value, 0, 10, "Camera index")
    
    @staticmethod
    def validate_resolution_dimension(value: Any, field_name: str = "Resolution") -> ValidationResult:
        """Validate resolution width/height."""
        return SettingsValidator.validate_integer_range(value, 240, 4096, field_name)
    
    @staticmethod
    def validate_fps(value: Any) -> ValidationResult:
        """Validate FPS value."""
        return SettingsValidator.validate_integer_range(value, 1, 120, "FPS")
    
    @staticmethod
    def validate_chatbot_persona(value: str) -> ValidationResult:
        """Validate chatbot persona text."""
        if not isinstance(value, str):
            return ValidationResult(False, "Persona must be a string", "")
        
        # Allow empty string (use default)
        if not value.strip():
            return ValidationResult(True)
        
        # Check length constraints
        if len(value) > 5000:
            return ValidationResult(
                False, 
                "Persona text is too long (maximum 5000 characters)", 
                value[:5000]
            )
        
        return ValidationResult(True)
    
    @staticmethod
    def validate_directory_path(value: str, field_name: str = "Directory") -> ValidationResult:
        """Validate directory path format."""
        if not isinstance(value, str):
            return ValidationResult(False, f"{field_name} must be a string", "")
        
        # Allow empty string
        if not value:
            return ValidationResult(True)
        
        # Basic path validation (no invalid characters)
        invalid_chars = '<>"|?*'
        if any(char in value for char in invalid_chars):
            return ValidationResult(False, f"{field_name} contains invalid characters", "")
        
        return ValidationResult(True)
    
    @staticmethod
    def validate_all_settings(config_dict: Dict[str, Any]) -> Dict[str, ValidationResult]:
        """Validate all settings in a configuration dictionary."""
        results = {}
        
        # General Application Settings
        if "app_theme" in config_dict:
            results["app_theme"] = SettingsValidator.validate_theme(config_dict["app_theme"])
        
        if "language" in config_dict:
            results["language"] = SettingsValidator.validate_language(config_dict["language"])
        
        if "auto_save_config" in config_dict:
            results["auto_save_config"] = SettingsValidator.validate_boolean(
                config_dict["auto_save_config"], "Auto-save config"
            )
        
        if "startup_fullscreen" in config_dict:
            results["startup_fullscreen"] = SettingsValidator.validate_boolean(
                config_dict["startup_fullscreen"], "Startup fullscreen"
            )
        
        if "remember_window_state" in config_dict:
            results["remember_window_state"] = SettingsValidator.validate_boolean(
                config_dict["remember_window_state"], "Remember window state"
            )
        
        if "window_width" in config_dict:
            results["window_width"] = SettingsValidator.validate_integer_range(
                config_dict["window_width"], 800, 3840, "Window width"
            )
        
        if "window_height" in config_dict:
            results["window_height"] = SettingsValidator.validate_integer_range(
                config_dict["window_height"], 600, 2160, "Window height"
            )
        
        if "performance_mode" in config_dict:
            results["performance_mode"] = SettingsValidator.validate_string_choice(
                config_dict["performance_mode"], 
                SettingsValidator.VALID_PERFORMANCE_MODES,
                "Performance mode"
            )
        
        if "max_memory_usage_mb" in config_dict:
            results["max_memory_usage_mb"] = SettingsValidator.validate_integer_range(
                config_dict["max_memory_usage_mb"], 512, 16384, "Memory limit"
            )
        
        if "log_level" in config_dict:
            results["log_level"] = SettingsValidator.validate_string_choice(
                config_dict["log_level"],
                SettingsValidator.VALID_LOG_LEVELS,
                "Log level"
            )
        
        # Webcam Settings
        if "last_webcam_index" in config_dict:
            results["last_webcam_index"] = SettingsValidator.validate_camera_index(
                config_dict["last_webcam_index"]
            )
        
        if "camera_width" in config_dict:
            results["camera_width"] = SettingsValidator.validate_resolution_dimension(
                config_dict["camera_width"], "Camera width"
            )
        
        if "camera_height" in config_dict:
            results["camera_height"] = SettingsValidator.validate_resolution_dimension(
                config_dict["camera_height"], "Camera height"
            )
        
        if "camera_fps" in config_dict:
            results["camera_fps"] = SettingsValidator.validate_fps(config_dict["camera_fps"])
        
        if "camera_brightness" in config_dict:
            results["camera_brightness"] = SettingsValidator.validate_integer_range(
                config_dict["camera_brightness"], -100, 100, "Camera brightness"
            )
        
        if "camera_contrast" in config_dict:
            results["camera_contrast"] = SettingsValidator.validate_integer_range(
                config_dict["camera_contrast"], -100, 100, "Camera contrast"
            )
        
        if "camera_saturation" in config_dict:
            results["camera_saturation"] = SettingsValidator.validate_integer_range(
                config_dict["camera_saturation"], -100, 100, "Camera saturation"
            )
        
        if "camera_recording_format" in config_dict:
            results["camera_recording_format"] = SettingsValidator.validate_string_choice(
                config_dict["camera_recording_format"],
                SettingsValidator.VALID_RECORDING_FORMATS,
                "Recording format"
            )
        
        if "camera_buffer_size" in config_dict:
            results["camera_buffer_size"] = SettingsValidator.validate_integer_range(
                config_dict["camera_buffer_size"], 1, 30, "Buffer size"
            )
        
        # Image Analysis Settings
        if "detection_confidence_threshold" in config_dict:
            results["detection_confidence_threshold"] = SettingsValidator.validate_float_range(
                config_dict["detection_confidence_threshold"], 0.0, 1.0, "Confidence threshold"
            )
        
        if "detection_iou_threshold" in config_dict:
            results["detection_iou_threshold"] = SettingsValidator.validate_float_range(
                config_dict["detection_iou_threshold"], 0.0, 1.0, "IoU threshold"
            )
        
        if "export_format" in config_dict:
            results["export_format"] = SettingsValidator.validate_string_choice(
                config_dict["export_format"],
                SettingsValidator.VALID_EXPORT_FORMATS,
                "Export format"
            )
        
        if "export_quality" in config_dict:
            results["export_quality"] = SettingsValidator.validate_integer_range(
                config_dict["export_quality"], 1, 100, "Export quality"
            )
        
        if "difference_sensitivity" in config_dict:
            results["difference_sensitivity"] = SettingsValidator.validate_float_range(
                config_dict["difference_sensitivity"], 0.0, 1.0, "Difference sensitivity"
            )
        
        # Gemini Settings
        if "gemini_api_key" in config_dict:
            results["gemini_api_key"] = SettingsValidator.validate_gemini_api_key(
                config_dict["gemini_api_key"]
            )
        
        if "gemini_model" in config_dict:
            results["gemini_model"] = SettingsValidator.validate_string_choice(
                config_dict["gemini_model"],
                SettingsValidator.VALID_GEMINI_MODELS,
                "Gemini model"
            )
        
        if "gemini_timeout" in config_dict:
            results["gemini_timeout"] = SettingsValidator.validate_integer_range(
                config_dict["gemini_timeout"], 5, 300, "Timeout"
            )
        
        if "gemini_temperature" in config_dict:
            results["gemini_temperature"] = SettingsValidator.validate_float_range(
                config_dict["gemini_temperature"], 0.0, 1.0, "Temperature"
            )
        
        if "gemini_max_tokens" in config_dict:
            results["gemini_max_tokens"] = SettingsValidator.validate_integer_range(
                config_dict["gemini_max_tokens"], 100, 8192, "Max tokens"
            )
        
        if "chat_history_limit" in config_dict:
            results["chat_history_limit"] = SettingsValidator.validate_integer_range(
                config_dict["chat_history_limit"], 10, 1000, "Chat history limit"
            )
        
        if "response_format" in config_dict:
            results["response_format"] = SettingsValidator.validate_string_choice(
                config_dict["response_format"],
                SettingsValidator.VALID_RESPONSE_FORMATS,
                "Response format"
            )
        
        if "requests_per_minute" in config_dict:
            results["requests_per_minute"] = SettingsValidator.validate_integer_range(
                config_dict["requests_per_minute"], 1, 60, "Requests per minute"
            )
        
        if "context_window_size" in config_dict:
            results["context_window_size"] = SettingsValidator.validate_integer_range(
                config_dict["context_window_size"], 1000, 32000, "Context window size"
            )
        
        if "chatbot_persona" in config_dict:
            results["chatbot_persona"] = SettingsValidator.validate_chatbot_persona(
                config_dict["chatbot_persona"]
            )
        
        # Additional Enhanced Settings
        if "auto_save_interval_minutes" in config_dict:
            results["auto_save_interval_minutes"] = SettingsValidator.validate_integer_range(
                config_dict["auto_save_interval_minutes"], 1, 60, "Auto-save interval"
            )
        
        if "update_check_interval_days" in config_dict:
            results["update_check_interval_days"] = SettingsValidator.validate_integer_range(
                config_dict["update_check_interval_days"], 1, 30, "Update check interval"
            )
        
        if "analysis_history_days" in config_dict:
            results["analysis_history_days"] = SettingsValidator.validate_integer_range(
                config_dict["analysis_history_days"], 1, 365, "Analysis history days"
            )
        
        if "chat_export_format" in config_dict:
            results["chat_export_format"] = SettingsValidator.validate_string_choice(
                config_dict["chat_export_format"],
                SettingsValidator.VALID_CHAT_EXPORT_FORMATS,
                "Chat export format"
            )
        
        if "reference_image_path" in config_dict:
            results["reference_image_path"] = SettingsValidator.validate_directory_path(
                config_dict["reference_image_path"], "Reference image path"
            )
        
        # Validate directory paths
        for dir_key in ["default_data_dir", "default_models_dir", "default_results_dir"]:
            if dir_key in config_dict:
                results[dir_key] = SettingsValidator.validate_directory_path(
                    config_dict[dir_key], dir_key.replace("_", " ").title()
                )
        
        return results


class SettingsPresets:
    """Predefined settings configurations for different use cases."""
    
    PERFORMANCE_PRESET = {
        "performance_mode": "Performance",
        "max_memory_usage_mb": 4096,
        "camera_buffer_size": 1,
        "enable_noise_reduction": False,
        "enable_contrast_enhancement": False,
        "requests_per_minute": 30,
    }
    
    QUALITY_PRESET = {
        "performance_mode": "Balanced",
        "max_memory_usage_mb": 2048,
        "camera_buffer_size": 5,
        "enable_noise_reduction": True,
        "enable_contrast_enhancement": True,
        "detection_confidence_threshold": 0.3,
        "export_quality": 100,
    }
    
    POWER_SAVING_PRESET = {
        "performance_mode": "Power_Saving",
        "max_memory_usage_mb": 1024,
        "camera_buffer_size": 10,
        "camera_fps": 15,
        "use_gpu": False,
        "requests_per_minute": 5,
    }
    
    DEVELOPER_PRESET = {
        "debug": True,
        "enable_logging": True,
        "log_level": "DEBUG",
        "chat_auto_save": True,
        "enable_conversation_memory": True,
        "auto_save_config": True,
    }
    
    @staticmethod
    def get_preset(preset_name: str) -> Dict[str, Any]:
        """Get a predefined settings preset."""
        presets = {
            "performance": SettingsPresets.PERFORMANCE_PRESET,
            "quality": SettingsPresets.QUALITY_PRESET,
            "power_saving": SettingsPresets.POWER_SAVING_PRESET,
            "developer": SettingsPresets.DEVELOPER_PRESET,
        }
        
        return presets.get(preset_name.lower(), {})
    
    @staticmethod
    def get_available_presets() -> List[str]:
        """Get list of available preset names."""
        return ["Performance", "Quality", "Power Saving", "Developer"]


__all__ = [
    "ValidationResult",
    "SettingsValidator", 
    "SettingsPresets"
]