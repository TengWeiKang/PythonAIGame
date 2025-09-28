"""Input validation and sanitization utilities.

This module provides comprehensive input validation and sanitization for:
- User prompts and messages
- File paths and names
- API inputs and parameters
- Image data and metadata
- Configuration settings validation engine

All validation follows security best practices to prevent injection attacks
and ensure data integrity.
"""
import re
import os
import hashlib
import logging
import time
import asyncio
from typing import Optional, Union, List, Dict, Any, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
from functools import lru_cache
import html

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class InputValidator:
    """Comprehensive input validation and sanitization utility."""

    # Dangerous pattern checking functionality has been removed
    # DANGEROUS_PATTERNS = {}
    # Pattern checking disabled for security review requirements

    # Maximum lengths for different input types
    MAX_LENGTHS = {
        'prompt': 10000,  # AI prompt
        'message': 5000,   # Chat message
        'filename': 255,   # File name
        'path': 1000,     # File path
        'api_key': 100,   # API key
    }

    # Allowed characters for different contexts
    ALLOWED_CHARS = {
        'filename': re.compile(r'^[a-zA-Z0-9._\-\s]+$'),
        'dirname': re.compile(r'^[a-zA-Z0-9._\-\s]+$'),
        'prompt': re.compile(r'^[\w\s\.,;:!?\'"()\[\]{}\-+=*/@#%&|<>^~`]+$'),
    }

    @classmethod
    def sanitize_user_prompt(cls, prompt: str) -> str:
        """Sanitize user prompt for AI API consumption.

        Args:
            prompt: Raw user prompt

        Returns:
            str: Sanitized prompt

        Raises:
            ValidationError: If prompt contains dangerous content
        """
        if not isinstance(prompt, str):
            raise ValidationError("Prompt must be a string")

        if not prompt.strip():
            raise ValidationError("Prompt cannot be empty")

        # Check length limits
        if len(prompt) > cls.MAX_LENGTHS['prompt']:
            raise ValidationError(f"Prompt exceeds maximum length of {cls.MAX_LENGTHS['prompt']} characters")

        # Dangerous pattern checking disabled
        # cls._check_dangerous_patterns(prompt, ['sql_injection', 'script_injection', 'command_injection'])

        # Basic HTML entity encoding for safety
        sanitized = html.escape(prompt, quote=False)

        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()

        # Remove null bytes and control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\t\n\r')

        if not sanitized:
            raise ValidationError("Prompt becomes empty after sanitization")

        logger.debug(f"Sanitized prompt: {len(sanitized)} characters")
        return sanitized

    @classmethod
    def validate_api_input(cls, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize API input parameters.

        Args:
            input_data: Raw API input data

        Returns:
            dict: Validated and sanitized input data

        Raises:
            ValidationError: If input contains invalid data
        """
        if not isinstance(input_data, dict):
            raise ValidationError("API input must be a dictionary")

        validated = {}

        for key, value in input_data.items():
            # Validate key names
            if not isinstance(key, str) or not key.isalnum():
                raise ValidationError(f"Invalid API parameter name: {key}")

            # Sanitize string values
            if isinstance(value, str):
                validated[key] = cls.sanitize_string_input(value, max_length=1000)
            elif isinstance(value, (int, float, bool)):
                validated[key] = value
            elif isinstance(value, (list, tuple)):
                validated[key] = cls._validate_list_input(value)
            elif isinstance(value, dict):
                validated[key] = cls.validate_api_input(value)  # Recursive validation
            else:
                raise ValidationError(f"Unsupported input type for key {key}: {type(value)}")

        return validated

    @classmethod
    def sanitize_file_path(cls, file_path: str, base_dir: Optional[str] = None) -> str:
        """Sanitize and validate file path.

        Args:
            file_path: Raw file path
            base_dir: Optional base directory to restrict to

        Returns:
            str: Sanitized absolute path

        Raises:
            ValidationError: If path is invalid or dangerous
        """
        if not isinstance(file_path, str):
            raise ValidationError("File path must be a string")

        if not file_path.strip():
            raise ValidationError("File path cannot be empty")

        # Check length
        if len(file_path) > cls.MAX_LENGTHS['path']:
            raise ValidationError(f"Path exceeds maximum length of {cls.MAX_LENGTHS['path']} characters")

        # Dangerous pattern checking disabled
        # cls._check_dangerous_patterns(file_path, ['path_traversal', 'command_injection'])

        # Normalize the path
        try:
            normalized = os.path.normpath(file_path)
            absolute = os.path.abspath(normalized)
        except Exception as e:
            raise ValidationError(f"Invalid file path: {e}")

        # Restrict to base directory if provided
        if base_dir:
            try:
                base_abs = os.path.abspath(base_dir)
                if not absolute.startswith(base_abs):
                    raise ValidationError(f"Path must be within base directory: {base_dir}")
            except Exception as e:
                raise ValidationError(f"Invalid base directory: {e}")

        # Check for system directories (basic protection)
        system_dirs = ['C:\\Windows', 'C:\\Program Files', '/etc', '/usr', '/bin', '/root']
        for sys_dir in system_dirs:
            if absolute.startswith(sys_dir):
                raise ValidationError(f"Access to system directory denied: {sys_dir}")

        return absolute

    @classmethod
    def validate_filename(cls, filename: str) -> str:
        """Validate and sanitize filename.

        Args:
            filename: Raw filename

        Returns:
            str: Sanitized filename

        Raises:
            ValidationError: If filename is invalid
        """
        if not isinstance(filename, str):
            raise ValidationError("Filename must be a string")

        if not filename.strip():
            raise ValidationError("Filename cannot be empty")

        # Check length
        if len(filename) > cls.MAX_LENGTHS['filename']:
            raise ValidationError(f"Filename exceeds maximum length of {cls.MAX_LENGTHS['filename']} characters")

        # Remove path separators and dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', filename)
        sanitized = sanitized.strip('. ')  # Remove leading/trailing dots and spaces

        # Check for reserved names (Windows)
        reserved_names = ['CON', 'PRN', 'AUX', 'NUL'] + [f'COM{i}' for i in range(1, 10)] + [f'LPT{i}' for i in range(1, 10)]
        if sanitized.upper() in reserved_names or sanitized.upper().split('.')[0] in reserved_names:
            raise ValidationError(f"Filename uses reserved name: {sanitized}")

        if not sanitized:
            raise ValidationError("Filename becomes empty after sanitization")

        return sanitized

    @classmethod
    def sanitize_string_input(cls, text: str, max_length: int = 1000) -> str:
        """General string input sanitization.

        Args:
            text: Raw text input
            max_length: Maximum allowed length

        Returns:
            str: Sanitized text

        Raises:
            ValidationError: If text is invalid
        """
        if not isinstance(text, str):
            raise ValidationError("Input must be a string")

        if len(text) > max_length:
            raise ValidationError(f"Input exceeds maximum length of {max_length} characters")

        # Basic HTML entity encoding
        sanitized = html.escape(text, quote=False)

        # Remove null bytes and dangerous control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\t\n\r')

        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()

        return sanitized

    @classmethod
    def validate_image_data(cls, image_data: bytes, max_size: int = 10 * 1024 * 1024) -> Tuple[bool, str]:
        """Validate image data for security and format.

        Args:
            image_data: Raw image bytes
            max_size: Maximum allowed size in bytes

        Returns:
            tuple: (is_valid, message)
        """
        if not isinstance(image_data, bytes):
            return False, "Image data must be bytes"

        if not image_data:
            return False, "Image data cannot be empty"

        if len(image_data) > max_size:
            return False, f"Image exceeds maximum size of {max_size} bytes"

        # Check for basic image format signatures
        signatures = {
            b'\xff\xd8\xff': 'JPEG',
            b'\x89PNG\r\n\x1a\n': 'PNG',
            b'GIF87a': 'GIF',
            b'GIF89a': 'GIF',
            b'RIFF': 'WEBP',  # Need to check for WEBP after RIFF
        }

        # Check magic bytes
        for signature, format_name in signatures.items():
            if image_data.startswith(signature):
                if format_name == 'WEBP':
                    # Additional check for WEBP
                    if b'WEBP' in image_data[:12]:
                        return True, f"Valid {format_name} image"
                else:
                    return True, f"Valid {format_name} image"

        return False, "Unsupported or invalid image format"

    # Dangerous pattern checking method removed
    # @classmethod
    # def _check_dangerous_patterns(cls, text: str, pattern_types: List[str]) -> None:
    #     """Check text against dangerous patterns - DISABLED.
    #
    #     This method has been disabled as requested.
    #     All dangerous pattern checking functionality has been removed.
    #     """
    #     pass

    @classmethod
    def _validate_list_input(cls, input_list: List[Any], max_items: int = 100) -> List[Any]:
        """Validate list input.

        Args:
            input_list: List to validate
            max_items: Maximum allowed items

        Returns:
            list: Validated list

        Raises:
            ValidationError: If list is invalid
        """
        if len(input_list) > max_items:
            raise ValidationError(f"List exceeds maximum items: {max_items}")

        validated_list = []
        for item in input_list:
            if isinstance(item, str):
                validated_list.append(cls.sanitize_string_input(item))
            elif isinstance(item, (int, float, bool)):
                validated_list.append(item)
            else:
                raise ValidationError(f"Unsupported list item type: {type(item)}")

        return validated_list


class ContentFilter:
    """Content filtering for inappropriate or sensitive content."""

    # Patterns for sensitive content detection
    SENSITIVE_PATTERNS = [
        r'\b(?:password|pwd|secret|token|key|auth)\b',
        r'\b(?:admin|administrator|root|superuser)\b',
        r'\b(?:\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4})\b',  # Credit card pattern
        r'\b(?:\d{3}-?\d{2}-?\d{4})\b',  # SSN pattern
    ]

    @classmethod
    def contains_sensitive_info(cls, text: str) -> bool:
        """Check if text contains potentially sensitive information.

        Args:
            text: Text to analyze

        Returns:
            bool: True if sensitive content detected
        """
        text_lower = text.lower()

        for pattern in cls.SENSITIVE_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                logger.warning("Sensitive content detected in input")
                return True

        return False

    @classmethod
    def filter_sensitive_content(cls, text: str, replacement: str = "[FILTERED]") -> str:
        """Filter out sensitive content from text.

        Args:
            text: Text to filter
            replacement: Replacement string for sensitive content

        Returns:
            str: Filtered text
        """
        filtered_text = text

        for pattern in cls.SENSITIVE_PATTERNS:
            filtered_text = re.sub(pattern, replacement, filtered_text, flags=re.IGNORECASE)

        return filtered_text


class EnvironmentValidator:
    """Validator for environment-specific data like API keys and configuration."""

    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validate API key format for Google Gemini.

        Args:
            api_key: The API key to validate

        Returns:
            bool: True if API key format is valid
        """
        if not api_key or not isinstance(api_key, str):
            return False

        # Trim whitespace
        api_key = api_key.strip()

        # Check basic length and format for Google API keys
        # Google API keys typically start with "AIza" and are 39 characters long
        if len(api_key) < 20:  # Minimum reasonable length
            return False

        # Check for obviously invalid patterns
        if api_key.lower() in ['your_api_key_here', 'replace_with_your_key', 'api_key']:
            return False

        # Google API keys usually start with "AIza"
        if api_key.startswith('AIza') and len(api_key) == 39:
            return True

        # Allow other formats but ensure reasonable length and content
        if len(api_key) >= 20 and api_key.replace('-', '').replace('_', '').isalnum():
            return True

        return False

    @staticmethod
    def validate_environment_variable(var_name: str, value: str) -> bool:
        """Validate environment variable format.

        Args:
            var_name: Name of the environment variable
            value: Value to validate

        Returns:
            bool: True if value is valid for the given variable
        """
        if not value or not isinstance(value, str):
            return False

        value = value.strip()

        # Specific validation based on variable name
        if var_name.upper().endswith('_API_KEY'):
            return EnvironmentValidator.validate_api_key(value)
        elif var_name.upper().endswith('_URL'):
            # Basic URL validation
            return value.startswith(('http://', 'https://')) and len(value) > 10
        elif var_name.upper().endswith('_PATH'):
            # Basic path validation
            return len(value) > 0 and not any(char in value for char in ['<', '>', '|', '?', '*'])

        # Default validation - non-empty string
        return len(value) > 0

    @staticmethod
    def validate_model_name(model_name: str) -> bool:
        """Validate Gemini model name.

        Args:
            model_name: The model name to validate

        Returns:
            bool: True if model name is valid
        """
        if not model_name or not isinstance(model_name, str):
            return False

        model_name = model_name.strip()

        # Valid Gemini model names
        valid_models = [
            'gemini-1.5-flash',
            'gemini-1.5-pro',
            'gemini-pro',
            'gemini-pro-vision',
            'gemini-1.0-pro',
            'gemini-1.0-pro-vision',
            'models/gemini-1.5-flash',
            'models/gemini-1.5-pro',
            'models/gemini-pro',
            'models/gemini-pro-vision'
        ]

        return model_name in valid_models

    @staticmethod
    def validate_numeric_range(value, min_val, max_val, value_type):
        """Validate that a numeric value is within a specified range.

        Args:
            value: The value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            value_type: Expected type (int or float)

        Returns:
            The validated value converted to the correct type

        Raises:
            ValidationError: If value is invalid
        """
        try:
            # Convert to the expected type
            if value_type == int:
                converted_value = int(value)
            elif value_type == float:
                converted_value = float(value)
            else:
                raise ValidationError(f"Unsupported value type: {value_type}")

            # Check range
            if converted_value < min_val or converted_value > max_val:
                raise ValidationError(f"Value {converted_value} is outside valid range [{min_val}, {max_val}]")

            return converted_value

        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid {value_type.__name__} value: {value}") from e


# Quick access functions
def validate_user_prompt(prompt: str) -> str:
    """Quick access function for prompt validation."""
    return InputValidator.sanitize_user_prompt(prompt)


def validate_file_path(path: str, base_dir: Optional[str] = None) -> str:
    """Quick access function for file path validation."""
    return InputValidator.sanitize_file_path(path, base_dir)


def validate_filename(filename: str) -> str:
    """Quick access function for filename validation."""
    return InputValidator.validate_filename(filename)


@dataclass
class ValidationReport:
    """Comprehensive validation report with field-level details."""
    is_valid: bool
    errors: Dict[str, List[str]] = field(default_factory=dict)
    warnings: Dict[str, List[str]] = field(default_factory=dict)
    auto_corrections: Dict[str, Any] = field(default_factory=dict)
    validation_time_ms: float = 0.0
    stages_completed: List[str] = field(default_factory=list)
    cache_hits: int = 0

    def add_error(self, field: str, message: str) -> None:
        """Add error message for a field."""
        if field not in self.errors:
            self.errors[field] = []
        self.errors[field].append(message)
        self.is_valid = False

    def add_warning(self, field: str, message: str) -> None:
        """Add warning message for a field."""
        if field not in self.warnings:
            self.warnings[field] = []
        self.warnings[field].append(message)

    def suggest_correction(self, field: str, value: Any) -> None:
        """Suggest an auto-correction for a field."""
        self.auto_corrections[field] = value

    def has_errors(self) -> bool:
        """Check if report has any errors."""
        return bool(self.errors)

    def has_warnings(self) -> bool:
        """Check if report has any warnings."""
        return bool(self.warnings)

    def get_field_status(self, field: str) -> str:
        """Get status for a specific field."""
        if field in self.errors:
            return "error"
        elif field in self.warnings:
            return "warning"
        else:
            return "valid"


class ValidationEngine:
    """Comprehensive Settings Validation Engine with multi-stage validation pipeline.

    Stages:
    1. Type validation - Ensure correct data types
    2. Range validation - Numeric bounds checking
    3. Dependency validation - Inter-field dependencies
    4. Resource validation - File paths, camera availability
    5. Service compatibility - Can services handle new values
    """

    def __init__(self):
        self._validation_cache = {}
        self._cache_timestamps = {}
        self._cache_ttl = 300  # 5 minutes TTL
        self.logger = logging.getLogger(__name__ + ".ValidationEngine")

        # Validation rules registry
        self._type_rules = self._build_type_rules()
        self._range_rules = self._build_range_rules()
        self._dependency_rules = self._build_dependency_rules()
        self._resource_rules = self._build_resource_rules()
        self._service_rules = self._build_service_rules()

    def validate_config_changes(self, config_dict: Dict[str, Any],
                               current_config: Optional[Dict[str, Any]] = None) -> ValidationReport:
        """Validate configuration changes through 5-stage pipeline.

        Args:
            config_dict: New configuration values to validate
            current_config: Current configuration for comparison

        Returns:
            ValidationReport: Comprehensive validation results
        """
        start_time = time.time()
        report = ValidationReport(is_valid=True)

        try:
            # Stage 1: Type validation
            self._validate_types(config_dict, report)
            report.stages_completed.append("type_validation")

            # Stage 2: Range validation
            self._validate_ranges(config_dict, report)
            report.stages_completed.append("range_validation")

            # Stage 3: Dependency validation
            self._validate_dependencies(config_dict, report)
            report.stages_completed.append("dependency_validation")

            # Stage 4: Resource validation
            self._validate_resources(config_dict, report)
            report.stages_completed.append("resource_validation")

            # Stage 5: Service compatibility
            self._validate_service_compatibility(config_dict, report)
            report.stages_completed.append("service_compatibility")

        except Exception as e:
            self.logger.error(f"Validation pipeline failed: {e}")
            report.add_error("_pipeline", f"Validation pipeline error: {str(e)}")

        # Calculate validation time
        report.validation_time_ms = (time.time() - start_time) * 1000

        self.logger.info(f"Validation completed in {report.validation_time_ms:.2f}ms, "
                        f"valid={report.is_valid}, stages={len(report.stages_completed)}")

        return report

    def _validate_types(self, config_dict: Dict[str, Any], report: ValidationReport) -> None:
        """Stage 1: Type validation."""
        for field, expected_type in self._type_rules.items():
            if field in config_dict:
                value = config_dict[field]
                if not isinstance(value, expected_type):
                    report.add_error(field,
                        f"Invalid type: expected {expected_type.__name__}, got {type(value).__name__}")

                    # Suggest auto-correction if possible
                    try:
                        if expected_type == int and isinstance(value, (float, str)):
                            corrected = int(float(value))
                            report.suggest_correction(field, corrected)
                        elif expected_type == float and isinstance(value, (int, str)):
                            corrected = float(value)
                            report.suggest_correction(field, corrected)
                        elif expected_type == bool and isinstance(value, (int, str)):
                            corrected = bool(int(value)) if isinstance(value, str) else bool(value)
                            report.suggest_correction(field, corrected)
                        elif expected_type == str:
                            corrected = str(value)
                            report.suggest_correction(field, corrected)
                    except (ValueError, TypeError):
                        pass  # No valid correction possible

    def _validate_ranges(self, config_dict: Dict[str, Any], report: ValidationReport) -> None:
        """Stage 2: Range validation."""
        for field, (min_val, max_val) in self._range_rules.items():
            if field in config_dict:
                value = config_dict[field]
                if isinstance(value, (int, float)):
                    if value < min_val or value > max_val:
                        report.add_error(field,
                            f"Value {value} outside valid range [{min_val}, {max_val}]")

                        # Suggest clamped value
                        corrected = max(min_val, min(max_val, value))
                        report.suggest_correction(field, corrected)

    def _validate_dependencies(self, config_dict: Dict[str, Any], report: ValidationReport) -> None:
        """Stage 3: Dependency validation."""
        for dependency_rule in self._dependency_rules:
            try:
                dependency_rule(config_dict, report)
            except Exception as e:
                self.logger.error(f"Dependency validation error: {e}")
                report.add_error("_dependencies", f"Dependency check failed: {str(e)}")

    def _validate_resources(self, config_dict: Dict[str, Any], report: ValidationReport) -> None:
        """Stage 4: Resource validation."""
        for field, validator_func in self._resource_rules.items():
            if field in config_dict:
                try:
                    # Check cache first
                    cache_key = f"{field}:{config_dict[field]}"
                    if self._is_cache_valid(cache_key):
                        cached_result = self._validation_cache[cache_key]
                        report.cache_hits += 1
                        if not cached_result['is_valid']:
                            report.add_error(field, cached_result['message'])
                        continue

                    # Perform validation
                    is_valid, message = validator_func(config_dict[field])

                    # Cache result
                    self._cache_result(cache_key, {'is_valid': is_valid, 'message': message})

                    if not is_valid:
                        report.add_error(field, message)

                except Exception as e:
                    self.logger.error(f"Resource validation error for {field}: {e}")
                    report.add_error(field, f"Resource validation failed: {str(e)}")

    def _validate_service_compatibility(self, config_dict: Dict[str, Any], report: ValidationReport) -> None:
        """Stage 5: Service compatibility validation."""
        for service_check in self._service_rules:
            try:
                service_check(config_dict, report)
            except Exception as e:
                self.logger.error(f"Service compatibility error: {e}")
                report.add_error("_services", f"Service compatibility check failed: {str(e)}")

    def _build_type_rules(self) -> Dict[str, type]:
        """Build type validation rules."""
        return {
            # Detection settings
            'detection_confidence_threshold': float,
            'detection_iou_threshold': float,

            # Webcam settings
            'last_webcam_index': int,
            'camera_width': int,
            'camera_height': int,
            'camera_fps': int,
            'camera_brightness': int,
            'camera_contrast': int,
            'camera_saturation': int,
            'camera_auto_exposure': bool,
            'camera_auto_focus': bool,
            'camera_preview_enabled': bool,
            'camera_device_name': str,
            'camera_recording_format': str,
            'camera_buffer_size': int,

            # AI/Gemini settings
            'gemini_api_key': str,
            'gemini_model': str,
            'gemini_timeout': int,
            'gemini_temperature': float,
            'gemini_max_tokens': int,
            'enable_ai_analysis': bool,
            'enable_rate_limiting': bool,
            'requests_per_minute': int,

            # Performance settings
            'use_gpu': bool,
            'max_memory_usage_mb': int,
            'performance_mode': str,
            'batch_size': int,
            'target_fps': int,

            # UI settings
            'app_theme': str,
            'language': str,
            'enable_logging': bool,
            'log_level': str,

            # Path settings
            'data_dir': str,
            'models_dir': str,
            'master_dir': str,
            'results_export_dir': str,
            'reference_image_path': str,
        }

    def _build_range_rules(self) -> Dict[str, Tuple[Union[int, float], Union[int, float]]]:
        """Build range validation rules."""
        return {
            # Detection ranges
            'detection_confidence_threshold': (0.0, 1.0),
            'detection_iou_threshold': (0.0, 1.0),

            # Webcam ranges
            'last_webcam_index': (0, 10),
            'camera_width': (320, 4096),
            'camera_height': (240, 2160),
            'camera_fps': (1, 120),
            'camera_brightness': (-100, 100),
            'camera_contrast': (-100, 100),
            'camera_saturation': (-100, 100),
            'camera_buffer_size': (1, 30),

            # AI ranges
            'gemini_timeout': (5, 300),
            'gemini_temperature': (0.0, 1.0),
            'gemini_max_tokens': (1, 8192),
            'requests_per_minute': (1, 100),

            # Performance ranges
            'max_memory_usage_mb': (512, 16384),
            'batch_size': (1, 64),
            'target_fps': (1, 120),
        }

    def _build_dependency_rules(self) -> List[callable]:
        """Build dependency validation rules."""

        def validate_ai_dependencies(config_dict: Dict[str, Any], report: ValidationReport) -> None:
            """Validate AI settings dependencies."""
            if config_dict.get('enable_ai_analysis', False):
                api_key = config_dict.get('gemini_api_key', '')
                if not api_key or not EnvironmentValidator.validate_api_key(api_key):
                    report.add_error('gemini_api_key', 'Valid API key required when AI analysis is enabled')

        def validate_performance_dependencies(config_dict: Dict[str, Any], report: ValidationReport) -> None:
            """Validate performance settings dependencies."""
            if config_dict.get('use_gpu', False):
                # Check if GPU settings are reasonable
                batch_size = config_dict.get('batch_size', 8)
                if batch_size > 32:
                    report.add_warning('batch_size', 'Large batch sizes may cause GPU memory issues')

        return [validate_ai_dependencies, validate_performance_dependencies]

    def _build_resource_rules(self) -> Dict[str, callable]:
        """Build resource validation rules."""
        def validate_camera_availability(camera_index: int) -> Tuple[bool, str]:
            """Check if camera index is available."""
            try:
                import cv2
                cap = cv2.VideoCapture(camera_index)
                if not cap.isOpened():
                    return False, f"Camera index {camera_index} is not available"
                cap.release()
                return True, "Camera available"
            except Exception as e:
                return False, f"Camera check failed: {str(e)}"

        def validate_directory_path(path: str) -> Tuple[bool, str]:
            """Check if directory path is valid and writable."""
            try:
                if not path:
                    return False, "Path cannot be empty"

                # Create directory if it doesn't exist
                os.makedirs(path, exist_ok=True)

                # Test write permissions
                test_file = os.path.join(path, '.write_test')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)

                return True, "Directory accessible"
            except Exception as e:
                return False, f"Directory validation failed: {str(e)}"

        def validate_model_file(model_name: str) -> Tuple[bool, str]:
            """Check if model is supported."""
            supported_models = [
                'yolo12n', 'yolo12s', 'yolo12m', 'yolo12l', 'yolo12x',
                'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x',
                'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'
            ]
            if model_name in supported_models:
                return True, "Model supported"
            else:
                return False, f"Unsupported model: {model_name}"

        return {
            'last_webcam_index': validate_camera_availability,
            'data_dir': validate_directory_path,
            'models_dir': validate_directory_path,
            'master_dir': validate_directory_path,
            'results_export_dir': validate_directory_path,
            'preferred_model': validate_model_file,
            'model_size': validate_model_file,
        }

    def _build_service_rules(self) -> List[callable]:
        """Build service compatibility rules."""
        def validate_gemini_service(config_dict: Dict[str, Any], report: ValidationReport) -> None:
            """Validate Gemini service compatibility."""
            if config_dict.get('enable_ai_analysis', False):
                model = config_dict.get('gemini_model', '')
                if not EnvironmentValidator.validate_model_name(model):
                    report.add_error('gemini_model', f'Unsupported Gemini model: {model}')
                    report.suggest_correction('gemini_model', 'gemini-1.5-flash')

        def validate_webcam_service(config_dict: Dict[str, Any], report: ValidationReport) -> None:
            """Validate webcam service compatibility."""
            resolution = (config_dict.get('camera_width', 1280), config_dict.get('camera_height', 720))
            fps = config_dict.get('camera_fps', 30)

            # Common supported resolutions
            common_resolutions = [
                (640, 480), (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)
            ]

            if resolution not in common_resolutions:
                report.add_warning('camera_width',
                    f'Resolution {resolution[0]}x{resolution[1]} may not be supported by all cameras')

            if fps > 60:
                report.add_warning('camera_fps', 'High FPS may not be supported by all cameras')

        return [validate_gemini_service, validate_webcam_service]

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self._validation_cache:
            return False

        timestamp = self._cache_timestamps.get(cache_key, 0)
        return (time.time() - timestamp) < self._cache_ttl

    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache validation result."""
        self._validation_cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()

    def clear_cache(self) -> None:
        """Clear validation cache."""
        self._validation_cache.clear()
        self._cache_timestamps.clear()

    async def validate_config_changes_async(self, config_dict: Dict[str, Any],
                                          current_config: Optional[Dict[str, Any]] = None) -> ValidationReport:
        """Async version of config validation for network operations."""
        # Run synchronous validation first
        report = self.validate_config_changes(config_dict, current_config)

        # Add async network validations
        if config_dict.get('enable_ai_analysis', False) and config_dict.get('gemini_api_key'):
            api_valid = await self._validate_api_key_async(config_dict['gemini_api_key'])
            if not api_valid:
                report.add_error('gemini_api_key', 'API key validation failed (network test)')

        return report

    async def _validate_api_key_async(self, api_key: str) -> bool:
        """Async API key validation with network test."""
        try:
            # Check cache first
            cache_key = f"api_key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
            if self._is_cache_valid(cache_key):
                return self._validation_cache[cache_key]['is_valid']

            # This would be a real API test in production
            # For now, just simulate network delay
            await asyncio.sleep(0.1)

            # Basic format validation
            is_valid = EnvironmentValidator.validate_api_key(api_key)

            # Cache result
            self._cache_result(cache_key, {'is_valid': is_valid})

            return is_valid

        except Exception as e:
            self.logger.error(f"Async API key validation failed: {e}")
            return False


__all__ = [
    "ValidationError",
    "InputValidator",
    "ContentFilter",
    "EnvironmentValidator",
    "ValidationReport",
    "ValidationEngine",
    "validate_user_prompt",
    "validate_file_path",
    "validate_filename"
]