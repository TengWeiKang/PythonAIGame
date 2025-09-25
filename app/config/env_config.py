"""Secure environment variable configuration management.

This module provides secure loading of environment variables with validation,
fallbacks, and proper error handling. It follows security best practices by:
- Never storing sensitive data in plaintext files
- Validating all environment variable inputs
- Providing secure defaults
- Logging security-related configuration issues
"""
import os
import logging
import re
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnvironmentConfig:
    """Immutable environment configuration object."""

    # API Configuration
    gemini_api_key: Optional[str]
    gemini_model: str
    gemini_timeout: int
    gemini_temperature: float
    gemini_max_tokens: int
    gemini_rate_limiting: bool
    gemini_requests_per_minute: int

    # Application Configuration
    debug_logging: bool
    data_dir: Optional[str]
    models_dir: Optional[str]
    results_export_dir: Optional[str]

    # Security flags
    is_api_key_configured: bool
    has_secure_config: bool


class EnvironmentError(Exception):
    """Custom exception for environment configuration errors."""
    pass


class EnvironmentValidator:
    """Validates environment variable values for security and correctness."""

    # API key patterns for validation
    GEMINI_API_KEY_PATTERN = re.compile(r'^AIza[0-9A-Za-z-_]{35}$')

    # Valid model names
    VALID_GEMINI_MODELS = {
        'gemini-1.5-flash',
        'gemini-1.5-pro',
        'gemini-1.0-pro',
        'gemini-pro',
        'gemini-pro-vision'
    }

    @classmethod
    def validate_api_key(cls, api_key: str) -> bool:
        """Validate Gemini API key format.

        Args:
            api_key: The API key to validate

        Returns:
            bool: True if valid format, False otherwise
        """
        if not api_key or not isinstance(api_key, str):
            return False

        # Check basic format
        if not cls.GEMINI_API_KEY_PATTERN.match(api_key):
            logger.warning("API key does not match expected Gemini format")
            return False

        return True

    @classmethod
    def validate_model_name(cls, model_name: str) -> bool:
        """Validate Gemini model name.

        Args:
            model_name: The model name to validate

        Returns:
            bool: True if valid, False otherwise
        """
        return model_name in cls.VALID_GEMINI_MODELS

    @classmethod
    def sanitize_path(cls, path: str) -> str:
        """Sanitize file path for security.

        Args:
            path: The path to sanitize

        Returns:
            str: Sanitized path

        Raises:
            EnvironmentError: If path contains dangerous characters
        """
        if not path:
            raise EnvironmentError("Path cannot be empty")

        # Check for dangerous path traversal patterns
        dangerous_patterns = ['..', '~', '$', '`', ';', '|', '&', '<', '>', '"', "'"]
        for pattern in dangerous_patterns:
            if pattern in path:
                raise EnvironmentError(f"Path contains dangerous pattern: {pattern}")

        # Normalize path separators
        normalized = os.path.normpath(path)

        # Ensure it's not an absolute path to sensitive areas
        if os.path.isabs(normalized):
            sensitive_dirs = ['/etc', '/root', '/home', '/usr', '/bin', '/sbin', 'C:\\Windows', 'C:\\Program Files']
            for sensitive in sensitive_dirs:
                if normalized.startswith(sensitive):
                    raise EnvironmentError(f"Path points to sensitive directory: {sensitive}")

        return normalized

    @classmethod
    def validate_numeric_range(cls, value: Union[str, int, float],
                              min_val: Optional[Union[int, float]] = None,
                              max_val: Optional[Union[int, float]] = None,
                              value_type: type = int) -> Union[int, float]:
        """Validate numeric value within specified range.

        Args:
            value: The value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            value_type: Expected type (int or float)

        Returns:
            The validated numeric value

        Raises:
            EnvironmentError: If validation fails
        """
        try:
            numeric_value = value_type(value)
        except (ValueError, TypeError):
            raise EnvironmentError(f"Invalid {value_type.__name__} value: {value}")

        if min_val is not None and numeric_value < min_val:
            raise EnvironmentError(f"Value {numeric_value} below minimum {min_val}")

        if max_val is not None and numeric_value > max_val:
            raise EnvironmentError(f"Value {numeric_value} above maximum {max_val}")

        return numeric_value


def load_env_file(env_path: Optional[str] = None) -> Dict[str, str]:
    """Load environment variables from .env file.

    Args:
        env_path: Path to .env file. Defaults to .env in current directory.

    Returns:
        dict: Loaded environment variables
    """
    if env_path is None:
        env_path = ".env"

    env_vars = {}
    env_file_path = Path(env_path)

    if not env_file_path.exists():
        logger.info(f"Environment file {env_path} not found, using system environment only")
        return env_vars

    try:
        with open(env_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Parse KEY=VALUE format
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]

                    env_vars[key] = value
                else:
                    logger.warning(f"Invalid line format in {env_path}:{line_num}: {line}")

        logger.info(f"Loaded {len(env_vars)} variables from {env_path}")

    except Exception as e:
        logger.error(f"Error reading environment file {env_path}: {e}")

    return env_vars


def get_env_var(key: str, default: Optional[str] = None,
                required: bool = False, env_vars: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Get environment variable with validation.

    Args:
        key: Environment variable name
        default: Default value if not found
        required: Whether the variable is required
        env_vars: Pre-loaded environment variables dict

    Returns:
        The environment variable value or default

    Raises:
        EnvironmentError: If required variable is missing
    """
    # Try loaded env vars first, then system environment
    if env_vars and key in env_vars:
        value = env_vars[key]
    else:
        value = os.getenv(key, default)

    if required and (value is None or value.strip() == ""):
        raise EnvironmentError(f"Required environment variable '{key}' is not set")

    return value


def load_environment_config(env_file_path: Optional[str] = None) -> EnvironmentConfig:
    """Load and validate environment configuration.

    Args:
        env_file_path: Path to .env file

    Returns:
        EnvironmentConfig: Validated configuration object

    Raises:
        EnvironmentError: If critical configuration is invalid
    """
    logger.info("Loading environment configuration...")

    # Load environment file
    env_vars = load_env_file(env_file_path)
    validator = EnvironmentValidator()

    try:
        # API Configuration
        api_key = get_env_var("GEMINI_API_KEY", env_vars=env_vars)
        is_api_configured = False

        if api_key:
            if validator.validate_api_key(api_key):
                is_api_configured = True
                logger.info("Valid Gemini API key configured")
            else:
                logger.warning("Invalid Gemini API key format detected")
                api_key = None
        else:
            # Single, clear message for missing API key
            pass  # We'll handle this in the main application

        # Model configuration
        model = get_env_var("GEMINI_MODEL", "gemini-1.5-flash", env_vars=env_vars)
        if not validator.validate_model_name(model):
            logger.warning(f"Invalid model name '{model}', using default")
            model = "gemini-1.5-flash"

        # Timeout configuration
        timeout_str = get_env_var("GEMINI_TIMEOUT", "30", env_vars=env_vars)
        timeout = validator.validate_numeric_range(timeout_str, 5, 300, int)

        # Temperature configuration
        temp_str = get_env_var("GEMINI_TEMPERATURE", "0.7", env_vars=env_vars)
        temperature = validator.validate_numeric_range(temp_str, 0.0, 1.0, float)

        # Max tokens configuration
        tokens_str = get_env_var("GEMINI_MAX_TOKENS", "2048", env_vars=env_vars)
        max_tokens = validator.validate_numeric_range(tokens_str, 1, 8192, int)

        # Rate limiting configuration
        rate_limit_str = get_env_var("GEMINI_RATE_LIMITING", "true", env_vars=env_vars)
        rate_limiting = rate_limit_str.lower() in ('true', '1', 'yes', 'on')

        # Requests per minute configuration
        rpm_str = get_env_var("GEMINI_REQUESTS_PER_MINUTE", "15", env_vars=env_vars)
        requests_per_minute = validator.validate_numeric_range(rpm_str, 1, 100, int)

        # Debug logging
        debug_str = get_env_var("DEBUG_LOGGING", "false", env_vars=env_vars)
        debug_logging = debug_str.lower() in ('true', '1', 'yes', 'on')

        # Path configurations (optional overrides)
        data_dir = get_env_var("DATA_DIR", env_vars=env_vars)
        models_dir = get_env_var("MODELS_DIR", env_vars=env_vars)
        results_dir = get_env_var("RESULTS_EXPORT_DIR", env_vars=env_vars)

        # Validate paths if provided
        if data_dir:
            data_dir = validator.sanitize_path(data_dir)
        if models_dir:
            models_dir = validator.sanitize_path(models_dir)
        if results_dir:
            results_dir = validator.sanitize_path(results_dir)

        # Security assessment
        has_secure_config = (
            is_api_configured and
            rate_limiting and
            timeout <= 60  # Reasonable timeout limit
        )

        config = EnvironmentConfig(
            gemini_api_key=api_key,
            gemini_model=model,
            gemini_timeout=timeout,
            gemini_temperature=temperature,
            gemini_max_tokens=max_tokens,
            gemini_rate_limiting=rate_limiting,
            gemini_requests_per_minute=requests_per_minute,
            debug_logging=debug_logging,
            data_dir=data_dir,
            models_dir=models_dir,
            results_export_dir=results_dir,
            is_api_key_configured=is_api_configured,
            has_secure_config=has_secure_config
        )

        # Single consolidated message about configuration status
        if is_api_configured:
            if has_secure_config:
                logger.info("Environment configuration loaded with secure AI settings")
            else:
                logger.info("Environment configuration loaded - AI features enabled with basic security")
        else:
            logger.info("Environment configuration loaded - Configure API key to enable AI features")

        return config

    except Exception as e:
        logger.error(f"Failed to load environment configuration: {e}")
        raise EnvironmentError(f"Environment configuration error: {e}")


__all__ = [
    "EnvironmentConfig",
    "EnvironmentError",
    "EnvironmentValidator",
    "load_environment_config",
    "load_env_file",
    "get_env_var"
]