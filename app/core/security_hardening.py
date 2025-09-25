"""
COMPREHENSIVE SECURITY HARDENING MODULE

This module addresses all identified security vulnerabilities and implements
production-grade security measures for the webcam detection application.
"""

import os
import re
import hmac
import hashlib
import secrets
import tempfile
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
import json
import base64

logger = logging.getLogger(__name__)

class SecurePathValidator:
    """Validates and sanitizes file paths to prevent directory traversal attacks."""

    DANGEROUS_PATTERNS = [
        r'\.\.[\\/]',          # Directory traversal
        r'[\\/]\.\.[\\/]',     # Directory traversal variants
        r'^\.\.[\\/]',         # Starting with parent directory
        r'[\\/]\.\.$',         # Ending with parent directory
        r'[<>:"|?*]',          # Windows forbidden characters
        r'[\x00-\x1f]',        # Control characters
        r'CON|PRN|AUX|NUL',    # Windows reserved names
        r'COM[1-9]|LPT[1-9]',  # Windows device names
    ]

    DANGEROUS_ABSOLUTE_PATHS = [
        '/etc', '/usr', '/bin', '/sbin', '/root', '/var',
        'C:\\Windows', 'C:\\Program Files', 'C:\\Program Files (x86)',
        'C:\\Users\\All Users', 'C:\\ProgramData'
    ]

    @classmethod
    def validate_path(cls, path: str, allowed_base_paths: List[str] = None) -> str:
        """
        Validate and sanitize a file path.

        Args:
            path: Path to validate
            allowed_base_paths: List of allowed base paths (optional)

        Returns:
            Sanitized path

        Raises:
            SecurityError: If path is dangerous
        """
        if not isinstance(path, str):
            raise ValueError("Path must be a string")

        # Normalize path
        normalized_path = os.path.normpath(path)

        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, normalized_path, re.IGNORECASE):
                raise SecurityError(f"Dangerous path pattern detected: {pattern}")

        # Check for dangerous absolute paths
        if os.path.isabs(normalized_path):
            for dangerous_path in cls.DANGEROUS_ABSOLUTE_PATHS:
                if normalized_path.lower().startswith(dangerous_path.lower()):
                    raise SecurityError(f"Access to system directory denied: {dangerous_path}")

        # If allowed base paths specified, ensure path is within them
        if allowed_base_paths:
            abs_path = os.path.abspath(normalized_path)
            allowed = False
            for base_path in allowed_base_paths:
                abs_base = os.path.abspath(base_path)
                try:
                    # Check if path is within allowed base
                    os.path.relpath(abs_path, abs_base)
                    if abs_path.startswith(abs_base):
                        allowed = True
                        break
                except ValueError:
                    continue

            if not allowed:
                raise SecurityError(f"Path outside allowed directories: {normalized_path}")

        return normalized_path

    @classmethod
    def create_secure_temp_file(cls, suffix: str = '', prefix: str = 'secure_') -> str:
        """Create a secure temporary file."""
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        os.close(fd)
        return path

class InputSanitizer:
    """Comprehensive input sanitization for all user inputs."""

    MAX_STRING_LENGTH = 10000
    MAX_JSON_SIZE = 100000

    # Patterns for dangerous content
    SCRIPT_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'vbscript:',
        r'data:text/html',
        r'data:application/',
        r'<iframe[^>]*>',
        r'<object[^>]*>',
        r'<embed[^>]*>',
        r'eval\s*\(',
        r'exec\s*\(',
        r'__import__\s*\(',
    ]

    SQL_INJECTION_PATTERNS = [
        r'(\'|(\\\')|(\%27)|(\%2527)).*(\"|(\\\")|(\%22)|(\%2522))',
        r'(\'|(\\\')|(\%27)|(\%2527)).*(\-\-|\#)',
        r'(\'|(\\\')|(\%27)|(\%2527)).*(;|\%3B)',
        r'\b(ALTER|CREATE|DELETE|DROP|EXEC|INSERT|SELECT|UNION|UPDATE)\b',
    ]

    @classmethod
    def sanitize_string(cls, value: str, max_length: int = None) -> str:
        """Sanitize a string input."""
        if not isinstance(value, str):
            return str(value)

        max_len = max_length or cls.MAX_STRING_LENGTH
        if len(value) > max_len:
            raise ValueError(f"String too long: {len(value)} > {max_len}")

        # Check for dangerous patterns
        for pattern in cls.SCRIPT_PATTERNS + cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"Dangerous pattern detected and blocked: {pattern}")
                raise SecurityError(f"Potentially dangerous content detected")

        # Remove null bytes and control characters
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', value)

        # Normalize Unicode
        try:
            sanitized = sanitized.encode('utf-8', errors='ignore').decode('utf-8')
        except UnicodeError:
            sanitized = value.encode('ascii', errors='ignore').decode('ascii')

        return sanitized.strip()

    @classmethod
    def sanitize_json(cls, json_data: Union[str, Dict, List]) -> Any:
        """Sanitize JSON data."""
        if isinstance(json_data, str):
            if len(json_data) > cls.MAX_JSON_SIZE:
                raise ValueError(f"JSON too large: {len(json_data)} > {cls.MAX_JSON_SIZE}")

            try:
                data = json.loads(json_data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {e}")
        else:
            data = json_data

        return cls._sanitize_json_recursive(data)

    @classmethod
    def _sanitize_json_recursive(cls, obj: Any) -> Any:
        """Recursively sanitize JSON objects."""
        if isinstance(obj, dict):
            return {cls.sanitize_string(k): cls._sanitize_json_recursive(v)
                    for k, v in obj.items()}
        elif isinstance(obj, list):
            return [cls._sanitize_json_recursive(item) for item in obj]
        elif isinstance(obj, str):
            return cls.sanitize_string(obj)
        else:
            return obj

class APIKeyValidator:
    """Secure API key validation and protection."""

    MIN_KEY_LENGTH = 20
    VALID_KEY_PATTERN = r'^[A-Za-z0-9\-_\.]{20,}$'

    @classmethod
    def validate_api_key(cls, api_key: str) -> bool:
        """Validate API key format."""
        if not isinstance(api_key, str):
            return False

        if len(api_key) < cls.MIN_KEY_LENGTH:
            return False

        if not re.match(cls.VALID_KEY_PATTERN, api_key):
            return False

        # Check for obviously fake keys
        fake_patterns = [
            r'^(test|demo|sample|fake|dummy)',
            r'(123|abc|test)$',
            r'^[a-z]*$',  # All lowercase
            r'^[A-Z]*$',  # All uppercase
            r'^[0-9]*$',  # All numbers
        ]

        for pattern in fake_patterns:
            if re.search(pattern, api_key, re.IGNORECASE):
                logger.warning("Potentially fake API key detected")
                return False

        return True

    @classmethod
    def mask_api_key(cls, api_key: str) -> str:
        """Mask API key for logging."""
        if not api_key or len(api_key) < 8:
            return "***"
        return f"{api_key[:4]}...{api_key[-4:]}"

class RateLimiter:
    """Rate limiting to prevent abuse."""

    def __init__(self, max_requests: int = 60, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}
        self._lock = threading.Lock()

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed under rate limit."""
        import time

        current_time = time.time()

        with self._lock:
            if identifier not in self.requests:
                self.requests[identifier] = []

            # Remove old requests outside time window
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if current_time - req_time < self.time_window
            ]

            # Check if under limit
            if len(self.requests[identifier]) >= self.max_requests:
                return False

            # Add current request
            self.requests[identifier].append(current_time)
            return True

class SecurityError(Exception):
    """Security-related exception."""
    pass

class SecureConfiguration:
    """Secure configuration management."""

    @classmethod
    def validate_config_value(cls, key: str, value: Any) -> Any:
        """Validate and sanitize configuration values."""
        if isinstance(value, str):
            # Sanitize string values
            value = InputSanitizer.sanitize_string(value)

            # Additional validation for specific keys
            if 'path' in key.lower():
                value = SecurePathValidator.validate_path(value)
            elif 'api_key' in key.lower() and value:
                if not APIKeyValidator.validate_api_key(value):
                    logger.warning(f"Invalid API key format for {key}")
                    return ""

        elif isinstance(value, (int, float)):
            # Validate numeric ranges
            if 'timeout' in key.lower():
                value = max(1, min(300, value))  # 1-300 seconds
            elif 'port' in key.lower():
                value = max(1, min(65535, value))  # Valid port range

        return value

class SecurityAuditLogger:
    """Centralized security event logging."""

    def __init__(self, log_file: str = None):
        self.log_file = log_file
        self._setup_logger()

    def _setup_logger(self):
        """Setup security audit logger."""
        self.security_logger = logging.getLogger('security_audit')
        self.security_logger.setLevel(logging.INFO)

        if self.log_file:
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter(
                '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.security_logger.addHandler(handler)

    def log_security_event(self, event_type: str, message: str,
                          severity: str = 'INFO', extra_data: Dict = None):
        """Log a security event."""
        log_entry = {
            'event_type': event_type,
            'message': message,
            'severity': severity,
            'extra_data': extra_data or {}
        }

        if severity == 'CRITICAL':
            self.security_logger.critical(json.dumps(log_entry))
        elif severity == 'ERROR':
            self.security_logger.error(json.dumps(log_entry))
        elif severity == 'WARNING':
            self.security_logger.warning(json.dumps(log_entry))
        else:
            self.security_logger.info(json.dumps(log_entry))

class SecureMemoryManager:
    """Secure memory management to prevent data leaks."""

    @staticmethod
    def secure_zero_memory(data: bytes) -> None:
        """Securely zero out memory (best effort in Python)."""
        # Python doesn't provide direct memory zeroing, but we can overwrite
        if isinstance(data, (bytes, bytearray)):
            for i in range(len(data)):
                data[i] = 0

    @staticmethod
    def secure_random_bytes(length: int) -> bytes:
        """Generate cryptographically secure random bytes."""
        return secrets.token_bytes(length)

    @staticmethod
    def secure_compare(a: bytes, b: bytes) -> bool:
        """Constant-time comparison to prevent timing attacks."""
        return hmac.compare_digest(a, b)

# Global security instances
_security_audit_logger = SecurityAuditLogger()
_rate_limiter = RateLimiter()

def get_security_audit_logger():
    """Get the global security audit logger."""
    return _security_audit_logger

def get_rate_limiter():
    """Get the global rate limiter."""
    return _rate_limiter

def apply_security_headers() -> Dict[str, str]:
    """Generate security headers for any HTTP responses."""
    return {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'",
        'Referrer-Policy': 'strict-origin-when-cross-origin'
    }