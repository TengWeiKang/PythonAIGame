"""Advanced logging configuration with structured logging and security features.

This module provides production-ready logging with correlation IDs, security-safe
message formatting, and proper log rotation for the Python Game Detection System.
"""
import logging
import logging.handlers
import sys
import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path
import threading
from contextvars import ContextVar

# Context variable for correlation IDs
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)

# Thread-local storage for correlation IDs (fallback)
_thread_local = threading.local()


class CorrelationIDFilter(logging.Filter):
    """Filter to add correlation IDs to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to the log record."""
        # Try context variable first, then thread-local fallback
        corr_id = correlation_id.get()
        if corr_id is None:
            corr_id = getattr(_thread_local, 'correlation_id', None)

        record.correlation_id = corr_id or 'no-correlation-id'
        return True


class SecuritySafeFormatter(logging.Formatter):
    """Formatter that sanitizes sensitive information from log messages."""

    # Sensitive patterns to redact
    SENSITIVE_PATTERNS = [
        # API keys
        (r'(?i)(api[_-]?key["\s]*[:=]["\s]*)[a-zA-Z0-9_-]+', r'\1[REDACTED]'),
        (r'(?i)(gemini[_-]?api[_-]?key["\s]*[:=]["\s]*)[a-zA-Z0-9_-]+', r'\1[REDACTED]'),
        # Passwords
        (r'(?i)(password["\s]*[:=]["\s]*)[^\s"]+', r'\1[REDACTED]'),
        (r'(?i)(pass["\s]*[:=]["\s]*)[^\s"]+', r'\1[REDACTED]'),
        # Tokens
        (r'(?i)(token["\s]*[:=]["\s]*)[a-zA-Z0-9_-]+', r'\1[REDACTED]'),
        # Email addresses
        (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL_REDACTED]'),
        # File paths that might contain user info
        (r'[C-Z]:\\\\Users\\\\[^\\s\\\\]+', '[USER_PATH_REDACTED]'),
        (r'/home/[^/\s]+', '[HOME_PATH_REDACTED]'),
    ]

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with sensitive data redaction."""
        # Format the record normally first
        formatted = super().format(record)

        # Sanitize sensitive information
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            import re
            formatted = re.sub(pattern, replacement, formatted)

        return formatted


class StructuredFormatter(SecuritySafeFormatter):
    """Structured JSON formatter for production logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'correlation_id': getattr(record, 'correlation_id', 'no-correlation-id'),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread_id': record.thread,
            'process_id': record.process,
        }

        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info) if record.exc_info else None
            }

        # Add extra fields if present
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'getMessage', 'exc_info',
                'exc_text', 'stack_info', 'correlation_id'
            }:
                extra_fields[key] = value

        if extra_fields:
            log_entry['extra'] = extra_fields

        # Apply security sanitization to the JSON string
        json_str = json.dumps(log_entry, default=str)
        return super(SecuritySafeFormatter, self).format(
            logging.LogRecord(
                record.name, record.levelno, record.pathname, record.lineno,
                json_str, (), None
            )
        )


class HumanReadableFormatter(SecuritySafeFormatter):
    """Human-readable formatter for development and console output."""

    def __init__(self, include_correlation_id: bool = True):
        self.include_correlation_id = include_correlation_id
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s'
            + (' - %(correlation_id)s' if include_correlation_id else '')
            + ' - %(message)s'
        )
        super().__init__(format_string)


class LoggingManager:
    """Central logging manager for the application."""

    def __init__(self):
        self._configured = False
        self._log_dir: Optional[Path] = None
        self._handlers: Dict[str, logging.Handler] = {}

    def configure(
        self,
        log_level: str = 'INFO',
        log_dir: Optional[Union[str, Path]] = None,
        enable_file_logging: bool = True,
        enable_console_logging: bool = True,
        structured_logging: bool = False,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        application_name: str = 'python-game-detection'
    ) -> None:
        """Configure logging for the application.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
            enable_file_logging: Enable logging to files
            enable_console_logging: Enable console logging
            structured_logging: Use structured JSON logging
            max_file_size: Maximum size of log files before rotation
            backup_count: Number of backup files to keep
            application_name: Name used for log files
        """
        if self._configured:
            return

        # Set up log directory
        if log_dir:
            self._log_dir = Path(log_dir)
            self._log_dir.mkdir(parents=True, exist_ok=True)
        elif enable_file_logging:
            self._log_dir = Path('logs')
            self._log_dir.mkdir(parents=True, exist_ok=True)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))

        # Clear existing handlers
        root_logger.handlers.clear()

        # Add correlation ID filter to root logger
        correlation_filter = CorrelationIDFilter()

        # Console handler
        if enable_console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))

            if structured_logging:
                console_formatter = StructuredFormatter()
            else:
                console_formatter = HumanReadableFormatter(include_correlation_id=True)

            console_handler.setFormatter(console_formatter)
            console_handler.addFilter(correlation_filter)
            root_logger.addHandler(console_handler)
            self._handlers['console'] = console_handler

        # File handlers
        if enable_file_logging and self._log_dir:
            # Application log file
            app_log_file = self._log_dir / f'{application_name}.log'
            app_handler = logging.handlers.RotatingFileHandler(
                app_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            app_handler.setLevel(getattr(logging, log_level.upper()))

            if structured_logging:
                app_formatter = StructuredFormatter()
            else:
                app_formatter = HumanReadableFormatter(include_correlation_id=True)

            app_handler.setFormatter(app_formatter)
            app_handler.addFilter(correlation_filter)
            root_logger.addHandler(app_handler)
            self._handlers['application'] = app_handler

            # Error log file (only ERROR and CRITICAL)
            error_log_file = self._log_dir / f'{application_name}-errors.log'
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(app_formatter)
            error_handler.addFilter(correlation_filter)
            root_logger.addHandler(error_handler)
            self._handlers['errors'] = error_handler

            # Security audit log
            security_log_file = self._log_dir / f'{application_name}-security.log'
            security_handler = logging.handlers.RotatingFileHandler(
                security_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            security_handler.setLevel(logging.INFO)
            security_handler.setFormatter(StructuredFormatter())
            security_handler.addFilter(correlation_filter)

            # Add security logger
            security_logger = logging.getLogger('security')
            security_logger.addHandler(security_handler)
            security_logger.setLevel(logging.INFO)
            security_logger.propagate = False
            self._handlers['security'] = security_handler

        # Configure specific loggers
        self._configure_specific_loggers()

        self._configured = True
        logging.info(f"Logging configured - Level: {log_level}, File: {enable_file_logging}, Console: {enable_console_logging}")

    def _configure_specific_loggers(self) -> None:
        """Configure specific loggers with appropriate levels."""
        # Suppress verbose third-party library logs
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)

        # Set application logger levels
        logging.getLogger('app.services').setLevel(logging.INFO)
        logging.getLogger('app.core').setLevel(logging.INFO)
        logging.getLogger('app.ui').setLevel(logging.INFO)

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name."""
        return logging.getLogger(name)

    def set_correlation_id(self, corr_id: Optional[str] = None) -> str:
        """Set correlation ID for current context."""
        if corr_id is None:
            corr_id = str(uuid.uuid4())

        # Set in context variable
        correlation_id.set(corr_id)

        # Set in thread-local storage as fallback
        _thread_local.correlation_id = corr_id

        return corr_id

    def get_correlation_id(self) -> Optional[str]:
        """Get current correlation ID."""
        corr_id = correlation_id.get()
        if corr_id is None:
            corr_id = getattr(_thread_local, 'correlation_id', None)
        return corr_id

    def clear_correlation_id(self) -> None:
        """Clear correlation ID from current context."""
        correlation_id.set(None)
        if hasattr(_thread_local, 'correlation_id'):
            del _thread_local.correlation_id

    def log_security_event(
        self,
        event_type: str,
        description: str,
        severity: str = 'INFO',
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a security-related event."""
        security_logger = logging.getLogger('security')

        event_data = {
            'event_type': event_type,
            'description': description,
            'severity': severity,
            'user_id': user_id,
            'ip_address': ip_address,
            'correlation_id': self.get_correlation_id(),
            'timestamp': datetime.utcnow().isoformat() + 'Z',
        }

        if additional_data:
            event_data['additional_data'] = additional_data

        security_logger.info(f"SECURITY_EVENT: {json.dumps(event_data)}")

    def shutdown(self) -> None:
        """Shutdown logging and close all handlers."""
        for handler in self._handlers.values():
            handler.close()
        self._handlers.clear()
        self._configured = False


# Global logging manager instance
logging_manager = LoggingManager()


def configure_logging(**kwargs) -> None:
    """Configure application logging."""
    logging_manager.configure(**kwargs)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging_manager.get_logger(name)


def set_correlation_id(corr_id: Optional[str] = None) -> str:
    """Set correlation ID for current context."""
    return logging_manager.set_correlation_id(corr_id)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return logging_manager.get_correlation_id()


def log_security_event(**kwargs) -> None:
    """Log a security event."""
    logging_manager.log_security_event(**kwargs)


# Context manager for correlation IDs
class CorrelationContext:
    """Context manager for correlation IDs."""

    def __init__(self, corr_id: Optional[str] = None):
        self.corr_id = corr_id
        self.previous_corr_id: Optional[str] = None

    def __enter__(self) -> str:
        """Enter context and set correlation ID."""
        self.previous_corr_id = get_correlation_id()
        return set_correlation_id(self.corr_id)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and restore previous correlation ID."""
        if self.previous_corr_id is not None:
            set_correlation_id(self.previous_corr_id)
        else:
            logging_manager.clear_correlation_id()


def with_correlation_id(corr_id: Optional[str] = None):
    """Decorator to set correlation ID for a function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with CorrelationContext(corr_id):
                return func(*args, **kwargs)
        return wrapper
    return decorator