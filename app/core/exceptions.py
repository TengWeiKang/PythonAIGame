"""Custom exceptions for the application."""

class DetectionError(Exception):
    """Base exception for detection-related errors."""
    pass

class ConfigError(Exception):
    """Configuration-related errors."""
    pass

class ModelError(Exception):
    """Model loading/inference errors."""
    pass

class WebcamError(Exception):
    """Webcam access errors."""
    pass

class ValidationError(Exception):
    """Data validation errors."""
    pass

class ServiceError(Exception):
    """Service operation errors."""
    pass