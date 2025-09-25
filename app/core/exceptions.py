"""Custom exceptions for the application."""

class ApplicationError(Exception):
    """Base application error."""
    pass

class DetectionError(ApplicationError):
    """Base exception for detection-related errors."""
    pass

class ConfigError(ApplicationError):
    """Configuration-related errors."""
    pass

class ConfigurationError(ApplicationError):
    """Configuration application errors."""
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

class ServiceError(ApplicationError):
    """Service operation errors."""
    pass

class AIServiceError(ServiceError):
    """AI service specific errors."""
    pass

class RateLimitError(ServiceError):
    """Rate limiting errors."""
    pass

class SecurityError(ApplicationError):
    """Security-related errors."""
    pass