"""Core domain entities and constants."""

from .entities import Detection, MasterObject, MatchResult, PipelineState, BBox
from .exceptions import DetectionError, ConfigError, ModelError
from .constants import APP_NAME, VERSION, SUPPORTED_IMAGE_FORMATS

__all__ = [
    "Detection", "MasterObject", "MatchResult", "PipelineState", "BBox",
    "DetectionError", "ConfigError", "ModelError",
    "APP_NAME", "VERSION", "SUPPORTED_IMAGE_FORMATS"
]