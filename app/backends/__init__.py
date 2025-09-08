"""Backend implementations for different model types."""

from .base_backend import BaseBackend
from .yolo_backend import YoloBackend

__all__ = ["BaseBackend", "YoloBackend"]