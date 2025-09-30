"""Services package."""

from .webcam_service import WebcamService
from .inference_service import InferenceService
from .training_service import TrainingService, TrainingObject
from .gemini_service import GeminiService
from .reference_manager import ReferenceManager

__all__ = [
    'WebcamService',
    'InferenceService',
    'TrainingService',
    'TrainingObject',
    'GeminiService',
    'ReferenceManager',
]