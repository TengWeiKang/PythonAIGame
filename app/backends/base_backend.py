"""Base backend interface for model implementations."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from ..core.entities import Detection

class BaseBackend(ABC):
    """Abstract base class for model backends."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_loaded = False
        self.model_info = {}

    @abstractmethod
    def load_model(self, model_path_or_name: str) -> bool:
        """Load a model from path or model name."""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray, **kwargs) -> List[Detection]:
        """Run inference on an image."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        pass

    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.is_loaded

    def unload_model(self) -> None:
        """Unload the current model to free memory."""
        self.is_loaded = False
        self.model_info = {}

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get list of supported model formats."""
        pass

    @abstractmethod
    def validate_model(self, model_path: str) -> bool:
        """Validate if a model file is compatible with this backend."""
        pass