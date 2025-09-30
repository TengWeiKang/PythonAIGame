"""Base classes for AI and ML services.

This module provides specialized base classes for AI/ML services with common
functionality like model management, inference patterns, and data validation.
"""
from __future__ import annotations

import abc
import hashlib
import time
from typing import Any, Dict, List, Optional, Union, Callable, Protocol
from pathlib import Path
import numpy as np

from .base_service import FullFeaturedService, BaseService
from .exceptions import ModelError, ValidationError, ServiceError
from ..utils.validation import InputValidator, ContentFilter


class ModelProvider(Protocol):
    """Protocol for model providers."""

    def load_model(self, model_path: str) -> Any:
        """Load a model from path."""
        ...

    def predict(self, *args, **kwargs) -> Any:
        """Make predictions with the model."""
        ...


class AIServiceBase(FullFeaturedService):
    """Base class for AI/ML services.

    Provides common functionality for AI services including:
    - Model loading and management
    - Input validation and sanitization
    - Prediction caching
    - Performance monitoring
    - Error handling specific to AI operations
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 cache_predictions: bool = True,
                 validate_inputs: bool = True,
                 **kwargs):
        """Initialize AI service base.

        Args:
            model_path: Path to model file (optional)
            cache_predictions: Whether to cache predictions
            validate_inputs: Whether to validate inputs before processing
            **kwargs: Additional arguments for base service
        """
        # Set up caching if requested
        if cache_predictions:
            kwargs['cache_size'] = kwargs.get('cache_size', 500)

        super().__init__(**kwargs)

        self._model_path = model_path
        self._model = None
        self._model_metadata = {}
        self._validate_inputs = validate_inputs

        # AI-specific metrics
        self._prediction_count = 0
        self._average_inference_time = 0.0
        self._last_inference_time = 0.0

        # Model performance tracking
        self._model_accuracy_samples = []
        self._confidence_threshold = 0.5

    @property
    def model(self) -> Any:
        """Get the loaded model."""
        return self._model

    @property
    def model_path(self) -> Optional[str]:
        """Get the model path."""
        return self._model_path

    @property
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def load_model(self, model_path: str, **kwargs) -> None:
        """Load AI model with validation and caching.

        Args:
            model_path: Path to the model file
            **kwargs: Additional model loading parameters

        Raises:
            ModelError: If model loading fails
            ValidationError: If model path is invalid
        """
        if not model_path or not isinstance(model_path, str):
            raise ValidationError("Model path must be a non-empty string")

        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise ModelError(f"Model file does not exist: {model_path}")

        self.logger.info(f"Loading model from {model_path}")

        with self.operation_context("load_model"):
            try:
                # Check cache first
                cache_key = f"model:{hashlib.md5(model_path.encode()).hexdigest()}"
                cached_model = self._get_from_cache(cache_key)

                if cached_model is not None:
                    self._model = cached_model
                    self.logger.debug("Model loaded from cache")
                else:
                    # Load model using subclass implementation
                    self._model = self._load_model_impl(model_path, **kwargs)

                    # Cache the model if it's not too large
                    if self._should_cache_model(self._model):
                        self._put_in_cache(cache_key, self._model)

                # Store metadata
                self._model_path = model_path
                self._model_metadata = self._extract_model_metadata(self._model)

                self.logger.info(f"Model loaded successfully: {self._model_metadata.get('info', '')}")

            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                raise ModelError(f"Failed to load model from {model_path}: {e}") from e

    def predict(self, *args, **kwargs) -> Any:
        """Make predictions with the loaded model.

        Args:
            *args, **kwargs: Arguments for prediction

        Returns:
            Prediction results

        Raises:
            ModelError: If model is not loaded
            ValidationError: If inputs are invalid
        """
        if not self.is_model_loaded:
            raise ModelError("Model not loaded. Call load_model() first.")

        with self.operation_context("predict"):
            start_time = time.time()

            try:
                # Validate inputs if enabled
                if self._validate_inputs:
                    self._validate_prediction_inputs(*args, **kwargs)

                # Check cache for predictions if enabled
                cache_key = None
                if hasattr(self, '_cache'):
                    cache_key = self._generate_prediction_cache_key(*args, **kwargs)
                    cached_result = self._get_from_cache(cache_key)
                    if cached_result is not None:
                        return cached_result

                # Make prediction using subclass implementation
                result = self._predict_impl(*args, **kwargs)

                # Cache result if we have a cache key
                if cache_key:
                    self._put_in_cache(cache_key, result)

                # Update metrics
                inference_time = time.time() - start_time
                self._update_inference_metrics(inference_time)

                return result

            except ValidationError:
                # Re-raise validation errors
                raise
            except Exception as e:
                self.logger.error(f"Prediction failed: {e}")
                raise ModelError(f"Prediction failed: {e}") from e

    def set_confidence_threshold(self, threshold: float) -> None:
        """Set confidence threshold for predictions.

        Args:
            threshold: Confidence threshold (0.0 to 1.0)

        Raises:
            ValidationError: If threshold is invalid
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValidationError("Confidence threshold must be between 0.0 and 1.0")

        self._confidence_threshold = threshold
        self.logger.debug(f"Confidence threshold set to {threshold}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            'model_path': self._model_path,
            'is_loaded': self.is_model_loaded,
            'metadata': self._model_metadata.copy(),
            'confidence_threshold': self._confidence_threshold,
            'prediction_count': self._prediction_count,
            'average_inference_time': self._average_inference_time,
            'last_inference_time': self._last_inference_time
        }

    def _get_health_details(self) -> Dict[str, Any]:
        """Include AI-specific health information."""
        details = super()._get_health_details()
        details.update({
            'model_loaded': self.is_model_loaded,
            'model_path': self._model_path,
            'prediction_count': self._prediction_count,
            'average_inference_time_ms': self._average_inference_time * 1000
        })
        return details

    def _health_check(self) -> bool:
        """AI service health check."""
        base_healthy = super()._health_check()

        # Check if model is loaded if we expect it to be
        model_healthy = True
        if self._model_path and not self.is_model_loaded:
            model_healthy = False

        return base_healthy and model_healthy

    def _update_inference_metrics(self, inference_time: float) -> None:
        """Update inference performance metrics."""
        self._prediction_count += 1
        self._last_inference_time = inference_time

        # Update rolling average
        alpha = 0.1  # Smoothing factor
        if self._average_inference_time == 0:
            self._average_inference_time = inference_time
        else:
            self._average_inference_time = (
                alpha * inference_time + (1 - alpha) * self._average_inference_time
            )

    def _generate_prediction_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key for prediction.

        Override this method to provide custom cache key generation.
        """
        # Simple hash-based key generation
        key_data = str(args) + str(sorted(kwargs.items()))
        return f"predict:{hashlib.md5(key_data.encode()).hexdigest()}"

    # Abstract methods that subclasses must implement
    @abc.abstractmethod
    def _load_model_impl(self, model_path: str, **kwargs) -> Any:
        """Load the model implementation.

        Args:
            model_path: Path to model file
            **kwargs: Additional loading parameters

        Returns:
            Loaded model object
        """
        pass

    @abc.abstractmethod
    def _predict_impl(self, *args, **kwargs) -> Any:
        """Make prediction implementation.

        Args:
            *args, **kwargs: Prediction arguments

        Returns:
            Prediction results
        """
        pass

    # Optional methods that can be overridden
    def _validate_prediction_inputs(self, *args, **kwargs) -> None:
        """Validate inputs before prediction.

        Override this method to provide input validation.

        Raises:
            ValidationError: If inputs are invalid
        """
        pass

    def _extract_model_metadata(self, model: Any) -> Dict[str, Any]:
        """Extract metadata from loaded model.

        Args:
            model: Loaded model object

        Returns:
            Dictionary with model metadata
        """
        return {
            'type': type(model).__name__,
            'info': str(model) if hasattr(model, '__str__') else 'Unknown model'
        }

    def _should_cache_model(self, model: Any) -> bool:
        """Determine if model should be cached.

        Args:
            model: Model object to check

        Returns:
            True if model should be cached
        """
        # By default, don't cache large models to avoid memory issues
        return True


class ImageProcessingServiceBase(AIServiceBase):
    """Base class for image processing AI services.

    Specialized for services that work with images, providing common
    functionality for image validation, preprocessing, and caching.
    """

    def __init__(self,
                 supported_formats: Optional[List[str]] = None,
                 max_image_size: tuple = (4096, 4096),
                 **kwargs):
        """Initialize image processing service.

        Args:
            supported_formats: List of supported image formats
            max_image_size: Maximum image dimensions (width, height)
            **kwargs: Additional arguments for base service
        """
        super().__init__(**kwargs)

        self._supported_formats = supported_formats or ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self._max_image_size = max_image_size

    def validate_image_input(self, image: Union[np.ndarray, str, Path]) -> np.ndarray:
        """Validate and convert image input.

        Args:
            image: Image as numpy array or path to image file

        Returns:
            Validated numpy array image

        Raises:
            ValidationError: If image is invalid
        """
        if isinstance(image, (str, Path)):
            image_path = Path(image)

            # Validate file existence and format
            if not image_path.exists():
                raise ValidationError(f"Image file does not exist: {image_path}")

            if image_path.suffix.lower() not in self._supported_formats:
                raise ValidationError(f"Unsupported image format: {image_path.suffix}")

            # Load image
            try:
                import cv2
                image_array = cv2.imread(str(image_path))
                if image_array is None:
                    raise ValidationError(f"Failed to load image: {image_path}")
            except Exception as e:
                raise ValidationError(f"Error loading image {image_path}: {e}")

        elif isinstance(image, np.ndarray):
            image_array = image
        else:
            raise ValidationError("Image must be numpy array or path to image file")

        # Validate image properties
        if image_array.size == 0:
            raise ValidationError("Image is empty")

        if len(image_array.shape) not in [2, 3]:
            raise ValidationError(f"Invalid image dimensions: {image_array.shape}")

        # Check size limits
        height, width = image_array.shape[:2]
        if width > self._max_image_size[0] or height > self._max_image_size[1]:
            raise ValidationError(
                f"Image too large: {width}x{height}, max: {self._max_image_size}"
            )

        return image_array

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input.

        Override this method to provide custom preprocessing.

        Args:
            image: Input image

        Returns:
            Preprocessed image
        """
        return image

    def _generate_prediction_cache_key(self, image: np.ndarray, *args, **kwargs) -> str:
        """Generate cache key for image prediction."""
        # Generate simple hash for image using md5 of image data
        image_hash = hashlib.md5(image.tobytes()).hexdigest()

        # Include other arguments
        other_args = str(args[1:]) + str(sorted(kwargs.items())) if len(args) > 1 or kwargs else ""

        # Combine into cache key
        key_data = f"{image_hash}:{other_args}"
        return f"predict:{hashlib.md5(key_data.encode()).hexdigest()}"

    def _validate_prediction_inputs(self, image: np.ndarray, *args, **kwargs) -> None:
        """Validate image prediction inputs."""
        self.validate_image_input(image)


class TextProcessingServiceBase(AIServiceBase):
    """Base class for text processing AI services.

    Specialized for services that work with text, providing common
    functionality for text validation, sanitization, and processing.
    """

    def __init__(self,
                 max_text_length: int = 10000,
                 filter_sensitive_content: bool = True,
                 **kwargs):
        """Initialize text processing service.

        Args:
            max_text_length: Maximum allowed text length
            filter_sensitive_content: Whether to filter sensitive content
            **kwargs: Additional arguments for base service
        """
        super().__init__(**kwargs)

        self._max_text_length = max_text_length
        self._filter_sensitive_content = filter_sensitive_content

    def validate_text_input(self, text: str) -> str:
        """Validate and sanitize text input.

        Args:
            text: Input text to validate

        Returns:
            Validated and sanitized text

        Raises:
            ValidationError: If text is invalid
        """
        if not isinstance(text, str):
            raise ValidationError("Text input must be a string")

        if not text.strip():
            raise ValidationError("Text input cannot be empty")

        if len(text) > self._max_text_length:
            raise ValidationError(f"Text too long: {len(text)} chars, max: {self._max_text_length}")

        # Sanitize the text
        sanitized_text = InputValidator.sanitize_string_input(text, max_length=self._max_text_length)

        # Filter sensitive content if enabled
        if self._filter_sensitive_content:
            if ContentFilter.contains_sensitive_info(sanitized_text):
                self.logger.warning("Sensitive content detected in text input")
                sanitized_text = ContentFilter.filter_sensitive_content(sanitized_text)

        return sanitized_text

    def _validate_prediction_inputs(self, text: str, *args, **kwargs) -> None:
        """Validate text prediction inputs."""
        self.validate_text_input(text)


class MultimodalServiceBase(ImageProcessingServiceBase, TextProcessingServiceBase):
    """Base class for multimodal AI services that handle both images and text.

    Combines functionality from both image and text processing services.
    """

    def validate_multimodal_input(self,
                                 image: np.ndarray,
                                 text: str) -> tuple[np.ndarray, str]:
        """Validate both image and text inputs.

        Args:
            image: Input image
            text: Input text

        Returns:
            Tuple of validated (image, text)

        Raises:
            ValidationError: If either input is invalid
        """
        validated_image = self.validate_image_input(image)
        validated_text = self.validate_text_input(text)

        return validated_image, validated_text

    def _validate_prediction_inputs(self, image: np.ndarray, text: str, *args, **kwargs) -> None:
        """Validate multimodal prediction inputs."""
        self.validate_multimodal_input(image, text)