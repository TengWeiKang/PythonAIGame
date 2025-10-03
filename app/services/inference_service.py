"""YOLO inference service for object detection."""

import logging
from typing import Optional, List, Dict, Any
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class InferenceService:
    """Service for YOLO model inference and object detection."""

    def __init__(self, model_path: str = "yolo12n.pt", confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
        """Initialize inference service.

        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self._model = None
        self._model_loaded = False

    def load_model(self, custom_model_path: Optional[str] = None) -> bool:
        """Load YOLO model. Thread-safe for reloading models.

        Args:
            custom_model_path: Optional custom model path to load instead of default

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            from ultralytics import YOLO

            # Use custom model path if provided, otherwise use default
            model_to_load = custom_model_path if custom_model_path else self.model_path

            # Check if it's a path to a file or a model name
            model_path_obj = Path(model_to_load)
            if not model_path_obj.exists() and not model_to_load.endswith('.pt'):
                # Might be a model name like 'yolo12n', let YOLO handle it
                logger.info(f"Loading YOLO model by name: {model_to_load}")
            elif not model_path_obj.exists():
                logger.error(f"Model file not found: {model_to_load}")
                return False
            else:
                logger.info(f"Loading YOLO model from file: {model_to_load}")

            # Load the new model (replaces existing model if any)
            self._model = YOLO(model_to_load)
            self._model_loaded = True
            logger.info(f"Model loaded successfully: {model_to_load}")

            # Update the current model path if loading was successful
            if custom_model_path:
                self.model_path = custom_model_path
                logger.info(f"Model path updated to: {custom_model_path}")

            return True

        except ImportError:
            logger.error("Ultralytics package not installed. Install with: pip install ultralytics")
            self._model_loaded = False
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._model_loaded = False
            return False

    def detect(self, image: np.ndarray, max_size: int = 640) -> List[Dict[str, Any]]:
        """Run object detection on image with automatic resizing and coordinate transformation.

        Args:
            image: Input image as numpy array (BGR format)
            max_size: Maximum size for YOLO inference (default 640x640)

        Returns:
            List of detection dictionaries with keys:
                - class_name: Object class name
                - confidence: Detection confidence (0-1)
                - bbox: Bounding box [x1, y1, x2, y2] in ORIGINAL image coordinates
                - class_id: Class ID number
        """
        if not self._model_loaded:
            logger.warning("Model not loaded, attempting to load...")
            if not self.load_model():
                return []

        try:
            import cv2

            # Get original image dimensions
            original_height, original_width = image.shape[:2]
            logger.debug(f"Original image size: {original_width}x{original_height}")

            # Calculate resize parameters if image exceeds max_size
            if original_height > max_size or original_width > max_size:
                # Calculate scale factor to fit within max_size while maintaining aspect ratio
                scale = max_size / max(original_height, original_width)
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)

                # Resize image for inference
                resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite("output.jpg", resized_image)
                # Calculate scale factors for coordinate transformation back to original size
                scale_x = original_width / new_width
                scale_y = original_height / new_height

                logger.debug(f"Resized image to: {new_width}x{new_height} (scale_x={scale_x:.3f}, scale_y={scale_y:.3f})")
            else:
                # No resize needed - image already within limits
                resized_image = image
                scale_x = 1.0
                scale_y = 1.0
                logger.debug("No resize needed - image within size limits")

            # Run inference on resized image
            results = self._model(
                resized_image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=100,
                verbose=True
            )

            # Parse results and scale coordinates back to original image size
            detections = []
            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)

                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        # Get bounding box coordinates from resized image
                        x1, y1, x2, y2 = box

                        # Scale coordinates back to original image size
                        x1_original = int(x1 * scale_x)
                        y1_original = int(y1 * scale_y)
                        x2_original = int(x2 * scale_x)
                        y2_original = int(y2 * scale_y)

                        # Ensure coordinates are within image bounds
                        x1_original = max(0, min(x1_original, original_width))
                        y1_original = max(0, min(y1_original, original_height))
                        x2_original = max(0, min(x2_original, original_width))
                        y2_original = max(0, min(y2_original, original_height))

                        detection = {
                            'class_name': result.names[cls_id],
                            'confidence': float(conf),
                            'bbox': [x1_original, y1_original, x2_original, y2_original],
                            'class_id': int(cls_id)
                        }
                        detections.append(detection)

                    logger.debug(f"Detected {len(detections)} objects with coordinates scaled to original size")

            return detections

        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []

    def detect_with_visualization(self, image: np.ndarray) -> tuple[np.ndarray, List[Dict[str, Any]]]:
        """Run detection and return annotated image.

        Args:
            image: Input image as numpy array

        Returns:
            Tuple of (annotated_image, detections_list)
        """
        detections = self.detect(image)
        annotated_image = self._draw_detections(image.copy(), detections)
        return annotated_image, detections

    def _draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detection bounding boxes and labels on image.

        Args:
            image: Input image
            detections: List of detection dictionaries

        Returns:
            Annotated image
        """
        import cv2

        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            class_name = det['class_name']
            confidence = det['confidence']

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # Draw label background
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), (0, 255, 0), -1)

            # Draw label text
            cv2.putText(image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return image

    def is_loaded(self) -> bool:
        """Check if model is loaded.

        Returns:
            True if model is loaded, False otherwise
        """
        return self._model_loaded

    def update_thresholds(self, confidence: Optional[float] = None,
                         iou: Optional[float] = None):
        """Update detection thresholds.

        Args:
            confidence: New confidence threshold (0-1)
            iou: New IoU threshold (0-1)
        """
        if confidence is not None:
            self.confidence_threshold = max(0.0, min(1.0, confidence))
        if iou is not None:
            self.iou_threshold = max(0.0, min(1.0, iou))