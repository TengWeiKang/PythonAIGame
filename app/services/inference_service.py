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
            logger.info(f"[INFERENCE] Input image size: {original_width}x{original_height}")

            # Calculate resize parameters if image exceeds max_size
            if original_height > max_size or original_width > max_size:
                # Calculate scale factor to fit within max_size while maintaining aspect ratio
                scale = max_size / max(original_height, original_width)
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)

                # Resize image for inference
                resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                # Calculate scale factors for coordinate transformation back to original size
                scale_x = original_width / new_width
                scale_y = original_height / new_height

                logger.info(f"[INFERENCE] Image resized for inference: {new_width}x{new_height}")
                logger.info(f"[INFERENCE] Scale factors: scale_x={scale_x:.6f}, scale_y={scale_y:.6f}")
            else:
                # No resize needed - image already within limits
                resized_image = image
                scale_x = 1.0
                scale_y = 1.0
                logger.info(f"[INFERENCE] No resize needed - image within {max_size}x{max_size} limits")

            # Run inference on resized image
            logger.info(f"[INFERENCE] Running YOLO inference with conf={self.confidence_threshold}, iou={self.iou_threshold}")
            results = self._model(
                resized_image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=100,
                verbose=False  # Changed to False to reduce noise
            )

            # Parse results and scale coordinates back to original image size
            detections = []
            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)

                    logger.info(f"[INFERENCE] YOLO returned {len(boxes)} detections")

                    for idx, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                        # Get bounding box coordinates from resized image
                        x1, y1, x2, y2 = box

                        logger.info(f"[INFERENCE] Detection {idx}: Raw YOLO bbox (on resized image): "
                                  f"x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}")
                        logger.info(f"[INFERENCE] Detection {idx}: Raw bbox size: {x2-x1:.2f}x{y2-y1:.2f} pixels")

                        # Scale coordinates back to original image size
                        x1_original = int(x1 * scale_x)
                        y1_original = int(y1 * scale_y)
                        x2_original = int(x2 * scale_x)
                        y2_original = int(y2 * scale_y)

                        logger.info(f"[INFERENCE] Detection {idx}: Scaled bbox (before clamping): "
                                  f"x1={x1_original}, y1={y1_original}, x2={x2_original}, y2={y2_original}")
                        logger.info(f"[INFERENCE] Detection {idx}: Scaled bbox size: {x2_original-x1_original}x{y2_original-y1_original} pixels")

                        # Ensure coordinates are within image bounds
                        x1_clamped = max(0, min(x1_original, original_width))
                        y1_clamped = max(0, min(y1_original, original_height))
                        x2_clamped = max(0, min(x2_original, original_width))
                        y2_clamped = max(0, min(y2_original, original_height))

                        # Check if clamping changed coordinates
                        if (x1_clamped != x1_original or y1_clamped != y1_original or
                            x2_clamped != x2_original or y2_clamped != y2_original):
                            logger.warning(f"[INFERENCE] Detection {idx}: Bbox clamped to image bounds! "
                                        f"Original: ({x1_original},{y1_original})-({x2_original},{y2_original}), "
                                        f"Clamped: ({x1_clamped},{y1_clamped})-({x2_clamped},{y2_clamped})")

                        # Calculate bbox size as percentage of image
                        bbox_width_pct = ((x2_clamped - x1_clamped) / original_width) * 100
                        bbox_height_pct = ((y2_clamped - y1_clamped) / original_height) * 100

                        logger.info(f"[INFERENCE] Detection {idx}: Bbox size relative to original image: "
                                  f"{bbox_width_pct:.1f}% width x {bbox_height_pct:.1f}% height")

                        # Get class name
                        class_name = result.names[cls_id]

                        detection = {
                            'class_name': class_name,
                            'confidence': float(conf),
                            'bbox': [x1_clamped, y1_clamped, x2_clamped, y2_clamped],
                            'class_id': int(cls_id)
                        }
                        detections.append(detection)

                        logger.info(f"[INFERENCE] Detection {idx}: Final detection: "
                                  f"class='{class_name}', conf={conf:.3f}, "
                                  f"bbox=[{x1_clamped}, {y1_clamped}, {x2_clamped}, {y2_clamped}]")

                    logger.info(f"[INFERENCE] Total detections returned: {len(detections)}")
                else:
                    logger.info(f"[INFERENCE] No objects detected (empty boxes)")

            else:
                logger.info(f"[INFERENCE] No results from YOLO model")

            return detections

        except Exception as e:
            logger.error(f"Error during detection: {e}", exc_info=True)
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

    def get_class_names(self) -> List[str]:
        """Get list of all class names that the model can detect.

        Returns:
            List of class names, or empty list if model not loaded
        """
        if not self._model_loaded or self._model is None:
            logger.warning("Model not loaded, cannot get class names")
            return []

        try:
            # Access the model's class names dictionary
            # result.names is a dict like {0: 'person', 1: 'bicycle', ...}
            names_dict = self._model.names
            # Return sorted list of class names by class ID
            class_names = [names_dict[i] for i in sorted(names_dict.keys())]
            logger.info(f"Retrieved {len(class_names)} class names from model")
            return class_names
        except Exception as e:
            logger.error(f"Error getting class names: {e}")
            return []