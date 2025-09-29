"""Enhanced inference service for model loading and prediction."""
from __future__ import annotations
import os
import time
import random
from typing import List, Dict, Optional, Any
import numpy as np
from ..core.entities import Detection, BBox
from ..core.exceptions import ModelError
from ..utils.geometry import xywh_to_xyxy
from ..config.settings import Config

# Try to import ultralytics if present
HAS_ULTRALYTICS = False
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

class InferenceService:
    """Enhanced model wrapper for object detection inference."""
    
    def __init__(self, config: Config, weights_path: Optional[str] = None):
        self.config = config
        self.weights = weights_path or os.path.join(config.models_dir, "best.pt")
        self.model = None
        self.loaded_source = None  # path or model name
        self.is_loaded = False
        
        if HAS_ULTRALYTICS:
            self._attempt_load()
        else:
            print("[InferenceService] Ultralytics not found; using dummy detections when debug enabled.")

    def _attempt_load(self) -> None:
        """Attempt focused model loading: prioritize 'model.pt' exclusively if it exists, otherwise use fallbacks."""
        candidates: List[str] = []

        # 1. Check for trained "model.pt" first and load EXCLUSIVELY if it exists
        model_pt_path = os.path.join(self.config.models_dir, "model.pt")
        if os.path.isfile(model_pt_path):
            try:
                print(f"[InferenceService] Found trained model: {model_pt_path}")
                print(f"[InferenceService] Loading trained model exclusively...")
                self.model = YOLO(model_pt_path)
                self.loaded_source = model_pt_path
                self.is_loaded = True
                print(f"[InferenceService] Successfully loaded trained model: {model_pt_path}")
                return
            except Exception as e:
                print(f"[InferenceService] Failed to load trained model {model_pt_path}: {e}")
                raise ModelError(f"Failed to load trained model {model_pt_path}: {e}")

        # 2. If no "model.pt" exists, fall back to original loading logic
        print(f"[InferenceService] No trained model found at {model_pt_path}, using fallback models...")

        # Explicit weights path if file exists
        if os.path.isfile(self.weights):
            candidates.append(self.weights)

        # Configured model_size (e.g. yolo11n)
        if self.config.model_size:
            candidates.append(self.config.model_size)

        # Fallback list
        candidates.extend(['yolo11n', 'yolov8n'])

        tried = []
        for candidate in candidates:
            if candidate in tried:
                continue
            tried.append(candidate)

            try:
                print(f"[InferenceService] Loading fallback model: {candidate}")
                self.model = YOLO(candidate)
                self.loaded_source = candidate
                self.is_loaded = True
                print(f"[InferenceService] Successfully loaded fallback model: {candidate}")
                return
            except Exception as e:
                print(f"[InferenceService] Failed to load {candidate}: {e}")
                continue

        raise ModelError(f"Failed to load any model from candidates: {candidates}")

    def predict(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Detection]:
        """Run inference on frame and return detections."""
        if not self.is_loaded or not self.model:
            if self.config.debug:
                return self._generate_dummy_detections(frame)
            return []
        
        try:
            results = self.model(frame, conf=conf_threshold, verbose=False)
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extract box data
                        xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Convert to our format
                        bbox: BBox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                        
                        detection = Detection(
                            class_id=cls,
                            score=conf,
                            bbox=bbox
                        )
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"[InferenceService] Prediction error: {e}")
            if self.config.debug:
                return self._generate_dummy_detections(frame)
            return []

    def _generate_dummy_detections(self, frame: np.ndarray) -> List[Detection]:
        """Generate dummy detections for testing when debug is enabled."""
        if not self.config.debug:
            return []
        
        h, w = frame.shape[:2]
        detections = []
        
        # Generate 1-3 random detections
        num_detections = random.randint(1, 3)
        for i in range(num_detections):
            # Random bbox
            x1 = random.randint(0, w // 2)
            y1 = random.randint(0, h // 2)
            x2 = random.randint(x1 + 50, min(x1 + 200, w))
            y2 = random.randint(y1 + 50, min(y1 + 200, h))
            
            detection = Detection(
                class_id=random.randint(0, 79),  # COCO has 80 classes
                score=random.uniform(0.5, 0.9),
                bbox=(x1, y1, x2, y2)
            )
            detections.append(detection)
        
        return detections

    def reload_model(self, new_weights_path: Optional[str] = None) -> bool:
        """Reload the model with new weights or refresh to load newly trained model.pt."""
        if new_weights_path:
            self.weights = new_weights_path

        self.model = None
        self.loaded_source = None
        self.is_loaded = False

        try:
            if HAS_ULTRALYTICS:
                self._attempt_load()
                return True
        except Exception as e:
            print(f"[InferenceService] Failed to reload model: {e}")
            return False

        return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'has_ultralytics': HAS_ULTRALYTICS,
            'is_loaded': self.is_loaded,
            'loaded_source': self.loaded_source,
            'weights_path': self.weights,
            'model_type': type(self.model).__name__ if self.model else None
        }

    def __del__(self):
        """Cleanup resources."""
        if self.model:
            del self.model