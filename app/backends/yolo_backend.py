"""YOLO backend implementation using Ultralytics."""
import os
from typing import List, Dict, Any, Optional
import numpy as np
from .base_backend import BaseBackend
from ..core.entities import Detection, BBox
from ..core.exceptions import ModelError

# Try to import ultralytics
HAS_ULTRALYTICS = False
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

class YoloBackend(BaseBackend):
    """YOLO backend using Ultralytics implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.model_path = None

    def load_model(self, model_path_or_name: str) -> bool:
        """Load a YOLO model from path or model name."""
        if not HAS_ULTRALYTICS:
            raise ModelError("Ultralytics not installed. Cannot use YOLO backend.")
        
        try:
            self.model = YOLO(model_path_or_name)
            self.model_path = model_path_or_name
            self.is_loaded = True
            
            # Store model info
            self.model_info = {
                'backend': 'ultralytics',
                'model_type': 'YOLO',
                'model_path': model_path_or_name,
                'device': str(self.model.device) if hasattr(self.model, 'device') else 'unknown'
            }
            
            print(f"[YoloBackend] Successfully loaded: {model_path_or_name}")
            return True
            
        except Exception as e:
            self.is_loaded = False
            raise ModelError(f"Failed to load YOLO model {model_path_or_name}: {e}")

    def predict(self, image: np.ndarray, **kwargs) -> List[Detection]:
        """Run YOLO inference on an image."""
        if not self.is_loaded or not self.model:
            raise ModelError("No model loaded")
        
        # Extract parameters with defaults
        conf_threshold = kwargs.get('conf', 0.5)
        iou_threshold = kwargs.get('iou', 0.45)
        verbose = kwargs.get('verbose', False)
        
        try:
            results = self.model(
                image,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=verbose
            )
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    # Convert to numpy for easier processing
                    xyxy = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    conf = boxes.conf.cpu().numpy()  # confidence scores
                    cls = boxes.cls.cpu().numpy()   # class indices
                    
                    for i in range(len(xyxy)):
                        bbox: BBox = (
                            int(xyxy[i][0]),
                            int(xyxy[i][1]),
                            int(xyxy[i][2]),
                            int(xyxy[i][3])
                        )
                        
                        detection = Detection(
                            class_id=int(cls[i]),
                            score=float(conf[i]),
                            bbox=bbox
                        )
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            raise ModelError(f"YOLO prediction failed: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded YOLO model."""
        if not self.is_loaded:
            return {'status': 'not_loaded'}
        
        info = self.model_info.copy()
        
        # Add additional runtime info if available
        if self.model and hasattr(self.model, 'names'):
            info['num_classes'] = len(self.model.names)
            info['class_names'] = self.model.names
        
        return info

    def get_supported_formats(self) -> List[str]:
        """Get list of supported YOLO model formats."""
        return ['.pt', '.onnx', '.torchscript', '.pb', '.tflite', '.engine']

    def validate_model(self, model_path: str) -> bool:
        """Validate if a model file is a valid YOLO model."""
        if not HAS_ULTRALYTICS:
            return False
        
        # Check file extension
        _, ext = os.path.splitext(model_path.lower())
        if ext not in self.get_supported_formats():
            return False
        
        # Check if file exists (for local files)
        if os.path.isfile(model_path) and not os.access(model_path, os.R_OK):
            return False
        
        # Try to load model to validate
        try:
            test_model = YOLO(model_path)
            del test_model  # Clean up
            return True
        except Exception:
            return False

    def export_model(self, format: str = 'onnx', **kwargs) -> str:
        """Export the loaded model to different format."""
        if not self.is_loaded or not self.model:
            raise ModelError("No model loaded for export")
        
        try:
            exported_path = self.model.export(format=format, **kwargs)
            return exported_path
        except Exception as e:
            raise ModelError(f"Model export failed: {e}")

    def benchmark(self, **kwargs) -> Dict[str, Any]:
        """Benchmark the loaded model."""
        if not self.is_loaded or not self.model:
            raise ModelError("No model loaded for benchmarking")
        
        try:
            # Use YOLO's built-in benchmark functionality
            results = self.model.benchmark(**kwargs)
            return results
        except Exception as e:
            raise ModelError(f"Benchmarking failed: {e}")

    def unload_model(self) -> None:
        """Unload the current YOLO model."""
        if self.model:
            del self.model
            self.model = None
        
        super().unload_model()
        self.model_path = None