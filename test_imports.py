"""Test script to validate all imports work correctly."""

import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_imports():
    """Test all the refactored imports."""
    try:
        # Test core imports
        print("Testing core imports...")
        from app.core.entities import Detection, MasterObject, MatchResult, PipelineState
        from app.core.exceptions import DetectionError, ConfigError, ModelError
        from app.core.constants import APP_NAME, VERSION
        print("+ Core imports successful")
        
        # Test config imports
        print("Testing config imports...")
        from app.config.settings import Config, load_config, save_config
        from app.config.defaults import DEFAULT_CONFIG
        cfg = load_config()
        print(f"+ Config imports successful - Model: {cfg.model_size}")
        
        # Test utils imports
        print("Testing utils imports...")
        from app.utils.geometry import xywh_to_xyxy, iou_xyxy, centroid_distance, angle_delta
        from app.utils.crypto_utils import encrypt_api_key, decrypt_api_key
        from app.utils.file_utils import read_yolo_labels, save_yolo_labels
        from app.utils.image_utils import resize_image, convert_color_space
        print("+ Utils imports successful")
        
        # Test services imports
        print("Testing services imports...")
        from app.services.webcam_service import WebcamService
        from app.services.inference_service import InferenceService
        from app.services.detection_service import DetectionService
        from app.services.annotation_service import AnnotationService
        from app.services.training_service import TrainingService
        print("+ Services imports successful")
        
        # Test backends imports
        print("Testing backends imports...")
        from app.backends.base_backend import BaseBackend
        from app.backends.yolo_backend import YoloBackend
        print("+ Backends imports successful")
        
        # Test UI imports (without creating windows)
        print("Testing UI imports...")
        from app.ui.components.status_bar import StatusBar
        from app.ui.dialogs.settings_dialog import SettingsDialog
        from app.ui.dialogs.webcam_dialog import WebcamDialog
        from app.ui.styles.themes import ThemeManager
        print("+ UI imports successful")
        
        # Test service creation
        print("Testing service creation...")
        webcam_service = WebcamService()
        inference_service = InferenceService(cfg)
        annotation_service = AnnotationService(cfg)
        training_service = TrainingService(cfg)
        
        print(f"+ Services created successfully")
        print(f"  - Webcam service: OK")
        print(f"  - Inference service: {'Loaded' if inference_service.is_loaded else 'Not loaded'}")
        print(f"  - Annotation service: OK")
        print(f"  - Training service: OK")
        
        # Test core functionality
        print("Testing core functionality...")
        
        # Test geometry functions
        bbox = (0.5, 0.5, 0.2, 0.2)  # center format
        xyxy_bbox = xywh_to_xyxy(bbox, 640, 480)  # convert to absolute
        print(f"+ Geometry conversion: {bbox} -> {xyxy_bbox}")
        
        # Test entity creation
        detection = Detection(class_id=0, score=0.8, bbox=(100, 100, 200, 200))
        master = MasterObject(class_id=0, name="test", bbox_norm=bbox)
        print(f"+ Entity creation successful")
        
        print("\nSUCCESS: All tests passed! Refactored structure is working correctly.")
        return True
        
    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)