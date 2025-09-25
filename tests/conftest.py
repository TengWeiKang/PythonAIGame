"""Pytest configuration and shared fixtures for the webcam detection application.

This module provides comprehensive test configuration, fixtures, and utilities
for testing the Python Game Detection System.
"""
import os
import sys
import tempfile
import pytest
import logging
import shutil
import json
from pathlib import Path
from typing import Dict, Any, Optional, Generator
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import cv2

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import application modules
from app.config.settings import Config
from app.core.entities import Detection, MasterObject
from app.core.exceptions import WebcamError, ModelError, ConfigError
from app.services.improved_webcam_service import ImprovedWebcamService


# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Disable some verbose loggers during testing
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def project_root():
    """Provide project root directory path."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide test data directory path."""
    test_data_path = PROJECT_ROOT / "tests" / "data"
    test_data_path.mkdir(exist_ok=True)
    return test_data_path


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config():
    """Provide a mock configuration object for testing."""
    config = Mock(spec=Config)

    # Set common configuration values
    config.data_dir = "test_data"
    config.models_dir = "test_models"
    config.results_export_dir = "test_results"
    config.master_dir = "test_master"
    config.img_size = 640
    config.target_fps = 30
    config.use_gpu = False
    config.debug = True
    config.camera_width = 640
    config.camera_height = 480
    config.camera_fps = 30
    config.detection_confidence_threshold = 0.5
    config.detection_iou_threshold = 0.45
    config.gemini_api_key = ""
    config.gemini_model = "gemini-1.5-flash"
    config.gemini_timeout = 30
    config.enable_ai_analysis = False

    return config


@pytest.fixture
def real_config(temp_dir):
    """Provide a real configuration object with temporary directories."""
    config_data = {
        "data_dir": str(temp_dir / "data"),
        "models_dir": str(temp_dir / "models"),
        "results_export_dir": str(temp_dir / "results"),
        "master_dir": str(temp_dir / "master"),
        "img_size": 640,
        "target_fps": 30,
        "use_gpu": False,
        "debug": True,
        "camera_width": 640,
        "camera_height": 480,
        "camera_fps": 30,
        "detection_confidence_threshold": 0.5,
        "detection_iou_threshold": 0.45,
        "gemini_api_key": "",
        "gemini_model": "gemini-1.5-flash",
        "gemini_timeout": 30,
        "enable_ai_analysis": False
    }

    # Create directories
    for dir_key in ["data_dir", "models_dir", "results_export_dir", "master_dir"]:
        Path(config_data[dir_key]).mkdir(parents=True, exist_ok=True)

    # Save config to file
    config_file = temp_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)

    # Load as Config object
    from app.config.settings import load_config
    return load_config(str(config_file))


@pytest.fixture
def sample_image():
    """Provide a sample image for testing."""
    # Create a simple test image (100x100 RGB)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Add some patterns to make it more realistic
    # Blue square in top-left
    image[10:40, 10:40] = [255, 0, 0]  # BGR format

    # Green circle in center
    cv2.circle(image, (50, 50), 15, (0, 255, 0), -1)

    # Red rectangle in bottom-right
    image[60:90, 60:90] = [0, 0, 255]

    return image


@pytest.fixture
def sample_large_image():
    """Provide a larger sample image for testing."""
    # Create a 640x480 test image
    image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # Add some recognizable patterns
    cv2.rectangle(image, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.circle(image, (400, 300), 50, (0, 255, 0), -1)

    return image


@pytest.fixture
def sample_detections():
    """Provide sample detection objects for testing."""
    return [
        Detection(
            class_id=0,
            score=0.85,
            bbox=(100, 100, 200, 200)
        ),
        Detection(
            class_id=1,
            score=0.92,
            bbox=(300, 150, 450, 300)
        ),
        Detection(
            class_id=0,
            score=0.78,
            bbox=(50, 350, 180, 450)
        )
    ]


@pytest.fixture
def sample_master_objects():
    """Provide sample master objects for testing."""
    return [
        MasterObject(
            class_id=0,
            name="Test Object 1",
            bbox_norm=(0.25, 0.25, 0.2, 0.2)  # Normalized coordinates
        ),
        MasterObject(
            class_id=1,
            name="Test Object 2",
            bbox_norm=(0.65, 0.4, 0.25, 0.3)
        )
    ]


@pytest.fixture
def mock_webcam_service(mock_config):
    """Provide a mock webcam service for testing."""
    service = Mock(spec=ImprovedWebcamService)
    service.is_webcam_opened.return_value = False
    service.get_current_fps.return_value = 30.0
    service.get_camera_properties.return_value = {
        'width': 640,
        'height': 480,
        'fps': 30,
        'backend': 'DSHOW',
        'device_index': 0
    }
    service.read_frame.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    return service


@pytest.fixture
def mock_opencv_capture():
    """Provide a mock OpenCV VideoCapture object."""
    cap = Mock()
    cap.isOpened.return_value = True
    cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    cap.get.return_value = 640  # Mock property getter
    cap.set.return_value = True
    cap.release.return_value = None
    cap.getBackendName.return_value = "DSHOW"
    return cap


@pytest.fixture
def mock_gemini_service():
    """Provide a mock Gemini AI service for testing."""
    from app.services.gemini_service import GeminiService

    service = Mock(spec=GeminiService)
    service.is_configured.return_value = False
    service.analyze_single_image.return_value = "This is a test analysis result."
    service.compare_images.return_value = "These images are similar with minor differences."
    service.send_message.return_value = "Test AI response."

    return service


@pytest.fixture
def api_key_env():
    """Provide environment variables for API keys during testing."""
    with patch.dict(os.environ, {
        'GEMINI_API_KEY': 'test_api_key_for_testing_only',
        'TEST_MODE': 'true'
    }):
        yield


@pytest.fixture(scope="session")
def test_model_file(test_data_dir):
    """Create a dummy model file for testing."""
    model_path = test_data_dir / "test_model.pt"

    # Create a minimal dummy file
    model_path.write_text("dummy_model_data")

    return model_path


@pytest.fixture
def mock_yolo_model():
    """Provide a mock YOLO model for testing."""
    model = Mock()
    model.predict.return_value = [Mock()]  # Mock results
    model.train.return_value = Mock()
    model.val.return_value = Mock()
    model.export.return_value = "exported_model.onnx"

    return model


class TestDataGenerator:
    """Utility class for generating test data."""

    @staticmethod
    def create_test_image(width: int = 640, height: int = 480, pattern: str = "random") -> np.ndarray:
        """Create a test image with specified dimensions and pattern.

        Args:
            width: Image width
            height: Image height
            pattern: Pattern type ('random', 'gradient', 'checkerboard', 'solid')

        Returns:
            Generated test image
        """
        if pattern == "random":
            return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        elif pattern == "gradient":
            image = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(height):
                image[i, :, :] = int(255 * i / height)
            return image
        elif pattern == "checkerboard":
            image = np.zeros((height, width, 3), dtype=np.uint8)
            square_size = 20
            for i in range(0, height, square_size):
                for j in range(0, width, square_size):
                    if (i // square_size + j // square_size) % 2 == 0:
                        image[i:i+square_size, j:j+square_size] = 255
            return image
        elif pattern == "solid":
            return np.full((height, width, 3), 128, dtype=np.uint8)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

    @staticmethod
    def create_detection_list(count: int = 3, image_width: int = 640, image_height: int = 480) -> list:
        """Create a list of random detections.

        Args:
            count: Number of detections to create
            image_width: Width of the image for coordinate bounds
            image_height: Height of the image for coordinate bounds

        Returns:
            List of Detection objects
        """
        detections = []

        for i in range(count):
            # Generate random bounding box
            x1 = np.random.randint(0, image_width // 2)
            y1 = np.random.randint(0, image_height // 2)
            x2 = np.random.randint(x1 + 20, image_width)
            y2 = np.random.randint(y1 + 20, image_height)

            detection = Detection(
                class_id=i % 3,  # Cycle through class IDs
                score=np.random.uniform(0.5, 1.0),
                bbox=(x1, y1, x2, y2)
            )

            detections.append(detection)

        return detections


@pytest.fixture
def test_data_generator():
    """Provide the TestDataGenerator utility."""
    return TestDataGenerator


# Performance testing fixtures
@pytest.fixture
def performance_monitor():
    """Provide a performance monitor for timing tests."""
    from app.core.performance import PerformanceMonitor
    return PerformanceMonitor.instance()


# Integration test fixtures
@pytest.fixture(scope="session")
def integration_test_env():
    """Set up environment for integration tests."""
    # Only run integration tests if explicitly requested
    if not os.getenv("RUN_INTEGRATION_TESTS"):
        pytest.skip("Integration tests disabled. Set RUN_INTEGRATION_TESTS=1 to enable.")

    return True


# Parametrized test data
@pytest.fixture(params=[
    {"width": 640, "height": 480},
    {"width": 1280, "height": 720},
    {"width": 1920, "height": 1080},
])
def camera_resolutions(request):
    """Provide different camera resolution parameters for testing."""
    return request.param


@pytest.fixture(params=[15, 30, 60])
def camera_fps_values(request):
    """Provide different FPS values for testing."""
    return request.param


@pytest.fixture(params=[0.3, 0.5, 0.7])
def confidence_thresholds(request):
    """Provide different confidence threshold values for testing."""
    return request.param


# Error simulation fixtures
@pytest.fixture
def simulate_webcam_error():
    """Simulate webcam errors for testing error handling."""
    def _simulate_error(error_type="access"):
        if error_type == "access":
            return WebcamError("Camera access denied")
        elif error_type == "not_found":
            return WebcamError("Camera device not found")
        elif error_type == "busy":
            return WebcamError("Camera device is busy")
        else:
            return WebcamError("Unknown camera error")

    return _simulate_error


@pytest.fixture
def simulate_model_error():
    """Simulate model errors for testing error handling."""
    def _simulate_error(error_type="load"):
        if error_type == "load":
            return ModelError("Failed to load model")
        elif error_type == "predict":
            return ModelError("Prediction failed")
        elif error_type == "format":
            return ModelError("Invalid model format")
        else:
            return ModelError("Unknown model error")

    return _simulate_error


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically clean up test files after each test."""
    yield

    # Clean up any temporary files created during tests
    temp_patterns = [
        "test_*.tmp",
        "*.test",
        "*_test_*"
    ]

    project_root = Path(__file__).parent.parent
    for pattern in temp_patterns:
        for file_path in project_root.glob(pattern):
            try:
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
            except (OSError, PermissionError):
                pass  # Ignore cleanup errors


# Test markers and utilities
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Add custom markers
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "service: mark test as service test")
    config.addinivalue_line("markers", "ui: mark test as UI test")
    config.addinivalue_line("markers", "security: mark test as security test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "regression: mark test as regression test")
    config.addinivalue_line("markers", "external: mark test as requiring external services")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "webcam: mark test as requiring webcam access")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and skip conditions."""
    for item in items:
        # Add markers based on test file location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "service" in str(item.fspath):
            item.add_marker(pytest.mark.service)
        elif "ui" in str(item.fspath):
            item.add_marker(pytest.mark.ui)

        # Skip tests based on environment
        if item.get_closest_marker("external") and not os.getenv("RUN_EXTERNAL_TESTS"):
            item.add_marker(pytest.mark.skip(reason="External tests disabled"))

        if item.get_closest_marker("gpu") and not os.getenv("RUN_GPU_TESTS"):
            item.add_marker(pytest.mark.skip(reason="GPU tests disabled"))

        if item.get_closest_marker("webcam") and not os.getenv("RUN_WEBCAM_TESTS"):
            item.add_marker(pytest.mark.skip(reason="Webcam tests disabled"))


# Test reporting
@pytest.fixture(scope="session", autouse=True)
def test_report():
    """Generate test report information."""
    import time
    start_time = time.time()

    yield

    end_time = time.time()
    duration = end_time - start_time

    print(f"\n=== Test Session Summary ===")
    print(f"Total duration: {duration:.2f} seconds")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"Ended: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")