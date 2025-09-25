"""Test and demonstrate the Atomic Application System for settings updates.

This script demonstrates:
1. Transactional settings updates with rollback
2. Service dependency management
3. Performance metrics tracking
4. Parallel and sequential operation execution
5. State snapshot and restoration
"""

import sys
import logging
import time
import random
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config.settings_manager import SettingsManager
from app.config.settings import Config
from app.config.atomic_applier import (
    TransactionalSettingsApplier,
    TransactionalService,
    UpdateType,
    ServicePriority
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class MockWebcamService:
    """Mock webcam service for testing."""

    def __init__(self):
        self.camera_index = 0
        self.width = 640
        self.height = 480
        self.fps = 30
        self.is_capturing_flag = False
        self.state_history = []

    def create_snapshot(self) -> Dict[str, Any]:
        """Create snapshot of current state."""
        return {
            'camera_index': self.camera_index,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'is_capturing': self.is_capturing_flag
        }

    def restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Restore state from snapshot."""
        self.camera_index = snapshot.get('camera_index', 0)
        self.width = snapshot.get('width', 640)
        self.height = snapshot.get('height', 480)
        self.fps = snapshot.get('fps', 30)
        self.is_capturing_flag = snapshot.get('is_capturing', False)
        logger.info(f"Webcam restored to: {self.width}x{self.height}@{self.fps}fps")

    def is_capturing(self) -> bool:
        return self.is_capturing_flag

    def stop_capture(self):
        logger.debug("Stopping webcam capture")
        self.is_capturing_flag = False

    def start_capture(self):
        logger.debug("Starting webcam capture")
        self.is_capturing_flag = True

    def set_camera(self, index: int):
        logger.debug(f"Setting camera index to {index}")
        self.camera_index = index

    def set_resolution(self, width: int, height: int):
        logger.debug(f"Setting resolution to {width}x{height}")
        # Simulate potential failure
        if width > 1920 or height > 1080:
            if random.random() < 0.3:  # 30% chance of failure for high resolutions
                raise ValueError(f"Resolution {width}x{height} not supported by camera")
        self.width = width
        self.height = height

    def set_fps(self, fps: int):
        logger.debug(f"Setting FPS to {fps}")
        self.fps = fps

    def is_opened(self) -> bool:
        return True


class MockDetectionService:
    """Mock detection service for testing."""

    def __init__(self):
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.4
        self.model_loaded = True

    def create_snapshot(self) -> Dict[str, Any]:
        return {
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'model_loaded': self.model_loaded
        }

    def restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self.confidence_threshold = snapshot.get('confidence_threshold', 0.5)
        self.iou_threshold = snapshot.get('iou_threshold', 0.4)
        self.model_loaded = snapshot.get('model_loaded', True)
        logger.info(f"Detection restored: conf={self.confidence_threshold}, iou={self.iou_threshold}")

    def set_confidence_threshold(self, threshold: float):
        logger.debug(f"Setting confidence threshold to {threshold}")
        # Simulate processing delay
        time.sleep(0.05)
        self.confidence_threshold = threshold

    def set_iou_threshold(self, threshold: float):
        logger.debug(f"Setting IoU threshold to {threshold}")
        time.sleep(0.05)
        self.iou_threshold = threshold


class MockGeminiService:
    """Mock Gemini AI service for testing."""

    def __init__(self):
        self.api_key = ""
        self.model = "gemini-1.5-flash"
        self.temperature = 0.7
        self.rate_limit_enabled = False

    def create_snapshot(self) -> Dict[str, Any]:
        return {
            'api_key': self.api_key,
            'model': self.model,
            'temperature': self.temperature,
            'rate_limit_enabled': self.rate_limit_enabled
        }

    def restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self.api_key = snapshot.get('api_key', '')
        self.model = snapshot.get('model', 'gemini-1.5-flash')
        self.temperature = snapshot.get('temperature', 0.7)
        self.rate_limit_enabled = snapshot.get('rate_limit_enabled', False)
        logger.info(f"Gemini restored: model={self.model}, temp={self.temperature}")

    def update_config(self, **kwargs):
        logger.debug(f"Updating Gemini config: {kwargs}")
        # Simulate API validation
        if 'api_key' in kwargs:
            if kwargs['api_key'] and not kwargs['api_key'].startswith('AIza'):
                raise ValueError("Invalid API key format")
            self.api_key = kwargs['api_key']

        if 'model' in kwargs:
            self.model = kwargs['model']

        if 'temperature' in kwargs:
            self.temperature = kwargs['temperature']

    def set_rate_limit(self, enabled: bool, requests_per_minute: int = 15):
        logger.debug(f"Setting rate limit: enabled={enabled}, rpm={requests_per_minute}")
        self.rate_limit_enabled = enabled


def test_successful_atomic_update():
    """Test successful atomic update of all services."""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Successful Atomic Update")
    logger.info("="*60)

    # Create settings manager
    settings_manager = SettingsManager("test_config.json")

    # Create mock services
    webcam_service = MockWebcamService()
    detection_service = MockDetectionService()
    gemini_service = MockGeminiService()

    # Register services
    settings_manager.register_service('webcam', webcam_service)
    settings_manager.register_service('detection', detection_service)
    settings_manager.register_service('gemini', gemini_service)

    # Get current config and modify it
    config = settings_manager.config
    config.camera_width = 1280
    config.camera_height = 720
    config.camera_fps = 60
    config.detection_confidence_threshold = 0.7
    config.detection_iou_threshold = 0.5
    config.gemini_temperature = 0.9

    # Apply settings atomically
    logger.info("Applying new settings atomically...")
    success = settings_manager.apply_settings(config, use_atomic=True)

    # Verify results
    assert success, "Settings application should succeed"
    assert webcam_service.width == 1280, f"Width should be 1280, got {webcam_service.width}"
    assert webcam_service.height == 720, f"Height should be 720, got {webcam_service.height}"
    assert detection_service.confidence_threshold == 0.7
    assert gemini_service.temperature == 0.9

    logger.info("✓ All settings applied successfully!")

    # Check performance metrics
    applier = settings_manager._transactional_applier
    logger.info(f"Performance: {applier.metrics.get_summary()}")


def test_rollback_on_failure():
    """Test rollback when one service fails."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Rollback on Service Failure")
    logger.info("="*60)

    # Create settings manager
    settings_manager = SettingsManager("test_config_rollback.json")

    # Create mock services
    webcam_service = MockWebcamService()
    detection_service = MockDetectionService()

    # Set initial values
    webcam_service.width = 640
    webcam_service.height = 480
    detection_service.confidence_threshold = 0.5

    # Register services
    settings_manager.register_service('webcam', webcam_service)
    settings_manager.register_service('detection', detection_service)

    # Get config and set invalid resolution (likely to fail)
    config = settings_manager.config
    config.camera_width = 4096  # Very high resolution
    config.camera_height = 2160  # May trigger failure
    config.detection_confidence_threshold = 0.8

    logger.info(f"Initial state: webcam={webcam_service.width}x{webcam_service.height}, "
                f"detection={detection_service.confidence_threshold}")

    # Apply settings (may fail and rollback)
    logger.info("Applying settings that may fail...")
    success = settings_manager.apply_settings(config, use_atomic=True)

    if not success:
        logger.info("✓ Settings application failed as expected, checking rollback...")

        # Verify rollback occurred
        assert webcam_service.width == 640, "Width should be rolled back to 640"
        assert webcam_service.height == 480, "Height should be rolled back to 480"
        assert detection_service.confidence_threshold == 0.5, "Threshold should be rolled back"

        logger.info("✓ All services rolled back successfully!")
    else:
        logger.info("Settings applied successfully (no failure occurred)")


def test_partial_update():
    """Test updating only specific service categories."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Partial Update (Only Changed Categories)")
    logger.info("="*60)

    # Create settings manager
    settings_manager = SettingsManager("test_config_partial.json")

    # Create mock services
    webcam_service = MockWebcamService()
    detection_service = MockDetectionService()
    gemini_service = MockGeminiService()

    # Register services
    settings_manager.register_service('webcam', webcam_service)
    settings_manager.register_service('detection', detection_service)
    settings_manager.register_service('gemini', gemini_service)

    # Get config and modify only detection settings
    config = settings_manager.config
    config.detection_confidence_threshold = 0.6
    config.detection_iou_threshold = 0.3

    # Apply settings
    logger.info("Applying only detection settings...")
    start_time = time.time()
    success = settings_manager.apply_settings(config, use_atomic=True)
    duration = (time.time() - start_time) * 1000

    assert success, "Partial update should succeed"
    assert detection_service.confidence_threshold == 0.6

    logger.info(f"✓ Partial update completed in {duration:.1f}ms")

    # Check that only detection was updated
    applier = settings_manager._transactional_applier
    logger.info(f"Performance: {applier.metrics.get_summary()}")


def test_context_manager():
    """Test atomic update using context manager."""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Context Manager for Atomic Updates")
    logger.info("="*60)

    # Create settings manager
    settings_manager = SettingsManager("test_config_context.json")

    # Create and register services
    webcam_service = MockWebcamService()
    settings_manager.register_service('webcam', webcam_service)

    initial_width = webcam_service.width

    try:
        # Use context manager for atomic update
        with settings_manager.atomic_update() as config:
            config.camera_width = 1920
            config.camera_height = 1080
            # Context manager will save and apply on exit

        logger.info(f"✓ Context manager update succeeded: {webcam_service.width}x{webcam_service.height}")

    except Exception as e:
        logger.error(f"Context manager update failed: {e}")
        assert webcam_service.width == initial_width, "Should be rolled back"


def test_performance_metrics():
    """Test and display performance metrics."""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Performance Metrics Analysis")
    logger.info("="*60)

    # Create settings manager
    settings_manager = SettingsManager("test_config_perf.json")

    # Register multiple services
    for i in range(5):
        service = MockDetectionService()
        settings_manager.register_service(f'detection_{i}', service)

    # Perform multiple updates and collect metrics
    for iteration in range(3):
        config = settings_manager.config
        config.detection_confidence_threshold = 0.5 + (iteration * 0.1)

        logger.info(f"\nIteration {iteration + 1}:")
        success = settings_manager.apply_settings(config, use_atomic=True)

        if success:
            applier = settings_manager._transactional_applier
            logger.info(f"Metrics: {applier.metrics.get_summary()}")

    # Display performance history
    applier = settings_manager._transactional_applier
    logger.info("\nPerformance History:")
    for entry in applier.performance_history:
        metrics = entry['metrics']
        timestamp = entry['timestamp'].strftime('%H:%M:%S')
        logger.info(f"  {timestamp}: {metrics.get_summary()}")


def main():
    """Run all tests."""
    try:
        test_successful_atomic_update()
        test_rollback_on_failure()
        test_partial_update()
        test_context_manager()
        test_performance_metrics()

        logger.info("\n" + "="*60)
        logger.info("All tests completed successfully!")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()