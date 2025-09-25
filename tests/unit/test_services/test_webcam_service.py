"""Comprehensive unit tests for ImprovedWebcamService.

Tests cover initialization, camera operations, error handling, and edge cases.
"""
import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
from app.services.improved_webcam_service import ImprovedWebcamService
from app.core.exceptions import WebcamError
from app.config.settings import Config


class TestImprovedWebcamService:
    """Test suite for ImprovedWebcamService functionality."""

    def test_initialization_with_valid_config(self, mock_config):
        """Test service initialization with valid configuration."""
        service = ImprovedWebcamService(mock_config)

        assert service.config is mock_config
        assert service._cap is None
        assert service._last_fps_update == 0
        assert service._frame_count == 0
        assert service._current_fps == 0.0

    def test_initialization_with_invalid_config(self):
        """Test service initialization fails with invalid configuration."""
        with pytest.raises(TypeError):
            ImprovedWebcamService(None)

    @patch('cv2.VideoCapture')
    def test_open_webcam_success(self, mock_video_capture, mock_config):
        """Test successful webcam opening."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30
        }.get(prop, 0)
        mock_video_capture.return_value = mock_cap

        service = ImprovedWebcamService(mock_config)
        success = service.open_webcam(0)

        assert success is True
        assert service._cap is mock_cap
        mock_video_capture.assert_called_once()

    @patch('cv2.VideoCapture')
    def test_open_webcam_failure(self, mock_video_capture, mock_config):
        """Test webcam opening failure."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap

        service = ImprovedWebcamService(mock_config)
        success = service.open_webcam(0)

        assert success is False
        assert service._cap is None

    @patch('cv2.VideoCapture')
    def test_open_webcam_with_invalid_device_index(self, mock_video_capture, mock_config):
        """Test webcam opening with invalid device index."""
        mock_video_capture.side_effect = Exception("Device not found")

        service = ImprovedWebcamService(mock_config)
        success = service.open_webcam(-1)

        assert success is False
        assert service._cap is None

    def test_is_webcam_opened_true(self, mock_config):
        """Test webcam opened status when camera is active."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True

        service = ImprovedWebcamService(mock_config)
        service._cap = mock_cap

        assert service.is_webcam_opened() is True

    def test_is_webcam_opened_false(self, mock_config):
        """Test webcam opened status when camera is not active."""
        service = ImprovedWebcamService(mock_config)

        assert service.is_webcam_opened() is False

    def test_read_frame_success(self, mock_config):
        """Test successful frame reading."""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap = Mock()
        mock_cap.read.return_value = (True, test_frame)
        mock_cap.isOpened.return_value = True

        service = ImprovedWebcamService(mock_config)
        service._cap = mock_cap

        success, frame = service.read_frame()

        assert success is True
        assert np.array_equal(frame, test_frame)
        assert service._frame_count == 1

    def test_read_frame_failure(self, mock_config):
        """Test frame reading failure."""
        mock_cap = Mock()
        mock_cap.read.return_value = (False, None)
        mock_cap.isOpened.return_value = True

        service = ImprovedWebcamService(mock_config)
        service._cap = mock_cap

        success, frame = service.read_frame()

        assert success is False
        assert frame is None

    def test_read_frame_no_camera(self, mock_config):
        """Test frame reading when no camera is opened."""
        service = ImprovedWebcamService(mock_config)

        success, frame = service.read_frame()

        assert success is False
        assert frame is None

    def test_close_webcam(self, mock_config):
        """Test webcam closing."""
        mock_cap = Mock()

        service = ImprovedWebcamService(mock_config)
        service._cap = mock_cap

        service.close_webcam()

        mock_cap.release.assert_called_once()
        assert service._cap is None

    def test_close_webcam_no_camera(self, mock_config):
        """Test closing when no camera is opened."""
        service = ImprovedWebcamService(mock_config)

        # Should not raise exception
        service.close_webcam()

    def test_get_camera_properties(self, mock_config):
        """Test getting camera properties."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720,
            cv2.CAP_PROP_FPS: 60,
            cv2.CAP_PROP_BACKEND: 1400  # DSHOW
        }.get(prop, 0)
        mock_cap.getBackendName.return_value = "DSHOW"

        service = ImprovedWebcamService(mock_config)
        service._cap = mock_cap
        service._device_index = 0

        properties = service.get_camera_properties()

        expected = {
            'width': 1280,
            'height': 720,
            'fps': 60,
            'backend': 'DSHOW',
            'device_index': 0
        }
        assert properties == expected

    def test_get_camera_properties_no_camera(self, mock_config):
        """Test getting properties when no camera is opened."""
        service = ImprovedWebcamService(mock_config)

        properties = service.get_camera_properties()

        assert properties == {}

    @patch('time.time')
    def test_get_current_fps_calculation(self, mock_time, mock_config):
        """Test FPS calculation."""
        mock_time.side_effect = [0, 1, 2, 3]  # Simulate time progression

        mock_cap = Mock()
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.isOpened.return_value = True

        service = ImprovedWebcamService(mock_config)
        service._cap = mock_cap

        # Read multiple frames to calculate FPS
        service.read_frame()  # Frame 1 at t=0
        service.read_frame()  # Frame 2 at t=1
        service.read_frame()  # Frame 3 at t=2

        fps = service.get_current_fps()  # Called at t=3

        # Should calculate FPS based on frame count and time elapsed
        assert isinstance(fps, float)
        assert fps > 0

    def test_get_current_fps_no_frames(self, mock_config):
        """Test FPS when no frames have been read."""
        service = ImprovedWebcamService(mock_config)

        fps = service.get_current_fps()

        assert fps == 0.0

    def test_set_camera_property_success(self, mock_config):
        """Test setting camera property successfully."""
        mock_cap = Mock()
        mock_cap.set.return_value = True
        mock_cap.isOpened.return_value = True

        service = ImprovedWebcamService(mock_config)
        service._cap = mock_cap

        result = service.set_camera_property(cv2.CAP_PROP_FRAME_WIDTH, 1920)

        assert result is True
        mock_cap.set.assert_called_once_with(cv2.CAP_PROP_FRAME_WIDTH, 1920)

    def test_set_camera_property_failure(self, mock_config):
        """Test setting camera property failure."""
        mock_cap = Mock()
        mock_cap.set.return_value = False
        mock_cap.isOpened.return_value = True

        service = ImprovedWebcamService(mock_config)
        service._cap = mock_cap

        result = service.set_camera_property(cv2.CAP_PROP_FRAME_WIDTH, 1920)

        assert result is False

    def test_set_camera_property_no_camera(self, mock_config):
        """Test setting property when no camera is opened."""
        service = ImprovedWebcamService(mock_config)

        result = service.set_camera_property(cv2.CAP_PROP_FRAME_WIDTH, 1920)

        assert result is False

    @patch('cv2.VideoCapture')
    def test_context_manager_usage(self, mock_video_capture, mock_config):
        """Test using service as context manager."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap

        with ImprovedWebcamService(mock_config) as service:
            service.open_webcam(0)
            assert service.is_webcam_opened() is True

        # Should automatically close
        mock_cap.release.assert_called_once()

    def test_thread_safety_simulation(self, mock_config):
        """Test service behavior under simulated concurrent access."""
        import threading
        import time

        mock_cap = Mock()
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.isOpened.return_value = True

        service = ImprovedWebcamService(mock_config)
        service._cap = mock_cap

        results = []
        errors = []

        def read_frames():
            try:
                for _ in range(10):
                    success, frame = service.read_frame()
                    results.append(success)
                    time.sleep(0.001)  # Simulate processing time
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_frames) for _ in range(3)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0  # No exceptions should occur
        assert len(results) == 30  # All frame reads completed

    def test_memory_leak_simulation(self, mock_config):
        """Test for potential memory leaks with repeated operations."""
        import gc
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        mock_cap = Mock()
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.isOpened.return_value = True

        service = ImprovedWebcamService(mock_config)
        service._cap = mock_cap

        # Perform many operations
        for _ in range(1000):
            success, frame = service.read_frame()
            if frame is not None:
                # Simulate some processing
                processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                del processed

        gc.collect()
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 10MB)
        assert memory_increase < 10 * 1024 * 1024


@pytest.mark.integration
class TestWebcamServiceIntegration:
    """Integration tests requiring actual hardware or system resources."""

    @pytest.mark.webcam
    def test_real_webcam_detection(self, mock_config):
        """Test with real webcam if available (requires hardware)."""
        service = ImprovedWebcamService(mock_config)

        # Try to open default camera
        success = service.open_webcam(0)

        if success:
            # Test basic operations
            assert service.is_webcam_opened() is True

            success, frame = service.read_frame()
            if success:
                assert frame is not None
                assert frame.shape[2] == 3  # Color image
                assert frame.dtype == np.uint8

            properties = service.get_camera_properties()
            assert 'width' in properties
            assert 'height' in properties

            service.close_webcam()
            assert service.is_webcam_opened() is False
        else:
            pytest.skip("No webcam available for integration testing")


@pytest.mark.performance
class TestWebcamServicePerformance:
    """Performance tests for webcam service operations."""

    def test_frame_reading_performance(self, mock_config):
        """Test frame reading performance."""
        import time

        mock_cap = Mock()
        test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)  # Large frame
        mock_cap.read.return_value = (True, test_frame)
        mock_cap.isOpened.return_value = True

        service = ImprovedWebcamService(mock_config)
        service._cap = mock_cap

        start_time = time.time()

        for _ in range(100):
            service.read_frame()

        end_time = time.time()
        duration = end_time - start_time

        # Should be able to read 100 frames quickly
        assert duration < 1.0  # Less than 1 second

        fps = 100 / duration
        assert fps > 100  # Should achieve high mock FPS

    def test_fps_calculation_accuracy(self, mock_config):
        """Test accuracy of FPS calculations."""
        import time

        mock_cap = Mock()
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.isOpened.return_value = True

        service = ImprovedWebcamService(mock_config)
        service._cap = mock_cap

        # Read frames with controlled timing
        frame_interval = 1.0/30.0  # Target 30 FPS

        with patch('time.time') as mock_time:
            mock_time.side_effect = [i * frame_interval for i in range(100)]

            for _ in range(30):  # Read 30 frames
                service.read_frame()

            calculated_fps = service.get_current_fps()

            # Should be close to target FPS
            assert abs(calculated_fps - 30.0) < 5.0


# Test utilities and edge cases
class TestWebcamServiceEdgeCases:
    """Test edge cases and error conditions."""

    def test_rapid_open_close_cycles(self, mock_config):
        """Test rapid opening and closing of webcam."""
        with patch('cv2.VideoCapture') as mock_video_capture:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_video_capture.return_value = mock_cap

            service = ImprovedWebcamService(mock_config)

            for _ in range(50):
                assert service.open_webcam(0) is True
                service.close_webcam()
                assert service.is_webcam_opened() is False

    def test_invalid_property_values(self, mock_config):
        """Test setting invalid property values."""
        mock_cap = Mock()
        mock_cap.set.return_value = False  # Simulate property setting failure
        mock_cap.isOpened.return_value = True

        service = ImprovedWebcamService(mock_config)
        service._cap = mock_cap

        # Test various invalid values
        invalid_values = [-1, 0, 999999, 'invalid', None]

        for value in invalid_values:
            result = service.set_camera_property(cv2.CAP_PROP_FRAME_WIDTH, value)
            assert result is False

    def test_exception_handling_in_read_frame(self, mock_config):
        """Test exception handling during frame reading."""
        mock_cap = Mock()
        mock_cap.read.side_effect = Exception("Camera disconnected")
        mock_cap.isOpened.return_value = True

        service = ImprovedWebcamService(mock_config)
        service._cap = mock_cap

        success, frame = service.read_frame()

        assert success is False
        assert frame is None