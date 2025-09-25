"""Improved webcam service using modern base service architecture.

This is a refactored version of the WebcamService that uses the new base service
classes to provide better error handling, logging, health checks, and performance monitoring.
"""
from __future__ import annotations

import cv2
import subprocess
import os
import time
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
from pathlib import Path

from ..core.base_service import AsyncServiceMixin, CacheableServiceMixin, BaseService
from ..core.exceptions import WebcamError, ValidationError, ServiceError
from ..core.performance import performance_timer
from ..utils.validation import InputValidator


class ImprovedWebcamService(AsyncServiceMixin, BaseService):
    """Enhanced webcam management service with modern architecture.

    Features:
    - Comprehensive error handling and recovery
    - Performance optimization with frame buffering
    - Health monitoring and metrics
    - Proper resource cleanup
    - Device detection and validation
    - Configurable frame processing
    """

    def __init__(self,
                 config=None,
                 buffer_size: int = 3,
                 target_fps: int = 30,
                 frame_skip_threshold: int = 2,
                 **kwargs):
        """Initialize webcam service.

        Args:
            config: Service configuration
            buffer_size: Frame buffer size for smooth playback
            target_fps: Target frames per second
            frame_skip_threshold: Number of frames to skip for performance
            **kwargs: Additional base service arguments
        """
        super().__init__(config=config, service_name="WebcamService", **kwargs)

        # Webcam state
        self._cap = None
        self._device_index: Optional[int] = None
        self._is_opened = False

        # Performance optimizations
        self._buffer_size = buffer_size
        self._frame_buffer = deque(maxlen=buffer_size)
        self._buffer_lock = threading.Lock()
        self._buffering_active = False
        self._buffer_thread = None

        # Frame processing settings
        self._target_fps = target_fps
        self._frame_skip_threshold = frame_skip_threshold
        self._frame_skip_count = 0

        # Performance metrics
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._current_fps = 0.0
        self._total_frames_processed = 0
        self._dropped_frames = 0

        # Device configuration
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._fps_setting: Optional[int] = None

        # Available backends in order of preference
        self._backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]

    def _initialize(self) -> None:
        """Initialize webcam service resources."""
        self.logger.info("Initializing webcam service")

        # Test OpenCV functionality
        try:
            test_cap = cv2.VideoCapture()
            test_cap.release()
        except Exception as e:
            raise ServiceError(f"OpenCV not properly installed or configured: {e}")

        self.logger.debug("Webcam service initialized successfully")

    def _shutdown(self) -> None:
        """Cleanup webcam service resources."""
        self.logger.info("Shutting down webcam service")

        # Close webcam if open
        if self._is_opened:
            self.close_webcam()

        # Stop background threads
        self._stop_background_threads()

        self.logger.debug("Webcam service shutdown completed")

    def _health_check(self) -> bool:
        """Perform webcam service health check."""
        base_healthy = super()._health_check()

        # Check if webcam is responsive
        webcam_healthy = True
        if self._is_opened and self._cap:
            try:
                # Try to get a property to test if camera is responsive
                _ = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            except Exception:
                webcam_healthy = False

        return base_healthy and webcam_healthy

    def _get_health_details(self) -> Dict[str, Any]:
        """Get detailed health information."""
        details = super()._get_health_details()
        details.update({
            'webcam_opened': self._is_opened,
            'device_index': self._device_index,
            'current_fps': self._current_fps,
            'target_fps': self._target_fps,
            'total_frames': self._total_frames_processed,
            'dropped_frames': self._dropped_frames,
            'buffer_size': len(self._frame_buffer),
            'buffering_active': self._buffering_active
        })
        return details

    @performance_timer("webcam_open")
    def open_webcam(self,
                   index: int,
                   width: Optional[int] = None,
                   height: Optional[int] = None,
                   fps: Optional[int] = None) -> bool:
        """Open webcam with specified parameters.

        Args:
            index: Camera device index
            width: Frame width (optional)
            height: Frame height (optional)
            fps: Frames per second (optional)

        Returns:
            True if webcam opened successfully

        Raises:
            ValidationError: If parameters are invalid
            WebcamError: If webcam cannot be opened
        """
        # Validate inputs
        if not isinstance(index, int) or index < 0:
            raise ValidationError("Device index must be a non-negative integer")

        if width is not None and (not isinstance(width, int) or width <= 0):
            raise ValidationError("Width must be a positive integer")

        if height is not None and (not isinstance(height, int) or height <= 0):
            raise ValidationError("Height must be a positive integer")

        if fps is not None and (not isinstance(fps, int) or fps <= 0):
            raise ValidationError("FPS must be a positive integer")

        with self.operation_context("open_webcam"):
            # Close existing connection
            if self._is_opened:
                self.close_webcam()

            self.logger.info(f"Opening webcam device {index}")

            try:
                # Try different backends for better compatibility
                for backend in self._backends:
                    try:
                        self._cap = cv2.VideoCapture(index, backend)
                        if self._cap.isOpened():
                            self.logger.debug(f"Webcam opened with backend {backend}")
                            break
                        else:
                            self._cap.release()
                            self._cap = None
                    except Exception as e:
                        self.logger.debug(f"Backend {backend} failed: {e}")
                        continue

                if not self._cap or not self._cap.isOpened():
                    available_devices = self.list_available_devices()
                    raise WebcamError(
                        f"Failed to open webcam at index {index}. "
                        f"Available devices: {available_devices}"
                    )

                # Store device settings
                self._device_index = index
                self._width = width
                self._height = height
                self._fps_setting = fps or self._target_fps

                # Configure camera properties
                self._configure_camera_properties()

                # Start frame buffering for smooth playback
                self._start_frame_buffering()

                self._is_opened = True
                self.logger.info(f"Webcam {index} opened successfully")

                # Log camera properties
                props = self.get_camera_properties()
                self.logger.debug(f"Camera properties: {props}")

                return True

            except Exception as e:
                self._cleanup_failed_open()
                raise WebcamError(f"Error opening webcam {index}: {e}") from e

    def _configure_camera_properties(self) -> None:
        """Configure camera properties for optimal performance."""
        if not self._cap:
            return

        try:
            # Set resolution if specified
            if self._width:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            if self._height:
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

            # Set FPS if specified
            if self._fps_setting:
                self._cap.set(cv2.CAP_PROP_FPS, self._fps_setting)

            # Performance optimizations
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering lag

            # Try to set MJPEG codec for better performance
            try:
                self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            except Exception:
                pass  # Codec setting is optional

            # Additional optimization settings
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Disable auto exposure for consistency

        except Exception as e:
            self.logger.warning(f"Failed to configure some camera properties: {e}")

    def _start_frame_buffering(self) -> None:
        """Start background frame buffering for smooth playback."""
        if self._buffer_thread and self._buffer_thread.is_alive():
            return

        self._buffering_active = True
        self._buffer_thread = self._start_background_thread(
            target=self._buffering_worker,
            name="WebcamBuffering"
        )

    def _buffering_worker(self) -> None:
        """Background worker for frame buffering."""
        self.logger.debug("Frame buffering worker started")

        while self._buffering_active and self._cap and self._is_opened:
            if self._shutdown_event.is_set():
                break

            try:
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    with self._buffer_lock:
                        # Add frame to buffer (automatically handles overflow)
                        self._frame_buffer.append(frame.copy())

                # Control buffering rate - slightly faster than target FPS
                buffer_delay = 1.0 / (self._target_fps * 1.2)
                time.sleep(buffer_delay)

            except Exception as e:
                self.logger.error(f"Error in buffering worker: {e}")
                time.sleep(0.1)  # Brief pause before retry

        self.logger.debug("Frame buffering worker stopped")

    @performance_timer("webcam_read")
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from webcam with optimizations.

        Returns:
            Tuple of (success, frame) where success is True if frame was read successfully

        Raises:
            WebcamError: If webcam is not open
        """
        if not self._is_opened or not self._cap:
            raise WebcamError("Webcam not opened. Call open_webcam() first.")

        with self.operation_context("read_frame"):
            try:
                # Try to read from buffer first for smoother playback
                if self._buffering_active and self._frame_buffer:
                    return self._read_from_buffer()

                # Fall back to direct read
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    self._update_fps_metrics()
                    self._total_frames_processed += 1

                    # Apply frame skipping for performance if needed
                    if self._should_skip_frame():
                        return self.read_frame()  # Recursively get next frame

                    return ret, frame
                else:
                    self._dropped_frames += 1

                return ret, frame

            except Exception as e:
                self.logger.error(f"Error reading frame: {e}")
                return False, None

    def _read_from_buffer(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from internal buffer."""
        with self._buffer_lock:
            if self._frame_buffer:
                frame = self._frame_buffer.popleft()
                self._update_fps_metrics()
                self._total_frames_processed += 1
                return True, frame

        # Buffer empty, fall back to direct read
        if self._cap:
            ret, frame = self._cap.read()
            if ret:
                self._total_frames_processed += 1
            return ret, frame

        return False, None

    def _should_skip_frame(self) -> bool:
        """Determine if current frame should be skipped for performance."""
        self._frame_skip_count += 1

        if self._frame_skip_count >= self._frame_skip_threshold:
            self._frame_skip_count = 0
            return False  # Don't skip this frame

        return True  # Skip this frame

    def _update_fps_metrics(self) -> None:
        """Update FPS performance metrics."""
        current_time = time.time()
        self._frame_count += 1

        # Calculate FPS every second
        if current_time - self._last_fps_time >= 1.0:
            elapsed = current_time - self._last_fps_time
            self._current_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = current_time

            # Record FPS metric
            self._performance_monitor.record_operation_time("webcam_fps", self._current_fps)

    def close_webcam(self) -> None:
        """Close webcam connection with comprehensive resource cleanup."""
        if not self._is_opened:
            self.logger.debug("Webcam not opened, nothing to close")
            return

        self.logger.info("Closing webcam connection")

        with self.operation_context("close_webcam"):
            # Stop buffering
            self._buffering_active = False

            # Stop buffering thread
            if self._buffer_thread and self._buffer_thread.is_alive():
                self.logger.debug("Stopping buffering thread...")
                self._buffer_thread.join(timeout=2.0)

                if self._buffer_thread.is_alive():
                    self.logger.warning("Buffering thread did not stop gracefully")

            # Clear buffer
            with self._buffer_lock:
                buffer_size = len(self._frame_buffer)
                self._frame_buffer.clear()
                self.logger.debug(f"Cleared {buffer_size} frames from buffer")

            # Release camera with retry logic
            self._release_camera_with_retry()

            # Reset state
            self._cap = None
            self._device_index = None
            self._is_opened = False

            self.logger.info("Webcam closed successfully")

    def _release_camera_with_retry(self, max_attempts: int = 3) -> None:
        """Release camera with multiple attempts and error handling."""
        if not self._cap:
            return

        for attempt in range(max_attempts):
            try:
                if hasattr(self._cap, 'isOpened') and callable(self._cap.isOpened):
                    if self._cap.isOpened():
                        self.logger.debug(f"Camera release attempt {attempt + 1}/{max_attempts}")
                        self._cap.release()

                        # Brief pause for system to process
                        time.sleep(0.1)

                        # Verify release
                        if hasattr(self._cap, 'isOpened') and self._cap.isOpened():
                            if attempt < max_attempts - 1:
                                self.logger.warning(f"Camera still open after attempt {attempt + 1}")
                                continue
                            else:
                                self.logger.error("Failed to release camera after all attempts")
                        else:
                            self.logger.debug("Camera released successfully")
                            break
                    else:
                        self.logger.debug("Camera was already closed")
                        break
                else:
                    # Fallback for older OpenCV versions
                    self._cap.release()
                    break

            except Exception as e:
                if attempt < max_attempts - 1:
                    self.logger.warning(f"Camera release attempt {attempt + 1} failed: {e}")
                else:
                    self.logger.error(f"Failed to release camera after {max_attempts} attempts: {e}")

    def _cleanup_failed_open(self) -> None:
        """Clean up resources after failed open attempt."""
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        self._is_opened = False
        self._device_index = None

    def get_camera_properties(self) -> Dict[str, Any]:
        """Get current camera properties.

        Returns:
            Dictionary with camera properties

        Raises:
            WebcamError: If webcam is not open
        """
        if not self._is_opened or not self._cap:
            raise WebcamError("Webcam not opened")

        try:
            return {
                'width': int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': int(self._cap.get(cv2.CAP_PROP_FPS)),
                'backend': self._cap.getBackendName(),
                'device_index': self._device_index,
                'buffer_size': len(self._frame_buffer),
                'current_fps': self._current_fps
            }
        except Exception as e:
            raise WebcamError(f"Failed to get camera properties: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        total_frames = self._total_frames_processed
        drop_rate = self._dropped_frames / total_frames if total_frames > 0 else 0.0

        return {
            'total_frames_processed': total_frames,
            'dropped_frames': self._dropped_frames,
            'drop_rate': drop_rate,
            'current_fps': self._current_fps,
            'target_fps': self._target_fps,
            'buffer_utilization': len(self._frame_buffer) / self._buffer_size,
            'buffering_active': self._buffering_active
        }

    @staticmethod
    def list_available_devices(max_test: int = 5) -> List[Tuple[int, str]]:
        """List available webcam devices.

        Args:
            max_test: Maximum number of device indices to test

        Returns:
            List of tuples (device_index, device_name)
        """
        devices = []
        device_names = []

        # Try to get device names on Windows
        if os.name == 'nt':
            try:
                result = subprocess.run(
                    ["wmic", "path", "Win32_PnPEntity", "where", "Service='usbvideo'", "get", "Name"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                    if lines and lines[0].lower().startswith('name'):
                        device_names = lines[1:]
            except Exception:
                pass  # Fallback to generic names

        # Test device indices
        for i in range(max_test):
            cap = None
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    device_name = device_names[i] if i < len(device_names) else f"Camera {i}"
                    devices.append((i, device_name))
            except Exception:
                pass
            finally:
                if cap:
                    cap.release()

        return devices

    def is_webcam_opened(self) -> bool:
        """Check if webcam is currently opened.

        Returns:
            True if webcam is opened
        """
        return self._is_opened

    def get_current_fps(self) -> float:
        """Get current actual FPS.

        Returns:
            Current FPS as float
        """
        return self._current_fps

    # Context manager support
    def __enter__(self) -> 'ImprovedWebcamService':
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        if self._is_opened:
            self.close_webcam()