"""Webcam service for camera capture and frame management."""

import cv2
import threading
import time
import logging
from typing import Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


class WebcamService:
    """Service for managing webcam capture and streaming."""

    def __init__(self, camera_index: int = 0, width: int = 1920, height: int = 1080, fps: int = 30):
        """Initialize webcam service.

        Args:
            camera_index: Camera device index
            width: Frame width
            height: Frame height
            fps: Target frames per second
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = fps

        self._capture: Optional[cv2.VideoCapture] = None
        self._is_streaming = False
        self._stream_thread: Optional[threading.Thread] = None
        self._current_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._frame_callback: Optional[Callable[[np.ndarray], None]] = None
        self._actual_fps = 0.0
        self._frame_count = 0
        self._fps_start_time = time.time()

    def start_stream(self, frame_callback: Optional[Callable[[np.ndarray], None]] = None) -> bool:
        """Start webcam streaming.

        Args:
            frame_callback: Optional callback function called for each frame

        Returns:
            True if stream started successfully, False otherwise
        """
        if self._is_streaming:
            logger.warning("Stream already running")
            return True

        try:
            self._capture = cv2.VideoCapture(self.camera_index)

            if not self._capture.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return False

            # Set camera properties
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._capture.set(cv2.CAP_PROP_FPS, self.target_fps)

            # Get actual properties
            actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Camera opened: {actual_width}x{actual_height}")

            self._frame_callback = frame_callback
            self._is_streaming = True
            self._fps_start_time = time.time()
            self._frame_count = 0

            # Start streaming thread
            self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
            self._stream_thread.start()

            return True

        except Exception as e:
            logger.error(f"Error starting stream: {e}")
            self._cleanup()
            return False

    def stop_stream(self):
        """Stop webcam streaming and release resources."""
        if not self._is_streaming:
            return

        self._is_streaming = False

        # Wait for thread to finish
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=2.0)

        self._cleanup()
        logger.info("Stream stopped")

    def _stream_loop(self):
        """Main streaming loop running in separate thread."""
        frame_delay = 1.0 / self.target_fps

        while self._is_streaming:
            loop_start = time.time()

            try:
                if self._capture and self._capture.isOpened():
                    ret, frame = self._capture.read()

                    if ret and frame is not None:
                        # Update current frame
                        with self._frame_lock:
                            self._current_frame = frame.copy()

                        # Update FPS
                        self._frame_count += 1
                        elapsed = time.time() - self._fps_start_time
                        if elapsed >= 1.0:
                            self._actual_fps = self._frame_count / elapsed
                            self._frame_count = 0
                            self._fps_start_time = time.time()

                        # Call callback if provided
                        if self._frame_callback:
                            try:
                                self._frame_callback(frame.copy())
                            except Exception as e:
                                logger.error(f"Error in frame callback: {e}")
                    else:
                        logger.warning("Failed to read frame from camera")
                        time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in stream loop: {e}")
                time.sleep(0.1)

            # Maintain target FPS
            loop_time = time.time() - loop_start
            sleep_time = max(0, frame_delay - loop_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame.

        Returns:
            Current frame as numpy array, or None if no frame available
        """
        with self._frame_lock:
            return self._current_frame.copy() if self._current_frame is not None else None

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture and return a single frame.

        Returns:
            Captured frame as numpy array, or None on failure
        """
        return self.get_current_frame()

    def get_fps(self) -> float:
        """Get actual FPS of the stream.

        Returns:
            Current FPS value
        """
        return self._actual_fps

    def get_resolution(self) -> tuple[int, int]:
        """Get current camera resolution.

        Returns:
            Tuple of (width, height)
        """
        if self._capture and self._capture.isOpened():
            width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (0, 0)

    def is_streaming(self) -> bool:
        """Check if stream is active.

        Returns:
            True if streaming, False otherwise
        """
        return self._is_streaming

    def _cleanup(self):
        """Clean up camera resources."""
        if self._capture:
            self._capture.release()
            self._capture = None

        with self._frame_lock:
            self._current_frame = None

        self._frame_callback = None