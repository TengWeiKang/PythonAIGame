"""Webcam service for camera capture and frame management."""

import cv2
import threading
import time
import logging
from typing import Optional, Callable, List, Dict, Any
import numpy as np
import platform

logger = logging.getLogger(__name__)


class WebcamService:
    """Service for managing webcam capture and streaming."""

    def __init__(self, camera_index: int = 0, width: int = 1920, height: int = 1080, fps: int = 30, codec: str = 'Auto'):
        """Initialize webcam service.

        Args:
            camera_index: Camera device index
            width: Frame width
            height: Frame height
            fps: Target frames per second
            codec: Video codec (Auto, MJPG, YUYV, H264, etc.)
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = fps
        self.codec = codec
        self._camera_name: Optional[str] = None

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
            self._capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

            if not self._capture.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return False

            # Set camera properties
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._capture.set(cv2.CAP_PROP_FPS, self.target_fps)

            # Set codec if not Auto
            if self.codec and self.codec != 'Auto':
                fourcc_code = self._codec_to_fourcc(self.codec)
                if fourcc_code is not None:
                    try:
                        self._capture.set(cv2.CAP_PROP_FOURCC, fourcc_code)
                        logger.info(f"Video codec set to: {self.codec}")
                    except Exception as e:
                        logger.warning(f"Failed to set codec {self.codec}: {e}. Using default codec.")
                else:
                    logger.warning(f"Invalid codec '{self.codec}', using Auto (default)")

            # Get actual properties
            actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Try to get camera name
            self._camera_name = self._get_camera_name(self.camera_index)
            logger.info(f"Camera opened: {actual_width}x{actual_height} - {self._camera_name}")

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
        self._camera_name = None

    def get_camera_name(self) -> str:
        """Get the name of the current camera device.

        Returns:
            Camera name string, or "Camera {index}" if name cannot be determined
        """
        if self._camera_name:
            return self._camera_name
        return self._get_camera_name(self.camera_index)

    def _get_camera_name(self, camera_index: int) -> str:
        """Get the name of a camera device.

        Args:
            camera_index: Camera device index

        Returns:
            Camera name string
        """
        try:
            # Try platform-specific methods
            if platform.system() == "Windows":
                return self._get_camera_name_windows(camera_index)
            elif platform.system() == "Linux":
                return self._get_camera_name_linux(camera_index)
            elif platform.system() == "Darwin":  # macOS
                return self._get_camera_name_macos(camera_index)
            else:
                return f"Camera {camera_index}"
        except Exception as e:
            logger.warning(f"Failed to get camera name: {e}")
            return f"Camera {camera_index}"

    def _get_camera_name_windows(self, camera_index: int) -> str:
        """Get camera name on Windows using DirectShow.

        Args:
            camera_index: Camera device index

        Returns:
            Camera name string
        """
        try:
            # Try to use Windows Management Instrumentation (WMI)
            import subprocess
            result = subprocess.run(
                ['powershell', '-Command',
                 'Get-PnpDevice -Class Camera | Select-Object -Property FriendlyName | ConvertTo-Json'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                import json
                cameras = json.loads(result.stdout)
                if isinstance(cameras, list) and camera_index < len(cameras):
                    name = cameras[camera_index].get('FriendlyName', f'Camera {camera_index}')
                    return name
                elif isinstance(cameras, dict):
                    name = cameras.get('FriendlyName', f'Camera {camera_index}')
                    return name
        except Exception as e:
            logger.debug(f"Failed to get Windows camera name via PowerShell: {e}")

        # Fallback to generic name
        return f"Camera {camera_index}"

    def _get_camera_name_linux(self, camera_index: int) -> str:
        """Get camera name on Linux using v4l2.

        Args:
            camera_index: Camera device index

        Returns:
            Camera name string
        """
        try:
            import subprocess
            device_path = f"/dev/video{camera_index}"
            result = subprocess.run(
                ['v4l2-ctl', '-d', device_path, '--info'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Card type' in line:
                        return line.split(':', 1)[1].strip()
        except Exception as e:
            logger.debug(f"Failed to get Linux camera name via v4l2-ctl: {e}")

        return f"Camera {camera_index}"

    def _get_camera_name_macos(self, camera_index: int) -> str:
        """Get camera name on macOS.

        Args:
            camera_index: Camera device index

        Returns:
            Camera name string
        """
        try:
            import subprocess
            result = subprocess.run(
                ['system_profiler', 'SPCameraDataType'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                cameras = []
                for line in result.stdout.split('\n'):
                    line = line.strip()
                    if line and ':' in line and 'Camera' not in line:
                        cameras.append(line.split(':')[0].strip())

                if camera_index < len(cameras):
                    return cameras[camera_index]
        except Exception as e:
            logger.debug(f"Failed to get macOS camera name: {e}")

        return f"Camera {camera_index}"

    def _codec_to_fourcc(self, codec: str) -> Optional[int]:
        """Convert codec string to OpenCV FourCC code.

        Args:
            codec: Codec name string (MJPG, YUYV, H264, etc.)

        Returns:
            FourCC code as integer, or None if invalid
        """
        try:
            # Map common codec names to FourCC codes
            codec_map = {
                'MJPG': cv2.VideoWriter_fourcc(*'MJPG'),
                'YUYV': cv2.VideoWriter_fourcc(*'YUYV'),
                'H264': cv2.VideoWriter_fourcc(*'H264'),
                'VP8': cv2.VideoWriter_fourcc(*'VP8 '),
                'I420': cv2.VideoWriter_fourcc(*'I420'),
                'RGB3': cv2.VideoWriter_fourcc(*'RGB3'),
                'GREY': cv2.VideoWriter_fourcc(*'GREY'),
                'NV12': cv2.VideoWriter_fourcc(*'NV12'),
                'UYVY': cv2.VideoWriter_fourcc(*'UYVY'),
            }

            return codec_map.get(codec.upper())

        except Exception as e:
            logger.error(f"Error converting codec to FourCC: {e}")
            return None

    def set_codec(self, codec: str):
        """Set video codec for camera.

        Args:
            codec: Codec name (Auto, MJPG, YUYV, H264, etc.)
        """
        self.codec = codec
        logger.info(f"Codec changed to: {codec}")

        # If camera is currently streaming, restart with new codec
        if self._is_streaming:
            logger.info("Restarting stream to apply new codec...")
            callback = self._frame_callback
            self.stop_stream()
            self.start_stream(callback)

    def set_camera_index(self, camera_index: int):
        """Set camera device index.

        Args:
            camera_index: Camera device index
        """
        if self.camera_index == camera_index:
            return  # No change needed

        self.camera_index = camera_index
        logger.info(f"Camera index changed to: {camera_index}")

        # If camera is currently streaming, restart with new camera
        if self._is_streaming:
            logger.info("Restarting stream to apply new camera...")
            callback = self._frame_callback
            self.stop_stream()
            self.start_stream(callback)

    @staticmethod
    def list_available_cameras(max_cameras: int = 10) -> List[Dict[str, Any]]:
        """List all available camera devices.

        Args:
            max_cameras: Maximum number of cameras to check (default 10)

        Returns:
            List of dictionaries with camera info: {'index': int, 'name': str, 'available': bool}
        """
        cameras = []

        for i in range(max_cameras):
            cap = None
            try:
                logger.info("Checking camera index %d", i)
                cap = cv2.VideoCapture(i)
                logger.info("Check camera index %d complete", i)
                if cap.isOpened():
                    # Camera is available
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    # Get camera name
                    temp_service = WebcamService(camera_index=i)
                    camera_name = temp_service._get_camera_name(i)

                    cameras.append({
                        'index': i,
                        'name': camera_name,
                        'available': True,
                        'width': width,
                        'height': height
                    })
                    cap.release()
            except Exception as e:
                logger.debug(f"Failed to check camera {i}: {e}")
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except:
                        pass

        return cameras