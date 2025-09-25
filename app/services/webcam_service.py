"""Enhanced webcam service with better error handling and device management."""
from __future__ import annotations
import cv2
import subprocess
import os
import threading
import time
import numpy as np
import logging
from typing import List, Tuple, Optional
from collections import deque

from ..core.exceptions import WebcamError
from ..core.performance import performance_timer, PerformanceMonitor

# Initialize logger for this module
logger = logging.getLogger(__name__)

class WebcamService:
    """Enhanced webcam management service with performance optimizations."""
    
    def __init__(self):
        self.cap = None
        self.index = None
        self._is_opened = False
        
        # Performance optimizations
        self._frame_buffer = deque(maxlen=3)  # Buffer for smooth playback
        self._buffer_lock = threading.Lock()
        self._buffer_thread = None
        self._buffering_active = False
        
        # Performance metrics
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._current_fps = 0.0
        
        # Frame processing optimizations
        self._frame_skip_count = 0
        self._target_fps = 30
        self._frame_skip_threshold = 2  # Skip every N frames if needed

    @performance_timer("webcam_open")
    def open(self, index: int, width: Optional[int] = None, 
             height: Optional[int] = None, fps: Optional[int] = None) -> bool:
        """Open webcam with specified parameters."""
        self.close()
        
        try:
            # Try different backends for better compatibility
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]
            
            for backend in backends:
                try:
                    self.cap = cv2.VideoCapture(index, backend)
                    if self.cap.isOpened():
                        break
                    else:
                        self.cap.release()
                        self.cap = None
                except (OSError, RuntimeError, Exception) as e:
                    # Backend not available or camera access denied
                    logger.debug(f"Failed to open camera with backend {backend}: {e}")
                    continue
            
            if not self.cap or not self.cap.isOpened():
                raise WebcamError(f"Failed to open webcam at index {index}")
            
            self.index = index
            self._is_opened = True
            
            # Set optimized camera properties
            if width:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps:
                self.cap.set(cv2.CAP_PROP_FPS, fps)
                self._target_fps = fps
            
            # Performance optimizations
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            # Start frame buffering thread for smooth playback
            self._start_buffering()
                
            return True
            
        except Exception as e:
            self.close()
            raise WebcamError(f"Error opening webcam: {e}")

    @performance_timer("webcam_read")
    def read(self) -> Tuple[bool, Optional[any]]:
        """Read frame from webcam with optimizations."""
        if not self.cap or not self._is_opened:
            return False, None
        
        try:
            # Use buffered frame if available for smoother playback
            if self._buffering_active:
                return self._read_from_buffer()
            
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Update FPS metrics
                self._update_fps_metrics()
                
                # Frame skipping for performance if needed
                self._frame_skip_count += 1
                if self._frame_skip_count >= self._frame_skip_threshold:
                    self._frame_skip_count = 0
                    return ret, frame
                else:
                    # Skip this frame, read next one
                    return self.read()
            
            return ret, frame
        except Exception:
            return False, None
    
    def _read_from_buffer(self) -> Tuple[bool, Optional[any]]:
        """Read frame from internal buffer."""
        with self._buffer_lock:
            if self._frame_buffer:
                return True, self._frame_buffer.popleft()
            else:
                # Fallback to direct read
                if self.cap:
                    ret, frame = self.cap.read()
                    return ret, frame
                return False, None
    
    def _start_buffering(self):
        """Start background frame buffering for smooth playback."""
        if self._buffer_thread is None or not self._buffer_thread.is_alive():
            self._buffering_active = True
            self._buffer_thread = threading.Thread(
                target=self._buffering_worker,
                name="WebcamBuffer",
                daemon=True
            )
            self._buffer_thread.start()
    
    def _buffering_worker(self):
        """Background worker for frame buffering."""
        while self._buffering_active and self.cap and self._is_opened:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self._buffer_lock:
                        self._frame_buffer.append(frame.copy())
                
                # Control buffering rate
                time.sleep(1.0 / (self._target_fps * 1.5))  # Slightly faster than target FPS
            except Exception as e:
                print(f"Buffering error: {e}")
                time.sleep(0.1)
    
    def _update_fps_metrics(self):
        """Update FPS performance metrics."""
        current_time = time.time()
        self._frame_count += 1
        
        if current_time - self._last_fps_time >= 1.0:
            self._current_fps = self._frame_count / (current_time - self._last_fps_time)
            self._frame_count = 0
            self._last_fps_time = current_time
            
            # Record FPS metric
            monitor = PerformanceMonitor.instance()
            monitor.record_operation_time("webcam_fps", self._current_fps)
    
    def get_current_fps(self) -> float:
        """Get current actual FPS."""
        return self._current_fps

    def close(self) -> None:
        """Close webcam connection with comprehensive resource cleanup.

        Implements proper resource leak prevention with retry logic and
        timeout handling to ensure all resources are properly released.
        """
        logger.info("Starting webcam service cleanup...")

        # Stop buffering thread with retry logic
        self._buffering_active = False
        if self._buffer_thread and self._buffer_thread.is_alive():
            logger.debug("Stopping buffering thread...")

            # First attempt: normal join with timeout
            self._buffer_thread.join(timeout=2.0)

            if self._buffer_thread.is_alive():
                logger.warning("Buffering thread did not stop gracefully, attempting forced cleanup...")

                # Second attempt: longer timeout
                self._buffer_thread.join(timeout=5.0)

                if self._buffer_thread.is_alive():
                    logger.error("Buffering thread failed to stop - this may indicate a resource leak")
                    # Note: In Python, we cannot forcefully terminate threads,
                    # but we can mark the thread as daemon and let the interpreter handle it
                    self._buffer_thread.daemon = True
                else:
                    logger.debug("Buffering thread stopped successfully on second attempt")
            else:
                logger.debug("Buffering thread stopped successfully")

        # Clear buffer with error handling
        try:
            with self._buffer_lock:
                buffer_size = len(self._frame_buffer)
                self._frame_buffer.clear()
                logger.debug(f"Cleared {buffer_size} frames from buffer")
        except Exception as e:
            logger.error(f"Error clearing frame buffer: {e}")

        # Close camera with multiple attempts and comprehensive error handling
        if self.cap:
            logger.debug("Releasing camera resources...")

            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    # Check if camera is still accessible
                    if hasattr(self.cap, 'isOpened') and callable(self.cap.isOpened):
                        if self.cap.isOpened():
                            logger.debug(f"Camera release attempt {attempt + 1}/{max_attempts}")
                            self.cap.release()

                            # Verify release was successful
                            import time
                            time.sleep(0.1)  # Brief pause for system to process

                            if hasattr(self.cap, 'isOpened') and self.cap.isOpened():
                                if attempt < max_attempts - 1:
                                    logger.warning(f"Camera still open after release attempt {attempt + 1}, retrying...")
                                    continue
                                else:
                                    logger.error("Failed to release camera after all attempts")
                            else:
                                logger.debug("Camera released successfully")
                                break
                        else:
                            logger.debug("Camera was already closed")
                            break
                    else:
                        # Fallback for older OpenCV versions or corrupted objects
                        logger.debug("Using fallback camera release method")
                        self.cap.release()
                        break

                except AttributeError as e:
                    logger.warning(f"Camera object corrupted during release: {e}")
                    break
                except Exception as e:
                    if attempt < max_attempts - 1:
                        logger.warning(f"Camera release attempt {attempt + 1} failed: {e}, retrying...")
                    else:
                        logger.error(f"Failed to release camera after {max_attempts} attempts: {e}")

        # Cleanup object references
        self.cap = None
        self.index = None
        self._is_opened = False

        # Force garbage collection to help with resource cleanup
        import gc
        gc.collect()

        logger.info("Webcam service cleanup completed")

    def is_opened(self) -> bool:
        """Check if webcam is currently opened."""
        return self._is_opened and self.cap is not None

    def get_properties(self) -> dict:
        """Get current webcam properties."""
        if not self.cap:
            return {}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
        }

    @staticmethod
    def list_devices(max_test: int = 5) -> List[Tuple[int, str]]:
        """List available webcam devices."""
        names = []
        
        # Try to get device names on Windows
        if os.name == 'nt':
            queries = [["wmic", "path", "Win32_PnPEntity", "where", "Service='usbvideo'", "get", "Name"]]
            seen = set()
            for cmd in queries:
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
                    if r.returncode != 0:
                        continue
                    lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
                    if lines and lines[0].lower().startswith('name'):
                        lines = lines[1:]
                    for l in lines:
                        if l and l not in seen:
                            names.append(l)
                            seen.add(l)
                except Exception:
                    continue
        
        # Test which device indices are available
        devices = []
        for i in range(max_test):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                label = names[i] if i < len(names) else f"Device {i}"
                devices.append((i, label))
                cap.release()
        
        return devices

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()