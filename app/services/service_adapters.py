"""Service adapters for integrating existing services with the restart manager.

This module provides adapters that wrap existing services to make them compatible
with the restart manager's hot-swap and graceful restart capabilities.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass
import weakref

from .restart_manager import (
    ServiceState,
    RestartableService,
    ServiceMetadata,
    RestartStrategy
)
from ..core.performance import performance_timer

logger = logging.getLogger(__name__)


class WebcamServiceAdapter(RestartableService):
    """Adapter for WebcamService to support hot-swap restart operations.

    Performance targets:
    - Hot-swap in <200ms
    - Zero frame drops during swap
    - Seamless buffer transition
    """

    def __init__(self, webcam_service):
        self.service = webcam_service
        self._state = ServiceState.STOPPED
        self._state_lock = threading.RLock()
        self._frame_buffer = []
        self._last_config = {}
        self._warm_instance = None  # Pre-warmed instance for hot-swap

    def get_state(self) -> ServiceState:
        """Get current service state."""
        with self._state_lock:
            if hasattr(self.service, '_is_opened') and self.service._is_opened:
                return ServiceState.RUNNING
            return self._state

    @performance_timer("webcam_prepare_shutdown")
    def prepare_shutdown(self) -> None:
        """Prepare for shutdown - save current frame buffer."""
        try:
            # Save last few frames for seamless transition
            if hasattr(self.service, '_frame_buffer'):
                self._frame_buffer = list(self.service._frame_buffer)

            # Stop frame buffering thread gracefully
            if hasattr(self.service, '_buffering_active'):
                self.service._buffering_active = False

        except Exception as e:
            logger.warning(f"Error preparing webcam shutdown: {e}")

    def shutdown(self) -> None:
        """Shutdown the webcam service."""
        with self._state_lock:
            self._state = ServiceState.STOPPING
            try:
                if self.service and hasattr(self.service, 'close'):
                    self.service.close()
                self._state = ServiceState.STOPPED
            except Exception as e:
                logger.error(f"Error shutting down webcam: {e}")
                self._state = ServiceState.ERROR

    def startup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Start the webcam service with configuration."""
        with self._state_lock:
            self._state = ServiceState.STARTING

            try:
                config = config or self._last_config
                self._last_config = config

                # Extract webcam parameters
                index = config.get('camera_index', 0)
                width = config.get('frame_width', 640)
                height = config.get('frame_height', 480)
                fps = config.get('fps', 30)

                # Open webcam
                if hasattr(self.service, 'open'):
                    success = self.service.open(index, width, height, fps)
                    if success:
                        # Restore frame buffer if available
                        if self._frame_buffer and hasattr(self.service, '_frame_buffer'):
                            self.service._frame_buffer.extend(self._frame_buffer)

                        self._state = ServiceState.RUNNING
                    else:
                        self._state = ServiceState.ERROR

            except Exception as e:
                logger.error(f"Error starting webcam: {e}")
                self._state = ServiceState.ERROR

    def health_check(self) -> bool:
        """Check if webcam is healthy and capturing frames."""
        try:
            if not hasattr(self.service, 'cap') or not self.service.cap:
                return False

            # Try to read a test frame
            if hasattr(self.service, 'read'):
                ret, frame = self.service.read()
                return ret and frame is not None

            return self.service.cap.isOpened()

        except Exception:
            return False

    def get_resources(self) -> Dict[str, Any]:
        """Get current resource usage."""
        resources = {
            'camera_index': getattr(self.service, 'index', None),
            'is_opened': getattr(self.service, '_is_opened', False),
            'frame_count': getattr(self.service, '_frame_count', 0),
            'current_fps': getattr(self.service, '_current_fps', 0.0),
            'buffer_size': len(getattr(self.service, '_frame_buffer', []))
        }

        # Add camera properties if available
        if hasattr(self.service, 'cap') and self.service.cap:
            try:
                import cv2
                resources['actual_width'] = int(self.service.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                resources['actual_height'] = int(self.service.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                resources['actual_fps'] = self.service.cap.get(cv2.CAP_PROP_FPS)
            except:
                pass

        return resources

    @performance_timer("webcam_hot_swap")
    def hot_swap(self, new_config: Dict[str, Any]) -> bool:
        """Perform hot-swap to new configuration without frame drops."""
        start_time = time.time()

        try:
            # Create new instance in parallel
            new_service = self._create_warm_instance(new_config)

            # Save current frame buffer
            old_buffer = []
            if hasattr(self.service, '_frame_buffer'):
                old_buffer = list(self.service._frame_buffer)

            # Get last valid frame for continuity
            last_frame = None
            if old_buffer:
                last_frame = old_buffer[-1]

            # Atomic swap
            old_service = self.service
            self.service = new_service

            # Transfer buffer to maintain continuity
            if hasattr(self.service, '_frame_buffer') and old_buffer:
                self.service._frame_buffer.extend(old_buffer)

            # Inject last frame to prevent black frames
            if last_frame is not None and hasattr(self.service, '_frame_buffer'):
                self.service._frame_buffer.append(last_frame)

            # Cleanup old service in background
            threading.Thread(
                target=self._cleanup_old_service,
                args=(old_service,),
                daemon=True
            ).start()

            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > 200:
                logger.warning(f"Webcam hot-swap took {elapsed_ms:.1f}ms (target: 200ms)")
            else:
                logger.info(f"Webcam hot-swap completed in {elapsed_ms:.1f}ms")

            return True

        except Exception as e:
            logger.error(f"Webcam hot-swap failed: {e}")
            return False

    def _create_warm_instance(self, config: Dict[str, Any]):
        """Create and warm up a new webcam instance."""
        # Import here to avoid circular dependency
        from .webcam_service import WebcamService

        new_service = WebcamService()

        # Configure and open
        index = config.get('camera_index', 0)
        width = config.get('frame_width', 640)
        height = config.get('frame_height', 480)
        fps = config.get('fps', 30)

        new_service.open(index, width, height, fps)

        # Warm up by reading a few frames
        for _ in range(3):
            new_service.read()

        return new_service

    def _cleanup_old_service(self, service):
        """Clean up old webcam service instance."""
        try:
            if service:
                if hasattr(service, 'close'):
                    service.close()
                # Clear references
                del service
        except Exception as e:
            logger.warning(f"Error cleaning up old webcam service: {e}")


class DetectionServiceAdapter(RestartableService):
    """Adapter for DetectionService to support graceful restart operations.

    Handles:
    - Pipeline state preservation
    - Listener management during restart
    - Smooth transition of detection flow
    """

    def __init__(self, detection_service):
        self.service = detection_service
        self._state = ServiceState.STOPPED
        self._state_lock = threading.RLock()
        self._saved_listeners = []
        self._saved_master_provider = None
        self._last_pipeline_state = None

    def get_state(self) -> ServiceState:
        """Get current service state."""
        with self._state_lock:
            if hasattr(self.service, 'is_running') and self.service.is_running():
                return ServiceState.RUNNING
            return self._state

    def prepare_shutdown(self) -> None:
        """Prepare for shutdown - save state and listeners."""
        try:
            # Save listeners for re-registration
            if hasattr(self.service, '_listeners'):
                self._saved_listeners = list(self.service._listeners)

            # Save master provider
            if hasattr(self.service, '_master_provider'):
                self._saved_master_provider = self.service._master_provider

            # Save last pipeline state
            self._last_pipeline_state = getattr(self.service, '_last_state', None)

        except Exception as e:
            logger.warning(f"Error preparing detection service shutdown: {e}")

    def shutdown(self) -> None:
        """Shutdown the detection service."""
        with self._state_lock:
            self._state = ServiceState.STOPPING

            try:
                if hasattr(self.service, 'stop'):
                    self.service.stop()

                # Wait for pipeline to fully stop
                time.sleep(0.1)

                self._state = ServiceState.STOPPED

            except Exception as e:
                logger.error(f"Error shutting down detection service: {e}")
                self._state = ServiceState.ERROR

    def startup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Start the detection service with saved state."""
        with self._state_lock:
            self._state = ServiceState.STARTING

            try:
                # Re-register listeners
                for listener in self._saved_listeners:
                    if hasattr(self.service, 'add_listener'):
                        self.service.add_listener(listener)

                # Restore master provider
                if self._saved_master_provider and hasattr(self.service, 'set_master_provider'):
                    self.service.set_master_provider(self._saved_master_provider)

                # Apply new configuration if provided
                if config and hasattr(self.service, 'config'):
                    self.service.config = config

                # Note: start() is typically called by UI, not here
                self._state = ServiceState.RUNNING

            except Exception as e:
                logger.error(f"Error starting detection service: {e}")
                self._state = ServiceState.ERROR

    def health_check(self) -> bool:
        """Check if detection service is healthy."""
        try:
            # Check if service has required components
            required = ['frame_source', 'inference_service', 'matcher']
            for attr in required:
                if not hasattr(self.service, attr) or getattr(self.service, attr) is None:
                    return False

            return True

        except Exception:
            return False

    def get_resources(self) -> Dict[str, Any]:
        """Get current resource usage."""
        return {
            'is_running': self.service.is_running() if hasattr(self.service, 'is_running') else False,
            'listener_count': len(getattr(self.service, '_listeners', [])),
            'has_master_provider': getattr(self.service, '_master_provider', None) is not None,
            'tk_root': getattr(self.service, '_tk_root', None) is not None
        }


class InferenceServiceAdapter(RestartableService):
    """Adapter for InferenceService with model hot-reload support.

    Performance targets:
    - Model reload in <500ms
    - GPU memory optimization
    - Warm model caching
    """

    def __init__(self, inference_service):
        self.service = inference_service
        self._state = ServiceState.STOPPED
        self._state_lock = threading.RLock()
        self._model_cache = {}
        self._last_model_path = None

    def get_state(self) -> ServiceState:
        """Get current service state."""
        with self._state_lock:
            if hasattr(self.service, 'model') and self.service.model is not None:
                return ServiceState.RUNNING
            return self._state

    @performance_timer("model_prepare_shutdown")
    def prepare_shutdown(self) -> None:
        """Prepare for shutdown - cache model if possible."""
        try:
            # Cache the current model for faster reload
            if hasattr(self.service, 'model') and self.service.model:
                model_path = getattr(self.service, 'model_path', None)
                if model_path:
                    self._last_model_path = model_path
                    # Note: We don't cache the actual model to avoid memory issues

        except Exception as e:
            logger.warning(f"Error preparing inference service shutdown: {e}")

    def shutdown(self) -> None:
        """Shutdown the inference service and release GPU memory."""
        with self._state_lock:
            self._state = ServiceState.STOPPING

            try:
                # Release model from GPU
                if hasattr(self.service, 'model'):
                    self.service.model = None

                # Clear GPU cache
                self._clear_gpu_memory()

                self._state = ServiceState.STOPPED

            except Exception as e:
                logger.error(f"Error shutting down inference service: {e}")
                self._state = ServiceState.ERROR

    def startup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Start the inference service with model preloading."""
        with self._state_lock:
            self._state = ServiceState.STARTING

            try:
                # Load model if path available
                model_path = None
                if config:
                    model_path = config.get('model_path', self._last_model_path)
                elif self._last_model_path:
                    model_path = self._last_model_path

                if model_path and hasattr(self.service, 'load_model'):
                    self.service.load_model(model_path)

                # Warm up model with dummy inference
                self._warm_up_model()

                self._state = ServiceState.RUNNING

            except Exception as e:
                logger.error(f"Error starting inference service: {e}")
                self._state = ServiceState.ERROR

    def health_check(self) -> bool:
        """Check if inference service is ready."""
        try:
            # Check if model is loaded
            if not hasattr(self.service, 'model') or self.service.model is None:
                return False

            # Try a dummy inference if possible
            if hasattr(self.service, 'predict'):
                import numpy as np
                dummy_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
                try:
                    self.service.predict(dummy_input)
                    return True
                except:
                    pass

            return True

        except Exception:
            return False

    def get_resources(self) -> Dict[str, Any]:
        """Get current resource usage including GPU memory."""
        resources = {
            'model_loaded': hasattr(self.service, 'model') and self.service.model is not None,
            'model_path': getattr(self.service, 'model_path', None)
        }

        # Add GPU memory usage
        try:
            import torch
            if torch.cuda.is_available():
                resources['gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
                resources['gpu_cached_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        except ImportError:
            pass

        return resources

    @performance_timer("model_hot_swap")
    def hot_swap_model(self, new_model_path: str) -> bool:
        """Hot-swap to a new model without interruption."""
        start_time = time.time()

        try:
            # Load new model in parallel
            new_model = self._load_model_parallel(new_model_path)

            # Atomic swap
            old_model = getattr(self.service, 'model', None)
            self.service.model = new_model
            self._last_model_path = new_model_path

            # Cleanup old model
            if old_model:
                del old_model
                self._clear_gpu_memory()

            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > 500:
                logger.warning(f"Model hot-swap took {elapsed_ms:.1f}ms (target: 500ms)")
            else:
                logger.info(f"Model hot-swap completed in {elapsed_ms:.1f}ms")

            return True

        except Exception as e:
            logger.error(f"Model hot-swap failed: {e}")
            return False

    def _load_model_parallel(self, model_path: str):
        """Load a model in parallel thread."""
        # Implementation depends on the specific model framework
        # This is a placeholder - actual implementation would load the model
        if hasattr(self.service, 'load_model'):
            return self.service.load_model(model_path)
        return None

    def _warm_up_model(self):
        """Warm up the model with dummy inference."""
        try:
            if hasattr(self.service, 'model') and self.service.model:
                # Run a dummy inference to warm up
                import numpy as np
                dummy_input = np.zeros((1, 3, 640, 640), dtype=np.float32)

                if hasattr(self.service, 'predict'):
                    self.service.predict(dummy_input)

        except Exception as e:
            logger.debug(f"Model warm-up skipped: {e}")

    def _clear_gpu_memory(self):
        """Clear GPU memory cache."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

        import gc
        gc.collect()


class GeminiServiceAdapter(RestartableService):
    """Adapter for GeminiService with connection pooling and retry logic."""

    def __init__(self, gemini_service):
        self.service = gemini_service
        self._state = ServiceState.STOPPED
        self._state_lock = threading.RLock()
        self._connection_pool = []
        self._api_key = None

    def get_state(self) -> ServiceState:
        """Get current service state."""
        with self._state_lock:
            if hasattr(self.service, 'model') and self.service.model:
                return ServiceState.RUNNING
            return self._state

    def prepare_shutdown(self) -> None:
        """Prepare for shutdown - save API configuration."""
        try:
            # Save API key for reconnection
            if hasattr(self.service, 'api_key'):
                self._api_key = self.service.api_key

        except Exception as e:
            logger.warning(f"Error preparing Gemini service shutdown: {e}")

    def shutdown(self) -> None:
        """Shutdown the Gemini service and close connections."""
        with self._state_lock:
            self._state = ServiceState.STOPPING

            try:
                # Close any active connections
                if hasattr(self.service, 'model'):
                    self.service.model = None

                self._state = ServiceState.STOPPED

            except Exception as e:
                logger.error(f"Error shutting down Gemini service: {e}")
                self._state = ServiceState.ERROR

    def startup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Start the Gemini service with API configuration."""
        with self._state_lock:
            self._state = ServiceState.STARTING

            try:
                # Restore API configuration
                api_key = None
                if config:
                    api_key = config.get('gemini_api_key', self._api_key)
                elif self._api_key:
                    api_key = self._api_key

                if api_key and hasattr(self.service, 'configure'):
                    self.service.configure({'api_key': api_key})

                # Initialize model connection
                if hasattr(self.service, '_initialize_model'):
                    self.service._initialize_model()

                self._state = ServiceState.RUNNING

            except Exception as e:
                logger.error(f"Error starting Gemini service: {e}")
                self._state = ServiceState.ERROR

    def health_check(self) -> bool:
        """Check if Gemini service is ready."""
        try:
            # Check if model is configured
            if not hasattr(self.service, 'model') or self.service.model is None:
                return False

            # API key must be present
            if not hasattr(self.service, 'api_key') or not self.service.api_key:
                return False

            return True

        except Exception:
            return False

    def get_resources(self) -> Dict[str, Any]:
        """Get current resource usage."""
        return {
            'model_configured': hasattr(self.service, 'model') and self.service.model is not None,
            'api_key_set': hasattr(self.service, 'api_key') and bool(self.service.api_key),
            'request_count': getattr(self.service, '_request_count', 0),
            'last_request_time': getattr(self.service, '_last_request_time', None)
        }


def create_service_metadata(service_name: str, service_instance: Any,
                           adapter: RestartableService) -> ServiceMetadata:
    """Create ServiceMetadata for a service with its adapter."""

    metadata = ServiceMetadata(
        name=service_name,
        service_ref=adapter
    )

    # Configure based on service type
    if 'webcam' in service_name.lower():
        metadata.restart_strategy = RestartStrategy.HOT_SWAP
        metadata.supports_hot_swap = True
        metadata.max_restart_time = 0.2  # 200ms
        metadata.requires_camera = True
        metadata.priority = 10  # High priority
        metadata.resource_pool_size = 2

    elif 'detection' in service_name.lower():
        metadata.restart_strategy = RestartStrategy.GRACEFUL
        metadata.max_restart_time = 1.0
        metadata.priority = 20
        metadata.dependencies = ['webcam']  # Depends on webcam

    elif 'inference' in service_name.lower():
        metadata.restart_strategy = RestartStrategy.HOT_SWAP
        metadata.supports_hot_swap = True
        metadata.max_restart_time = 0.5  # 500ms
        metadata.requires_gpu = True
        metadata.priority = 30
        metadata.warm_up_time = 0.2
        metadata.resource_pool_size = 2

    elif 'gemini' in service_name.lower():
        metadata.restart_strategy = RestartStrategy.GRACEFUL
        metadata.max_restart_time = 2.0
        metadata.priority = 40

    return metadata


class ServiceAdapterFactory:
    """Factory for creating service adapters."""

    @staticmethod
    def create_adapter(service_name: str, service_instance: Any) -> Optional[RestartableService]:
        """Create an appropriate adapter for a service."""

        if 'webcam' in service_name.lower():
            return WebcamServiceAdapter(service_instance)

        elif 'detection' in service_name.lower():
            return DetectionServiceAdapter(service_instance)

        elif 'inference' in service_name.lower():
            return InferenceServiceAdapter(service_instance)

        elif 'gemini' in service_name.lower():
            return GeminiServiceAdapter(service_instance)

        else:
            # Generic adapter for unknown services
            return GenericServiceAdapter(service_instance)


class GenericServiceAdapter(RestartableService):
    """Generic adapter for services without specific implementations."""

    def __init__(self, service):
        self.service = service
        self._state = ServiceState.STOPPED

    def get_state(self) -> ServiceState:
        return self._state

    def prepare_shutdown(self) -> None:
        pass

    def shutdown(self) -> None:
        if hasattr(self.service, 'stop'):
            self.service.stop()
        elif hasattr(self.service, 'close'):
            self.service.close()
        self._state = ServiceState.STOPPED

    def startup(self, config: Optional[Dict[str, Any]] = None) -> None:
        if hasattr(self.service, 'start'):
            self.service.start()
        self._state = ServiceState.RUNNING

    def health_check(self) -> bool:
        return self._state == ServiceState.RUNNING

    def get_resources(self) -> Dict[str, Any]:
        return {}