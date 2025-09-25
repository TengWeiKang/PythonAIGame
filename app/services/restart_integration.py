"""Integration module for adding restart capabilities to existing services.

This module provides adapters and wrappers to make existing services compatible
with the restart manager, including hot-swap support for critical services.
"""

from __future__ import annotations

import asyncio
import threading
import time
import logging
from typing import Any, Dict, Optional, Type
from abc import ABC, abstractmethod
import weakref
import cv2

from .restart_manager import (
    RestartableService, ServiceState, ServiceMetadata, RestartStrategy,
    get_restart_manager, ServiceHealthCheck
)
from .webcam_service import WebcamService
from .gemini_service import GeminiService
from .detection_service import DetectionService
from .inference_service import InferenceService

from ..core.performance import performance_timer
from ..core.exceptions import WebcamError, ConfigurationError

logger = logging.getLogger(__name__)


class RestartableServiceAdapter(ABC):
    """Base adapter for making services restartable."""

    def __init__(self, service: Any):
        self.service = service
        self._state = ServiceState.STOPPED
        self._config = {}
        self._lock = threading.RLock()
        self._resource_snapshot = {}

    def get_state(self) -> ServiceState:
        """Get current service state."""
        return self._state

    def prepare_shutdown(self) -> None:
        """Prepare for shutdown by saving state."""
        with self._lock:
            self._state = ServiceState.STOPPING
            self._save_state()

    @abstractmethod
    def _save_state(self) -> None:
        """Save service-specific state."""
        pass

    def shutdown(self) -> None:
        """Shutdown the service."""
        with self._lock:
            self._state = ServiceState.STOPPING
            try:
                self._perform_shutdown()
                self._state = ServiceState.STOPPED
            except Exception as e:
                logger.error(f"Shutdown failed: {e}")
                self._state = ServiceState.ERROR

    @abstractmethod
    def _perform_shutdown(self) -> None:
        """Perform service-specific shutdown."""
        pass

    def startup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Start the service with configuration."""
        with self._lock:
            self._state = ServiceState.STARTING
            try:
                if config:
                    self._config = config
                self._perform_startup()
                self._state = ServiceState.RUNNING
            except Exception as e:
                logger.error(f"Startup failed: {e}")
                self._state = ServiceState.ERROR
                raise

    @abstractmethod
    def _perform_startup(self) -> None:
        """Perform service-specific startup."""
        pass

    def health_check(self) -> bool:
        """Check if service is healthy."""
        return self._state == ServiceState.RUNNING and self._is_healthy()

    @abstractmethod
    def _is_healthy(self) -> bool:
        """Service-specific health check."""
        pass

    def get_resources(self) -> Dict[str, Any]:
        """Get current resource usage."""
        return {
            'state': self._state.value,
            'config': self._config,
            **self._get_service_resources()
        }

    @abstractmethod
    def _get_service_resources(self) -> Dict[str, Any]:
        """Get service-specific resources."""
        pass


class RestartableWebcamService(RestartableServiceAdapter):
    """Webcam service with hot-swap support for zero-downtime restarts."""

    def __init__(self, service: Optional[WebcamService] = None):
        if service is None:
            service = WebcamService()
        super().__init__(service)
        self._frame_buffer = []
        self._swap_thread = None
        self._new_service = None

    def _save_state(self) -> None:
        """Save current webcam state including frame buffer."""
        if hasattr(self.service, 'cap') and self.service.cap:
            self._resource_snapshot = {
                'index': self.service.index,
                'width': self.service.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                'height': self.service.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                'fps': self.service.cap.get(cv2.CAP_PROP_FPS),
                'is_opened': self.service._is_opened
            }

            # Buffer last few frames for continuity
            for _ in range(3):
                ret, frame = self.service.read()
                if ret and frame is not None:
                    self._frame_buffer.append(frame.copy())

    def _perform_shutdown(self) -> None:
        """Gracefully shutdown webcam."""
        if self.service:
            self.service.close()

    def _perform_startup(self) -> None:
        """Start webcam with saved or new configuration."""
        if self._resource_snapshot:
            # Restore previous state
            self.service.open(
                index=int(self._resource_snapshot.get('index', 0)),
                width=int(self._resource_snapshot.get('width', 640)),
                height=int(self._resource_snapshot.get('height', 480)),
                fps=int(self._resource_snapshot.get('fps', 30))
            )
        elif self._config:
            # Use new configuration
            self.service.open(
                index=self._config.get('camera_index', 0),
                width=self._config.get('width', 640),
                height=self._config.get('height', 480),
                fps=self._config.get('fps', 30)
            )

    def _is_healthy(self) -> bool:
        """Check if webcam is capturing frames."""
        if not self.service or not self.service._is_opened:
            return False

        # Try to read a frame
        ret, _ = self.service.read()
        return ret

    def _get_service_resources(self) -> Dict[str, Any]:
        """Get webcam resource information."""
        resources = {
            'camera_index': getattr(self.service, 'index', None),
            'is_opened': getattr(self.service, '_is_opened', False),
            'buffer_size': len(self._frame_buffer)
        }

        if hasattr(self.service, '_current_fps'):
            resources['current_fps'] = self.service._current_fps

        return resources

    @performance_timer("webcam_hot_swap")
    def hot_swap(self, new_config: Dict[str, Any]) -> bool:
        """Perform hot-swap with zero frame drops."""
        try:
            # Create new service instance
            self._new_service = WebcamService()

            # Start new camera in parallel
            self._new_service.open(
                index=new_config.get('camera_index', 0),
                width=new_config.get('width', 640),
                height=new_config.get('height', 480),
                fps=new_config.get('fps', 30)
            )

            # Verify new service is working
            ret, test_frame = self._new_service.read()
            if not ret or test_frame is None:
                self._new_service.close()
                return False

            # Atomic swap
            with self._lock:
                old_service = self.service
                self.service = self._new_service
                self._new_service = None

            # Clean up old service in background
            threading.Thread(
                target=lambda: old_service.close(),
                daemon=True
            ).start()

            return True

        except Exception as e:
            logger.error(f"Hot-swap failed for webcam: {e}")
            if self._new_service:
                self._new_service.close()
            return False


class RestartableGeminiService(RestartableServiceAdapter):
    """Gemini service with connection pooling for fast restarts."""

    def __init__(self, service: Optional[GeminiService] = None):
        if service is None:
            service = GeminiService()
        super().__init__(service)
        self._session_pool = []
        self._active_requests = 0

    def _save_state(self) -> None:
        """Save Gemini service state."""
        self._resource_snapshot = {
            'api_key': self.service.api_key,
            'model_name': self.service.model_name,
            'temperature': self.service.temperature,
            'max_tokens': self.service.max_tokens,
            'persona': self.service.persona,
            'timeout': self.service.timeout
        }

    def _perform_shutdown(self) -> None:
        """Shutdown Gemini service and clear sessions."""
        # Wait for active requests to complete (with timeout)
        start_time = time.time()
        while self._active_requests > 0 and time.time() - start_time < 5.0:
            time.sleep(0.1)

        # Clear session pool
        for session in self._session_pool:
            try:
                if hasattr(session, 'close'):
                    session.close()
            except Exception:
                pass
        self._session_pool.clear()

        # Clear service state
        if self.service:
            self.service.model = None
            self.service.chat_session = None

    def _perform_startup(self) -> None:
        """Start Gemini service with configuration."""
        config = self._resource_snapshot if self._resource_snapshot else self._config

        if config and 'api_key' in config:
            # Reinitialize service
            self.service.api_key = config['api_key']
            self.service.model_name = config.get('model_name', 'gemini-1.5-flash')
            self.service.temperature = config.get('temperature', 0.7)
            self.service.max_tokens = config.get('max_tokens', 2048)
            self.service.timeout = config.get('timeout', 30)

            # Initialize model if API key is present
            if self.service.api_key:
                self.service._init_model()

    def _is_healthy(self) -> bool:
        """Check if Gemini service is initialized and ready."""
        return (
            self.service is not None and
            self.service.api_key is not None and
            (self.service.model is not None or not self.service.api_key)
        )

    def _get_service_resources(self) -> Dict[str, Any]:
        """Get Gemini service resources."""
        return {
            'model_name': getattr(self.service, 'model_name', None),
            'has_api_key': bool(getattr(self.service, 'api_key', None)),
            'active_requests': self._active_requests,
            'session_pool_size': len(self._session_pool)
        }

    def track_request(self) -> None:
        """Track active API request."""
        self._active_requests += 1

    def release_request(self) -> None:
        """Release tracked API request."""
        self._active_requests = max(0, self._active_requests - 1)


class RestartableDetectionService(RestartableServiceAdapter):
    """Detection service with model preloading for fast restarts."""

    def __init__(self, service: Optional[DetectionService] = None):
        if service is None:
            service = DetectionService()
        super().__init__(service)
        self._model_cache = {}
        self._warm_model = None

    def _save_state(self) -> None:
        """Save detection service state."""
        if hasattr(self.service, 'backend') and self.service.backend:
            self._resource_snapshot = {
                'model_path': getattr(self.service.backend, 'model_path', None),
                'device': getattr(self.service.backend, 'device', 'cpu'),
                'confidence_threshold': getattr(self.service, 'confidence_threshold', 0.5)
            }

    def _perform_shutdown(self) -> None:
        """Shutdown detection service."""
        if hasattr(self.service, 'backend') and self.service.backend:
            # Release model resources
            if hasattr(self.service.backend, 'model'):
                self.service.backend.model = None

            # Clear CUDA cache if using GPU
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def _perform_startup(self) -> None:
        """Start detection service with model."""
        config = self._resource_snapshot if self._resource_snapshot else self._config

        if config and 'model_path' in config:
            # Use warm model if available
            if self._warm_model and self._warm_model.get('path') == config['model_path']:
                if hasattr(self.service, 'backend'):
                    self.service.backend.model = self._warm_model['model']
            else:
                # Load model normally
                if hasattr(self.service, 'load_model'):
                    self.service.load_model(
                        config['model_path'],
                        device=config.get('device', 'cpu')
                    )

    def _is_healthy(self) -> bool:
        """Check if detection model is loaded."""
        return (
            hasattr(self.service, 'backend') and
            self.service.backend is not None and
            hasattr(self.service.backend, 'model') and
            self.service.backend.model is not None
        )

    def _get_service_resources(self) -> Dict[str, Any]:
        """Get detection service resources."""
        resources = {
            'has_model': self._is_healthy(),
            'model_cache_size': len(self._model_cache)
        }

        if hasattr(self.service, 'backend'):
            backend = self.service.backend
            if hasattr(backend, 'device'):
                resources['device'] = backend.device
            if hasattr(backend, 'model_path'):
                resources['model_path'] = backend.model_path

        return resources

    def preload_model(self, model_path: str, device: str = 'cpu') -> None:
        """Preload a model for fast switching."""
        try:
            # Load model in background
            from ..backends.yolo_backend import YOLOBackend
            backend = YOLOBackend()
            backend.load_model(model_path, device=device)

            self._warm_model = {
                'path': model_path,
                'model': backend.model,
                'device': device
            }
        except Exception as e:
            logger.warning(f"Failed to preload model: {e}")


class ServiceRestartIntegration:
    """Integration point for registering all services with restart manager."""

    @staticmethod
    def register_all_services(
        webcam: Optional[WebcamService] = None,
        gemini: Optional[GeminiService] = None,
        detection: Optional[DetectionService] = None,
        inference: Optional[InferenceService] = None
    ) -> None:
        """Register all services with the restart manager."""
        manager = get_restart_manager()

        # Register webcam service with hot-swap support
        if webcam:
            adapter = RestartableWebcamService(webcam)
            metadata = ServiceMetadata(
                name="webcam",
                service_ref=adapter,
                dependencies=[],
                dependents=["detection", "inference"],
                restart_strategy=RestartStrategy.HOT_SWAP,
                priority=10,
                max_restart_time=0.2,
                requires_camera=True,
                supports_hot_swap=True,
                warm_up_time=0.1,
                resource_pool_size=2
            )
            manager.register_service(metadata)

        # Register Gemini service
        if gemini:
            adapter = RestartableGeminiService(gemini)
            metadata = ServiceMetadata(
                name="gemini",
                service_ref=adapter,
                dependencies=[],
                dependents=[],
                restart_strategy=RestartStrategy.GRACEFUL,
                priority=30,
                max_restart_time=2.0,
                supports_hot_swap=False,
                warm_up_time=0.5
            )
            manager.register_service(metadata)

        # Register detection service
        if detection:
            adapter = RestartableDetectionService(detection)
            metadata = ServiceMetadata(
                name="detection",
                service_ref=adapter,
                dependencies=["webcam"],
                dependents=["inference"],
                restart_strategy=RestartStrategy.GRACEFUL,
                priority=20,
                max_restart_time=1.0,
                requires_gpu=True,
                supports_hot_swap=False,
                warm_up_time=0.5
            )
            manager.register_service(metadata)

        # Register inference service
        if inference:
            # Use basic adapter for inference
            adapter = RestartableServiceAdapter(inference)
            metadata = ServiceMetadata(
                name="inference",
                service_ref=adapter,
                dependencies=["webcam", "detection"],
                dependents=[],
                restart_strategy=RestartStrategy.GRACEFUL,
                priority=25,
                max_restart_time=0.5,
                requires_gpu=True,
                supports_hot_swap=False
            )
            manager.register_service(metadata)

        logger.info("All services registered with restart manager")

    @staticmethod
    def create_restart_callback(ui_callback: Optional[callable] = None) -> callable:
        """Create a progress callback for UI updates."""
        def callback(event: str, service: Optional[str], **kwargs):
            message = ""
            if event == "stopping":
                message = f"Stopping {service}..."
            elif event == "starting":
                message = f"Starting {service}..."
            elif event == "ready":
                message = f"{service} is ready"
            elif event == "complete":
                success = kwargs.get('success', False)
                message = "Restart complete" if success else "Restart failed"

            if ui_callback:
                ui_callback(message)
            else:
                logger.info(message)

        return callback