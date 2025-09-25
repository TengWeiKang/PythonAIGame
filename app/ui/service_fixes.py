"""Service access safety fixes for ModernMainWindow.

This module provides patches for safe service access to prevent NoneType errors.
"""

from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


class SafeServiceMixin:
    """Mixin class providing safe service access methods."""

    def _safe_gemini_configured(self) -> bool:
        """Safely check if Gemini service is configured.

        Returns:
            bool: True if service exists and is configured, False otherwise
        """
        try:
            return (
                hasattr(self, 'gemini_service') and
                self.gemini_service is not None and
                hasattr(self.gemini_service, 'is_configured') and
                self.gemini_service.is_configured()
            )
        except Exception as e:
            logger.warning(f"Error checking Gemini configuration: {e}")
            return False

    def _safe_webcam_opened(self) -> bool:
        """Safely check if webcam is opened.

        Returns:
            bool: True if webcam service exists and is opened, False otherwise
        """
        try:
            return (
                hasattr(self, 'webcam_service') and
                self.webcam_service is not None and
                hasattr(self.webcam_service, 'is_opened') and
                self.webcam_service.is_opened()
            )
        except Exception as e:
            logger.warning(f"Error checking webcam status: {e}")
            return False

    def _safe_service_call(self, service_name: str, method_name: str,
                          *args, default=None, **kwargs) -> Any:
        """Safely call a method on a service with fallback.

        Args:
            service_name: Name of the service attribute
            method_name: Name of the method to call
            *args: Positional arguments for the method
            default: Default value to return on failure
            **kwargs: Keyword arguments for the method

        Returns:
            Result of method call or default value
        """
        try:
            service = getattr(self, service_name, None)
            if service is None:
                logger.debug(f"Service {service_name} is not available")
                return default

            method = getattr(service, method_name, None)
            if method is None:
                logger.warning(f"Method {method_name} not found on {service_name}")
                return default

            return method(*args, **kwargs)

        except Exception as e:
            logger.error(f"Error calling {service_name}.{method_name}: {e}")
            return default

    def _ensure_service_available(self, service_name: str) -> bool:
        """Check if a service is available and initialized.

        Args:
            service_name: Name of the service to check

        Returns:
            bool: True if service is available, False otherwise
        """
        return (
            hasattr(self, service_name) and
            getattr(self, service_name) is not None
        )

    def _get_service_or_none(self, service_name: str) -> Optional[Any]:
        """Get a service instance or None if not available.

        Args:
            service_name: Name of the service

        Returns:
            Service instance or None
        """
        if hasattr(self, service_name):
            return getattr(self, service_name)
        return None


def patch_process_chat_message(original_method):
    """Decorator to patch _process_chat_message with safe service access.

    Args:
        original_method: The original method to patch

    Returns:
        Patched method with safe service access
    """
    def patched_method(self, message: str):
        # Safe check for Gemini service configuration
        if not self._safe_gemini_configured():
            self._add_chat_message("System",
                "Gemini API not configured. Please go to Settings > Chatbot to set up your API key.")
            return

        # Rest of the original method logic
        return original_method(self, message)

    return patched_method


def patch_analyze_single_image(original_method):
    """Decorator to patch _analyze_single_image with safe service access.

    Args:
        original_method: The original method to patch

    Returns:
        Patched method with safe service access
    """
    def patched_method(self):
        if not self._safe_gemini_configured():
            self._add_chat_message("System",
                "Gemini API not configured. Please configure the API in settings.")
            return

        if not self._captured_image:
            self._add_chat_message("System",
                "No image captured. Please capture an image first.")
            return

        # Safe async call
        def on_result(success, result, error=None):
            if success:
                self._add_chat_message("Gemini", result)
            else:
                self._add_chat_message("System", f"Analysis failed: {error}")

        self._safe_service_call(
            'gemini_service',
            'analyze_single_image_async',
            self._captured_image,
            on_result
        )

    return patched_method


def patch_compare_images(original_method):
    """Decorator to patch _compare_images with safe service access.

    Args:
        original_method: The original method to patch

    Returns:
        Patched method with safe service access
    """
    def patched_method(self):
        if not self._safe_gemini_configured():
            self._add_chat_message("System",
                "Gemini API not configured. Please configure the API in settings.")
            return

        if not self._reference_image or not self._captured_image:
            self._add_chat_message("System",
                "Both reference and captured images are required for comparison.")
            return

        # Safe async call
        def on_result(success, result, error=None):
            if success:
                self._add_chat_message("Gemini", result)
            else:
                self._add_chat_message("System", f"Comparison failed: {error}")

        self._safe_service_call(
            'gemini_service',
            'compare_images_async',
            self._reference_image,
            self._captured_image,
            on_result
        )

    return patched_method


def safe_cleanup_destructor(self):
    """Safe cleanup for __del__ method.

    Args:
        self: Instance being destroyed
    """
    try:
        # Safe performance monitoring stop
        if hasattr(self, 'stop_performance_monitoring'):
            try:
                self.stop_performance_monitoring()
            except Exception as e:
                logger.debug(f"Error stopping performance monitoring: {e}")

        # Safe streaming stop
        if hasattr(self, '_is_streaming'):
            self._is_streaming = False

        # Safe webcam service close
        webcam = self._get_service_or_none('webcam_service')
        if webcam and hasattr(webcam, 'close'):
            try:
                webcam.close()
            except Exception as e:
                logger.debug(f"Error closing webcam: {e}")

        # Safe Gemini service cleanup
        gemini = self._get_service_or_none('gemini_service')
        if gemini and hasattr(gemini, 'cleanup_threads'):
            try:
                gemini.cleanup_threads()
            except Exception as e:
                logger.debug(f"Error cleaning up Gemini threads: {e}")

        # Safe canvas cleanup
        if hasattr(self, '_video_canvas'):
            canvas = self._video_canvas
            if canvas and hasattr(canvas, 'stop_async_rendering'):
                try:
                    canvas.stop_async_rendering()
                except Exception as e:
                    logger.debug(f"Error stopping canvas rendering: {e}")

        # Safe threading manager shutdown
        threading_mgr = self._get_service_or_none('threading_manager')
        if threading_mgr and hasattr(threading_mgr, 'shutdown_all'):
            try:
                threading_mgr.shutdown_all(timeout=5.0)
            except Exception as e:
                logger.debug(f"Error shutting down threading manager: {e}")

    except Exception as e:
        # Catch-all to prevent exceptions during garbage collection
        logger.debug(f"Unexpected error during cleanup: {e}")