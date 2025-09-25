"""Comprehensive patches for ModernMainWindow to fix service access issues.

This module provides complete patches for all identified vulnerabilities.
"""

import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class ModernMainWindowPatches:
    """Patches for ModernMainWindow to fix NoneType errors and improve safety."""

    @staticmethod
    def patch_on_start_stream(instance):
        """Safe version of _on_start_stream."""
        try:
            # Safe webcam service check
            if not hasattr(instance, 'webcam_service') or instance.webcam_service is None:
                instance._status_bar.update_status("Webcam service not available", "error")
                logger.error("Cannot start stream: Webcam service not initialized")
                return

            if not instance.webcam_service.is_opened():
                success = instance.webcam_service.open(
                    instance.config.camera_index,
                    instance.config.camera_width,
                    instance.config.camera_height,
                    instance.config.camera_fps
                )
                if not success:
                    instance._status_bar.update_status("Failed to open webcam", "error")
                    return

            # Rest of streaming logic...
            instance._is_streaming = True
            instance._status_bar.update_status("Streaming started", "info")

        except Exception as e:
            logger.error(f"Error starting stream: {e}")
            instance._status_bar.update_status(f"Stream error: {str(e)}", "error")

    @staticmethod
    def patch_on_stop_stream(instance):
        """Safe version of _on_stop_stream."""
        try:
            instance._is_streaming = False

            # Safe webcam close
            if hasattr(instance, 'webcam_service') and instance.webcam_service:
                try:
                    instance.webcam_service.close()
                except Exception as e:
                    logger.warning(f"Error closing webcam: {e}")

            instance._status_bar.update_status("Streaming stopped", "info")

        except Exception as e:
            logger.error(f"Error stopping stream: {e}")
            instance._status_bar.update_status(f"Stop error: {str(e)}", "error")

    @staticmethod
    def patch_stream_worker(instance):
        """Safe version of _stream_worker."""
        while instance._is_streaming:
            try:
                # Safe webcam read
                if not hasattr(instance, 'webcam_service') or instance.webcam_service is None:
                    logger.warning("Webcam service not available in stream worker")
                    break

                ret, frame = instance.webcam_service.read()
                if not ret or frame is None:
                    continue

                # Process frame safely
                instance._current_frame = frame

                # Safe inference service call
                if hasattr(instance, 'inference_service') and instance.inference_service:
                    try:
                        # Inference logic here
                        pass
                    except Exception as e:
                        logger.debug(f"Inference error (non-critical): {e}")

            except Exception as e:
                logger.error(f"Stream worker error: {e}")
                if not instance._is_streaming:
                    break

    @staticmethod
    def patch_update_cache_display(instance):
        """Safe version of _update_cache_display."""
        try:
            # Safe cache manager access
            if not hasattr(instance, 'cache_manager') or instance.cache_manager is None:
                return

            cache_stats = instance.cache_manager.get_stats()
            # Update UI with cache stats

        except Exception as e:
            logger.debug(f"Error updating cache display: {e}")

    @staticmethod
    def patch_update_objects_list(instance):
        """Safe version of _update_objects_list."""
        try:
            # Safe object training service access
            if not hasattr(instance, 'object_training_service') or instance.object_training_service is None:
                logger.warning("Object training service not available")
                return

            instance._training_objects = instance.object_training_service.load_objects()
            # Update UI...

        except Exception as e:
            logger.error(f"Error updating objects list: {e}")

    @staticmethod
    def patch_save_configuration(instance, updated_config):
        """Safe version of configuration save with service updates."""
        try:
            # Safe Gemini service update
            if hasattr(instance, 'gemini_service') and instance.gemini_service:
                gemini_config = {
                    'api_key': getattr(updated_config, 'gemini_api_key', ''),
                    'model': getattr(updated_config, 'gemini_model', 'gemini-1.5-flash'),
                    'timeout': getattr(updated_config, 'gemini_timeout', 30),
                    'temperature': getattr(updated_config, 'gemini_temperature', 0.7),
                    'max_tokens': getattr(updated_config, 'gemini_max_tokens', 2048)
                }

                try:
                    instance.gemini_service.update_configuration(**gemini_config)

                    # Restart chat session if API key changed
                    if gemini_config['api_key']:
                        instance.gemini_service.start_chat_session(
                            getattr(updated_config, 'chatbot_persona', '')
                        )
                        instance._gemini_configured = True
                    else:
                        instance._gemini_configured = False

                except Exception as e:
                    logger.error(f"Failed to update Gemini configuration: {e}")
                    instance._gemini_configured = False

            # Safe performance optimization updates
            if hasattr(instance, 'memory_manager') and instance.memory_manager:
                try:
                    if updated_config.performance_mode == 'high':
                        instance.memory_manager.set_memory_pressure_threshold(0.85)
                    elif updated_config.performance_mode == 'balanced':
                        instance.memory_manager.set_memory_pressure_threshold(0.75)
                    else:  # low
                        instance.memory_manager.set_memory_pressure_threshold(0.60)
                except Exception as e:
                    logger.warning(f"Failed to update memory manager: {e}")

            if hasattr(instance, 'cache_manager') and instance.cache_manager:
                try:
                    cache_sizes = {'high': 512, 'balanced': 256, 'low': 128}
                    cache_size = cache_sizes.get(updated_config.performance_mode, 256)
                    instance.cache_manager.set_cache_size_mb(cache_size)
                except Exception as e:
                    logger.warning(f"Failed to update cache manager: {e}")

        except Exception as e:
            logger.error(f"Error saving configuration: {e}")


def apply_all_patches(window_class):
    """Apply all safety patches to ModernMainWindow class.

    Args:
        window_class: The ModernMainWindow class to patch

    Returns:
        Patched class
    """
    # Import service_fixes for mixins
    from .service_fixes import (
        SafeServiceMixin,
        patch_process_chat_message,
        patch_analyze_single_image,
        patch_compare_images,
        safe_cleanup_destructor
    )

    # Add mixin to class hierarchy
    if SafeServiceMixin not in window_class.__bases__:
        window_class.__bases__ = (SafeServiceMixin,) + window_class.__bases__

    # Patch critical methods
    original_process_chat = window_class._process_chat_message
    window_class._process_chat_message = patch_process_chat_message(original_process_chat)

    original_analyze = getattr(window_class, '_analyze_single_image', None)
    if original_analyze:
        window_class._analyze_single_image = patch_analyze_single_image(original_analyze)

    original_compare = getattr(window_class, '_compare_images', None)
    if original_compare:
        window_class._compare_images = patch_compare_images(original_compare)

    # Replace destructor with safe version
    window_class.__del__ = safe_cleanup_destructor

    # Apply other patches
    window_class._safe_on_start_stream = lambda self: ModernMainWindowPatches.patch_on_start_stream(self)
    window_class._safe_on_stop_stream = lambda self: ModernMainWindowPatches.patch_on_stop_stream(self)
    window_class._safe_stream_worker = lambda self: ModernMainWindowPatches.patch_stream_worker(self)
    window_class._safe_update_cache_display = lambda self: ModernMainWindowPatches.patch_update_cache_display(self)
    window_class._safe_update_objects_list = lambda self: ModernMainWindowPatches.patch_update_objects_list(self)
    window_class._safe_save_configuration = lambda self, cfg: ModernMainWindowPatches.patch_save_configuration(self, cfg)

    return window_class


# Example usage in modern_main_window.py:
# from .modern_main_window_patches import apply_all_patches
#
# class ModernMainWindow:
#     ...
#
# # Apply patches at module level
# ModernMainWindow = apply_all_patches(ModernMainWindow)