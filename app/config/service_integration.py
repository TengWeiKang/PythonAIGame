"""Service integration utilities for real-time settings application.

Provides extension methods and utilities to integrate existing services
with the enhanced settings manager for real-time configuration updates.
"""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..services.webcam_service import WebcamService
    from ..services.gemini_service import AsyncGeminiService
    from .settings import Config

logger = logging.getLogger(__name__)


class WebcamServiceIntegrator:
    """Integration utilities for WebcamService with settings manager."""
    
    @staticmethod
    def apply_settings(service: 'WebcamService', config: Config) -> bool:
        """Apply webcam settings to service instance.
        
        Args:
            service: WebcamService instance
            config: Configuration object with settings
            
        Returns:
            bool: True if settings applied successfully
        """
        try:
            success = True
            
            # Check if camera needs to be changed
            current_index = getattr(service, 'current_camera_index', -1)
            if config.last_webcam_index != current_index:
                if hasattr(service, 'set_camera'):
                    if not service.set_camera(config.last_webcam_index):
                        logger.warning(f"Failed to switch to camera {config.last_webcam_index}")
                        success = False
                else:
                    logger.warning("WebcamService.set_camera method not available")
            
            # Apply resolution settings
            if hasattr(service, 'set_resolution'):
                if not service.set_resolution(config.camera_width, config.camera_height):
                    logger.warning(f"Failed to set resolution to {config.camera_width}x{config.camera_height}")
                    success = False
            else:
                # Try to set properties directly
                if hasattr(service, 'cap') and service.cap is not None:
                    import cv2
                    service.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
                    service.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
            
            # Apply FPS settings
            if hasattr(service, 'set_fps'):
                if not service.set_fps(config.camera_fps):
                    logger.warning(f"Failed to set FPS to {config.camera_fps}")
                    success = False
            else:
                # Try to set property directly
                if hasattr(service, 'cap') and service.cap is not None:
                    import cv2
                    service.cap.set(cv2.CAP_PROP_FPS, config.camera_fps)
            
            # Apply camera controls if supported
            WebcamServiceIntegrator._apply_camera_controls(service, config)
            
            logger.info(f"Webcam settings applied: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to apply webcam settings: {e}")
            return False
    
    @staticmethod
    def _apply_camera_controls(service: 'WebcamService', config: Config) -> None:
        """Apply camera control settings."""
        try:
            if not hasattr(service, 'cap') or service.cap is None:
                return
            
            import cv2
            
            # Apply brightness
            if hasattr(service, 'set_brightness'):
                service.set_brightness(config.camera_brightness)
            else:
                service.cap.set(cv2.CAP_PROP_BRIGHTNESS, config.camera_brightness / 100.0)
            
            # Apply contrast
            if hasattr(service, 'set_contrast'):
                service.set_contrast(config.camera_contrast)
            else:
                service.cap.set(cv2.CAP_PROP_CONTRAST, config.camera_contrast / 100.0)
            
            # Apply saturation
            if hasattr(service, 'set_saturation'):
                service.set_saturation(config.camera_saturation)
            else:
                service.cap.set(cv2.CAP_PROP_SATURATION, config.camera_saturation / 100.0)
            
            # Apply auto settings
            if hasattr(service, 'set_auto_exposure'):
                service.set_auto_exposure(config.camera_auto_exposure)
            else:
                service.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 
                              0.75 if config.camera_auto_exposure else 0.25)
            
            if hasattr(service, 'set_auto_focus'):
                service.set_auto_focus(config.camera_auto_focus)
            else:
                service.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if config.camera_auto_focus else 0)
            
        except Exception as e:
            logger.warning(f"Failed to apply camera controls: {e}")


class GeminiServiceIntegrator:
    """Integration utilities for GeminiService with settings manager."""
    
    @staticmethod
    def apply_settings(service: 'AsyncGeminiService', config: Config) -> bool:
        """Apply Gemini AI settings to service instance.
        
        Args:
            service: AsyncGeminiService instance
            config: Configuration object with settings
            
        Returns:
            bool: True if settings applied successfully
        """
        try:
            success = True
            
            # Update API configuration
            if hasattr(service, 'update_config'):
                try:
                    service.update_config(
                        api_key=config.gemini_api_key,
                        model=config.gemini_model,
                        temperature=config.gemini_temperature,
                        max_tokens=config.gemini_max_tokens,
                        timeout=config.gemini_timeout
                    )
                except Exception as e:
                    logger.warning(f"Failed to update Gemini config via update_config: {e}")
                    success = False
            else:
                # Try to set individual properties
                if hasattr(service, 'api_key'):
                    service.api_key = config.gemini_api_key
                if hasattr(service, 'model'):
                    service.model = config.gemini_model
                if hasattr(service, 'temperature'):
                    service.temperature = config.gemini_temperature
                if hasattr(service, 'max_tokens'):
                    service.max_tokens = config.gemini_max_tokens
                if hasattr(service, 'timeout'):
                    service.timeout = config.gemini_timeout
            
            # Update rate limiting
            if hasattr(service, 'set_rate_limit'):
                try:
                    service.set_rate_limit(
                        enabled=config.enable_rate_limiting,
                        requests_per_minute=config.requests_per_minute
                    )
                except Exception as e:
                    logger.warning(f"Failed to set rate limit: {e}")
                    success = False
            
            # Update context settings
            if hasattr(service, 'set_context_window'):
                try:
                    service.set_context_window(config.context_window_size)
                except Exception as e:
                    logger.warning(f"Failed to set context window: {e}")
                    success = False
            
            # Update conversation memory
            if hasattr(service, 'enable_conversation_memory'):
                service.enable_conversation_memory = config.enable_conversation_memory
            
            logger.info(f"Gemini settings applied: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to apply Gemini settings: {e}")
            return False


class DetectionServiceIntegrator:
    """Integration utilities for DetectionService with settings manager."""
    
    @staticmethod
    def apply_settings(service: Any, config: Config) -> bool:
        """Apply detection/analysis settings to service instance.
        
        Args:
            service: DetectionService instance
            config: Configuration object with settings
            
        Returns:
            bool: True if settings applied successfully
        """
        try:
            success = True
            
            # Update detection thresholds
            if hasattr(service, 'set_confidence_threshold'):
                try:
                    service.set_confidence_threshold(config.detection_confidence_threshold)
                except Exception as e:
                    logger.warning(f"Failed to set confidence threshold: {e}")
                    success = False
            elif hasattr(service, 'confidence_threshold'):
                service.confidence_threshold = config.detection_confidence_threshold
            
            if hasattr(service, 'set_iou_threshold'):
                try:
                    service.set_iou_threshold(config.detection_iou_threshold)
                except Exception as e:
                    logger.warning(f"Failed to set IoU threshold: {e}")
                    success = False
            elif hasattr(service, 'iou_threshold'):
                service.iou_threshold = config.detection_iou_threshold
            
            # Update ROI settings
            if config.enable_roi:
                if hasattr(service, 'set_roi'):
                    try:
                        service.set_roi(config.roi_x, config.roi_y, 
                                       config.roi_width, config.roi_height)
                    except Exception as e:
                        logger.warning(f"Failed to set ROI: {e}")
                        success = False
                elif hasattr(service, 'roi'):
                    service.roi = (config.roi_x, config.roi_y, 
                                  config.roi_width, config.roi_height)
            else:
                if hasattr(service, 'clear_roi'):
                    try:
                        service.clear_roi()
                    except Exception as e:
                        logger.warning(f"Failed to clear ROI: {e}")
                elif hasattr(service, 'roi'):
                    service.roi = None
            
            # Update model preference
            if hasattr(service, 'set_model'):
                try:
                    service.set_model(config.preferred_model)
                except Exception as e:
                    logger.warning(f"Failed to set model: {e}")
                    success = False
            elif hasattr(service, 'model'):
                service.model = config.preferred_model
            
            logger.info(f"Detection settings applied: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to apply detection settings: {e}")
            return False


class MainWindowIntegrator:
    """Integration utilities for MainWindow with settings manager."""
    
    @staticmethod
    def apply_settings(main_window: Any, config: Config) -> bool:
        """Apply UI/theme settings to main window instance.
        
        Args:
            main_window: MainWindow instance
            config: Configuration object with settings
            
        Returns:
            bool: True if settings applied successfully
        """
        try:
            success = True
            
            # Apply theme
            if hasattr(main_window, 'apply_theme'):
                try:
                    main_window.apply_theme(config.app_theme)
                except Exception as e:
                    logger.warning(f"Failed to apply theme: {e}")
                    success = False
            elif hasattr(main_window, 'set_theme'):
                try:
                    main_window.set_theme(config.app_theme)
                except Exception as e:
                    logger.warning(f"Failed to set theme: {e}")
                    success = False
            
            
            # Apply performance settings
            if hasattr(main_window, 'set_performance_mode'):
                try:
                    main_window.set_performance_mode(config.performance_mode)
                except Exception as e:
                    logger.warning(f"Failed to set performance mode: {e}")
                    success = False
            
            logger.info(f"Main window settings applied: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to apply main window settings: {e}")
            return False


# Registry of integrator classes
SERVICE_INTEGRATORS = {
    'webcam': WebcamServiceIntegrator,
    'gemini': GeminiServiceIntegrator, 
    'detection': DetectionServiceIntegrator,
    'main_window': MainWindowIntegrator,
}


def apply_settings_to_service(service_name: str, service: Any, config: Config) -> bool:
    """Apply settings to a service using the appropriate integrator.
    
    Args:
        service_name: Name of the service ('webcam', 'gemini', etc.)
        service: Service instance
        config: Configuration object
        
    Returns:
        bool: True if settings applied successfully
    """
    integrator_class = SERVICE_INTEGRATORS.get(service_name)
    if integrator_class:
        return integrator_class.apply_settings(service, config)
    else:
        logger.warning(f"No integrator available for service: {service_name}")
        return False


__all__ = [
    'WebcamServiceIntegrator',
    'GeminiServiceIntegrator', 
    'DetectionServiceIntegrator',
    'MainWindowIntegrator',
    'SERVICE_INTEGRATORS',
    'apply_settings_to_service'
]