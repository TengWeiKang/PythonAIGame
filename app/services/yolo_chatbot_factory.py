"""
Factory for creating and configuring YOLO + Chatbot integration services.

This factory provides easy setup and configuration of the integrated
YOLO object detection and AI chatbot analysis system.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from ..backends.yolo_backend import YoloBackend
from .gemini_service import AsyncGeminiService
from .yolo_comparison_service import YoloComparisonService
from .integrated_analysis_service import IntegratedAnalysisService
from ..core.exceptions import WebcamError, DetectionError

logger = logging.getLogger(__name__)


class YoloChatbotFactory:
    """Factory for creating integrated YOLO + Chatbot services."""

    @staticmethod
    def create_integrated_service(config: Dict[str, Any],
                                 model_path: Optional[str] = None,
                                 api_key: Optional[str] = None) -> Tuple[IntegratedAnalysisService, Dict[str, Any]]:
        """
        Create a fully configured integrated analysis service.

        Args:
            config: Application configuration dictionary
            model_path: Optional path to YOLO model (uses config default if None)
            api_key: Optional Gemini API key (uses config default if None)

        Returns:
            Tuple of (IntegratedAnalysisService, status_info)

        Raises:
            WebcamError: If service creation fails
        """
        status = {
            'yolo_backend': False,
            'gemini_service': False,
            'integrated_service': False,
            'ready_for_use': False,
            'errors': [],
            'warnings': []
        }

        try:
            logger.info("Creating integrated YOLO + Chatbot service...")

            # Step 1: Create YOLO backend
            try:
                yolo_backend = YoloChatbotFactory._create_yolo_backend(config, model_path)
                status['yolo_backend'] = True
                logger.info("YOLO backend created successfully")
            except Exception as e:
                error_msg = f"YOLO backend creation failed: {e}"
                status['errors'].append(error_msg)
                logger.error(error_msg)
                raise WebcamError(error_msg)

            # Step 2: Create Gemini service
            try:
                gemini_service = YoloChatbotFactory._create_gemini_service(config, api_key)
                status['gemini_service'] = True
                logger.info("Gemini service created successfully")

                if not gemini_service.is_configured():
                    warning_msg = "Gemini service created but not properly configured (check API key)"
                    status['warnings'].append(warning_msg)
                    logger.warning(warning_msg)

            except Exception as e:
                error_msg = f"Gemini service creation failed: {e}"
                status['errors'].append(error_msg)
                logger.error(error_msg)
                raise WebcamError(error_msg)

            # Step 3: Create integrated service
            try:
                integrated_service = IntegratedAnalysisService(
                    yolo_backend=yolo_backend,
                    gemini_service=gemini_service,
                    config=config
                )
                status['integrated_service'] = True
                logger.info("Integrated analysis service created successfully")

                # Configure persona if provided
                persona = config.get('chatbot_persona', '')
                if persona and gemini_service.is_configured():
                    try:
                        gemini_service.start_chat_session(persona)
                        logger.info("Chatbot persona configured")
                    except Exception as e:
                        warning_msg = f"Failed to set chatbot persona: {e}"
                        status['warnings'].append(warning_msg)
                        logger.warning(warning_msg)

                status['ready_for_use'] = True
                logger.info("Integrated service ready for use")

                return integrated_service, status

            except Exception as e:
                error_msg = f"Integrated service creation failed: {e}"
                status['errors'].append(error_msg)
                logger.error(error_msg)
                raise WebcamError(error_msg)

        except Exception as e:
            logger.error(f"Factory creation failed: {e}")
            raise

    @staticmethod
    def _create_yolo_backend(config: Dict[str, Any], model_path: Optional[str] = None) -> YoloBackend:
        """Create and configure YOLO backend."""
        try:
            # Create backend
            yolo_backend = YoloBackend(config)

            # Determine model to load
            if model_path:
                target_model = model_path
            else:
                target_model = config.get('preferred_model', 'yolo11n.pt')

            # Try to load the model
            success = yolo_backend.load_model(target_model)
            if not success:
                raise DetectionError(f"Failed to load YOLO model: {target_model}")

            # Verify model is working
            model_info = yolo_backend.get_model_info()
            logger.info(f"YOLO model loaded: {model_info}")

            return yolo_backend

        except Exception as e:
            logger.error(f"YOLO backend creation failed: {e}")
            raise

    @staticmethod
    def _create_gemini_service(config: Dict[str, Any], api_key: Optional[str] = None) -> AsyncGeminiService:
        """Create and configure Gemini service."""
        try:
            # Use provided API key or get from config
            effective_api_key = api_key or config.get('gemini_api_key', '')

            # Create service with configuration
            gemini_service = AsyncGeminiService(
                api_key=effective_api_key,
                model=config.get('gemini_model', 'gemini-1.5-flash'),
                timeout=config.get('gemini_timeout', 30),
                temperature=config.get('gemini_temperature', 0.7),
                max_tokens=config.get('gemini_max_tokens', 2048)
            )

            # Configure rate limiting if specified
            if config.get('enable_rate_limiting', True):
                gemini_service.set_rate_limit(
                    enabled=True,
                    requests_per_minute=config.get('requests_per_minute', 15)
                )

            return gemini_service

        except Exception as e:
            logger.error(f"Gemini service creation failed: {e}")
            raise

    @staticmethod
    def create_yolo_comparison_only(config: Dict[str, Any],
                                   model_path: Optional[str] = None) -> Tuple[YoloComparisonService, Dict[str, Any]]:
        """
        Create only the YOLO comparison service (no AI chatbot).

        Args:
            config: Application configuration dictionary
            model_path: Optional path to YOLO model

        Returns:
            Tuple of (YoloComparisonService, status_info)
        """
        status = {
            'yolo_backend': False,
            'comparison_service': False,
            'ready_for_use': False,
            'errors': [],
            'warnings': []
        }

        try:
            logger.info("Creating YOLO comparison service (without AI chatbot)...")

            # Create YOLO backend
            yolo_backend = YoloChatbotFactory._create_yolo_backend(config, model_path)
            status['yolo_backend'] = True

            # Create dummy Gemini service (for interface compatibility)
            dummy_gemini = AsyncGeminiService(api_key=None)

            # Create comparison service
            comparison_service = YoloComparisonService(
                yolo_backend=yolo_backend,
                gemini_service=dummy_gemini,  # Won't be used for AI responses
                config=config
            )
            status['comparison_service'] = True
            status['ready_for_use'] = True

            status['warnings'].append("AI chatbot responses not available - using YOLO comparison only")
            logger.info("YOLO comparison service created successfully (no AI responses)")

            return comparison_service, status

        except Exception as e:
            error_msg = f"YOLO comparison service creation failed: {e}"
            status['errors'].append(error_msg)
            logger.error(error_msg)
            raise WebcamError(error_msg)

    @staticmethod
    def validate_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration for YOLO + Chatbot integration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Dict with validation results
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'missing_required': [],
            'recommendations': []
        }

        # Check required YOLO settings
        required_yolo_settings = [
            'detection_confidence_threshold',
            'detection_iou_threshold',
            'preferred_model'
        ]

        for setting in required_yolo_settings:
            if setting not in config:
                validation_result['missing_required'].append(setting)
                validation_result['valid'] = False

        # Check Gemini API settings
        if not config.get('gemini_api_key'):
            validation_result['warnings'].append("No Gemini API key configured - AI responses will be unavailable")

        gemini_settings = {
            'gemini_model': 'gemini-1.5-flash',
            'gemini_timeout': 30,
            'gemini_temperature': 0.7,
            'gemini_max_tokens': 2048
        }

        for setting, default_value in gemini_settings.items():
            if setting not in config:
                validation_result['recommendations'].append(f"Consider setting {setting} (default: {default_value})")

        # Check confidence thresholds
        confidence = config.get('detection_confidence_threshold', 0.5)
        if not 0.1 <= confidence <= 1.0:
            validation_result['errors'].append("detection_confidence_threshold must be between 0.1 and 1.0")
            validation_result['valid'] = False

        iou = config.get('detection_iou_threshold', 0.45)
        if not 0.1 <= iou <= 1.0:
            validation_result['errors'].append("detection_iou_threshold must be between 0.1 and 1.0")
            validation_result['valid'] = False

        # Check model path
        preferred_model = config.get('preferred_model', '')
        if preferred_model:
            # Check if it's a file path
            if '/' in preferred_model or '\\' in preferred_model:
                model_path = Path(preferred_model)
                if not model_path.exists():
                    validation_result['warnings'].append(f"Preferred model file not found: {preferred_model}")

        # Check data directories
        data_dirs = ['models_dir', 'data_dir', 'results_export_dir']
        for dir_setting in data_dirs:
            if dir_setting in config:
                dir_path = Path(config[dir_setting])
                if not dir_path.exists():
                    validation_result['recommendations'].append(f"Directory does not exist: {dir_path}")

        return validation_result

    @staticmethod
    def get_recommended_config() -> Dict[str, Any]:
        """Get recommended configuration for YOLO + Chatbot integration."""
        return {
            # YOLO Detection Settings
            'detection_confidence_threshold': 0.5,
            'detection_iou_threshold': 0.45,
            'preferred_model': 'yolo11n.pt',  # Fast, lightweight model for real-time use
            'use_gpu': True,

            # Gemini AI Settings
            'gemini_model': 'gemini-1.5-flash',  # Fast model for real-time responses
            'gemini_timeout': 30,
            'gemini_temperature': 0.7,
            'gemini_max_tokens': 2048,
            'enable_rate_limiting': True,
            'requests_per_minute': 15,

            # Integration Settings
            'enable_image_comparison': True,
            'enable_scene_analysis': True,
            'response_format': 'Educational',  # 'Educational', 'Technical', 'Detailed'
            'chatbot_persona': (
                "You are a helpful AI assistant for image analysis and object detection. "
                "Your role is to help users understand computer vision results, compare images, "
                "and provide educational feedback about what the AI can detect and analyze."
            ),

            # Performance Settings
            'master_tolerance_px': 40,
            'target_fps': 30,
            'enable_logging': True,
            'log_level': 'INFO',

            # Data Directories
            'data_dir': 'data',
            'models_dir': 'data/models',
            'results_export_dir': 'data/results'
        }

    @staticmethod
    def create_from_file(config_file_path: str,
                        model_path: Optional[str] = None,
                        api_key: Optional[str] = None) -> Tuple[IntegratedAnalysisService, Dict[str, Any]]:
        """
        Create integrated service from configuration file.

        Args:
            config_file_path: Path to JSON configuration file
            model_path: Optional override for model path
            api_key: Optional override for API key

        Returns:
            Tuple of (IntegratedAnalysisService, status_info)
        """
        try:
            import json
            config_path = Path(config_file_path)

            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_file_path}")

            with open(config_path, 'r') as f:
                config = json.load(f)

            logger.info(f"Configuration loaded from: {config_file_path}")

            return YoloChatbotFactory.create_integrated_service(config, model_path, api_key)

        except Exception as e:
            logger.error(f"Failed to create service from file: {e}")
            raise WebcamError(f"Configuration file error: {e}")