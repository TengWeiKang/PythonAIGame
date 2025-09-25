"""
Integrated Analysis Service for YOLO + Chatbot Integration.

This service coordinates between YOLO object detection, image comparison,
and AI chatbot analysis to provide comprehensive educational feedback.
"""

import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
import numpy as np

from ..core.entities import Detection, ChatbotContext, ComparisonMetrics
from ..core.exceptions import DetectionError, WebcamError
from ..core.performance import performance_timer
from ..backends.yolo_backend import YoloBackend
from .gemini_service import AsyncGeminiService
from .yolo_comparison_service import YoloComparisonService, YoloComparisonResult
from .image_analysis_service import ImageAnalysisService, ImageAnalysisResult

logger = logging.getLogger(__name__)

@dataclass
class IntegratedAnalysisResult:
    """Complete integrated analysis result combining YOLO detection and AI chatbot response."""
    timestamp: str
    user_message: str
    yolo_comparison: Optional[YoloComparisonResult]
    image_analysis: Optional[ImageAnalysisResult]
    chatbot_response: str
    analysis_duration_ms: float
    success: bool
    error_message: Optional[str] = None


class IntegratedAnalysisService:
    """
    Service that integrates YOLO object detection with AI chatbot analysis.

    This service coordinates between:
    1. YOLO object detection and comparison
    2. Scene analysis and description
    3. AI chatbot response generation
    4. Educational feedback synthesis
    """

    def __init__(self,
                 yolo_backend: YoloBackend,
                 gemini_service: AsyncGeminiService,
                 config: Dict[str, Any]):
        """
        Initialize the integrated analysis service.

        Args:
            yolo_backend: Configured YOLO backend for object detection
            gemini_service: Gemini service for AI analysis
            config: Service configuration dictionary
        """
        self.yolo_backend = yolo_backend
        self.gemini_service = gemini_service
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize sub-services
        self.yolo_comparison_service = YoloComparisonService(
            yolo_backend=yolo_backend,
            gemini_service=gemini_service,
            config=config
        )

        # Create inference service for image analysis (if needed)
        self.inference_service = None
        self._setup_inference_service()

        self.image_analysis_service = ImageAnalysisService(
            inference_service=self.inference_service,
            config=config
        )

        # Configuration
        self.enable_comparison = config.get('enable_image_comparison', True)
        self.enable_scene_analysis = config.get('enable_scene_analysis', True)
        self.chatbot_persona = config.get('chatbot_persona', '')
        self.response_format = config.get('response_format', 'Detailed')

        # Performance tracking
        self._stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'average_response_time_ms': 0.0,
            'last_analysis_time': None
        }

        # Callback for UI updates
        self._progress_callback: Optional[Callable[[str], None]] = None

    def _setup_inference_service(self) -> None:
        """Setup inference service for image analysis if not provided."""
        try:
            # Create a simple inference wrapper around YOLO backend
            from .inference_service import InferenceService

            class YoloInferenceAdapter:
                """Adapter to make YOLO backend compatible with InferenceService interface."""
                def __init__(self, yolo_backend: YoloBackend):
                    self.yolo_backend = yolo_backend

                @property
                def is_loaded(self) -> bool:
                    return self.yolo_backend.is_loaded

                def predict(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Detection]:
                    try:
                        return self.yolo_backend.predict(
                            image,
                            conf=confidence_threshold,
                            verbose=False
                        )
                    except Exception:
                        return []

            self.inference_service = YoloInferenceAdapter(self.yolo_backend)

        except Exception as e:
            self.logger.warning(f"Could not setup inference service: {e}")
            self.inference_service = None

    def set_progress_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for progress updates during analysis."""
        self._progress_callback = callback

    def _update_progress(self, message: str) -> None:
        """Update progress if callback is set."""
        if self._progress_callback:
            try:
                self._progress_callback(message)
            except Exception as e:
                self.logger.debug(f"Progress callback failed: {e}")

    def set_reference_image(self, reference_image: np.ndarray) -> bool:
        """
        Set reference image for comparison analysis.

        Args:
            reference_image: Reference image as numpy array

        Returns:
            bool: True if reference was set successfully
        """
        try:
            if not self.enable_comparison:
                self.logger.info("Image comparison disabled, skipping reference image setup")
                return True

            return self.yolo_comparison_service.set_reference_image(reference_image)

        except Exception as e:
            self.logger.error(f"Failed to set reference image: {e}")
            return False

    @performance_timer("integrated_analysis")
    async def analyze_with_chatbot(self,
                                  current_image: np.ndarray,
                                  user_message: str,
                                  include_comparison: bool = None,
                                  include_scene_analysis: bool = None) -> IntegratedAnalysisResult:
        """
        Perform complete integrated analysis with chatbot response.

        Args:
            current_image: Current image frame to analyze
            user_message: User's message/question
            include_comparison: Override for comparison analysis
            include_scene_analysis: Override for scene analysis

        Returns:
            IntegratedAnalysisResult: Complete analysis with chatbot response
        """
        analysis_start = time.time()

        # Use config defaults if not specified
        if include_comparison is None:
            include_comparison = self.enable_comparison
        if include_scene_analysis is None:
            include_scene_analysis = self.enable_scene_analysis

        try:
            self.logger.info(f"Starting integrated analysis for user message: '{user_message[:50]}...'")
            self._update_progress("Starting analysis...")

            yolo_comparison = None
            image_analysis = None

            # Validate inputs
            if current_image is None or current_image.size == 0:
                raise ValueError("Invalid current image provided")

            if not user_message or not user_message.strip():
                user_message = "Please analyze this image."

            # Step 1: YOLO Comparison Analysis (if enabled and reference is set)
            if include_comparison and self.yolo_comparison_service.get_reference_info() is not None:
                try:
                    self._update_progress("Performing YOLO object comparison...")
                    yolo_comparison = self.yolo_comparison_service.compare_with_current(
                        current_image, user_message
                    )
                    self.logger.debug("YOLO comparison completed successfully")
                except Exception as e:
                    self.logger.warning(f"YOLO comparison failed: {e}")

            # Step 2: Scene Analysis (if enabled)
            if include_scene_analysis and self.image_analysis_service:
                try:
                    self._update_progress("Analyzing scene characteristics...")
                    image_analysis = self.image_analysis_service.analyze_frame_comprehensive(
                        current_image, user_message
                    )
                    self.logger.debug("Scene analysis completed successfully")
                except Exception as e:
                    self.logger.warning(f"Scene analysis failed: {e}")

            # Step 3: Generate comprehensive prompt for chatbot
            self._update_progress("Generating AI response...")
            chatbot_prompt = self._create_comprehensive_prompt(
                user_message, yolo_comparison, image_analysis, current_image
            )

            # Step 4: Get AI response
            if not self.gemini_service.is_configured():
                chatbot_response = self._generate_fallback_response(
                    user_message, yolo_comparison, image_analysis
                )
                self.logger.warning("Gemini service not configured, using fallback response")
            else:
                try:
                    # Start chat session with persona if not already started
                    if self.chatbot_persona and not self.gemini_service.chat_session:
                        self.gemini_service.start_chat_session(self.chatbot_persona)

                    # Convert image to bytes for Gemini
                    import cv2
                    _, image_bytes = cv2.imencode('.jpg', current_image)
                    image_data = image_bytes.tobytes()

                    chatbot_response = self.gemini_service.send_message(
                        chatbot_prompt, image_data
                    )
                    self.logger.debug("Gemini response generated successfully")

                except Exception as e:
                    self.logger.error(f"Gemini API call failed: {e}")
                    chatbot_response = self._generate_fallback_response(
                        user_message, yolo_comparison, image_analysis
                    )

            # Calculate metrics
            analysis_duration = (time.time() - analysis_start) * 1000

            # Create result
            result = IntegratedAnalysisResult(
                timestamp=datetime.now().isoformat(),
                user_message=user_message,
                yolo_comparison=yolo_comparison,
                image_analysis=image_analysis,
                chatbot_response=chatbot_response,
                analysis_duration_ms=analysis_duration,
                success=True
            )

            # Update statistics
            self._update_stats(analysis_duration, success=True)

            self._update_progress("Analysis complete!")
            self.logger.info(f"Integrated analysis completed in {analysis_duration:.1f}ms")

            return result

        except Exception as e:
            analysis_duration = (time.time() - analysis_start) * 1000
            error_message = f"Integrated analysis failed: {e}"

            self.logger.error(error_message)
            self._update_stats(analysis_duration, success=False)

            # Return error result
            return IntegratedAnalysisResult(
                timestamp=datetime.now().isoformat(),
                user_message=user_message,
                yolo_comparison=None,
                image_analysis=None,
                chatbot_response=f"I apologize, but I encountered an error while analyzing the image: {str(e)}",
                analysis_duration_ms=analysis_duration,
                success=False,
                error_message=error_message
            )

    def _create_comprehensive_prompt(self,
                                   user_message: str,
                                   yolo_comparison: Optional[YoloComparisonResult],
                                   image_analysis: Optional[ImageAnalysisResult],
                                   current_image: np.ndarray) -> str:
        """Create a comprehensive prompt combining all analysis results."""
        try:
            prompt_parts = [
                f"User message: {user_message}",
                "",
                "COMPREHENSIVE IMAGE ANALYSIS CONTEXT:"
            ]

            # Add frame information
            height, width = current_image.shape[:2]
            prompt_parts.extend([
                f"- Image dimensions: {width}x{height}",
                f"- Analysis timestamp: {datetime.now().isoformat()}",
                ""
            ])

            # Add YOLO comparison results if available
            if yolo_comparison:
                prompt_parts.extend([
                    "OBJECT DETECTION COMPARISON:",
                    f"- Reference vs Current similarity: {yolo_comparison.scene_comparison.scene_similarity:.1%}",
                    f"- Objects in reference: {yolo_comparison.scene_comparison.reference_object_count}",
                    f"- Objects in current: {yolo_comparison.scene_comparison.current_object_count}",
                    f"- Changes detected: {', '.join(yolo_comparison.scene_comparison.dominant_changes)}",
                ])

                if yolo_comparison.object_comparisons:
                    prompt_parts.append("- Detailed object changes:")
                    for comp in yolo_comparison.object_comparisons[:5]:  # Limit to top 5
                        if comp.comparison_type == 'added' and comp.current_object:
                            prompt_parts.append(f"  • NEW: {comp.current_object.class_name} detected")
                        elif comp.comparison_type == 'missing' and comp.reference_object:
                            prompt_parts.append(f"  • REMOVED: {comp.reference_object.class_name} missing")
                        elif comp.comparison_type in ['moved', 'changed'] and comp.current_object:
                            prompt_parts.append(f"  • CHANGED: {comp.current_object.class_name} position/size modified")

                prompt_parts.append("")

            # Add scene analysis if available
            if image_analysis:
                prompt_parts.extend([
                    "SCENE ANALYSIS:",
                    f"- Scene description: {image_analysis.scene_description}",
                    f"- Object count: {len(image_analysis.objects)}",
                    f"- Lighting: {'bright' if image_analysis.scene_metrics.brightness > 0.7 else 'normal' if image_analysis.scene_metrics.brightness > 0.4 else 'dim'}",
                    f"- Image quality: {'excellent' if image_analysis.scene_metrics.sharpness > 0.7 else 'good' if image_analysis.scene_metrics.sharpness > 0.4 else 'fair'}",
                    f"- Motion detected: {'Yes' if image_analysis.scene_metrics.motion_detected else 'No'}",
                    f"- Dominant colors: {', '.join(image_analysis.scene_metrics.dominant_colors)}",
                ])

                if image_analysis.objects:
                    prompt_parts.append("- Detected objects:")
                    for obj in image_analysis.objects[:5]:  # Limit to top 5
                        position_desc = obj.metadata.get('position_description', 'unknown position')
                        size_desc = obj.metadata.get('object_size', 'unknown size')
                        confidence_pct = int(obj.confidence * 100)
                        prompt_parts.append(f"  • {obj.class_name} ({confidence_pct}% confidence) - {size_desc} in {position_desc}")

                prompt_parts.append("")

            # Add instructions based on response format
            if self.response_format == 'Educational':
                prompt_parts.extend([
                    "RESPONSE INSTRUCTIONS:",
                    "Please provide an educational response that:",
                    "1. Directly addresses the user's question or comment",
                    "2. Explains what you can see in the image using the analysis data",
                    "3. If comparison data is available, explains what has changed and why it matters",
                    "4. Provides educational context about object detection and computer vision",
                    "5. Encourages learning and curiosity about the technology",
                    "6. Uses simple, clear language appropriate for educational settings"
                ])
            elif self.response_format == 'Technical':
                prompt_parts.extend([
                    "RESPONSE INSTRUCTIONS:",
                    "Please provide a technical response that:",
                    "1. Addresses the user's specific technical question",
                    "2. References detection confidence scores and technical metrics",
                    "3. Explains the computer vision analysis process",
                    "4. Discusses accuracy, limitations, and interpretation of results",
                    "5. Uses appropriate technical terminology"
                ])
            else:  # Detailed (default)
                prompt_parts.extend([
                    "RESPONSE INSTRUCTIONS:",
                    "Please provide a detailed response that:",
                    "1. Thoroughly addresses the user's message",
                    "2. Describes what you observe in the image",
                    "3. Explains any changes or comparisons when relevant",
                    "4. Provides context and educational value",
                    "5. Maintains an engaging, helpful tone"
                ])

            return "\n".join(prompt_parts)

        except Exception as e:
            self.logger.error(f"Prompt creation failed: {e}")
            return f"User message: {user_message}\n\nPlease analyze the provided image and respond to the user's question."

    def _generate_fallback_response(self,
                                   user_message: str,
                                   yolo_comparison: Optional[YoloComparisonResult],
                                   image_analysis: Optional[ImageAnalysisResult]) -> str:
        """Generate a fallback response when AI service is unavailable."""
        try:
            response_parts = [
                f"I understand you asked: '{user_message}'"
            ]

            # Add comparison info if available
            if yolo_comparison:
                similarity = yolo_comparison.scene_comparison.scene_similarity
                changes = yolo_comparison.scene_comparison.dominant_changes

                response_parts.extend([
                    "",
                    f"Based on object detection analysis:",
                    f"- The current image is {similarity:.1%} similar to the reference",
                    f"- Changes detected: {', '.join(changes)}"
                ])

                if yolo_comparison.object_comparisons:
                    added = sum(1 for c in yolo_comparison.object_comparisons if c.comparison_type == 'added')
                    removed = sum(1 for c in yolo_comparison.object_comparisons if c.comparison_type == 'missing')
                    moved = sum(1 for c in yolo_comparison.object_comparisons if c.comparison_type in ['moved', 'changed'])

                    if added > 0:
                        response_parts.append(f"- {added} new objects detected")
                    if removed > 0:
                        response_parts.append(f"- {removed} objects no longer visible")
                    if moved > 0:
                        response_parts.append(f"- {moved} objects changed position or size")

            # Add scene info if available
            if image_analysis and image_analysis.objects:
                response_parts.extend([
                    "",
                    f"Current scene analysis:",
                    f"- {len(image_analysis.objects)} objects detected",
                    f"- Scene appears {image_analysis.scene_metrics.scene_complexity}",
                    f"- Lighting conditions: {'bright' if image_analysis.scene_metrics.brightness > 0.7 else 'normal' if image_analysis.scene_metrics.brightness > 0.4 else 'dim'}"
                ])

            if not yolo_comparison and not image_analysis:
                response_parts.append("\nI can see the image but detailed analysis is currently unavailable. Please check the system configuration.")

            response_parts.append("\nNote: AI analysis service is currently unavailable. This is a basic automated response based on computer vision detection only.")

            return "\n".join(response_parts)

        except Exception as e:
            self.logger.error(f"Fallback response generation failed: {e}")
            return f"I received your message: '{user_message}'. However, I'm currently unable to analyze the image. Please try again later."

    def _update_stats(self, processing_time_ms: float, success: bool) -> None:
        """Update performance statistics."""
        self._stats['total_analyses'] += 1
        self._stats['last_analysis_time'] = datetime.now().isoformat()

        if success:
            self._stats['successful_analyses'] += 1
        else:
            self._stats['failed_analyses'] += 1

        # Update running average for successful analyses
        if success:
            current_avg = self._stats['average_response_time_ms']
            successful_count = self._stats['successful_analyses']
            self._stats['average_response_time_ms'] = (current_avg * (successful_count - 1) + processing_time_ms) / successful_count

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        success_rate = (self._stats['successful_analyses'] / max(1, self._stats['total_analyses'])) * 100

        return {
            **self._stats,
            'success_rate_percent': success_rate,
            'yolo_comparison_stats': self.yolo_comparison_service.get_performance_stats(),
            'reference_image_set': self.yolo_comparison_service.get_reference_info() is not None,
            'gemini_configured': self.gemini_service.is_configured(),
            'services_available': {
                'yolo_backend': self.yolo_backend.is_loaded,
                'comparison_service': True,
                'image_analysis': self.inference_service is not None,
                'gemini_service': self.gemini_service.is_configured()
            }
        }

    def get_reference_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current reference image."""
        return self.yolo_comparison_service.get_reference_info()

    def clear_caches(self) -> None:
        """Clear all service caches to free memory."""
        self.yolo_comparison_service.clear_cache()
        if hasattr(self.image_analysis_service, 'clear_cache'):
            self.image_analysis_service.clear_cache()
        self.logger.info("All service caches cleared")

    def update_configuration(self, **kwargs) -> None:
        """Update service configuration."""
        # Update local configuration
        if 'enable_image_comparison' in kwargs:
            self.enable_comparison = kwargs['enable_image_comparison']
        if 'enable_scene_analysis' in kwargs:
            self.enable_scene_analysis = kwargs['enable_scene_analysis']
        if 'chatbot_persona' in kwargs:
            self.chatbot_persona = kwargs['chatbot_persona']
        if 'response_format' in kwargs:
            self.response_format = kwargs['response_format']

        # Update sub-service configurations
        self.yolo_comparison_service.update_configuration(**kwargs)

        if 'gemini_api_key' in kwargs or 'gemini_model' in kwargs:
            self.gemini_service.update_configuration(**kwargs)

        self.logger.info("Service configuration updated")

    async def analyze_async(self, current_image: np.ndarray, user_message: str) -> IntegratedAnalysisResult:
        """Asynchronous wrapper for the main analysis method."""
        return await self.analyze_with_chatbot(current_image, user_message)