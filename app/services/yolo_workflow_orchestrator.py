"""
YOLO Workflow Orchestrator Service.

This service provides a high-performance orchestration layer for coordinating
YOLO detection, reference comparison, and AI chatbot analysis in real-time.
It handles async operations, service coordination, and error boundaries.

Architecture Overview:
- Async/await patterns for non-blocking operations
- Thread-safe service coordination
- Configuration-driven feature enablement
- Performance monitoring integration
- Graceful error handling and fallback mechanisms
"""

import asyncio
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2

from ..core.entities import Detection
from ..core.exceptions import DetectionError, WebcamError
from ..core.performance import performance_timer, PerformanceMonitor
from ..backends.yolo_backend import YoloBackend
from ..utils.detection_formatter import format_detection_data
from .reference_manager import ReferenceImageManager, ComparisonResult
from .gemini_service import AsyncGeminiService
from .integrated_analysis_service import IntegratedAnalysisService, IntegratedAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class WorkflowConfig:
    """Configuration for YOLO workflow orchestration."""
    auto_yolo_analysis: bool = True
    reference_comparison_enabled: bool = True
    min_detection_confidence: float = 0.5
    max_objects_to_analyze: int = 10
    async_timeout_seconds: float = 5.0
    enable_performance_monitoring: bool = True
    cache_analysis_results: bool = True
    max_concurrent_operations: int = 3


@dataclass
class WorkflowResult:
    """Result of YOLO workflow orchestration."""
    timestamp: datetime
    user_message: str
    detections: List[Detection]
    comparison_result: Optional[ComparisonResult]
    formatted_data: str
    enhanced_prompt: str
    ai_response: Optional[str]
    workflow_time_ms: float
    success: bool
    error_message: Optional[str] = None
    performance_metrics: Dict[str, float] = None


class YoloWorkflowOrchestrator:
    """
    High-performance orchestration service for YOLO analysis workflow.

    This service coordinates:
    - Frame capture from webcam
    - YOLO object detection
    - Reference image comparison
    - Detection data formatting
    - AI chatbot integration
    """

    def __init__(self,
                 yolo_backend: Optional[YoloBackend] = None,
                 reference_manager: Optional[ReferenceImageManager] = None,
                 gemini_service: Optional[AsyncGeminiService] = None,
                 integrated_service: Optional[IntegratedAnalysisService] = None,
                 config: Optional[WorkflowConfig] = None):
        """
        Initialize the YOLO Workflow Orchestrator.

        Args:
            yolo_backend: YOLO backend for object detection
            reference_manager: Reference image management service
            gemini_service: Gemini AI service for chatbot
            integrated_service: Integrated analysis service (fallback)
            config: Workflow configuration
        """
        self.yolo_backend = yolo_backend
        self.reference_manager = reference_manager
        self.gemini_service = gemini_service
        self.integrated_service = integrated_service
        self.config = config or WorkflowConfig()

        # Performance monitoring (use singleton instance)
        self.performance_monitor = PerformanceMonitor.instance() if self.config.enable_performance_monitoring else None

        # Thread pool for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_operations)

        # Analysis cache
        self._analysis_cache = {} if self.config.cache_analysis_results else None
        self._cache_lock = threading.Lock()

        # State tracking
        self._is_running = False
        self._active_workflows = 0
        self._workflow_lock = threading.Lock()

        logger.info(f"YoloWorkflowOrchestrator initialized with config: {asdict(self.config)}")

    async def orchestrate_analysis(self,
                                  current_frame: np.ndarray,
                                  user_message: str,
                                  reference_frame: Optional[np.ndarray] = None) -> WorkflowResult:
        """
        Orchestrate the complete YOLO analysis workflow.

        Args:
            current_frame: Current video frame
            user_message: User's chat message
            reference_frame: Optional reference frame for comparison

        Returns:
            WorkflowResult: Complete workflow result
        """
        workflow_start = time.time()

        # Track active workflow
        with self._workflow_lock:
            self._active_workflows += 1

        try:
            # Check if auto-analysis is enabled
            if not self.config.auto_yolo_analysis:
                logger.info("Auto YOLO analysis disabled, skipping workflow")
                return self._create_bypass_result(user_message, workflow_start)

            # Validate inputs
            if current_frame is None or current_frame.size == 0:
                raise ValueError("Invalid current frame provided")

            # Create workflow context
            context = {
                'timestamp': datetime.now(),
                'user_message': user_message,
                'frame_shape': current_frame.shape,
                'performance_metrics': {}
            }

            # Step 1: Run YOLO detection (async)
            detections = await self._run_yolo_detection_async(current_frame, context)

            # Step 2: Reference comparison (if enabled and available)
            comparison_result = None
            if self.config.reference_comparison_enabled and self.reference_manager:
                comparison_result = await self._run_reference_comparison_async(
                    current_frame, detections, reference_frame, context
                )

            # Step 3: Format detection data
            formatted_data = await self._format_detection_data_async(
                detections, current_frame.shape[:2][::-1],
                comparison_result, context
            )

            # Step 4: Create enhanced prompt for AI
            enhanced_prompt = self._create_enhanced_prompt(
                user_message, formatted_data, comparison_result
            )

            # Step 5: Get AI response (if service available)
            ai_response = None
            if self.gemini_service and self.gemini_service.is_configured():
                ai_response = await self._get_ai_response_async(
                    enhanced_prompt, current_frame, context
                )

            # Calculate total workflow time
            workflow_time_ms = (time.time() - workflow_start) * 1000

            # Create successful result
            result = WorkflowResult(
                timestamp=context['timestamp'],
                user_message=user_message,
                detections=detections,
                comparison_result=comparison_result,
                formatted_data=formatted_data,
                enhanced_prompt=enhanced_prompt,
                ai_response=ai_response,
                workflow_time_ms=workflow_time_ms,
                success=True,
                performance_metrics=context['performance_metrics']
            )

            # Cache result if enabled
            if self._analysis_cache is not None:
                self._cache_analysis_result(user_message, result)

            # Log performance metrics
            if self.performance_monitor:
                self.performance_monitor.record_operation_time('workflow_orchestration', workflow_time_ms / 1000.0)
                for key, value in context['performance_metrics'].items():
                    self.performance_monitor.record_operation_time(f'workflow_{key}', value / 1000.0)

            logger.info(f"Workflow completed successfully in {workflow_time_ms:.1f}ms")
            return result

        except asyncio.TimeoutError:
            logger.error(f"Workflow timeout after {self.config.async_timeout_seconds}s")
            return self._create_error_result(
                user_message, "Workflow timeout", workflow_start
            )
        except Exception as e:
            logger.error(f"Workflow orchestration failed: {e}")
            return self._create_error_result(
                user_message, str(e), workflow_start
            )
        finally:
            # Track workflow completion
            with self._workflow_lock:
                self._active_workflows -= 1

    async def _run_yolo_detection_async(self,
                                       frame: np.ndarray,
                                       context: Dict) -> List[Detection]:
        """Run YOLO detection asynchronously."""
        start_time = time.time()

        try:
            if not self.yolo_backend or not self.yolo_backend.is_loaded:
                logger.warning("YOLO backend not available")
                return []

            # Run detection in executor to avoid blocking
            loop = asyncio.get_event_loop()
            detections = await loop.run_in_executor(
                self.executor,
                self._run_yolo_detection_sync,
                frame
            )

            # Filter by confidence and max objects
            detections = [d for d in detections
                         if d.confidence >= self.config.min_detection_confidence]
            detections = detections[:self.config.max_objects_to_analyze]

            # Record performance
            detection_time_ms = (time.time() - start_time) * 1000
            context['performance_metrics']['yolo_detection_ms'] = detection_time_ms
            logger.debug(f"YOLO detection completed in {detection_time_ms:.1f}ms, found {len(detections)} objects")

            return detections

        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            context['performance_metrics']['yolo_detection_error'] = 1
            return []

    def _run_yolo_detection_sync(self, frame: np.ndarray) -> List[Detection]:
        """Synchronous YOLO detection (runs in thread pool)."""
        return self.yolo_backend.predict(
            frame,
            conf=self.config.min_detection_confidence,
            iou=0.45,
            device='0' if self.yolo_backend.device == 'cuda' else 'cpu'
        )

    async def _run_reference_comparison_async(self,
                                             current_frame: np.ndarray,
                                             detections: List[Detection],
                                             reference_frame: Optional[np.ndarray],
                                             context: Dict) -> Optional[ComparisonResult]:
        """Run reference comparison asynchronously."""
        start_time = time.time()

        try:
            if not self.reference_manager:
                logger.debug("Reference manager not available")
                return None

            # Set reference if provided
            if reference_frame is not None:
                await self._set_reference_frame_async(reference_frame)

            # Check if reference is set
            if not self.reference_manager.has_active_reference():
                logger.debug("No reference image set for comparison")
                return None

            # Run comparison in executor
            loop = asyncio.get_event_loop()
            comparison_result = await loop.run_in_executor(
                self.executor,
                self.reference_manager.compare_detections,
                detections,
                self.reference_manager.get_active_reference_id()
            )

            # Record performance
            comparison_time_ms = (time.time() - start_time) * 1000
            context['performance_metrics']['comparison_ms'] = comparison_time_ms
            logger.debug(f"Reference comparison completed in {comparison_time_ms:.1f}ms")

            return comparison_result

        except Exception as e:
            logger.error(f"Reference comparison failed: {e}")
            context['performance_metrics']['comparison_error'] = 1
            return None

    async def _set_reference_frame_async(self, reference_frame: np.ndarray):
        """Set reference frame asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self.reference_manager.capture_reference,
            reference_frame,
            self.config.min_detection_confidence
        )

    async def _format_detection_data_async(self,
                                          detections: List[Detection],
                                          frame_dimensions: Tuple[int, int],
                                          comparison_result: Optional[ComparisonResult],
                                          context: Dict) -> str:
        """Format detection data asynchronously."""
        start_time = time.time()

        try:
            # Format in executor to avoid blocking
            loop = asyncio.get_event_loop()
            formatted_data = await loop.run_in_executor(
                self.executor,
                format_detection_data,
                detections,
                frame_dimensions,
                None,  # yolo_comparison (using comparison_result instead)
                None,  # image_analysis
                True,  # include_coordinates
                True,  # include_angles
                True,  # include_confidence
                True   # include_size_info
            )

            # Add comparison summary if available
            if comparison_result:
                formatted_data += self._format_comparison_summary(comparison_result)

            # Record performance
            format_time_ms = (time.time() - start_time) * 1000
            context['performance_metrics']['format_ms'] = format_time_ms

            return formatted_data

        except Exception as e:
            logger.error(f"Detection formatting failed: {e}")
            return f"Detection data formatting failed: {str(e)}"

    def _format_comparison_summary(self, comparison: ComparisonResult) -> str:
        """Format comparison result summary."""
        summary_parts = [
            "",
            "📊 Reference Comparison Results:",
            f"Overall Similarity: {comparison.overall_similarity:.1%}",
            f"Objects Missing: {len(comparison.objects_missing)}",
            f"Objects Added: {len(comparison.objects_added)}",
            f"Scene Change Score: {comparison.scene_change_score:.1%}"
        ]
        return "\n".join(summary_parts)

    def _create_enhanced_prompt(self,
                               user_message: str,
                               formatted_data: str,
                               comparison_result: Optional[ComparisonResult]) -> str:
        """Create enhanced prompt with YOLO data for AI."""
        prompt_parts = [user_message]

        # Add detection data context
        if formatted_data:
            prompt_parts.append("\n[Current Scene Analysis:]")
            prompt_parts.append(formatted_data)

        # Add comparison context if available
        if comparison_result and comparison_result.overall_similarity < 0.8:
            prompt_parts.append("\n[Scene Changes Detected:]")
            if comparison_result.objects_missing:
                prompt_parts.append(f"- {len(comparison_result.objects_missing)} objects are missing")
            if comparison_result.objects_added:
                prompt_parts.append(f"- {len(comparison_result.objects_added)} new objects detected")

        return "\n".join(prompt_parts)

    async def _get_ai_response_async(self,
                                    enhanced_prompt: str,
                                    current_frame: np.ndarray,
                                    context: Dict) -> Optional[str]:
        """Get AI response asynchronously."""
        start_time = time.time()

        try:
            # Convert frame to bytes for Gemini
            _, image_bytes = cv2.imencode('.jpg', current_frame)
            image_data = image_bytes.tobytes()

            # Send to Gemini with timeout
            response = await asyncio.wait_for(
                self.gemini_service.send_message_async(enhanced_prompt, image_data),
                timeout=self.config.async_timeout_seconds
            )

            # Record performance
            ai_time_ms = (time.time() - start_time) * 1000
            context['performance_metrics']['ai_response_ms'] = ai_time_ms
            logger.debug(f"AI response received in {ai_time_ms:.1f}ms")

            return response

        except asyncio.TimeoutError:
            logger.error(f"AI response timeout after {self.config.async_timeout_seconds}s")
            context['performance_metrics']['ai_timeout'] = 1
            return None
        except Exception as e:
            logger.error(f"AI response failed: {e}")
            context['performance_metrics']['ai_error'] = 1
            return None

    def _cache_analysis_result(self, key: str, result: WorkflowResult):
        """Cache analysis result for reuse."""
        with self._cache_lock:
            # Simple LRU: keep last 10 results
            if len(self._analysis_cache) >= 10:
                # Remove oldest
                oldest_key = next(iter(self._analysis_cache))
                del self._analysis_cache[oldest_key]

            self._analysis_cache[key] = result

    def get_cached_result(self, key: str) -> Optional[WorkflowResult]:
        """Get cached analysis result."""
        with self._cache_lock:
            return self._analysis_cache.get(key)

    def _create_bypass_result(self, user_message: str, start_time: float) -> WorkflowResult:
        """Create result when bypassing YOLO workflow."""
        return WorkflowResult(
            timestamp=datetime.now(),
            user_message=user_message,
            detections=[],
            comparison_result=None,
            formatted_data="",
            enhanced_prompt=user_message,
            ai_response=None,
            workflow_time_ms=(time.time() - start_time) * 1000,
            success=True,
            error_message="YOLO analysis bypassed"
        )

    def _create_error_result(self, user_message: str, error: str, start_time: float) -> WorkflowResult:
        """Create error result."""
        return WorkflowResult(
            timestamp=datetime.now(),
            user_message=user_message,
            detections=[],
            comparison_result=None,
            formatted_data="",
            enhanced_prompt=user_message,
            ai_response=None,
            workflow_time_ms=(time.time() - start_time) * 1000,
            success=False,
            error_message=error
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.performance_monitor:
            return {}

        return {
            'active_workflows': self._active_workflows,
            'cached_results': len(self._analysis_cache) if self._analysis_cache else 0,
            'metrics': self.performance_monitor.get_summary()
        }

    def shutdown(self):
        """Shutdown the orchestrator and cleanup resources."""
        logger.info("Shutting down YoloWorkflowOrchestrator")

        # Wait for active workflows to complete
        timeout = 10
        start_time = time.time()
        while self._active_workflows > 0 and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        # Shutdown executor
        self.executor.shutdown(wait=True, timeout=5)

        # Clear cache
        if self._analysis_cache:
            self._analysis_cache.clear()

        logger.info("YoloWorkflowOrchestrator shutdown complete")