"""
YOLO-based Image Comparison Service for ChatBot Integration.

This service provides comprehensive image comparison using YOLO object detection,
comparing reference images with live webcam frames and providing structured data
for AI-powered chatbot analysis and educational feedback.
"""

import time
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
import numpy as np
import cv2

from ..core.entities import Detection, BBox
from ..core.exceptions import DetectionError, WebcamError
# Performance monitoring removed for simplification
from ..backends.yolo_backend import YoloBackend
from .gemini_service import AsyncGeminiService

logger = logging.getLogger(__name__)


# Simple LRU Cache implementation
class LRUCache:
    """Simple Least Recently Used cache."""

    def __init__(self, max_size: int = 100):
        from collections import OrderedDict
        self.cache = OrderedDict()
        self.max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, key: str):
        """Get item from cache."""
        if key in self.cache:
            self._hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        self._misses += 1
        return None

    def put(self, key: str, value):
        """Put item in cache."""
        if key in self.cache:
            # Move to end
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value
            # Remove oldest if cache is full
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def _cache(self):
        """Provide _cache property for compatibility."""
        return self.cache

@dataclass
class ObjectComparison:
    """Detailed comparison between detected objects."""
    reference_object: Optional[Detection]
    current_object: Optional[Detection]
    comparison_type: str  # 'match', 'missing', 'added', 'moved', 'changed'
    confidence_difference: float
    position_change: Optional[Dict[str, float]]  # x_delta, y_delta, distance
    size_change: Optional[Dict[str, float]]  # width_ratio, height_ratio, area_ratio
    match_score: float  # 0.0 to 1.0

@dataclass
class SceneComparison:
    """Overall scene-level comparison metrics."""
    reference_object_count: int
    current_object_count: int
    objects_added: int
    objects_removed: int
    objects_moved: int
    objects_unchanged: int
    scene_similarity: float  # 0.0 to 1.0
    dominant_changes: List[str]

@dataclass
class YoloComparisonResult:
    """Complete YOLO-based comparison result for chatbot integration."""
    timestamp: str
    reference_image_hash: str
    current_image_hash: str
    frame_dimensions: Dict[str, int]
    object_comparisons: List[ObjectComparison]
    scene_comparison: SceneComparison
    analysis_duration_ms: float
    confidence_threshold: float
    chatbot_summary: str


class YoloComparisonService:
    """
    Service for comparing images using YOLO object detection.

    This service performs detailed object-level comparison between reference images
    and live frames, providing structured data optimized for AI chatbot analysis
    and educational feedback.
    """

    def __init__(self, yolo_backend: YoloBackend, gemini_service: AsyncGeminiService, config: Dict[str, Any]):
        """
        Initialize the YOLO comparison service.

        Args:
            yolo_backend: Configured YOLO backend for object detection
            gemini_service: Gemini service for AI analysis
            config: Service configuration dictionary
        """
        self.yolo_backend = yolo_backend
        self.gemini_service = gemini_service
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Configuration parameters
        self.confidence_threshold = config.get('detection_confidence_threshold', 0.5)
        self.iou_threshold = config.get('detection_iou_threshold', 0.45)
        self.position_tolerance = config.get('master_tolerance_px', 40)
        self.size_change_threshold = 0.2  # 20% size change threshold

        # Caching for performance
        self._comparison_cache = LRUCache(max_size=50)
        self._image_hash_cache = LRUCache(max_size=100)

        # Reference image management
        self._reference_detections = None
        self._reference_image_hash = None
        self._reference_frame_dimensions = None

        # Performance tracking
        self._stats = {
            'comparisons_performed': 0,
            'cache_hits': 0,
            'average_processing_time_ms': 0.0,
            'last_comparison_time': None
        }

    def set_reference_image(self, reference_image: np.ndarray) -> bool:
        """
        Set the reference image for comparison.

        Args:
            reference_image: Reference image as numpy array

        Returns:
            bool: True if reference was set successfully

        Raises:
            DetectionError: If YOLO detection fails
            ValueError: If image is invalid
        """
        try:
            if reference_image is None or reference_image.size == 0:
                raise ValueError("Reference image is empty or invalid")

            self.logger.info("Setting new reference image for comparison")

            # Generate image hash for caching
            self._reference_image_hash = self._generate_image_hash(reference_image)
            self._reference_frame_dimensions = {
                'width': reference_image.shape[1],
                'height': reference_image.shape[0]
            }

            # Perform YOLO detection on reference image
            self._reference_detections = self._detect_objects(reference_image)

            self.logger.info(f"Reference image set with {len(self._reference_detections)} detected objects")
            return True

        except Exception as e:
            self.logger.error(f"Failed to set reference image: {e}")
            self._reference_detections = None
            self._reference_image_hash = None
            self._reference_frame_dimensions = None
            raise DetectionError(f"Failed to set reference image: {e}")

    def compare_with_current(self, current_image: np.ndarray, user_message: str = "") -> YoloComparisonResult:
        """
        Compare current image with the reference image.

        Args:
            current_image: Current image frame as numpy array
            user_message: Optional user message for context

        Returns:
            YoloComparisonResult: Comprehensive comparison result

        Raises:
            DetectionError: If comparison fails
            ValueError: If no reference image is set
        """
        comparison_start = time.time()

        try:
            if self._reference_detections is None:
                raise ValueError("No reference image set. Call set_reference_image() first.")

            if current_image is None or current_image.size == 0:
                raise ValueError("Current image is empty or invalid")

            # Generate current image hash for caching
            current_image_hash = self._generate_image_hash(current_image)

            # Check cache for recent comparison
            cache_key = f"{self._reference_image_hash}:{current_image_hash}:{self.confidence_threshold}"
            cached_result = self._comparison_cache.get(cache_key)
            if cached_result is not None:
                self._stats['cache_hits'] += 1
                self.logger.debug("Using cached comparison result")
                return cached_result

            self.logger.info("Performing YOLO-based image comparison")

            # Detect objects in current image
            current_detections = self._detect_objects(current_image)

            # Perform detailed comparison
            object_comparisons = self._compare_objects(
                self._reference_detections,
                current_detections,
                current_image.shape[:2]
            )

            # Calculate scene-level metrics
            scene_comparison = self._calculate_scene_comparison(
                self._reference_detections,
                current_detections,
                object_comparisons
            )

            # Generate chatbot-friendly summary
            chatbot_summary = self._generate_chatbot_summary(
                object_comparisons,
                scene_comparison,
                user_message
            )

            # Create comprehensive result
            analysis_duration = (time.time() - comparison_start) * 1000

            result = YoloComparisonResult(
                timestamp=datetime.now().isoformat(),
                reference_image_hash=self._reference_image_hash,
                current_image_hash=current_image_hash,
                frame_dimensions={
                    'width': current_image.shape[1],
                    'height': current_image.shape[0]
                },
                object_comparisons=object_comparisons,
                scene_comparison=scene_comparison,
                analysis_duration_ms=analysis_duration,
                confidence_threshold=self.confidence_threshold,
                chatbot_summary=chatbot_summary
            )

            # Cache the result
            self._comparison_cache.put(cache_key, result)

            # Update statistics
            self._update_stats(analysis_duration)

            self.logger.info(f"Comparison completed in {analysis_duration:.1f}ms - "
                           f"Found {len(object_comparisons)} object changes")

            return result

        except Exception as e:
            self.logger.error(f"Image comparison failed: {e}")
            raise DetectionError(f"Image comparison failed: {e}")

    def _detect_objects(self, image: np.ndarray) -> List[Detection]:
        """
        Perform YOLO object detection on an image.

        Args:
            image: Input image array

        Returns:
            List[Detection]: List of detected objects

        Raises:
            DetectionError: If detection fails
        """
        try:
            if not self.yolo_backend.is_loaded:
                raise DetectionError("YOLO model not loaded")

            # Perform inference
            detections = self.yolo_backend.predict(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )

            # Add class names if available
            model_info = self.yolo_backend.get_model_info()
            class_names = model_info.get('class_names', {})

            for detection in detections:
                if hasattr(detection, 'class_name'):
                    continue  # Already has class name

                class_name = class_names.get(detection.class_id, f"class_{detection.class_id}")
                detection.class_name = class_name

            return detections

        except Exception as e:
            self.logger.error(f"Object detection failed: {e}")
            raise DetectionError(f"Object detection failed: {e}")

    def _compare_objects(self, reference_detections: List[Detection],
                        current_detections: List[Detection],
                        frame_shape: Tuple[int, int]) -> List[ObjectComparison]:
        """
        Compare detected objects between reference and current images.

        Args:
            reference_detections: Objects detected in reference image
            current_detections: Objects detected in current image
            frame_shape: Frame dimensions (height, width)

        Returns:
            List[ObjectComparison]: Detailed object comparisons
        """
        comparisons = []
        frame_height, frame_width = frame_shape

        # Track which current detections have been matched
        matched_current = set()

        try:
            # Find matches for reference objects
            for ref_detection in reference_detections:
                best_match = None
                best_score = 0.0
                best_current_idx = -1

                # Look for the best matching current detection
                for i, curr_detection in enumerate(current_detections):
                    if i in matched_current:
                        continue

                    # Only match objects of the same class
                    if ref_detection.class_id != curr_detection.class_id:
                        continue

                    # Calculate match score based on position and size similarity
                    match_score = self._calculate_match_score(
                        ref_detection, curr_detection, frame_width, frame_height
                    )

                    if match_score > best_score and match_score > 0.3:  # Minimum match threshold
                        best_match = curr_detection
                        best_score = match_score
                        best_current_idx = i

                if best_match is not None:
                    # Found a match
                    matched_current.add(best_current_idx)

                    # Calculate detailed comparison metrics
                    position_change = self._calculate_position_change(
                        ref_detection, best_match, frame_width, frame_height
                    )
                    size_change = self._calculate_size_change(ref_detection, best_match)

                    # Determine comparison type
                    comparison_type = self._determine_comparison_type(
                        position_change, size_change, best_score
                    )

                    comparison = ObjectComparison(
                        reference_object=ref_detection,
                        current_object=best_match,
                        comparison_type=comparison_type,
                        confidence_difference=best_match.score - ref_detection.score,
                        position_change=position_change,
                        size_change=size_change,
                        match_score=best_score
                    )
                else:
                    # Missing object
                    comparison = ObjectComparison(
                        reference_object=ref_detection,
                        current_object=None,
                        comparison_type='missing',
                        confidence_difference=0.0,
                        position_change=None,
                        size_change=None,
                        match_score=0.0
                    )

                comparisons.append(comparison)

            # Handle unmatched current objects (newly added)
            for i, curr_detection in enumerate(current_detections):
                if i not in matched_current:
                    comparison = ObjectComparison(
                        reference_object=None,
                        current_object=curr_detection,
                        comparison_type='added',
                        confidence_difference=0.0,
                        position_change=None,
                        size_change=None,
                        match_score=0.0
                    )
                    comparisons.append(comparison)

        except Exception as e:
            self.logger.error(f"Object comparison failed: {e}")

        return comparisons

    def _calculate_match_score(self, ref_detection: Detection, curr_detection: Detection,
                              frame_width: int, frame_height: int) -> float:
        """
        Calculate similarity score between two detections.

        Args:
            ref_detection: Reference detection
            curr_detection: Current detection
            frame_width: Frame width for normalization
            frame_height: Frame height for normalization

        Returns:
            float: Match score between 0.0 and 1.0
        """
        try:
            # Position similarity (normalized)
            ref_center_x = (ref_detection.bbox[0] + ref_detection.bbox[2]) / 2 / frame_width
            ref_center_y = (ref_detection.bbox[1] + ref_detection.bbox[3]) / 2 / frame_height
            curr_center_x = (curr_detection.bbox[0] + curr_detection.bbox[2]) / 2 / frame_width
            curr_center_y = (curr_detection.bbox[1] + curr_detection.bbox[3]) / 2 / frame_height

            position_distance = np.sqrt((ref_center_x - curr_center_x)**2 + (ref_center_y - curr_center_y)**2)
            position_score = max(0.0, 1.0 - position_distance * 2)  # Scale distance penalty

            # Size similarity
            ref_width = ref_detection.bbox[2] - ref_detection.bbox[0]
            ref_height = ref_detection.bbox[3] - ref_detection.bbox[1]
            curr_width = curr_detection.bbox[2] - curr_detection.bbox[0]
            curr_height = curr_detection.bbox[3] - curr_detection.bbox[1]

            width_ratio = min(ref_width, curr_width) / max(ref_width, curr_width)
            height_ratio = min(ref_height, curr_height) / max(ref_height, curr_height)
            size_score = (width_ratio + height_ratio) / 2

            # Confidence similarity
            conf_diff = abs(ref_detection.score - curr_detection.score)
            conf_score = max(0.0, 1.0 - conf_diff)

            # Weighted combination
            total_score = (position_score * 0.5 + size_score * 0.3 + conf_score * 0.2)
            return min(1.0, max(0.0, total_score))

        except Exception as e:
            self.logger.debug(f"Match score calculation failed: {e}")
            return 0.0

    def _calculate_position_change(self, ref_detection: Detection, curr_detection: Detection,
                                  frame_width: int, frame_height: int) -> Dict[str, float]:
        """Calculate position change metrics."""
        try:
            # Calculate center points
            ref_center_x = (ref_detection.bbox[0] + ref_detection.bbox[2]) / 2
            ref_center_y = (ref_detection.bbox[1] + ref_detection.bbox[3]) / 2
            curr_center_x = (curr_detection.bbox[0] + curr_detection.bbox[2]) / 2
            curr_center_y = (curr_detection.bbox[1] + curr_detection.bbox[3]) / 2

            # Calculate changes
            x_delta = curr_center_x - ref_center_x
            y_delta = curr_center_y - ref_center_y
            distance = np.sqrt(x_delta**2 + y_delta**2)

            return {
                'x_delta': float(x_delta),
                'y_delta': float(y_delta),
                'distance': float(distance),
                'x_delta_normalized': float(x_delta / frame_width),
                'y_delta_normalized': float(y_delta / frame_height),
                'distance_normalized': float(distance / np.sqrt(frame_width**2 + frame_height**2))
            }
        except Exception:
            return {'x_delta': 0.0, 'y_delta': 0.0, 'distance': 0.0}

    def _calculate_size_change(self, ref_detection: Detection, curr_detection: Detection) -> Dict[str, float]:
        """Calculate size change metrics."""
        try:
            ref_width = ref_detection.bbox[2] - ref_detection.bbox[0]
            ref_height = ref_detection.bbox[3] - ref_detection.bbox[1]
            curr_width = curr_detection.bbox[2] - curr_detection.bbox[0]
            curr_height = curr_detection.bbox[3] - curr_detection.bbox[1]

            width_ratio = curr_width / ref_width if ref_width > 0 else 1.0
            height_ratio = curr_height / ref_height if ref_height > 0 else 1.0
            area_ratio = (curr_width * curr_height) / (ref_width * ref_height) if (ref_width * ref_height) > 0 else 1.0

            return {
                'width_ratio': float(width_ratio),
                'height_ratio': float(height_ratio),
                'area_ratio': float(area_ratio),
                'width_change_percent': float((width_ratio - 1.0) * 100),
                'height_change_percent': float((height_ratio - 1.0) * 100),
                'area_change_percent': float((area_ratio - 1.0) * 100)
            }
        except Exception:
            return {'width_ratio': 1.0, 'height_ratio': 1.0, 'area_ratio': 1.0}

    def _determine_comparison_type(self, position_change: Dict[str, float],
                                  size_change: Dict[str, float], match_score: float) -> str:
        """Determine the type of change between objects."""
        try:
            # Check for significant position change
            position_threshold = self.position_tolerance / 100.0  # Convert to normalized threshold
            has_moved = position_change.get('distance_normalized', 0) > position_threshold

            # Check for significant size change
            size_threshold = self.size_change_threshold
            width_changed = abs(size_change.get('width_ratio', 1.0) - 1.0) > size_threshold
            height_changed = abs(size_change.get('height_ratio', 1.0) - 1.0) > size_threshold
            has_size_change = width_changed or height_changed

            # Determine type based on changes
            if has_moved and has_size_change:
                return 'changed'
            elif has_moved:
                return 'moved'
            elif has_size_change:
                return 'changed'
            elif match_score > 0.8:
                return 'match'
            else:
                return 'match'  # Default for lower confidence matches

        except Exception:
            return 'match'

    def _calculate_scene_comparison(self, reference_detections: List[Detection],
                                   current_detections: List[Detection],
                                   object_comparisons: List[ObjectComparison]) -> SceneComparison:
        """Calculate scene-level comparison metrics."""
        try:
            # Count different types of changes
            objects_added = sum(1 for comp in object_comparisons if comp.comparison_type == 'added')
            objects_removed = sum(1 for comp in object_comparisons if comp.comparison_type == 'missing')
            objects_moved = sum(1 for comp in object_comparisons if comp.comparison_type == 'moved')
            objects_unchanged = sum(1 for comp in object_comparisons if comp.comparison_type == 'match')

            # Calculate overall scene similarity
            total_ref_objects = len(reference_detections)
            total_curr_objects = len(current_detections)

            if total_ref_objects == 0 and total_curr_objects == 0:
                scene_similarity = 1.0
            elif total_ref_objects == 0 or total_curr_objects == 0:
                scene_similarity = 0.0
            else:
                # Base similarity on proportion of unchanged objects
                unchanged_ratio = objects_unchanged / max(total_ref_objects, total_curr_objects)
                # Penalize for added/removed objects
                change_penalty = (objects_added + objects_removed) / (total_ref_objects + total_curr_objects)
                scene_similarity = max(0.0, unchanged_ratio - change_penalty * 0.5)

            # Identify dominant changes
            dominant_changes = []
            if objects_added > 0:
                dominant_changes.append(f"{objects_added} object(s) added")
            if objects_removed > 0:
                dominant_changes.append(f"{objects_removed} object(s) removed")
            if objects_moved > 0:
                dominant_changes.append(f"{objects_moved} object(s) moved")
            if not dominant_changes:
                dominant_changes.append("No significant changes detected")

            return SceneComparison(
                reference_object_count=total_ref_objects,
                current_object_count=total_curr_objects,
                objects_added=objects_added,
                objects_removed=objects_removed,
                objects_moved=objects_moved,
                objects_unchanged=objects_unchanged,
                scene_similarity=float(scene_similarity),
                dominant_changes=dominant_changes
            )

        except Exception as e:
            self.logger.error(f"Scene comparison calculation failed: {e}")
            return SceneComparison(0, 0, 0, 0, 0, 0, 0.0, ["Analysis failed"])

    def _generate_chatbot_summary(self, object_comparisons: List[ObjectComparison],
                                 scene_comparison: SceneComparison, user_message: str) -> str:
        """Generate a summary optimized for chatbot consumption."""
        try:
            summary_parts = []

            # Scene overview
            summary_parts.append(f"Scene Analysis: {scene_comparison.scene_similarity:.1%} similarity to reference")

            # Object changes summary
            if scene_comparison.objects_added > 0:
                added_objects = [comp.current_object.class_name for comp in object_comparisons
                               if comp.comparison_type == 'added' and comp.current_object]
                summary_parts.append(f"Added: {', '.join(added_objects)}")

            if scene_comparison.objects_removed > 0:
                removed_objects = [comp.reference_object.class_name for comp in object_comparisons
                                 if comp.comparison_type == 'missing' and comp.reference_object]
                summary_parts.append(f"Removed: {', '.join(removed_objects)}")

            if scene_comparison.objects_moved > 0:
                moved_objects = [comp.current_object.class_name for comp in object_comparisons
                               if comp.comparison_type == 'moved' and comp.current_object]
                summary_parts.append(f"Moved: {', '.join(moved_objects)}")

            # Detailed changes for significant modifications
            significant_changes = []
            for comp in object_comparisons:
                if comp.comparison_type in ['moved', 'changed'] and comp.position_change:
                    distance = comp.position_change.get('distance_normalized', 0)
                    if distance > 0.1:  # Significant movement
                        obj_name = comp.current_object.class_name if comp.current_object else "object"
                        significant_changes.append(f"{obj_name} moved {distance:.1%} of frame distance")

            if significant_changes:
                summary_parts.extend(significant_changes)

            if not summary_parts:
                summary_parts.append("No significant changes detected between reference and current image")

            return " | ".join(summary_parts)

        except Exception as e:
            self.logger.error(f"Chatbot summary generation failed: {e}")
            return "Image comparison completed with errors"

    def _generate_image_hash(self, image: np.ndarray) -> str:
        """Generate a hash for image caching."""
        try:
            # Create a simple hash based on image properties
            height, width = image.shape[:2]
            mean_values = np.mean(image, axis=(0, 1))
            hash_input = f"{height}x{width}:{mean_values.tolist()}"
            return hashlib.md5(hash_input.encode()).hexdigest()[:16]
        except Exception:
            return str(int(time.time() * 1000))

    def _update_stats(self, processing_time_ms: float) -> None:
        """Update performance statistics."""
        self._stats['comparisons_performed'] += 1
        self._stats['last_comparison_time'] = datetime.now().isoformat()

        # Update running average
        current_avg = self._stats['average_processing_time_ms']
        count = self._stats['comparisons_performed']
        self._stats['average_processing_time_ms'] = (current_avg * (count - 1) + processing_time_ms) / count

    def format_for_chatbot(self, comparison_result: YoloComparisonResult, user_message: str) -> str:
        """
        Format comparison result for optimal chatbot consumption.

        Args:
            comparison_result: YOLO comparison result
            user_message: User's message for context

        Returns:
            str: Formatted prompt for chatbot
        """
        try:
            prompt_parts = [
                f"User message: {user_message}",
                "",
                "YOLO Object Detection Comparison Analysis:",
                f"- Timestamp: {comparison_result.timestamp}",
                f"- Frame dimensions: {comparison_result.frame_dimensions['width']}x{comparison_result.frame_dimensions['height']}",
                f"- Detection confidence threshold: {comparison_result.confidence_threshold}",
                f"- Analysis duration: {comparison_result.analysis_duration_ms:.1f}ms",
                "",
                "Scene Comparison:",
                f"- Overall similarity: {comparison_result.scene_comparison.scene_similarity:.1%}",
                f"- Reference objects: {comparison_result.scene_comparison.reference_object_count}",
                f"- Current objects: {comparison_result.scene_comparison.current_object_count}",
                f"- Changes: {', '.join(comparison_result.scene_comparison.dominant_changes)}",
                ""
            ]

            if comparison_result.object_comparisons:
                prompt_parts.append("Detailed Object Analysis:")
                for i, comp in enumerate(comparison_result.object_comparisons, 1):
                    if comp.comparison_type == 'added' and comp.current_object:
                        prompt_parts.append(f"  {i}. NEW: {comp.current_object.class_name} "
                                          f"(confidence: {comp.current_object.score:.1%})")
                    elif comp.comparison_type == 'missing' and comp.reference_object:
                        prompt_parts.append(f"  {i}. MISSING: {comp.reference_object.class_name} "
                                          f"(was confidence: {comp.reference_object.score:.1%})")
                    elif comp.comparison_type in ['moved', 'changed'] and comp.current_object and comp.position_change:
                        obj_name = comp.current_object.class_name
                        distance = comp.position_change.get('distance_normalized', 0)
                        prompt_parts.append(f"  {i}. CHANGED: {obj_name} moved "
                                          f"{distance:.1%} of frame distance, "
                                          f"match score: {comp.match_score:.1%}")
                    elif comp.comparison_type == 'match' and comp.current_object:
                        prompt_parts.append(f"  {i}. UNCHANGED: {comp.current_object.class_name} "
                                          f"(match score: {comp.match_score:.1%})")
                prompt_parts.append("")

            prompt_parts.extend([
                f"Quick Summary: {comparison_result.chatbot_summary}",
                "",
                "Please provide an educational response that:",
                "1. Addresses the user's specific question or comment",
                "2. References the detected changes and object positions when relevant",
                "3. Explains what the comparison results mean in practical terms",
                "4. Provides helpful context about object detection confidence and accuracy"
            ])

            return "\n".join(prompt_parts)

        except Exception as e:
            self.logger.error(f"Chatbot formatting failed: {e}")
            return f"User message: {user_message}\n\nComparison analysis completed with errors."

    def get_reference_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current reference image."""
        if self._reference_detections is None:
            return None

        return {
            'image_hash': self._reference_image_hash,
            'dimensions': self._reference_frame_dimensions,
            'object_count': len(self._reference_detections),
            'objects': [
                {
                    'class_name': getattr(det, 'class_name', f'class_{det.class_id}'),
                    'confidence': det.score,
                    'bbox': det.bbox
                }
                for det in self._reference_detections
            ]
        }

    def clear_cache(self) -> None:
        """Clear comparison cache to free memory."""
        self._comparison_cache.clear()
        self._image_hash_cache.clear()
        self.logger.info("Comparison cache cleared")

    def update_configuration(self, **kwargs) -> None:
        """Update service configuration parameters."""
        if 'detection_confidence_threshold' in kwargs:
            self.confidence_threshold = kwargs['detection_confidence_threshold']
        if 'detection_iou_threshold' in kwargs:
            self.iou_threshold = kwargs['detection_iou_threshold']
        if 'master_tolerance_px' in kwargs:
            self.position_tolerance = kwargs['master_tolerance_px']

        # Clear cache when configuration changes
        self.clear_cache()
        self.logger.info("Configuration updated and cache cleared")