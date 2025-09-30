"""
Comprehensive Image Analysis Service for ChatBot Integration.

This service provides detailed image analysis including object detection,
scene analysis, and contextual information for AI-powered chat responses.
"""

import time
import logging
import threading
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import json

from ..core.entities import Detection, BBox
from ..core.exceptions import DetectionError
from .inference_service import InferenceService

logger = logging.getLogger(__name__)

@dataclass
class ObjectInfo:
    """Comprehensive object information for chat integration."""
    class_name: str
    confidence: float
    position: Dict[str, float]  # x, y, relative_x, relative_y
    dimensions: Dict[str, float]  # width, height, area, area_percentage
    angle: Optional[float] = None
    metadata: Dict[str, Any] = None

@dataclass
class SceneMetrics:
    """Scene-level analysis metrics."""
    total_objects: int
    dominant_colors: List[str]
    brightness: float
    contrast: float
    sharpness: float
    motion_detected: bool
    scene_complexity: str  # simple, moderate, complex

@dataclass
class ImageAnalysisResult:
    """Complete image analysis result for ChatBot integration."""
    timestamp: str
    frame_dimensions: Dict[str, int]  # width, height
    scene_description: str
    objects: List[ObjectInfo]
    scene_metrics: SceneMetrics
    analysis_duration_ms: float

class ImageAnalysisService:
    """
    Advanced image analysis service that combines object detection,
    scene analysis, and contextual information for ChatBot integration.
    """

    def __init__(self, inference_service: InferenceService, config):
        self.inference_service = inference_service
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Frame comparison for motion detection
        self._previous_frame = None
        self._motion_threshold = 25.0

        # Analysis cache to avoid redundant processing
        self._last_analysis_time = 0
        self._analysis_cache = None
        self._cache_duration = 2.0  # seconds

        # Color analysis setup
        self._color_names = {
            'red': ([0, 50, 50], [10, 255, 255]),
            'orange': ([10, 50, 50], [25, 255, 255]),
            'yellow': ([25, 50, 50], [35, 255, 255]),
            'green': ([35, 50, 50], [85, 255, 255]),
            'blue': ([85, 50, 50], [125, 255, 255]),
            'purple': ([125, 50, 50], [165, 255, 255]),
            'pink': ([165, 50, 50], [180, 255, 255]),
            'white': ([0, 0, 200], [180, 30, 255]),
            'gray': ([0, 0, 50], [180, 30, 200]),
            'black': ([0, 0, 0], [180, 255, 50])
        }

    def analyze_frame_comprehensive(self, frame: np.ndarray, user_message: str = "") -> ImageAnalysisResult:
        """
        Perform comprehensive image analysis for ChatBot integration.

        Args:
            frame: Input image frame
            user_message: User's chat message for context-aware analysis

        Returns:
            Complete analysis result with objects, scene metrics, and metadata
        """
        analysis_start = time.time()

        try:
            # Check cache for recent analysis
            current_time = time.time()
            if (current_time - self._last_analysis_time < self._cache_duration and
                self._analysis_cache is not None):
                self.logger.debug("Using cached analysis result")
                return self._analysis_cache

            self.logger.info("Starting comprehensive image analysis")

            # Basic frame information
            height, width = frame.shape[:2]
            timestamp = datetime.now().isoformat()

            # 1. Object Detection
            objects = self._detect_objects(frame, width, height)

            # 2. Scene Analysis
            scene_metrics = self._analyze_scene(frame)

            # 3. Motion Detection
            motion_detected = self._detect_motion(frame)
            scene_metrics.motion_detected = motion_detected

            # 4. Scene Description
            scene_description = self._generate_scene_description(objects, scene_metrics, width, height)

            # Create comprehensive result
            analysis_duration = (time.time() - analysis_start) * 1000

            result = ImageAnalysisResult(
                timestamp=timestamp,
                frame_dimensions={"width": width, "height": height},
                scene_description=scene_description,
                objects=objects,
                scene_metrics=scene_metrics,
                analysis_duration_ms=analysis_duration
            )

            # Cache the result
            self._analysis_cache = result
            self._last_analysis_time = current_time

            self.logger.info(f"Analysis completed in {analysis_duration:.1f}ms - Found {len(objects)} objects")

            return result

        except Exception as e:
            self.logger.error(f"Image analysis failed: {e}")
            # Return minimal result on error
            return ImageAnalysisResult(
                timestamp=datetime.now().isoformat(),
                frame_dimensions={"width": frame.shape[1], "height": frame.shape[0]},
                scene_description="Unable to analyze image",
                objects=[],
                scene_metrics=SceneMetrics(0, [], 0.5, 0.5, 0.5, False, "unknown"),
                analysis_duration_ms=(time.time() - analysis_start) * 1000
            )

    def _detect_objects(self, frame: np.ndarray, width: int, height: int) -> List[ObjectInfo]:
        """Detect objects and extract comprehensive information."""
        objects = []

        try:
            # Use existing inference service
            if self.inference_service and self.inference_service.is_loaded:
                detections = self.inference_service.predict(frame, conf_threshold=0.3)

                for detection in detections:
                    # Calculate position information
                    bbox = detection.bbox
                    center_x = bbox.x + bbox.width / 2
                    center_y = bbox.y + bbox.height / 2

                    position = {
                        "x": float(center_x),
                        "y": float(center_y),
                        "relative_x": float(center_x / width),
                        "relative_y": float(center_y / height)
                    }

                    # Calculate dimensions
                    area = bbox.width * bbox.height
                    area_percentage = (area / (width * height)) * 100

                    dimensions = {
                        "width": float(bbox.width),
                        "height": float(bbox.height),
                        "area": float(area),
                        "area_percentage": float(area_percentage)
                    }

                    # Calculate angle (basic orientation estimation)
                    angle = self._estimate_object_angle(frame, bbox)

                    # Additional metadata
                    metadata = {
                        "estimated_distance": self._estimate_distance(bbox, height),
                        "object_size": self._classify_object_size(area_percentage),
                        "position_description": self._describe_position(position["relative_x"], position["relative_y"])
                    }

                    obj_info = ObjectInfo(
                        class_name=detection.class_name,
                        confidence=float(detection.confidence),
                        position=position,
                        dimensions=dimensions,
                        angle=angle,
                        metadata=metadata
                    )

                    objects.append(obj_info)

        except Exception as e:
            self.logger.error(f"Object detection failed: {e}")

        return objects

    def _analyze_scene(self, frame: np.ndarray) -> SceneMetrics:
        """Analyze overall scene characteristics."""
        try:
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Brightness analysis
            brightness = np.mean(gray) / 255.0

            # Contrast analysis (standard deviation of grayscale)
            contrast = np.std(gray) / 255.0

            # Sharpness analysis (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness = min(laplacian_var / 1000.0, 1.0)  # Normalize

            # Dominant colors analysis
            dominant_colors = self._analyze_dominant_colors(hsv)

            # Scene complexity based on edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            if edge_density < 0.1:
                complexity = "simple"
            elif edge_density < 0.25:
                complexity = "moderate"
            else:
                complexity = "complex"

            return SceneMetrics(
                total_objects=0,  # Will be updated by caller
                dominant_colors=dominant_colors,
                brightness=float(brightness),
                contrast=float(contrast),
                sharpness=float(sharpness),
                motion_detected=False,  # Will be updated by motion detection
                scene_complexity=complexity
            )

        except Exception as e:
            self.logger.error(f"Scene analysis failed: {e}")
            return SceneMetrics(0, ["unknown"], 0.5, 0.5, 0.5, False, "unknown")

    def _analyze_dominant_colors(self, hsv_frame: np.ndarray) -> List[str]:
        """Analyze dominant colors in the frame."""
        dominant_colors = []

        try:
            for color_name, (lower, upper) in self._color_names.items():
                lower = np.array(lower)
                upper = np.array(upper)
                mask = cv2.inRange(hsv_frame, lower, upper)
                color_percentage = np.sum(mask > 0) / mask.size

                if color_percentage > 0.05:  # 5% threshold
                    dominant_colors.append(color_name)

            # Sort by prevalence (approximate)
            if not dominant_colors:
                dominant_colors = ["mixed"]

        except Exception as e:
            self.logger.error(f"Color analysis failed: {e}")
            dominant_colors = ["unknown"]

        return dominant_colors[:3]  # Return top 3 colors

    def _detect_motion(self, current_frame: np.ndarray) -> bool:
        """Detect motion by comparing with previous frame."""
        try:
            if self._previous_frame is None:
                self._previous_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                return False

            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

            # Calculate frame difference
            diff = cv2.absdiff(self._previous_frame, current_gray)
            mean_diff = np.mean(diff)

            # Update previous frame
            self._previous_frame = current_gray.copy()

            return mean_diff > self._motion_threshold

        except Exception as e:
            self.logger.error(f"Motion detection failed: {e}")
            return False

    def _estimate_object_angle(self, frame: np.ndarray, bbox: BBox) -> Optional[float]:
        """Estimate object orientation angle."""
        try:
            # Extract object region
            x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height)
            roi = frame[y:y+h, x:x+w]

            if roi.size == 0:
                return None

            # Convert to grayscale and find contours
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)

                # Fit ellipse to get orientation
                if len(largest_contour) >= 5:
                    ellipse = cv2.fitEllipse(largest_contour)
                    angle = ellipse[2]  # Angle of rotation
                    return float(angle)

            return None

        except Exception as e:
            self.logger.debug(f"Angle estimation failed: {e}")
            return None

    def _estimate_distance(self, bbox: BBox, frame_height: int) -> str:
        """Estimate relative distance based on object size and position."""
        try:
            # Simple distance estimation based on object height relative to frame
            relative_height = bbox.height / frame_height

            if relative_height > 0.6:
                return "very close"
            elif relative_height > 0.3:
                return "close"
            elif relative_height > 0.15:
                return "medium distance"
            elif relative_height > 0.05:
                return "far"
            else:
                return "very far"

        except Exception:
            return "unknown distance"

    def _classify_object_size(self, area_percentage: float) -> str:
        """Classify object size relative to frame."""
        if area_percentage > 50:
            return "very large"
        elif area_percentage > 25:
            return "large"
        elif area_percentage > 10:
            return "medium"
        elif area_percentage > 2:
            return "small"
        else:
            return "very small"

    def _describe_position(self, rel_x: float, rel_y: float) -> str:
        """Generate human-readable position description."""
        horizontal = "center"
        if rel_x < 0.3:
            horizontal = "left side"
        elif rel_x > 0.7:
            horizontal = "right side"

        vertical = "center"
        if rel_y < 0.3:
            vertical = "top"
        elif rel_y > 0.7:
            vertical = "bottom"

        if horizontal == "center" and vertical == "center":
            return "center of frame"
        elif horizontal == "center":
            return f"{vertical} center"
        elif vertical == "center":
            return f"{horizontal}"
        else:
            return f"{vertical} {horizontal}"

    def _generate_scene_description(self, objects: List[ObjectInfo],
                                  scene_metrics: SceneMetrics,
                                  width: int, height: int) -> str:
        """Generate natural language scene description."""
        try:
            scene_metrics.total_objects = len(objects)

            # Build description components
            parts = []

            # Basic scene info
            resolution_desc = f"{width}x{height}"
            parts.append(f"Scene captured at {resolution_desc} resolution")

            # Lighting and quality
            if scene_metrics.brightness > 0.7:
                lighting = "bright"
            elif scene_metrics.brightness > 0.4:
                lighting = "well-lit"
            else:
                lighting = "dim"

            if scene_metrics.sharpness > 0.6:
                quality = "sharp"
            elif scene_metrics.sharpness > 0.3:
                quality = "clear"
            else:
                quality = "slightly blurry"

            parts.append(f"with {lighting} lighting and {quality} image quality")

            # Objects description
            if len(objects) == 0:
                parts.append("No distinct objects detected")
            elif len(objects) == 1:
                obj = objects[0]
                parts.append(f"Contains 1 {obj.class_name} located in the {obj.metadata.get('position_description', 'frame')}")
            else:
                # Group objects by class
                object_counts = {}
                for obj in objects:
                    object_counts[obj.class_name] = object_counts.get(obj.class_name, 0) + 1

                object_desc = []
                for class_name, count in object_counts.items():
                    if count == 1:
                        object_desc.append(f"1 {class_name}")
                    else:
                        object_desc.append(f"{count} {class_name}s")

                if len(object_desc) == 1:
                    parts.append(f"Contains {object_desc[0]}")
                else:
                    parts.append(f"Contains {', '.join(object_desc[:-1])} and {object_desc[-1]}")

            # Colors and motion
            if scene_metrics.dominant_colors and scene_metrics.dominant_colors != ["unknown"]:
                color_desc = ", ".join(scene_metrics.dominant_colors[:2])
                parts.append(f"Predominantly {color_desc} tones")

            if scene_metrics.motion_detected:
                parts.append("Motion detected in scene")

            return ". ".join(parts) + "."

        except Exception as e:
            self.logger.error(f"Scene description generation failed: {e}")
            return "Scene analysis completed"

    def format_for_chatbot(self, analysis_result: ImageAnalysisResult, user_message: str) -> str:
        """
        Format analysis result for ChatBot consumption.

        Returns a structured prompt that includes both user message and image context.
        """
        try:
            # Create structured data for AI
            context_data = {
                "user_message": user_message,
                "image_analysis": {
                    "timestamp": analysis_result.timestamp,
                    "scene_description": analysis_result.scene_description,
                    "frame_info": analysis_result.frame_dimensions,
                    "objects": [asdict(obj) for obj in analysis_result.objects],
                    "scene_metrics": asdict(analysis_result.scene_metrics),
                    "analysis_time_ms": analysis_result.analysis_duration_ms
                }
            }

            # Create AI-friendly prompt
            prompt_parts = [
                f"User message: {user_message}",
                "",
                "Current visual context from live video feed:",
                f"- {analysis_result.scene_description}",
                ""
            ]

            if analysis_result.objects:
                prompt_parts.append("Detected objects:")
                for i, obj in enumerate(analysis_result.objects, 1):
                    position_desc = obj.metadata.get('position_description', 'unknown position')
                    size_desc = obj.metadata.get('object_size', 'unknown size')
                    confidence_pct = int(obj.confidence * 100)

                    prompt_parts.append(
                        f"  {i}. {obj.class_name} ({confidence_pct}% confidence) - "
                        f"{size_desc} object in {position_desc}, "
                        f"covering {obj.dimensions['area_percentage']:.1f}% of frame"
                    )
                prompt_parts.append("")

            # Scene characteristics
            metrics = analysis_result.scene_metrics
            prompt_parts.extend([
                "Scene characteristics:",
                f"- Lighting: {'bright' if metrics.brightness > 0.7 else 'normal' if metrics.brightness > 0.4 else 'dim'}",
                f"- Image quality: {'excellent' if metrics.sharpness > 0.7 else 'good' if metrics.sharpness > 0.4 else 'fair'}",
                f"- Complexity: {metrics.scene_complexity}",
                f"- Colors: {', '.join(metrics.dominant_colors)}",
                f"- Motion: {'detected' if metrics.motion_detected else 'static scene'}",
                ""
            ])

            prompt_parts.append(
                "Please respond to the user's message with full awareness of the current visual context. "
                "Reference specific objects, their positions, and scene characteristics when relevant to provide "
                "an intelligent, contextual response."
            )

            return "\n".join(prompt_parts)

        except Exception as e:
            self.logger.error(f"ChatBot formatting failed: {e}")
            return f"User message: {user_message}\n\nVisual context: Unable to analyze current video feed."

    async def analyze_frame_async(self, frame: np.ndarray, user_message: str = "") -> ImageAnalysisResult:
        """Asynchronous version of comprehensive frame analysis."""
        return await asyncio.to_thread(self.analyze_frame_comprehensive, frame, user_message)