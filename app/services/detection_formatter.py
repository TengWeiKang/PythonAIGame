"""
Detection Data Formatter for Gemini AI Analysis.

This module provides a robust, type-safe formatter that converts YOLO detection data,
reference comparisons, and analysis context into structured prompts for Gemini AI.
Optimized for sub-2ms performance with comprehensive validation and intelligent
prompt generation.

Key Features:
- Type-safe data structures with comprehensive validation
- Context-aware prompt generation for different analysis types
- Flexible JSON metadata alongside markdown prompts
- Performance optimized for real-time webcam analysis
- Configurable formatting templates and output styles
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Protocol, Union, TypedDict
from enum import Enum
import json

from ..core.entities import Detection, ComparisonMetrics, BBox
from ..core.performance import performance_timer
from ..utils.validation import InputValidator, ValidationError, validate_user_prompt
from ..config.settings import Config

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of analysis requests for context-aware prompting."""
    DESCRIPTIVE = "descriptive"          # "What do you see?"
    COMPARATIVE = "comparative"          # When reference images available
    VERIFICATION = "verification"        # "Check if everything is in place"
    COUNT_ANALYSIS = "count_analysis"    # "How many objects are there?"
    EMPTY_FRAME = "empty_frame"         # When no objects detected
    CUSTOM = "custom"                   # User-defined analysis


class PromptTemplate(Protocol):
    """Protocol for prompt template objects."""
    def render(self, context: Dict[str, Any]) -> str: ...


@dataclass(slots=True, frozen=True)
class DetectionSummary:
    """Summary statistics for detection data."""
    total_objects: int
    unique_classes: int
    average_confidence: float
    frame_coverage_percent: float
    highest_confidence: float
    lowest_confidence: float
    class_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class FormattedDetection:
    """Formatted detection data for consistent display."""
    object_id: int
    class_name: str
    confidence: float
    position_description: str
    center_point: Tuple[int, int]
    size_pixels: Tuple[int, int]
    area_pixels: int
    aspect_ratio: float
    orientation: str
    bounding_box: BBox
    angle_degrees: Optional[float] = None


@dataclass(slots=True, frozen=True)
class FrameMetadata:
    """Frame metadata for context."""
    timestamp: str
    dimensions: Tuple[int, int]
    total_area: int
    analysis_type: AnalysisType
    processing_time_ms: Optional[float] = None


class DetectionMetadata(TypedDict, total=False):
    """Typed dict for detection metadata JSON output."""
    frame_metadata: Dict[str, Any]
    detection_summary: Dict[str, Any]
    formatted_detections: List[Dict[str, Any]]
    comparison_results: Optional[Dict[str, Any]]
    analysis_request: Dict[str, Any]
    generation_timestamp: str
    performance_metrics: Dict[str, float]


@dataclass(slots=True)
class FormattingConfig:
    """Configuration for formatting behavior."""
    include_coordinates: bool = True
    include_confidence: bool = True
    include_size_info: bool = True
    include_angles: bool = True
    max_objects_detailed: int = 20
    precision_digits: int = 1
    use_emoji_indicators: bool = True
    markdown_style: str = "structured"  # "structured", "compact", "verbose"
    performance_target_ms: float = 2.0


class DetectionDataFormatter:
    """
    High-performance, type-safe formatter for detection data to Gemini AI prompts.

    Converts YOLO detection data, reference comparisons, and user context into
    structured markdown prompts optimized for AI analysis with sub-2ms performance.
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize formatter with configuration.

        Args:
            config: Application configuration object
        """
        self.config = config
        self.formatting_config = FormattingConfig()
        self._template_cache: Dict[AnalysisType, str] = {}
        self._performance_metrics: Dict[str, float] = {}

        # Initialize prompt templates
        self._init_prompt_templates()

        logger.debug("DetectionDataFormatter initialized")

    def _init_prompt_templates(self) -> None:
        """Initialize prompt templates for different analysis types."""
        self._template_cache = {
            AnalysisType.DESCRIPTIVE: self._get_descriptive_template(),
            AnalysisType.COMPARATIVE: self._get_comparative_template(),
            AnalysisType.VERIFICATION: self._get_verification_template(),
            AnalysisType.COUNT_ANALYSIS: self._get_count_template(),
            AnalysisType.EMPTY_FRAME: self._get_empty_frame_template(),
            AnalysisType.CUSTOM: self._get_custom_template(),
        }

    @performance_timer("detection_formatter_format_for_gemini")
    def format_for_gemini(self,
                         user_message: str,
                         current_detections: List[Detection],
                         reference_detections: Optional[List[Detection]] = None,
                         comparison_results: Optional[ComparisonMetrics] = None,
                         context: Optional[Dict[str, Any]] = None,
                         frame_dimensions: Tuple[int, int] = (640, 480)) -> str:
        """
        Format detection data for Gemini AI analysis.

        Args:
            user_message: User's analysis request
            current_detections: Current frame detections
            reference_detections: Reference frame detections (optional)
            comparison_results: Comparison metrics (optional)
            context: Additional context information
            frame_dimensions: Frame dimensions (width, height)

        Returns:
            Formatted markdown prompt for Gemini AI

        Raises:
            ValidationError: If input data is invalid
        """
        start_time = time.perf_counter()

        try:
            # Validate inputs
            sanitized_message = self._validate_user_message(user_message)
            validated_detections = self._validate_detections(current_detections)
            validated_context = self._validate_context(context or {})

            # Determine analysis type
            analysis_type = self._determine_analysis_type(
                sanitized_message, validated_detections, reference_detections
            )

            # Create frame metadata
            frame_metadata = FrameMetadata(
                timestamp=datetime.now().isoformat(),
                dimensions=frame_dimensions,
                total_area=frame_dimensions[0] * frame_dimensions[1],
                analysis_type=analysis_type
            )

            # Format detections
            formatted_detections = self._format_detections(validated_detections, frame_dimensions)
            detection_summary = self._create_detection_summary(validated_detections, frame_dimensions)

            # Build context for template
            template_context = {
                "user_message": sanitized_message,
                "frame_metadata": frame_metadata,
                "detection_summary": detection_summary,
                "formatted_detections": formatted_detections,
                "comparison_results": comparison_results,
                "reference_detections": reference_detections,
                "context": validated_context,
                "analysis_type": analysis_type,
            }

            # Generate structured prompt
            prompt = self._build_structured_prompt(template_context)

            # Track performance
            processing_time = (time.perf_counter() - start_time) * 1000
            self._performance_metrics["last_format_time_ms"] = processing_time

            if processing_time > self.formatting_config.performance_target_ms:
                logger.warning(f"Formatting took {processing_time:.2f}ms, target: {self.formatting_config.performance_target_ms}ms")

            logger.debug(f"Successfully formatted prompt in {processing_time:.2f}ms")
            return prompt

        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Error formatting detection data after {processing_time:.2f}ms: {e}")
            raise ValidationError(f"Failed to format detection data: {e}") from e

    def create_json_metadata(self,
                           user_message: str,
                           current_detections: List[Detection],
                           reference_detections: Optional[List[Detection]] = None,
                           comparison_results: Optional[ComparisonMetrics] = None,
                           context: Optional[Dict[str, Any]] = None,
                           frame_dimensions: Tuple[int, int] = (640, 480)) -> DetectionMetadata:
        """
        Create JSON metadata for logging and analysis.

        Args:
            user_message: User's analysis request
            current_detections: Current frame detections
            reference_detections: Reference frame detections (optional)
            comparison_results: Comparison metrics (optional)
            context: Additional context information
            frame_dimensions: Frame dimensions (width, height)

        Returns:
            Structured metadata dictionary
        """
        try:
            # Validate inputs
            sanitized_message = self._validate_user_message(user_message)
            validated_detections = self._validate_detections(current_detections)

            # Create metadata components
            frame_metadata = FrameMetadata(
                timestamp=datetime.now().isoformat(),
                dimensions=frame_dimensions,
                total_area=frame_dimensions[0] * frame_dimensions[1],
                analysis_type=self._determine_analysis_type(
                    sanitized_message, validated_detections, reference_detections
                )
            )

            formatted_detections = self._format_detections(validated_detections, frame_dimensions)
            detection_summary = self._create_detection_summary(validated_detections, frame_dimensions)

            # Build metadata dict
            metadata: DetectionMetadata = {
                "frame_metadata": asdict(frame_metadata),
                "detection_summary": asdict(detection_summary),
                "formatted_detections": [asdict(det) for det in formatted_detections],
                "analysis_request": {
                    "original_message": sanitized_message,
                    "analysis_type": frame_metadata.analysis_type.value,
                    "has_reference": reference_detections is not None,
                    "has_comparison": comparison_results is not None,
                },
                "generation_timestamp": datetime.now().isoformat(),
                "performance_metrics": self._performance_metrics.copy(),
            }

            # Add optional components
            if comparison_results:
                metadata["comparison_results"] = asdict(comparison_results)

            return metadata

        except Exception as e:
            logger.error(f"Error creating JSON metadata: {e}")
            raise ValidationError(f"Failed to create metadata: {e}") from e

    def _build_structured_prompt(self, context: Dict[str, Any]) -> str:
        """Build structured markdown prompt from context."""
        analysis_type: AnalysisType = context["analysis_type"]
        template = self._template_cache.get(analysis_type, self._template_cache[AnalysisType.CUSTOM])

        # Build prompt sections
        sections = []

        # User query section
        sections.append(self._format_user_query_section(context["user_message"]))

        # Current frame analysis section
        sections.append(self._format_current_frame_section(context))

        # Reference comparison section (if applicable)
        if context.get("reference_detections") or context.get("comparison_results"):
            sections.append(self._format_comparison_section(context))

        # Analysis request section
        sections.append(self._create_analysis_request(context["user_message"], analysis_type, context))

        return "\n\n".join(sections)

    def _format_user_query_section(self, user_message: str) -> str:
        """Format the user query section."""
        return f"## User Query\n{user_message}"

    def _format_current_frame_section(self, context: Dict[str, Any]) -> str:
        """Format the current frame analysis section."""
        frame_metadata: FrameMetadata = context["frame_metadata"]
        detection_summary: DetectionSummary = context["detection_summary"]
        formatted_detections: List[FormattedDetection] = context["formatted_detections"]

        sections = []
        sections.append("## Current Frame Analysis")
        sections.append(f"**Timestamp:** {frame_metadata.timestamp}")
        sections.append(f"**Image Dimensions:** {frame_metadata.dimensions[0]}x{frame_metadata.dimensions[1]}")

        # Detection summary
        sections.append("\n### Detection Summary")
        sections.append(f"- Total Objects: {detection_summary.total_objects}")
        sections.append(f"- Unique Classes: {detection_summary.unique_classes}")
        if detection_summary.average_confidence > 0:
            sections.append(f"- Average Confidence: {detection_summary.average_confidence:.1f}%")
        sections.append(f"- Frame Coverage: {detection_summary.frame_coverage_percent:.1f}%")

        # Class distribution
        if detection_summary.class_distribution:
            class_dist = ", ".join([f"{cls}: {count}" for cls, count in detection_summary.class_distribution.items()])
            sections.append(f"- Class Distribution: {class_dist}")

        # Detailed objects
        if formatted_detections:
            sections.append("\n### Detected Objects")
            for det in formatted_detections[:self.formatting_config.max_objects_detailed]:
                sections.append(self._format_detection_section(det))

        return "\n".join(sections)

    def _format_detection_section(self, detection: FormattedDetection) -> str:
        """Format a single detection for display."""
        lines = []
        lines.append(f"**Object #{detection.object_id}: {detection.class_name}**")
        lines.append(f"- Confidence: {detection.confidence:.1f}%")
        lines.append(f"- Position: {detection.position_description}")
        lines.append(f"- Center: {detection.center_point}")
        lines.append(f"- Size: {detection.size_pixels[0]}x{detection.size_pixels[1]} pixels (Area: {detection.area_pixels:,} px²)")
        lines.append(f"- Aspect Ratio: {detection.aspect_ratio:.2f}")
        lines.append(f"- Orientation: {detection.orientation}")
        lines.append(f"- Bounding Box: {list(detection.bounding_box)}")

        if detection.angle_degrees is not None:
            lines.append(f"- Angle: {detection.angle_degrees:.1f}°")

        return "\n".join(lines)

    def _format_comparison_section(self, context: Dict[str, Any]) -> str:
        """Format comparison section if reference data is available."""
        sections = []
        sections.append("## Reference Comparison")

        comparison_results: Optional[ComparisonMetrics] = context.get("comparison_results")
        if comparison_results:
            sections.append(f"**Similarity Score:** {comparison_results.similarity_score:.1%}")
            sections.append(f"**Objects Added:** {comparison_results.objects_added}")
            sections.append(f"**Objects Removed:** {comparison_results.objects_removed}")
            sections.append(f"**Objects Moved:** {comparison_results.objects_moved}")
            sections.append(f"**Objects Unchanged:** {comparison_results.objects_unchanged}")
            sections.append(f"**Total Changes:** {comparison_results.total_changes}")
            sections.append(f"**Change Significance:** {comparison_results.change_significance}")

        return "\n".join(sections)

    def _create_analysis_request(self, user_message: str, analysis_type: AnalysisType, context: Dict[str, Any]) -> str:
        """Create the analysis request section."""
        sections = []
        sections.append("## Analysis Request")
        sections.append("Based on the detection data provided above, please:")

        if analysis_type == AnalysisType.DESCRIPTIVE:
            sections.append("1. Identify and describe all detected objects")
            sections.append("2. Explain their spatial relationships and positioning")
            sections.append("3. Note any interesting patterns or arrangements")
        elif analysis_type == AnalysisType.COMPARATIVE:
            sections.append("1. Compare current detections with reference data")
            sections.append("2. Identify and explain any differences")
            sections.append("3. Assess the significance of changes")
        elif analysis_type == AnalysisType.VERIFICATION:
            sections.append("1. Verify the presence and positioning of expected objects")
            sections.append("2. Check for any missing or misplaced items")
            sections.append("3. Provide a compliance assessment")
        elif analysis_type == AnalysisType.COUNT_ANALYSIS:
            sections.append("1. Count and categorize all detected objects")
            sections.append("2. Provide detailed quantity breakdowns")
            sections.append("3. Note any counting discrepancies or uncertainties")
        elif analysis_type == AnalysisType.EMPTY_FRAME:
            sections.append("1. Confirm no objects are detected")
            sections.append("2. Suggest possible reasons for empty detection")
            sections.append("3. Provide recommendations for better detection")

        sections.append(f"4. Address: '{user_message}'")

        return "\n".join(sections)

    def _validate_user_message(self, message: str) -> str:
        """Validate and sanitize user message."""
        return validate_user_prompt(message)

    def _validate_detections(self, detections: List[Detection]) -> List[Detection]:
        """Validate detection data."""
        if not isinstance(detections, list):
            raise ValidationError("Detections must be a list")

        validated = []
        for i, detection in enumerate(detections):
            if not isinstance(detection, Detection):
                raise ValidationError(f"Detection {i} is not a Detection object")

            # Validate detection fields
            if not isinstance(detection.bbox, tuple) or len(detection.bbox) != 4:
                raise ValidationError(f"Detection {i} has invalid bbox")

            if not (0.0 <= detection.score <= 1.0):
                raise ValidationError(f"Detection {i} has invalid confidence score")

            validated.append(detection)

        return validated

    def _validate_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate context dictionary."""
        if not isinstance(context, dict):
            raise ValidationError("Context must be a dictionary")

        # Sanitize string values in context
        sanitized = {}
        for key, value in context.items():
            if isinstance(value, str):
                sanitized[key] = InputValidator.sanitize_string_input(value, max_length=1000)
            else:
                sanitized[key] = value

        return sanitized

    def _determine_analysis_type(self, message: str, detections: List[Detection],
                                reference_detections: Optional[List[Detection]]) -> AnalysisType:
        """Determine the type of analysis based on input."""
        message_lower = message.lower()

        if not detections:
            return AnalysisType.EMPTY_FRAME

        if reference_detections:
            return AnalysisType.COMPARATIVE

        if any(word in message_lower for word in ["count", "how many", "number"]):
            return AnalysisType.COUNT_ANALYSIS

        if any(word in message_lower for word in ["check", "verify", "confirm", "ensure"]):
            return AnalysisType.VERIFICATION

        if any(word in message_lower for word in ["describe", "what", "see", "identify"]):
            return AnalysisType.DESCRIPTIVE

        return AnalysisType.CUSTOM

    def _format_detections(self, detections: List[Detection],
                          frame_dimensions: Tuple[int, int]) -> List[FormattedDetection]:
        """Format detections for consistent display."""
        formatted = []
        width, height = frame_dimensions

        for i, detection in enumerate(detections, 1):
            x1, y1, x2, y2 = detection.bbox
            center_x = x1 + (x2 - x1) // 2
            center_y = y1 + (y2 - y1) // 2
            obj_width = x2 - x1
            obj_height = y2 - y1
            area = obj_width * obj_height
            aspect_ratio = obj_width / obj_height if obj_height > 0 else 0

            # Determine position description
            if center_x < width * 0.33:
                h_pos = "left"
            elif center_x > width * 0.67:
                h_pos = "right"
            else:
                h_pos = "center"

            if center_y < height * 0.33:
                v_pos = "top"
            elif center_y > height * 0.67:
                v_pos = "bottom"
            else:
                v_pos = "middle"

            position_desc = f"{v_pos}-{h_pos}" if v_pos != "middle" or h_pos != "center" else "center"

            # Determine orientation
            if aspect_ratio > 1.5:
                orientation = "horizontal"
            elif aspect_ratio < 0.67:
                orientation = "vertical"
            else:
                orientation = "square"

            formatted_detection = FormattedDetection(
                object_id=i,
                class_name=detection.class_name or f"class_{detection.class_id}",
                confidence=detection.score * 100,
                position_description=position_desc,
                center_point=(center_x, center_y),
                size_pixels=(obj_width, obj_height),
                area_pixels=area,
                aspect_ratio=aspect_ratio,
                orientation=orientation,
                bounding_box=detection.bbox,
                angle_degrees=detection.angle
            )

            formatted.append(formatted_detection)

        return formatted

    def _create_detection_summary(self, detections: List[Detection],
                                 frame_dimensions: Tuple[int, int]) -> DetectionSummary:
        """Create summary statistics for detections."""
        if not detections:
            return DetectionSummary(
                total_objects=0,
                unique_classes=0,
                average_confidence=0.0,
                frame_coverage_percent=0.0,
                highest_confidence=0.0,
                lowest_confidence=0.0,
                class_distribution={}
            )

        confidences = [det.score * 100 for det in detections]
        total_area = 0
        class_counts: Dict[str, int] = {}

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            obj_area = (x2 - x1) * (y2 - y1)
            total_area += obj_area

            class_name = detection.class_name or f"class_{detection.class_id}"
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        frame_area = frame_dimensions[0] * frame_dimensions[1]
        coverage_percent = (total_area / frame_area) * 100 if frame_area > 0 else 0

        return DetectionSummary(
            total_objects=len(detections),
            unique_classes=len(class_counts),
            average_confidence=sum(confidences) / len(confidences),
            frame_coverage_percent=coverage_percent,
            highest_confidence=max(confidences),
            lowest_confidence=min(confidences),
            class_distribution=class_counts
        )

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for monitoring."""
        return self._performance_metrics.copy()

    def update_formatting_config(self, **kwargs) -> None:
        """Update formatting configuration."""
        for key, value in kwargs.items():
            if hasattr(self.formatting_config, key):
                setattr(self.formatting_config, key, value)
                logger.debug(f"Updated formatting config: {key} = {value}")

    # Template methods for different analysis types
    def _get_descriptive_template(self) -> str:
        return "descriptive_analysis"

    def _get_comparative_template(self) -> str:
        return "comparative_analysis"

    def _get_verification_template(self) -> str:
        return "verification_analysis"

    def _get_count_template(self) -> str:
        return "count_analysis"

    def _get_empty_frame_template(self) -> str:
        return "empty_frame_analysis"

    def _get_custom_template(self) -> str:
        return "custom_analysis"


__all__ = [
    "DetectionDataFormatter",
    "AnalysisType",
    "DetectionSummary",
    "FormattedDetection",
    "FrameMetadata",
    "DetectionMetadata",
    "FormattingConfig",
]