"""
Detection Data Formatter for User-Friendly Display.

This module provides functions to format YOLO detection data, object coordinates,
angles, and other detection metadata into user-friendly chat responses.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from ..core.entities import Detection, BBox
from ..services.yolo_comparison_service import YoloComparisonResult, ObjectComparison, SceneComparison
from ..services.image_analysis_service import ImageAnalysisResult

logger = logging.getLogger(__name__)


def format_detection_data(
    detections: List[Detection],
    frame_dimensions: Tuple[int, int],
    yolo_comparison: Optional[YoloComparisonResult] = None,
    image_analysis: Optional[ImageAnalysisResult] = None,
    include_coordinates: bool = True,
    include_angles: bool = True,
    include_confidence: bool = True,
    include_size_info: bool = True
) -> str:
    """
    Format detection data into a user-friendly display format.

    Args:
        detections: List of Detection objects from YOLO
        frame_dimensions: Tuple of (width, height) for the frame
        yolo_comparison: Optional comparison results
        image_analysis: Optional scene analysis results
        include_coordinates: Whether to show bounding box coordinates
        include_angles: Whether to show angle information
        include_confidence: Whether to show confidence scores
        include_size_info: Whether to show object dimensions

    Returns:
        Formatted string with detection information
    """
    try:
        if not detections:
            return format_no_detections()

        # Start building the formatted output
        output_parts = []

        # Header with detection count
        output_parts.append(f"🔍 Object Detection Results:")
        output_parts.append("")
        output_parts.append(f"📦 Objects Found: {len(detections)}")

        # Add frame information
        width, height = frame_dimensions
        output_parts.append(f"📺 Frame Size: {width}×{height} pixels")
        output_parts.append("")

        # Format each detection
        for i, detection in enumerate(detections, 1):
            detection_box = format_single_detection(
                detection, i,
                include_coordinates=include_coordinates,
                include_angles=include_angles,
                include_confidence=include_confidence,
                include_size_info=include_size_info
            )
            output_parts.append(detection_box)
            output_parts.append("")

        # Add comparison summary if available
        if yolo_comparison:
            comparison_summary = format_comparison_summary(yolo_comparison)
            if comparison_summary:
                output_parts.append(comparison_summary)
                output_parts.append("")

        # Add scene analysis summary if available
        if image_analysis:
            scene_summary = format_scene_summary(image_analysis)
            if scene_summary:
                output_parts.append(scene_summary)
                output_parts.append("")

        return "\n".join(output_parts).rstrip()

    except Exception as e:
        logger.error(f"Failed to format detection data: {e}")
        return f"🔍 Object Detection Results:\n❌ Error formatting detection data: {str(e)}"


def format_single_detection(
    detection: Detection,
    index: int,
    include_coordinates: bool = True,
    include_angles: bool = True,
    include_confidence: bool = True,
    include_size_info: bool = True
) -> str:
    """
    Format a single detection object for display.

    Args:
        detection: Detection object to format
        index: Object index number for display
        include_coordinates: Whether to show coordinates
        include_angles: Whether to show angle information
        include_confidence: Whether to show confidence score
        include_size_info: Whether to show size information

    Returns:
        Formatted string for the detection
    """
    try:
        # Extract detection data
        x1, y1, x2, y2 = detection.bbox
        confidence_pct = detection.score * 100
        class_name = detection.class_name or f"class_{detection.class_id}"

        # Calculate derived information
        width = x2 - x1
        height = y2 - y1
        center_x = x1 + width // 2
        center_y = y1 + height // 2
        area = width * height

        # Build the detection box
        box_parts = []

        # Header line with class name and confidence
        header = f"│ {index}. {class_name}"
        if include_confidence:
            header += f" ({confidence_pct:.1f}% confidence)"
        header += " " * max(0, 45 - len(header)) + "│"

        box_parts.append("┌─────────────────────────────────────────────┐")
        box_parts.append(header)

        # Coordinates information
        if include_coordinates:
            coord_line = f"│    📍 Position: ({x1}, {y1}) to ({x2}, {y2})"
            coord_line += " " * max(0, 45 - len(coord_line)) + "│"
            box_parts.append(coord_line)

        # Size information
        if include_size_info:
            size_line = f"│    📏 Size: {width}×{height} pixels"
            size_line += " " * max(0, 45 - len(size_line)) + "│"
            box_parts.append(size_line)

            # Add area for larger objects
            if area > 10000:
                area_line = f"│    📐 Area: {area:,} pixels²"
                area_line += " " * max(0, 45 - len(area_line)) + "│"
                box_parts.append(area_line)

        # Center point
        if include_coordinates:
            center_line = f"│    🎯 Center: ({center_x}, {center_y})"
            center_line += " " * max(0, 45 - len(center_line)) + "│"
            box_parts.append(center_line)

        # Angle information (if available)
        if include_angles and detection.angle is not None:
            angle_line = f"│    🔄 Angle: {detection.angle:.1f}°"
            angle_line += " " * max(0, 45 - len(angle_line)) + "│"
            box_parts.append(angle_line)

        box_parts.append("└─────────────────────────────────────────────┘")

        return "\n".join(box_parts)

    except Exception as e:
        logger.error(f"Failed to format single detection: {e}")
        return f"┌─────────────────────────────────────────────┐\n│ {index}. Error formatting detection        │\n└─────────────────────────────────────────────┘"


def format_comparison_summary(yolo_comparison: YoloComparisonResult) -> str:
    """
    Format YOLO comparison results summary.

    Args:
        yolo_comparison: YOLO comparison results

    Returns:
        Formatted comparison summary
    """
    try:
        scene = yolo_comparison.scene_comparison

        summary_parts = [
            "📊 Scene Comparison Summary:",
            f"🔄 Similarity: {scene.scene_similarity:.1%}",
            f"📈 Reference Objects: {scene.reference_object_count}",
            f"📉 Current Objects: {scene.current_object_count}"
        ]

        # Changes summary
        changes = []
        if scene.objects_added > 0:
            changes.append(f"➕ {scene.objects_added} added")
        if scene.objects_removed > 0:
            changes.append(f"➖ {scene.objects_removed} removed")
        if scene.objects_moved > 0:
            changes.append(f"🔄 {scene.objects_moved} moved")
        if scene.objects_unchanged > 0:
            changes.append(f"✅ {scene.objects_unchanged} unchanged")

        if changes:
            summary_parts.append(f"📋 Changes: {', '.join(changes)}")

        # Dominant changes
        if scene.dominant_changes:
            summary_parts.append(f"🎯 Key Changes: {', '.join(scene.dominant_changes)}")

        return "\n".join(summary_parts)

    except Exception as e:
        logger.error(f"Failed to format comparison summary: {e}")
        return "📊 Scene Comparison: Error formatting comparison data"


def format_scene_summary(image_analysis: ImageAnalysisResult) -> str:
    """
    Format scene analysis summary.

    Args:
        image_analysis: Image analysis results

    Returns:
        Formatted scene summary
    """
    try:
        summary_parts = [
            "🌟 Scene Analysis:",
            f"📝 Description: {image_analysis.scene_description}"
        ]

        # Scene metrics
        metrics = image_analysis.scene_metrics

        # Lighting assessment
        if metrics.brightness > 0.7:
            lighting = "bright"
        elif metrics.brightness > 0.4:
            lighting = "normal"
        else:
            lighting = "dim"
        summary_parts.append(f"💡 Lighting: {lighting}")

        # Image quality
        if metrics.sharpness > 0.7:
            quality = "excellent"
        elif metrics.sharpness > 0.4:
            quality = "good"
        else:
            quality = "fair"
        summary_parts.append(f"📸 Quality: {quality}")

        # Motion detection
        motion_status = "Yes" if metrics.motion_detected else "No"
        summary_parts.append(f"🏃 Motion: {motion_status}")

        # Colors
        if hasattr(metrics, 'dominant_colors') and metrics.dominant_colors:
            colors = ', '.join(metrics.dominant_colors)
            summary_parts.append(f"🎨 Colors: {colors}")

        return "\n".join(summary_parts)

    except Exception as e:
        logger.error(f"Failed to format scene summary: {e}")
        return "🌟 Scene Analysis: Error formatting scene data"


def format_no_detections() -> str:
    """
    Format message when no objects are detected.

    Returns:
        Formatted no detection message
    """
    return """🔍 Object Detection Results:

📦 Objects Found: 0

┌─────────────────────────────────────────────┐
│ No objects detected in the current frame   │
│                                             │
│ 💡 Tips for better detection:              │
│ • Ensure good lighting                     │
│ • Position objects clearly in view         │
│ • Check if objects are in the model's      │
│   training data                            │
└─────────────────────────────────────────────┘"""


def format_detection_coordinates_only(detections: List[Detection]) -> str:
    """
    Format only coordinate information for quick reference.

    Args:
        detections: List of Detection objects

    Returns:
        Compact coordinate summary
    """
    try:
        if not detections:
            return "📍 No object coordinates available"

        coord_parts = ["📍 Object Coordinates:"]

        for i, detection in enumerate(detections, 1):
            x1, y1, x2, y2 = detection.bbox
            class_name = detection.class_name or f"class_{detection.class_id}"
            center_x = x1 + (x2 - x1) // 2
            center_y = y1 + (y2 - y1) // 2

            coord_parts.append(f"  {i}. {class_name}: ({x1},{y1}) to ({x2},{y2}) center ({center_x},{center_y})")

        return "\n".join(coord_parts)

    except Exception as e:
        logger.error(f"Failed to format coordinates: {e}")
        return "📍 Error formatting coordinates"


def format_detection_summary_compact(detections: List[Detection]) -> str:
    """
    Format a compact summary of detections for inline display.

    Args:
        detections: List of Detection objects

    Returns:
        Compact detection summary
    """
    try:
        if not detections:
            return "No objects detected"

        # Group by class name
        class_counts = {}
        for detection in detections:
            class_name = detection.class_name or f"class_{detection.class_id}"
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # Format as compact list
        summary_items = []
        for class_name, count in class_counts.items():
            if count == 1:
                summary_items.append(class_name)
            else:
                summary_items.append(f"{count} {class_name}s")

        return f"Detected: {', '.join(summary_items)}"

    except Exception as e:
        logger.error(f"Failed to format compact summary: {e}")
        return "Detection summary unavailable"