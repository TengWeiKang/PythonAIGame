"""
ASCII-compatible Detection Data Formatter for User-Friendly Display.

This module provides functions to format YOLO detection data, object coordinates,
angles, and other detection metadata into user-friendly chat responses using only ASCII characters.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


def format_detection_data_ascii(
    detections: List[Any],
    frame_dimensions: Tuple[int, int],
    yolo_comparison: Optional[Any] = None,
    image_analysis: Optional[Any] = None,
    include_coordinates: bool = True,
    include_angles: bool = True,
    include_confidence: bool = True,
    include_size_info: bool = True
) -> str:
    """
    Format detection data into a user-friendly display format using ASCII characters only.
    """
    try:
        if not detections:
            return format_no_detections_ascii()

        # Start building the formatted output
        output_parts = []

        # Header with detection count
        output_parts.append("OBJECT DETECTION RESULTS:")
        output_parts.append("")
        output_parts.append(f"Objects Found: {len(detections)}")

        # Add frame information
        width, height = frame_dimensions
        output_parts.append(f"Frame Size: {width}x{height} pixels")
        output_parts.append("")

        # Format each detection
        for i, detection in enumerate(detections, 1):
            detection_box = format_single_detection_ascii(
                detection, i,
                include_coordinates=include_coordinates,
                include_angles=include_angles,
                include_confidence=include_confidence,
                include_size_info=include_size_info
            )
            output_parts.append(detection_box)
            output_parts.append("")

        return "\n".join(output_parts).rstrip()

    except Exception as e:
        logger.error(f"Failed to format detection data: {e}")
        return f"OBJECT DETECTION RESULTS:\nERROR: Error formatting detection data: {str(e)}"


def format_single_detection_ascii(
    detection: Any,
    index: int,
    include_coordinates: bool = True,
    include_angles: bool = True,
    include_confidence: bool = True,
    include_size_info: bool = True
) -> str:
    """
    Format a single detection object for display using ASCII characters.
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
        header = f"| {index}. {class_name}"
        if include_confidence:
            header += f" ({confidence_pct:.1f}% confidence)"
        header += " " * max(0, 45 - len(header)) + "|"

        box_parts.append("+---------------------------------------------+")
        box_parts.append(header)

        # Coordinates information
        if include_coordinates:
            coord_line = f"|    Position: ({x1}, {y1}) to ({x2}, {y2})"
            coord_line += " " * max(0, 45 - len(coord_line)) + "|"
            box_parts.append(coord_line)

        # Size information
        if include_size_info:
            size_line = f"|    Size: {width}x{height} pixels"
            size_line += " " * max(0, 45 - len(size_line)) + "|"
            box_parts.append(size_line)

            # Add area for larger objects
            if area > 10000:
                area_line = f"|    Area: {area:,} pixels^2"
                area_line += " " * max(0, 45 - len(area_line)) + "|"
                box_parts.append(area_line)

        # Center point
        if include_coordinates:
            center_line = f"|    Center: ({center_x}, {center_y})"
            center_line += " " * max(0, 45 - len(center_line)) + "|"
            box_parts.append(center_line)

        # Angle information (if available)
        if include_angles and hasattr(detection, 'angle') and detection.angle is not None:
            angle_line = f"|    Angle: {detection.angle:.1f} degrees"
            angle_line += " " * max(0, 45 - len(angle_line)) + "|"
            box_parts.append(angle_line)

        box_parts.append("+---------------------------------------------+")

        return "\n".join(box_parts)

    except Exception as e:
        logger.error(f"Failed to format single detection: {e}")
        return f"+---------------------------------------------+\n| {index}. Error formatting detection        |\n+---------------------------------------------+"


def format_no_detections_ascii() -> str:
    """
    Format message when no objects are detected using ASCII characters.
    """
    return """OBJECT DETECTION RESULTS:

Objects Found: 0

+---------------------------------------------+
| No objects detected in the current frame   |
|                                             |
| Tips for better detection:                 |
| * Ensure good lighting                     |
| * Position objects clearly in view         |
| * Check if objects are in the model's      |
|   training data                            |
+---------------------------------------------+"""


def format_detection_coordinates_only_ascii(detections: List[Any]) -> str:
    """
    Format only coordinate information for quick reference using ASCII.
    """
    try:
        if not detections:
            return "No object coordinates available"

        coord_parts = ["Object Coordinates:"]

        for i, detection in enumerate(detections, 1):
            x1, y1, x2, y2 = detection.bbox
            class_name = detection.class_name or f"class_{detection.class_id}"
            center_x = x1 + (x2 - x1) // 2
            center_y = y1 + (y2 - y1) // 2

            coord_parts.append(f"  {i}. {class_name}: ({x1},{y1}) to ({x2},{y2}) center ({center_x},{center_y})")

        return "\n".join(coord_parts)

    except Exception as e:
        logger.error(f"Failed to format coordinates: {e}")
        return "Error formatting coordinates"


def format_detection_summary_compact_ascii(detections: List[Any]) -> str:
    """
    Format a compact summary of detections for inline display using ASCII.
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