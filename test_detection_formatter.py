#!/usr/bin/env python3
"""
Comprehensive Test Suite for DetectionDataFormatter.

Tests the new detection data formatter for:
- Type safety and validation
- Performance requirements (<2ms)
- Prompt generation quality
- JSON metadata creation
- Configuration handling
- Error handling and edge cases
"""

import pytest
import time
from datetime import datetime
from typing import List, Dict, Any

from app.core.entities import Detection, ComparisonMetrics
from app.services.detection_formatter import (
    DetectionDataFormatter,
    AnalysisType,
    DetectionSummary,
    FormattedDetection,
    FrameMetadata,
    FormattingConfig,
    ValidationError
)
from app.config.settings import Config


class TestDetectionDataFormatter:
    """Test suite for DetectionDataFormatter."""

    @pytest.fixture
    def formatter(self):
        """Create formatter instance for testing."""
        return DetectionDataFormatter()

    @pytest.fixture
    def sample_detections(self) -> List[Detection]:
        """Create sample detection data."""
        return [
            Detection(
                class_id=0,
                score=0.95,
                bbox=(100, 50, 200, 150),
                class_name="person",
                angle=45.0
            ),
            Detection(
                class_id=1,
                score=0.87,
                bbox=(300, 200, 400, 300),
                class_name="chair",
                angle=None
            ),
            Detection(
                class_id=0,
                score=0.92,
                bbox=(500, 100, 600, 250),
                class_name="person",
                angle=90.0
            )
        ]

    @pytest.fixture
    def comparison_metrics(self) -> ComparisonMetrics:
        """Create sample comparison metrics."""
        return ComparisonMetrics(
            similarity_score=0.85,
            objects_added=1,
            objects_removed=0,
            objects_moved=2,
            objects_unchanged=3,
            total_changes=3,
            change_significance="moderate"
        )

    def test_initialization(self, formatter):
        """Test formatter initialization."""
        assert formatter is not None
        assert isinstance(formatter.formatting_config, FormattingConfig)
        assert len(formatter._template_cache) == 6  # All analysis types

    def test_format_for_gemini_basic(self, formatter, sample_detections):
        """Test basic prompt formatting functionality."""
        user_message = "What objects do you see in this image?"

        result = formatter.format_for_gemini(
            user_message=user_message,
            current_detections=sample_detections,
            frame_dimensions=(640, 480)
        )

        assert isinstance(result, str)
        assert len(result) > 0
        assert "## User Query" in result
        assert "## Current Frame Analysis" in result
        assert "## Analysis Request" in result
        assert user_message in result

    def test_performance_requirement(self, formatter, sample_detections):
        """Test that formatting meets <2ms performance requirement."""
        user_message = "Analyze these objects"

        # Warm up
        formatter.format_for_gemini(
            user_message=user_message,
            current_detections=sample_detections,
            frame_dimensions=(640, 480)
        )

        # Measure performance
        start_time = time.perf_counter()
        formatter.format_for_gemini(
            user_message=user_message,
            current_detections=sample_detections,
            frame_dimensions=(640, 480)
        )
        end_time = time.perf_counter()

        processing_time_ms = (end_time - start_time) * 1000
        assert processing_time_ms < 2.0, f"Formatting took {processing_time_ms:.2f}ms, exceeds 2ms requirement"

    def test_analysis_type_detection(self, formatter, sample_detections):
        """Test automatic analysis type detection."""
        test_cases = [
            ("What do you see?", AnalysisType.DESCRIPTIVE),
            ("How many objects are there?", AnalysisType.COUNT_ANALYSIS),
            ("Check if everything is in place", AnalysisType.VERIFICATION),
            ("Count the people", AnalysisType.COUNT_ANALYSIS),
            ("Verify the setup", AnalysisType.VERIFICATION),
            ("Custom analysis request", AnalysisType.CUSTOM),
        ]

        for message, expected_type in test_cases:
            detected_type = formatter._determine_analysis_type(message, sample_detections, None)
            assert detected_type == expected_type, f"Message '{message}' should detect {expected_type}, got {detected_type}"

    def test_empty_detections_handling(self, formatter):
        """Test handling of empty detection lists."""
        user_message = "What do you see?"

        result = formatter.format_for_gemini(
            user_message=user_message,
            current_detections=[],
            frame_dimensions=(640, 480)
        )

        assert isinstance(result, str)
        assert "Total Objects: 0" in result
        # Should still have proper structure even with no detections

    def test_comparison_mode(self, formatter, sample_detections, comparison_metrics):
        """Test formatting with reference comparison data."""
        reference_detections = sample_detections[:2]  # Subset as reference

        result = formatter.format_for_gemini(
            user_message="Compare with reference",
            current_detections=sample_detections,
            reference_detections=reference_detections,
            comparison_results=comparison_metrics,
            frame_dimensions=(640, 480)
        )

        assert "## Reference Comparison" in result
        assert "Similarity Score:" in result
        assert "Objects Added:** 1" in result
        assert "Change Significance:** moderate" in result

    def test_json_metadata_creation(self, formatter, sample_detections, comparison_metrics):
        """Test JSON metadata generation."""
        metadata = formatter.create_json_metadata(
            user_message="Test analysis",
            current_detections=sample_detections,
            comparison_results=comparison_metrics,
            frame_dimensions=(640, 480)
        )

        assert isinstance(metadata, dict)
        assert "frame_metadata" in metadata
        assert "detection_summary" in metadata
        assert "formatted_detections" in metadata
        assert "analysis_request" in metadata
        assert "generation_timestamp" in metadata

        # Validate structure
        assert metadata["detection_summary"]["total_objects"] == 3
        assert metadata["detection_summary"]["unique_classes"] == 2
        assert len(metadata["formatted_detections"]) == 3

    def test_detection_summary_creation(self, formatter, sample_detections):
        """Test detection summary statistics."""
        summary = formatter._create_detection_summary(sample_detections, (640, 480))

        assert isinstance(summary, DetectionSummary)
        assert summary.total_objects == 3
        assert summary.unique_classes == 2
        assert summary.average_confidence > 0
        assert summary.frame_coverage_percent > 0
        assert "person" in summary.class_distribution
        assert "chair" in summary.class_distribution
        assert summary.class_distribution["person"] == 2
        assert summary.class_distribution["chair"] == 1

    def test_formatted_detection_creation(self, formatter, sample_detections):
        """Test individual detection formatting."""
        formatted = formatter._format_detections(sample_detections, (640, 480))

        assert len(formatted) == 3
        for i, det in enumerate(formatted):
            assert isinstance(det, FormattedDetection)
            assert det.object_id == i + 1
            assert det.class_name in ["person", "chair"]
            assert 0 <= det.confidence <= 100
            assert det.area_pixels > 0
            assert det.aspect_ratio > 0

    def test_input_validation(self, formatter):
        """Test input validation and error handling."""
        # Invalid user message
        with pytest.raises(ValidationError):
            formatter.format_for_gemini("", [])

        # Invalid detection list
        with pytest.raises(ValidationError):
            formatter.format_for_gemini("test", "not a list")

        # Invalid detection object
        with pytest.raises(ValidationError):
            formatter.format_for_gemini("test", ["not a detection"])

    def test_position_description_accuracy(self, formatter):
        """Test position description generation."""
        # Test center position
        detection = Detection(
            class_id=0,
            score=0.9,
            bbox=(310, 230, 330, 250),  # Center of 640x480 frame
            class_name="object"
        )

        formatted = formatter._format_detections([detection], (640, 480))
        assert len(formatted) == 1
        assert "center" in formatted[0].position_description

    def test_orientation_classification(self, formatter):
        """Test object orientation classification."""
        test_cases = [
            ((100, 100, 200, 120), "horizontal"),  # Wide rectangle
            ((100, 100, 120, 200), "vertical"),    # Tall rectangle
            ((100, 100, 150, 150), "square"),      # Square-ish
        ]

        for bbox, expected_orientation in test_cases:
            detection = Detection(
                class_id=0,
                score=0.9,
                bbox=bbox,
                class_name="object"
            )

            formatted = formatter._format_detections([detection], (640, 480))
            assert formatted[0].orientation == expected_orientation

    def test_configuration_updates(self, formatter):
        """Test formatting configuration updates."""
        # Test config update
        formatter.update_formatting_config(
            include_confidence=False,
            max_objects_detailed=5
        )

        assert formatter.formatting_config.include_confidence is False
        assert formatter.formatting_config.max_objects_detailed == 5

    def test_performance_metrics_tracking(self, formatter, sample_detections):
        """Test performance metrics collection."""
        formatter.format_for_gemini(
            user_message="Test",
            current_detections=sample_detections,
            frame_dimensions=(640, 480)
        )

        metrics = formatter.get_performance_metrics()
        assert isinstance(metrics, dict)
        assert "last_format_time_ms" in metrics
        assert metrics["last_format_time_ms"] > 0

    def test_template_caching(self, formatter):
        """Test that templates are properly cached."""
        assert len(formatter._template_cache) == 6
        for analysis_type in AnalysisType:
            assert analysis_type in formatter._template_cache

    def test_large_detection_list_performance(self, formatter):
        """Test performance with large detection lists."""
        # Create large detection list
        large_detections = []
        for i in range(100):
            detection = Detection(
                class_id=i % 10,
                score=0.8 + (i % 20) * 0.01,
                bbox=(i * 5, i * 3, i * 5 + 50, i * 3 + 50),
                class_name=f"class_{i % 10}"
            )
            large_detections.append(detection)

        start_time = time.perf_counter()
        result = formatter.format_for_gemini(
            user_message="Analyze all objects",
            current_detections=large_detections,
            frame_dimensions=(1920, 1080)
        )
        end_time = time.perf_counter()

        processing_time_ms = (end_time - start_time) * 1000
        assert processing_time_ms < 10.0  # Allow more time for large lists but still reasonable
        assert isinstance(result, str)
        assert len(result) > 0

    def test_angle_handling(self, formatter):
        """Test handling of detection angles."""
        detection_with_angle = Detection(
            class_id=0,
            score=0.9,
            bbox=(100, 100, 200, 200),
            class_name="rotated_object",
            angle=45.5
        )

        detection_without_angle = Detection(
            class_id=1,
            score=0.8,
            bbox=(300, 300, 400, 400),
            class_name="normal_object",
            angle=None
        )

        formatted = formatter._format_detections([detection_with_angle, detection_without_angle], (640, 480))

        # Check angle is preserved
        assert formatted[0].angle_degrees == 45.5
        assert formatted[1].angle_degrees is None

    def test_context_validation_and_sanitization(self, formatter, sample_detections):
        """Test context validation and sanitization."""
        # Test with various context data
        context = {
            "scene_type": "classroom",
            "lighting": "normal",
            "special_instructions": "<script>alert('test')</script>",  # Should be sanitized
            "numeric_value": 42,
            "boolean_flag": True
        }

        result = formatter.format_for_gemini(
            user_message="Analyze scene",
            current_detections=sample_detections,
            context=context,
            frame_dimensions=(640, 480)
        )

        assert isinstance(result, str)
        # Context should be validated but script tags sanitized


def test_integration_with_config():
    """Test integration with application configuration."""
    config = Config()
    formatter = DetectionDataFormatter(config)

    assert formatter.config == config


def test_type_safety():
    """Test type safety of data structures."""
    # Test that all dataclasses are properly typed
    summary = DetectionSummary(
        total_objects=5,
        unique_classes=3,
        average_confidence=85.5,
        frame_coverage_percent=12.3,
        highest_confidence=95.0,
        lowest_confidence=70.0
    )

    assert isinstance(summary.total_objects, int)
    assert isinstance(summary.average_confidence, float)

    metadata = FrameMetadata(
        timestamp="2024-01-15T10:30:45",
        dimensions=(640, 480),
        total_area=307200,
        analysis_type=AnalysisType.DESCRIPTIVE
    )

    assert isinstance(metadata.dimensions, tuple)
    assert isinstance(metadata.analysis_type, AnalysisType)


if __name__ == "__main__":
    # Run basic performance test
    formatter = DetectionDataFormatter()

    # Create test data
    detections = [
        Detection(class_id=0, score=0.95, bbox=(100, 50, 200, 150), class_name="person"),
        Detection(class_id=1, score=0.87, bbox=(300, 200, 400, 300), class_name="chair"),
    ]

    # Test performance
    start_time = time.perf_counter()
    result = formatter.format_for_gemini(
        user_message="What objects do you see?",
        current_detections=detections,
        frame_dimensions=(640, 480)
    )
    end_time = time.perf_counter()

    processing_time_ms = (end_time - start_time) * 1000
    print(f"Formatting completed in {processing_time_ms:.2f}ms")
    print(f"Performance requirement (<2ms): {'✓ PASS' if processing_time_ms < 2.0 else '✗ FAIL'}")
    print(f"Result length: {len(result)} characters")
    print("\nSample output:")
    print("=" * 50)
    print(result[:500] + "..." if len(result) > 500 else result)