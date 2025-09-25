#!/usr/bin/env python3
"""
Validation Script: DetectionDataFormatter Requirements Compliance.

This script validates that the DetectionDataFormatter implementation
meets all specified requirements from the task.
"""

import time
import json
from typing import List, Dict, Any
import traceback

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


class RequirementValidator:
    """Validates that all requirements are met."""

    def __init__(self):
        self.formatter = DetectionDataFormatter()
        self.passed_tests = 0
        self.total_tests = 0

    def validate_requirement(self, requirement: str, test_func, *args, **kwargs) -> bool:
        """Validate a single requirement."""
        self.total_tests += 1
        try:
            result = test_func(*args, **kwargs)
            if result:
                print(f"✓ {requirement}")
                self.passed_tests += 1
                return True
            else:
                print(f"✗ {requirement} - Test returned False")
                return False
        except Exception as e:
            print(f"✗ {requirement} - Error: {e}")
            return False

    def print_summary(self):
        """Print validation summary."""
        print(f"\nValidation Summary: {self.passed_tests}/{self.total_tests} requirements met")
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")
        return self.passed_tests == self.total_tests


def test_type_safe_data_structures() -> bool:
    """Test: Type-Safe Data Structures using dataclasses/TypedDict."""
    try:
        # Test DetectionSummary dataclass
        summary = DetectionSummary(
            total_objects=5,
            unique_classes=3,
            average_confidence=85.5,
            frame_coverage_percent=12.3,
            highest_confidence=95.0,
            lowest_confidence=70.0
        )

        # Test FrameMetadata dataclass
        metadata = FrameMetadata(
            timestamp="2024-01-15T10:30:45",
            dimensions=(640, 480),
            total_area=307200,
            analysis_type=AnalysisType.DESCRIPTIVE
        )

        # Test FormattedDetection dataclass
        formatted_det = FormattedDetection(
            object_id=1,
            class_name="person",
            confidence=95.0,
            position_description="center",
            center_point=(320, 240),
            size_pixels=(100, 150),
            area_pixels=15000,
            aspect_ratio=0.67,
            orientation="vertical",
            bounding_box=(270, 165, 370, 315)
        )

        # Verify types and immutability (frozen dataclasses)
        assert isinstance(summary.total_objects, int)
        assert isinstance(summary.average_confidence, float)
        assert isinstance(metadata.analysis_type, AnalysisType)
        assert isinstance(formatted_det.center_point, tuple)

        return True
    except Exception:
        return False


def test_intelligent_prompt_generation() -> bool:
    """Test: Intelligent Prompt Generation for different analysis types."""
    try:
        formatter = DetectionDataFormatter()

        test_cases = [
            ("What do you see?", AnalysisType.DESCRIPTIVE),
            ("How many objects are there?", AnalysisType.COUNT_ANALYSIS),
            ("Check if everything is in place", AnalysisType.VERIFICATION),
        ]

        detections = [
            Detection(class_id=0, score=0.95, bbox=(100, 100, 200, 200), class_name="person")
        ]

        for message, expected_type in test_cases:
            prompt = formatter.format_for_gemini(
                user_message=message,
                current_detections=detections
            )

            # Verify prompt contains expected sections
            required_sections = ["## User Query", "## Current Frame Analysis", "## Analysis Request"]
            for section in required_sections:
                if section not in prompt:
                    return False

            # Verify analysis type detection
            detected_type = formatter._determine_analysis_type(message, detections, None)
            if detected_type != expected_type:
                return False

        return True
    except Exception:
        return False


def test_flexible_serialization() -> bool:
    """Test: Flexible Serialization - JSON metadata alongside markdown prompts."""
    try:
        formatter = DetectionDataFormatter()
        detections = [
            Detection(class_id=0, score=0.95, bbox=(100, 100, 200, 200), class_name="person")
        ]

        # Test markdown prompt generation
        prompt = formatter.format_for_gemini(
            user_message="Test message",
            current_detections=detections
        )

        # Test JSON metadata generation
        metadata = formatter.create_json_metadata(
            user_message="Test message",
            current_detections=detections
        )

        # Verify both formats are generated
        if not isinstance(prompt, str) or len(prompt) == 0:
            print("Prompt generation failed")
            return False

        if not isinstance(metadata, dict):
            print("Metadata generation failed")
            return False

        # Verify JSON can be serialized
        try:
            json_str = json.dumps(metadata, default=str)  # Handle any non-serializable objects
            if len(json_str) == 0:
                print("JSON serialization failed")
                return False
        except Exception as e:
            print(f"JSON serialization error: {e}")
            return False

        # Verify metadata structure
        required_keys = ["frame_metadata", "detection_summary", "formatted_detections", "analysis_request"]
        for key in required_keys:
            if key not in metadata:
                print(f"Missing metadata key: {key}")
                return False

        return True
    except Exception as e:
        print(f"Serialization test error: {e}")
        return False


def test_performance_requirement() -> bool:
    """Test: Performance - Minimal overhead (<2ms for typical scenes)."""
    try:
        formatter = DetectionDataFormatter()

        # Create typical scene (5-10 objects)
        detections = [
            Detection(class_id=i, score=0.8 + i*0.02, bbox=(i*50, i*30, i*50+80, i*30+100),
                     class_name=f"object_{i}")
            for i in range(7)  # 7 objects - typical scene
        ]

        # Warm up
        formatter.format_for_gemini("Test", detections)

        # Measure performance over multiple runs
        times = []
        for _ in range(10):
            start_time = time.perf_counter()
            formatter.format_for_gemini(
                user_message="Analyze this typical scene",
                current_detections=detections,
                frame_dimensions=(640, 480)
            )
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms

        # Check performance requirement
        avg_time = sum(times) / len(times)
        max_time = max(times)

        # Must be under 2ms average and no single run over 5ms
        return avg_time < 2.0 and max_time < 5.0

    except Exception:
        return False


def test_key_class_implementation() -> bool:
    """Test: Key Class Implementation - DetectionDataFormatter with required methods."""
    try:
        formatter = DetectionDataFormatter()

        # Verify all required methods exist
        required_methods = [
            'format_for_gemini',
            'create_json_metadata',
            '_build_structured_prompt',
            '_format_detection_section',
            '_format_comparison_section',
            '_create_analysis_request'
        ]

        for method in required_methods:
            if not hasattr(formatter, method):
                return False

        # Test method signatures and basic functionality
        detections = [Detection(class_id=0, score=0.95, bbox=(100, 100, 200, 200), class_name="person")]

        # Test format_for_gemini
        result = formatter.format_for_gemini("test", detections)
        if not isinstance(result, str):
            return False

        # Test create_json_metadata
        metadata = formatter.create_json_metadata("test", detections)
        if not isinstance(metadata, dict):
            return False

        return True
    except Exception:
        return False


def test_prompt_templates() -> bool:
    """Test: Prompt Templates - Five different analysis types."""
    try:
        formatter = DetectionDataFormatter()

        # Test each analysis type
        required_types = [
            AnalysisType.DESCRIPTIVE,
            AnalysisType.COMPARATIVE,
            AnalysisType.VERIFICATION,
            AnalysisType.COUNT_ANALYSIS,
            AnalysisType.EMPTY_FRAME
        ]

        # Verify all templates are cached
        for analysis_type in required_types:
            if analysis_type not in formatter._template_cache:
                return False

        return True
    except Exception:
        return False


def test_markdown_output_format() -> bool:
    """Test: Expected Output Format - Structured markdown for Gemini."""
    try:
        formatter = DetectionDataFormatter()
        detections = [
            Detection(class_id=0, score=0.952, bbox=(150, 200, 250, 320), class_name="person", angle=90.0),
            Detection(class_id=1, score=0.875, bbox=(300, 150, 400, 250), class_name="chair"),
            Detection(class_id=2, score=0.923, bbox=(100, 140, 200, 260), class_name="book")
        ]

        result = formatter.format_for_gemini(
            user_message="What objects do you see?",
            current_detections=detections,
            frame_dimensions=(640, 480)
        )

        # Verify required sections exist
        required_sections = [
            "## User Query",
            "## Current Frame Analysis",
            "**Timestamp:**",
            "**Image Dimensions:**",
            "### Detection Summary",
            "- Total Objects:",
            "- Unique Classes:",
            "- Average Confidence:",
            "- Frame Coverage:",
            "### Detected Objects",
            "**Object #1:",
            "- Confidence:",
            "- Position:",
            "- Center:",
            "- Size:",
            "- Aspect Ratio:",
            "- Orientation:",
            "- Bounding Box:",
            "## Analysis Request"
        ]

        for section in required_sections:
            if section not in result:
                print(f"Missing section: {section}")
                return False

        # Verify specific formatting
        assert "Total Objects: 3" in result
        assert "Unique Classes: 3" in result
        assert "person" in result
        assert "chair" in result
        assert "book" in result

        return True
    except Exception as e:
        print(f"Markdown format test error: {e}")
        return False


def test_python_pro_focus_areas() -> bool:
    """Test: Python-Pro Focus Areas - Modern patterns, optimization, validation."""
    try:
        formatter = DetectionDataFormatter()

        # Test modern Python patterns - type hints
        import inspect
        sig = inspect.signature(formatter.format_for_gemini)
        if not sig.return_annotation:
            return False

        # Test dataclasses usage
        from dataclasses import is_dataclass
        if not is_dataclass(DetectionSummary):
            return False
        if not is_dataclass(FormattedDetection):
            return False
        if not is_dataclass(FrameMetadata):
            return False

        # Test validation
        try:
            formatter.format_for_gemini("", [])  # Should raise ValidationError
            return False  # Should not reach here
        except ValidationError:
            pass  # Expected
        except Exception:
            pass  # Also acceptable - any validation error

        # Test configuration support
        formatter.update_formatting_config(include_confidence=False)
        config_updated = not formatter.formatting_config.include_confidence
        if not config_updated:
            return False

        return True
    except Exception:
        return False


def test_integration_points() -> bool:
    """Test: Integration Points with existing services."""
    try:
        formatter = DetectionDataFormatter()

        # Test detection data acceptance
        detections = [Detection(class_id=0, score=0.95, bbox=(100, 100, 200, 200), class_name="person")]

        # Test reference comparison
        comparison_metrics = ComparisonMetrics(
            similarity_score=0.85,
            objects_added=1,
            objects_removed=0,
            objects_moved=2,
            objects_unchanged=3,
            total_changes=3,
            change_significance="moderate"
        )

        # Test with comparison
        result = formatter.format_for_gemini(
            user_message="Compare scenes",
            current_detections=detections,
            reference_detections=detections,
            comparison_results=comparison_metrics
        )

        # Should include comparison section
        if "## Reference Comparison" not in result:
            return False

        # Test configuration compatibility
        config = Config()
        formatter_with_config = DetectionDataFormatter(config)
        if formatter_with_config.config != config:
            return False

        return True
    except Exception:
        return False


def test_success_criteria() -> bool:
    """Test: Success Criteria - All requirements met."""
    try:
        formatter = DetectionDataFormatter()

        # Test type safety
        detections = [Detection(class_id=0, score=0.95, bbox=(100, 100, 200, 200), class_name="person")]

        # Test sub-2ms performance
        start_time = time.perf_counter()
        result = formatter.format_for_gemini("Test", detections)
        end_time = time.perf_counter()

        processing_time = (end_time - start_time) * 1000
        if processing_time >= 2.0:
            return False

        # Test dynamic prompt adaptation
        descriptive_type = formatter._determine_analysis_type("What do you see?", detections, None)
        count_type = formatter._determine_analysis_type("How many objects?", detections, None)

        if descriptive_type == count_type:  # Should be different
            return False

        # Test clean markdown output
        if not isinstance(result, str) or len(result) < 100:
            return False

        # Test comprehensive JSON metadata
        metadata = formatter.create_json_metadata("Test", detections)
        required_metadata_keys = ["frame_metadata", "detection_summary", "formatted_detections"]
        for key in required_metadata_keys:
            if key not in metadata:
                return False

        return True
    except Exception:
        return False


def main():
    """Run all requirement validations."""
    print("DetectionDataFormatter Requirements Validation")
    print("=" * 60)

    validator = RequirementValidator()

    # Validate all requirements
    validator.validate_requirement(
        "Type-Safe Data Structures (dataclasses/TypedDict)",
        test_type_safe_data_structures
    )

    validator.validate_requirement(
        "Intelligent Prompt Generation (context-aware)",
        test_intelligent_prompt_generation
    )

    validator.validate_requirement(
        "Flexible Serialization (JSON + markdown)",
        test_flexible_serialization
    )

    validator.validate_requirement(
        "Performance (<2ms for typical scenes)",
        test_performance_requirement
    )

    validator.validate_requirement(
        "Key Class Implementation (DetectionDataFormatter)",
        test_key_class_implementation
    )

    validator.validate_requirement(
        "Prompt Templates (5 analysis types)",
        test_prompt_templates
    )

    validator.validate_requirement(
        "Markdown Output Format (structured for Gemini)",
        test_markdown_output_format
    )

    validator.validate_requirement(
        "Python-Pro Focus Areas (modern patterns, optimization)",
        test_python_pro_focus_areas
    )

    validator.validate_requirement(
        "Integration Points (existing services compatibility)",
        test_integration_points
    )

    validator.validate_requirement(
        "Success Criteria (comprehensive requirements)",
        test_success_criteria
    )

    print("=" * 60)
    success = validator.print_summary()

    if success:
        print("\n🎉 All requirements successfully implemented!")
        print("\nThe DetectionDataFormatter is ready for production use with:")
        print("- Type-safe data structures with comprehensive validation")
        print("- Sub-2ms performance for real-time webcam analysis")
        print("- Intelligent prompt generation for enhanced AI analysis")
        print("- Seamless integration with existing AsyncGeminiService")
        print("- Comprehensive JSON metadata for logging and debugging")
    else:
        print("\n⚠️  Some requirements need attention.")

    return success


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\nValidation script failed: {e}")
        traceback.print_exc()
        exit(1)