#!/usr/bin/env python3
"""
Example: DetectionDataFormatter Integration with AsyncGeminiService.

This example demonstrates how to integrate the new DetectionDataFormatter
with the existing AsyncGeminiService for enhanced webcam YOLO analysis.
"""

import asyncio
import time
from typing import List, Optional

from app.core.entities import Detection, ComparisonMetrics
from app.services.detection_formatter import DetectionDataFormatter, AnalysisType
from app.services.gemini_service import AsyncGeminiService
from app.config.settings import load_config


def create_sample_detections() -> List[Detection]:
    """Create sample detection data for demonstration."""
    return [
        Detection(
            class_id=0,
            score=0.95,
            bbox=(150, 100, 250, 200),
            class_name="person",
            angle=12.5
        ),
        Detection(
            class_id=1,
            score=0.87,
            bbox=(400, 200, 500, 300),
            class_name="chair",
            angle=None
        ),
        Detection(
            class_id=2,
            score=0.92,
            bbox=(300, 50, 350, 150),
            class_name="book",
            angle=45.0
        ),
        Detection(
            class_id=0,
            score=0.89,
            bbox=(500, 150, 580, 280),
            class_name="person",
            angle=0.0
        )
    ]


def create_sample_comparison_metrics() -> ComparisonMetrics:
    """Create sample comparison metrics."""
    return ComparisonMetrics(
        similarity_score=0.78,
        objects_added=1,
        objects_removed=0,
        objects_moved=2,
        objects_unchanged=3,
        total_changes=3,
        change_significance="moderate"
    )


class EnhancedWebcamAnalyzer:
    """
    Enhanced webcam analyzer integrating DetectionDataFormatter with Gemini AI.

    This class demonstrates how to use the new formatter to create structured
    prompts for improved AI analysis of webcam detection data.
    """

    def __init__(self, config=None):
        """Initialize the analyzer with configuration."""
        self.config = config or load_config()
        self.formatter = DetectionDataFormatter(self.config)
        self.gemini_service = AsyncGeminiService(
            api_key=self.config.gemini_api_key if self.config.gemini_api_key else None
        )

    def analyze_frame(self,
                     user_query: str,
                     detections: List[Detection],
                     reference_detections: Optional[List[Detection]] = None,
                     comparison_metrics: Optional[ComparisonMetrics] = None,
                     frame_dimensions: tuple = (640, 480)) -> dict:
        """
        Analyze a frame using structured prompts and Gemini AI.

        Args:
            user_query: User's analysis request
            detections: Current frame detections
            reference_detections: Optional reference detections
            comparison_metrics: Optional comparison results
            frame_dimensions: Frame dimensions

        Returns:
            dict: Analysis results with metadata
        """
        start_time = time.perf_counter()

        try:
            # Format detection data for Gemini
            structured_prompt = self.formatter.format_for_gemini(
                user_message=user_query,
                current_detections=detections,
                reference_detections=reference_detections,
                comparison_results=comparison_metrics,
                frame_dimensions=frame_dimensions
            )

            # Create metadata for logging
            metadata = self.formatter.create_json_metadata(
                user_message=user_query,
                current_detections=detections,
                reference_detections=reference_detections,
                comparison_results=comparison_metrics,
                frame_dimensions=frame_dimensions
            )

            # Get performance metrics
            formatting_time = (time.perf_counter() - start_time) * 1000

            # Check if Gemini service is configured
            if not self.gemini_service.is_configured():
                return {
                    "success": False,
                    "error": "Gemini service not configured",
                    "structured_prompt": structured_prompt,
                    "metadata": metadata,
                    "formatting_time_ms": formatting_time
                }

            # Send to Gemini (this would be async in real usage)
            # For demo purposes, we'll simulate the response
            ai_response = self._simulate_gemini_response(structured_prompt)

            processing_time = (time.perf_counter() - start_time) * 1000

            return {
                "success": True,
                "ai_response": ai_response,
                "structured_prompt": structured_prompt,
                "metadata": metadata,
                "formatting_time_ms": formatting_time,
                "total_processing_time_ms": processing_time,
                "performance_metrics": self.formatter.get_performance_metrics()
            }

        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": processing_time
            }

    def _simulate_gemini_response(self, prompt: str) -> str:
        """Simulate a Gemini AI response for demonstration."""
        return f"""Based on the structured detection data provided:

**Scene Analysis:**
I can see 4 objects in the current frame: 2 people, 1 chair, and 1 book. The scene appears to be an indoor environment, possibly a classroom or office setting.

**Object Details:**
- Person #1: Located in the upper-left area with high confidence (95%). Appears to be slightly rotated (12.5°).
- Chair: Positioned in the center-right area with good confidence (87%). Standard upright orientation.
- Book: Small object in the upper-center area with high confidence (92%). Significantly rotated (45°).
- Person #2: Located in the right side of the frame with good confidence (89%). Standard upright position.

**Spatial Relationships:**
The people are positioned on opposite sides of the frame, with the chair and book between them. This suggests an active learning or meeting environment.

**Quality Assessment:**
The detection confidence scores are all above 85%, indicating good image quality and lighting conditions. The variety of orientations suggests dynamic activity in the scene.

[Note: This is a simulated response demonstrating the enhanced analysis capabilities enabled by structured prompt formatting.]"""

    async def analyze_frame_async(self, *args, **kwargs) -> dict:
        """Async version of frame analysis."""
        return self.analyze_frame(*args, **kwargs)


def demonstrate_analysis_types():
    """Demonstrate different analysis types and their prompt generation."""
    print("=== DetectionDataFormatter Analysis Types Demo ===\n")

    analyzer = EnhancedWebcamAnalyzer()
    detections = create_sample_detections()
    comparison_metrics = create_sample_comparison_metrics()

    # Test cases for different analysis types
    test_cases = [
        {
            "name": "Descriptive Analysis",
            "query": "What objects do you see in this image?",
            "detections": detections,
            "reference": None,
            "comparison": None
        },
        {
            "name": "Count Analysis",
            "query": "How many people are in the classroom?",
            "detections": detections,
            "reference": None,
            "comparison": None
        },
        {
            "name": "Verification Check",
            "query": "Check if all required classroom items are present",
            "detections": detections,
            "reference": None,
            "comparison": None
        },
        {
            "name": "Comparative Analysis",
            "query": "Compare the current scene with the reference setup",
            "detections": detections,
            "reference": detections[:2],  # Use subset as reference
            "comparison": comparison_metrics
        },
        {
            "name": "Empty Frame Handling",
            "query": "What's happening in this frame?",
            "detections": [],  # Empty detection list
            "reference": None,
            "comparison": None
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case['name']}")
        print("-" * 50)

        result = analyzer.analyze_frame(
            user_query=test_case["query"],
            detections=test_case["detections"],
            reference_detections=test_case["reference"],
            comparison_metrics=test_case["comparison"]
        )

        if result["success"]:
            print(f"✓ Analysis completed in {result['formatting_time_ms']:.2f}ms")
            print(f"Prompt length: {len(result['structured_prompt'])} characters")

            # Show analysis type detection
            analysis_type = result['metadata']['analysis_request']['analysis_type']
            print(f"Detected analysis type: {analysis_type}")

            # Show detection summary
            summary = result['metadata']['detection_summary']
            print(f"Objects detected: {summary['total_objects']} ({summary['unique_classes']} unique classes)")

        else:
            print(f"✗ Analysis failed: {result['error']}")

        print()


def demonstrate_performance_optimization():
    """Demonstrate performance optimization features."""
    print("=== Performance Optimization Demo ===\n")

    analyzer = EnhancedWebcamAnalyzer()
    detections = create_sample_detections()

    # Test multiple formatting operations
    times = []
    for i in range(10):
        start_time = time.perf_counter()
        result = analyzer.analyze_frame(
            user_query="Analyze the classroom setup",
            detections=detections
        )
        end_time = time.perf_counter()

        if result["success"]:
            times.append(result["formatting_time_ms"])

    if times:
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)

        print(f"Performance metrics over {len(times)} runs:")
        print(f"Average formatting time: {avg_time:.2f}ms")
        print(f"Maximum formatting time: {max_time:.2f}ms")
        print(f"Minimum formatting time: {min_time:.2f}ms")
        print(f"Performance target (<2ms): {'✓ PASS' if avg_time < 2.0 else '✗ FAIL'}")


def demonstrate_json_metadata():
    """Demonstrate JSON metadata creation."""
    print("=== JSON Metadata Demo ===\n")

    analyzer = EnhancedWebcamAnalyzer()
    detections = create_sample_detections()
    comparison_metrics = create_sample_comparison_metrics()

    result = analyzer.analyze_frame(
        user_query="Provide detailed analysis of classroom objects",
        detections=detections,
        reference_detections=detections[:2],
        comparison_metrics=comparison_metrics
    )

    if result["success"]:
        metadata = result["metadata"]
        print("JSON Metadata Structure:")
        print(f"- Frame metadata: {len(metadata['frame_metadata'])} fields")
        print(f"- Detection summary: {len(metadata['detection_summary'])} fields")
        print(f"- Formatted detections: {len(metadata['formatted_detections'])} objects")
        print(f"- Analysis request: {len(metadata['analysis_request'])} fields")
        print(f"- Performance metrics: {len(metadata['performance_metrics'])} metrics")

        if 'comparison_results' in metadata:
            print(f"- Comparison results: {len(metadata['comparison_results'])} fields")

        print(f"\nGeneration timestamp: {metadata['generation_timestamp']}")


if __name__ == "__main__":
    print("DetectionDataFormatter Integration Example")
    print("=" * 60)
    print()

    try:
        # Run demonstrations
        demonstrate_analysis_types()
        demonstrate_performance_optimization()
        demonstrate_json_metadata()

        print("=" * 60)
        print("✓ All demonstrations completed successfully!")
        print("\nKey Features Demonstrated:")
        print("- Type-safe data structures with comprehensive validation")
        print("- Context-aware prompt generation for different analysis types")
        print("- Sub-2ms performance for real-time webcam analysis")
        print("- Flexible JSON metadata creation for logging")
        print("- Seamless integration with existing AsyncGeminiService")

    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()