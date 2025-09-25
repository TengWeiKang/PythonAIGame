#!/usr/bin/env python3
"""
Integration Guide: Using DetectionDataFormatter with Existing Services.

This guide shows how to integrate the new DetectionDataFormatter into
the existing webcam YOLO analysis workflow.
"""

from typing import List, Optional, Dict, Any
import logging

from app.core.entities import Detection, ComparisonMetrics
from app.services.detection_formatter import DetectionDataFormatter, AnalysisType
from app.services.gemini_service import AsyncGeminiService
from app.config.settings import load_config, Config

logger = logging.getLogger(__name__)


class WebcamAnalysisService:
    """
    Enhanced webcam analysis service integrating DetectionDataFormatter.

    This service demonstrates the recommended pattern for integrating the
    new formatter with existing YOLO detection and Gemini AI services.
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize the service with configuration."""
        self.config = config or load_config()

        # Initialize the new formatter
        self.formatter = DetectionDataFormatter(self.config)

        # Initialize existing Gemini service
        self.gemini_service = AsyncGeminiService(
            api_key=self.config.gemini_api_key if self.config.gemini_api_key else None,
            model=self.config.gemini_model,
            timeout=self.config.gemini_timeout,
            temperature=self.config.gemini_temperature,
            max_tokens=self.config.gemini_max_tokens
        )

        logger.info("WebcamAnalysisService initialized with DetectionDataFormatter")

    def analyze_webcam_frame(self,
                           user_query: str,
                           detections: List[Detection],
                           frame_dimensions: tuple = (640, 480),
                           reference_detections: Optional[List[Detection]] = None,
                           comparison_metrics: Optional[ComparisonMetrics] = None,
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze a webcam frame with enhanced structured prompting.

        Args:
            user_query: User's analysis request
            detections: YOLO detection results
            frame_dimensions: Frame width and height
            reference_detections: Optional reference frame detections
            comparison_metrics: Optional comparison results
            context: Additional context information

        Returns:
            Analysis results dictionary
        """
        try:
            # Step 1: Format detection data for Gemini
            structured_prompt = self.formatter.format_for_gemini(
                user_message=user_query,
                current_detections=detections,
                reference_detections=reference_detections,
                comparison_results=comparison_metrics,
                context=context,
                frame_dimensions=frame_dimensions
            )

            # Step 2: Create metadata for logging and debugging
            metadata = self.formatter.create_json_metadata(
                user_message=user_query,
                current_detections=detections,
                reference_detections=reference_detections,
                comparison_results=comparison_metrics,
                context=context,
                frame_dimensions=frame_dimensions
            )

            # Step 3: Check if Gemini service is available
            if not self.gemini_service.is_configured():
                logger.warning("Gemini service not configured, returning structured prompt only")
                return {
                    "success": True,
                    "ai_response": None,
                    "structured_prompt": structured_prompt,
                    "metadata": metadata,
                    "service_available": False
                }

            # Step 4: Send structured prompt to Gemini
            try:
                ai_response = self.gemini_service.send_message(structured_prompt)

                return {
                    "success": True,
                    "ai_response": ai_response,
                    "structured_prompt": structured_prompt,
                    "metadata": metadata,
                    "service_available": True,
                    "performance_metrics": self.formatter.get_performance_metrics()
                }

            except Exception as gemini_error:
                logger.error(f"Gemini API error: {gemini_error}")
                return {
                    "success": False,
                    "error": f"AI service error: {gemini_error}",
                    "structured_prompt": structured_prompt,
                    "metadata": metadata,
                    "service_available": True
                }

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "service_available": False
            }

    def update_formatter_config(self, **kwargs) -> None:
        """Update formatter configuration."""
        self.formatter.update_formatting_config(**kwargs)
        logger.info(f"Updated formatter configuration: {kwargs}")

    def get_analysis_capabilities(self) -> Dict[str, Any]:
        """Get information about analysis capabilities."""
        return {
            "formatter_available": True,
            "gemini_configured": self.gemini_service.is_configured(),
            "supported_analysis_types": [t.value for t in AnalysisType],
            "performance_target_ms": self.formatter.formatting_config.performance_target_ms,
            "max_objects_detailed": self.formatter.formatting_config.max_objects_detailed
        }


# Integration Examples

def example_basic_integration():
    """Example: Basic integration with existing detection pipeline."""
    print("=== Basic Integration Example ===")

    # Initialize service
    service = WebcamAnalysisService()

    # Simulate YOLO detection results
    detections = [
        Detection(class_id=0, score=0.95, bbox=(100, 100, 200, 200), class_name="person"),
        Detection(class_id=1, score=0.87, bbox=(300, 150, 400, 250), class_name="chair")
    ]

    # Analyze frame
    result = service.analyze_webcam_frame(
        user_query="What objects do you see in this classroom?",
        detections=detections,
        frame_dimensions=(640, 480)
    )

    if result["success"]:
        print("✓ Analysis completed successfully")
        print(f"Structured prompt length: {len(result['structured_prompt'])} characters")
        print(f"Gemini service available: {result['service_available']}")
        if result["ai_response"]:
            print(f"AI response length: {len(result['ai_response'])} characters")
    else:
        print(f"✗ Analysis failed: {result['error']}")


def example_comparison_analysis():
    """Example: Comparison analysis with reference frame."""
    print("\n=== Comparison Analysis Example ===")

    service = WebcamAnalysisService()

    # Current frame detections
    current_detections = [
        Detection(class_id=0, score=0.95, bbox=(100, 100, 200, 200), class_name="person"),
        Detection(class_id=1, score=0.87, bbox=(300, 150, 400, 250), class_name="chair"),
        Detection(class_id=2, score=0.92, bbox=(500, 200, 600, 300), class_name="book")
    ]

    # Reference frame detections (what we expect to see)
    reference_detections = [
        Detection(class_id=0, score=0.90, bbox=(120, 120, 220, 220), class_name="person"),
        Detection(class_id=1, score=0.85, bbox=(300, 150, 400, 250), class_name="chair")
    ]

    # Comparison metrics
    comparison_metrics = ComparisonMetrics(
        similarity_score=0.75,
        objects_added=1,  # book was added
        objects_removed=0,
        objects_moved=1,  # person moved slightly
        objects_unchanged=1,  # chair stayed
        total_changes=2,
        change_significance="moderate"
    )

    result = service.analyze_webcam_frame(
        user_query="Compare the current classroom setup with the reference arrangement",
        detections=current_detections,
        reference_detections=reference_detections,
        comparison_metrics=comparison_metrics,
        frame_dimensions=(640, 480)
    )

    if result["success"]:
        print("✓ Comparison analysis completed")
        # Show analysis type detection
        analysis_type = result['metadata']['analysis_request']['analysis_type']
        print(f"Detected analysis type: {analysis_type}")
        print(f"Total changes detected: {comparison_metrics.total_changes}")
    else:
        print(f"✗ Comparison analysis failed: {result['error']}")


def example_performance_monitoring():
    """Example: Performance monitoring and optimization."""
    print("\n=== Performance Monitoring Example ===")

    service = WebcamAnalysisService()

    # Create test data
    detections = [
        Detection(class_id=i, score=0.8 + i*0.02, bbox=(i*50, i*30, i*50+100, i*30+100),
                 class_name=f"object_{i}")
        for i in range(10)  # 10 objects for testing
    ]

    # Run multiple analyses to test performance
    times = []
    for i in range(5):
        result = service.analyze_webcam_frame(
            user_query=f"Analysis run {i+1}",
            detections=detections,
            frame_dimensions=(1920, 1080)  # High resolution
        )

        if result["success"] and "performance_metrics" in result:
            metrics = result["performance_metrics"]
            if "last_format_time_ms" in metrics:
                times.append(metrics["last_format_time_ms"])

    if times:
        avg_time = sum(times) / len(times)
        print(f"Average formatting time: {avg_time:.2f}ms")
        print(f"Performance target met: {'✓' if avg_time < 2.0 else '✗'}")

    # Show capabilities
    capabilities = service.get_analysis_capabilities()
    print(f"Analysis capabilities: {capabilities}")


def example_configuration_customization():
    """Example: Customizing formatter configuration."""
    print("\n=== Configuration Customization Example ===")

    service = WebcamAnalysisService()

    # Update formatter configuration
    service.update_formatter_config(
        include_coordinates=True,
        include_confidence=True,
        max_objects_detailed=5,
        use_emoji_indicators=True
    )

    # Test with configuration
    detections = [
        Detection(class_id=0, score=0.95, bbox=(100, 100, 200, 200), class_name="person", angle=45.0)
    ]

    result = service.analyze_webcam_frame(
        user_query="Test with custom configuration",
        detections=detections
    )

    if result["success"]:
        print("✓ Custom configuration applied successfully")
        print(f"Prompt includes angle info: {'Angle:' in result['structured_prompt']}")


# Main integration test
if __name__ == "__main__":
    print("DetectionDataFormatter Integration Guide")
    print("=" * 60)

    try:
        # Run all examples
        example_basic_integration()
        example_comparison_analysis()
        example_performance_monitoring()
        example_configuration_customization()

        print("\n" + "=" * 60)
        print("✓ All integration examples completed successfully!")

        print("\nNext Steps for Integration:")
        print("1. Import DetectionDataFormatter in your webcam service")
        print("2. Initialize formatter with your config")
        print("3. Replace direct Gemini calls with formatted prompts")
        print("4. Use metadata for logging and debugging")
        print("5. Monitor performance metrics for optimization")

    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()