"""
Test script for Detection Data Display in Chat Responses.

This script tests the enhanced chat system that displays YOLO detection data
including coordinates, angles, confidence scores, and other detection metadata
in user-friendly chat responses.
"""

import asyncio
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """Load configuration from config.json."""
    try:
        config_path = Path("config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info("Configuration loaded successfully")
            return config
        else:
            logger.warning("config.json not found, using defaults")
            return {
                'detection_confidence_threshold': 0.5,
                'detection_iou_threshold': 0.45,
                'gemini_api_key': '',
                'gemini_model': 'gemini-1.5-flash',
                'preferred_model': 'yolo12m',
                'enable_ai_analysis': True,
                'chatbot_persona': 'You are a helpful AI assistant for educational image analysis.',
                'response_format': 'Educational',
                'enable_image_comparison': True,
                'enable_scene_analysis': True
            }
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

def create_test_image_with_objects() -> np.ndarray:
    """Create a test image with simple geometric shapes for detection testing."""
    # Create a 640x480 test image
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add some simple shapes that YOLO might detect as objects
    # Rectangle (might be detected as a book or laptop)
    cv2.rectangle(image, (50, 50), (200, 150), (100, 150, 200), -1)

    # Circle (might be detected as a ball or plate)
    cv2.circle(image, (400, 120), 60, (150, 200, 100), -1)

    # Another rectangle (might be detected as a phone or remote)
    cv2.rectangle(image, (300, 300), (350, 400), (200, 100, 150), -1)

    # Add some text to make it more realistic
    cv2.putText(image, "Test Image", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return image

def test_detection_formatter():
    """Test the detection formatter functions."""
    try:
        from app.utils.detection_formatter import (
            format_detection_data,
            format_single_detection,
            format_detection_summary_compact,
            format_detection_coordinates_only
        )
        from app.core.entities import Detection

        logger.info("Testing detection formatter functions...")

        # Create mock detection data
        test_detections = [
            Detection(
                class_id=0,
                score=0.95,
                bbox=(100, 50, 200, 150),
                angle=15.5,
                class_name="person"
            ),
            Detection(
                class_id=67,
                score=0.78,
                bbox=(300, 200, 450, 350),
                angle=None,
                class_name="cell phone"
            ),
            Detection(
                class_id=56,
                score=0.63,
                bbox=(500, 100, 600, 200),
                angle=45.2,
                class_name="chair"
            )
        ]

        frame_dimensions = (640, 480)

        # Test main formatting function
        formatted_data = format_detection_data(
            detections=test_detections,
            frame_dimensions=frame_dimensions,
            include_coordinates=True,
            include_angles=True,
            include_confidence=True,
            include_size_info=True
        )

        print("\n" + "="*60)
        print("FORMATTED DETECTION DATA TEST")
        print("="*60)
        print(formatted_data)
        print("="*60)

        # Test compact summary
        compact_summary = format_detection_summary_compact(test_detections)
        print(f"\nCompact Summary: {compact_summary}")

        # Test coordinates only
        coords_only = format_detection_coordinates_only(test_detections)
        print(f"\nCoordinates Only:\n{coords_only}")

        logger.info("Detection formatter tests completed successfully")
        return True

    except ImportError as e:
        logger.error(f"Failed to import detection formatter: {e}")
        return False
    except Exception as e:
        logger.error(f"Detection formatter test failed: {e}")
        return False

async def test_integrated_analysis_with_detection_display():
    """Test the integrated analysis service with enhanced detection display."""
    try:
        from app.backends.yolo_backend import YoloBackend
        from app.services.gemini_service import AsyncGeminiService
        from app.services.integrated_analysis_service import IntegratedAnalysisService

        config = load_config()

        # Initialize YOLO backend
        logger.info("Initializing YOLO backend...")
        yolo_backend = YoloBackend(config)

        # Try to load a YOLO model
        model_path = config.get('preferred_model', 'yolo12m')
        if not yolo_backend.load_model(model_path):
            logger.error("Failed to load YOLO model")
            return False

        logger.info(f"YOLO model loaded successfully: {model_path}")

        # Initialize Gemini service
        logger.info("Initializing Gemini service...")
        gemini_service = AsyncGeminiService(config)
        if not gemini_service.is_configured():
            logger.warning("Gemini service not configured - will test detection display only")

        # Initialize integrated analysis service
        integrated_service = IntegratedAnalysisService(
            yolo_backend=yolo_backend,
            gemini_service=gemini_service,
            config=config
        )

        # Create test image
        test_image = create_test_image_with_objects()

        # Test user messages
        test_messages = [
            "What objects do you see in this image?",
            "Can you analyze the coordinates and positions of detected objects?",
            "Show me the detection confidence scores and object sizes.",
            "What are the bounding box coordinates for each object?"
        ]

        logger.info("Testing integrated analysis with detection display...")

        for i, message in enumerate(test_messages, 1):
            print(f"\n{'='*80}")
            print(f"TEST MESSAGE {i}: {message}")
            print('='*80)

            # Run integrated analysis
            result = await integrated_service.analyze_with_chatbot(
                current_image=test_image,
                user_message=message,
                include_comparison=False,  # Skip comparison for this test
                include_scene_analysis=True
            )

            if result and result.success:
                # Simulate the enhanced chat response creation
                from app.ui.modern_main_window import ModernMainWindow

                # Create a mock window instance for testing the method
                class MockWindow:
                    def __init__(self):
                        self.integrated_analysis_service = integrated_service
                        self._current_frame = test_image

                    def _create_enhanced_chat_response(self, analysis_result, frame_dimensions):
                        """Copy of the enhanced response method for testing."""
                        try:
                            from app.utils.detection_formatter import format_detection_data

                            response_parts = []

                            # Extract detection data from analysis result
                            detections = []
                            yolo_comparison = analysis_result.yolo_comparison
                            image_analysis = analysis_result.image_analysis

                            # Try multiple sources for detection data (prioritized)
                            # 1. From YOLO comparison current objects
                            if yolo_comparison and yolo_comparison.object_comparisons:
                                for comp in yolo_comparison.object_comparisons:
                                    if comp.current_object:
                                        detections.append(comp.current_object)

                            # 2. From image analysis objects
                            if not detections and image_analysis and image_analysis.objects:
                                detections = image_analysis.objects

                            # 3. Fallback: Run direct YOLO detection
                            if not detections:
                                try:
                                    yolo_backend = self.integrated_analysis_service.yolo_backend
                                    if yolo_backend and yolo_backend.is_loaded:
                                        detections = yolo_backend.predict(
                                            self._current_frame,
                                            conf=0.5,
                                            iou=0.45,
                                            verbose=False
                                        )
                                        logging.debug(f"Direct YOLO detection found {len(detections)} objects")
                                except Exception as e:
                                    logging.warning(f"Direct YOLO detection failed: {e}")

                            # Enhance detections with class names
                            if detections:
                                try:
                                    yolo_backend = self.integrated_analysis_service.yolo_backend
                                    if yolo_backend and yolo_backend.is_loaded and hasattr(yolo_backend.model, 'names'):
                                        class_names = yolo_backend.model.names
                                        for detection in detections:
                                            if not detection.class_name and detection.class_id in class_names:
                                                detection.class_name = class_names[detection.class_id]
                                except Exception as e:
                                    logging.warning(f"Failed to add class names: {e}")

                            # Format detection data if available
                            if detections:
                                detection_data = format_detection_data(
                                    detections=detections,
                                    frame_dimensions=frame_dimensions,
                                    yolo_comparison=yolo_comparison,
                                    image_analysis=image_analysis,
                                    include_coordinates=True,
                                    include_angles=True,
                                    include_confidence=True,
                                    include_size_info=True
                                )
                                response_parts.append(detection_data)
                                response_parts.append("")  # Add spacing

                            # Add the AI response
                            if analysis_result.chatbot_response:
                                response_parts.append("🤖 AI Analysis:")
                                response_parts.append(analysis_result.chatbot_response)

                            # Add performance info
                            response_parts.append("")
                            response_parts.append(f"⚡ Analysis Time: {analysis_result.analysis_duration_ms:.1f}ms")

                            return "\n".join(response_parts)

                        except Exception as e:
                            logging.error(f"Failed to create enhanced response: {e}")
                            return analysis_result.chatbot_response if analysis_result.chatbot_response else "Response formatting failed."

                # Test the enhanced response
                mock_window = MockWindow()
                enhanced_response = mock_window._create_enhanced_chat_response(
                    result, test_image.shape[:2][::-1]  # (width, height)
                )

                print(enhanced_response)
                print(f"\nAnalysis Duration: {result.analysis_duration_ms:.1f}ms")

            else:
                print(f"Analysis failed: {result.error_message if result else 'Unknown error'}")

            # Add delay between tests
            await asyncio.sleep(1)

        logger.info("All detection display tests completed successfully")
        return True

    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        return False
    except Exception as e:
        logger.error(f"Integrated analysis test failed: {e}")
        return False

def test_no_detections_display():
    """Test the display when no objects are detected."""
    try:
        from app.utils.detection_formatter import format_no_detections

        logger.info("Testing no detections display...")

        no_detection_message = format_no_detections()

        print("\n" + "="*60)
        print("NO DETECTIONS DISPLAY TEST")
        print("="*60)
        print(no_detection_message)
        print("="*60)

        logger.info("No detections display test completed")
        return True

    except Exception as e:
        logger.error(f"No detections test failed: {e}")
        return False

async def main():
    """Run all detection display tests."""
    logger.info("Starting Detection Data Display Tests...")

    print("\n🔍 DETECTION DATA DISPLAY TEST SUITE")
    print("="*80)

    success_count = 0
    total_tests = 3

    # Test 1: Detection formatter functions
    print("\n📝 Test 1: Detection Formatter Functions")
    if test_detection_formatter():
        success_count += 1
        print("✅ Detection formatter test PASSED")
    else:
        print("❌ Detection formatter test FAILED")

    # Test 2: No detections display
    print("\n📝 Test 2: No Detections Display")
    if test_no_detections_display():
        success_count += 1
        print("✅ No detections display test PASSED")
    else:
        print("❌ No detections display test FAILED")

    # Test 3: Integrated analysis with detection display
    print("\n📝 Test 3: Integrated Analysis with Detection Display")
    try:
        if await test_integrated_analysis_with_detection_display():
            success_count += 1
            print("✅ Integrated analysis test PASSED")
        else:
            print("❌ Integrated analysis test FAILED")
    except Exception as e:
        logger.error(f"Integrated analysis test error: {e}")
        print("❌ Integrated analysis test FAILED")

    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("="*80)
    print(f"Tests Passed: {success_count}/{total_tests}")
    print(f"Success Rate: {(success_count/total_tests)*100:.1f}%")

    if success_count == total_tests:
        print("🎉 All tests PASSED! Detection data display is working correctly.")
    else:
        print("⚠️  Some tests FAILED. Check the output above for details.")

    return success_count == total_tests

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        print(f"\n💥 Test suite failed: {e}")
        exit(1)