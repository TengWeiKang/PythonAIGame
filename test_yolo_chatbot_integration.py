"""
Test script for YOLO + Chatbot Integration.

This script tests the complete image comparison workflow including:
1. YOLO object detection on reference and current images
2. Object comparison and analysis
3. Chatbot integration with structured prompts
4. Educational feedback generation
"""

import asyncio
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any
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
                'chatbot_persona': 'You are a helpful AI assistant for image analysis.',
                'response_format': 'Educational'
            }
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

def create_test_images() -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic test images for demonstration."""
    logger.info("Creating synthetic test images...")

    # Reference image: Blue rectangle in center
    reference_img = np.zeros((480, 640, 3), dtype=np.uint8)
    reference_img.fill(50)  # Dark gray background
    cv2.rectangle(reference_img, (200, 150), (440, 330), (255, 0, 0), -1)  # Blue rectangle
    cv2.putText(reference_img, "REFERENCE", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Current image: Blue rectangle moved and green circle added
    current_img = np.zeros((480, 640, 3), dtype=np.uint8)
    current_img.fill(50)  # Dark gray background
    cv2.rectangle(current_img, (250, 100), (490, 280), (255, 0, 0), -1)  # Blue rectangle (moved)
    cv2.circle(current_img, (150, 350), 50, (0, 255, 0), -1)  # Green circle (added)
    cv2.putText(current_img, "CURRENT", (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    logger.info("Test images created successfully")
    return reference_img, current_img

def setup_yolo_backend(config: Dict[str, Any]):
    """Setup and load YOLO backend."""
    try:
        from app.backends.yolo_backend import YoloBackend

        logger.info("Setting up YOLO backend...")
        yolo_backend = YoloBackend(config)

        # Try to load a model
        model_name = config.get('preferred_model', 'yolo11n.pt')
        success = yolo_backend.load_model(model_name)

        if success:
            logger.info(f"YOLO backend loaded successfully with model: {model_name}")
            model_info = yolo_backend.get_model_info()
            logger.info(f"Model info: {model_info}")
            return yolo_backend
        else:
            logger.error("Failed to load YOLO model")
            return None

    except Exception as e:
        logger.error(f"YOLO backend setup failed: {e}")
        return None

def setup_gemini_service(config: Dict[str, Any]):
    """Setup Gemini service."""
    try:
        from app.services.gemini_service import AsyncGeminiService

        logger.info("Setting up Gemini service...")

        api_key = config.get('gemini_api_key', '')
        if not api_key:
            logger.warning("No Gemini API key found in config")

        gemini_service = AsyncGeminiService(
            api_key=api_key,
            model=config.get('gemini_model', 'gemini-1.5-flash'),
            temperature=config.get('gemini_temperature', 0.7),
            max_tokens=config.get('gemini_max_tokens', 2048)
        )

        if gemini_service.is_configured():
            logger.info("Gemini service configured successfully")
        else:
            logger.warning("Gemini service not properly configured")

        return gemini_service

    except Exception as e:
        logger.error(f"Gemini service setup failed: {e}")
        return None

async def test_yolo_comparison_service(yolo_backend, gemini_service, config, reference_img, current_img):
    """Test the YOLO comparison service."""
    try:
        from app.services.yolo_comparison_service import YoloComparisonService

        logger.info("Testing YOLO comparison service...")

        # Create comparison service
        comparison_service = YoloComparisonService(
            yolo_backend=yolo_backend,
            gemini_service=gemini_service,
            config=config
        )

        # Set reference image
        logger.info("Setting reference image...")
        success = comparison_service.set_reference_image(reference_img)
        if not success:
            logger.error("Failed to set reference image")
            return None

        # Get reference info
        ref_info = comparison_service.get_reference_info()
        logger.info(f"Reference image info: {ref_info}")

        # Perform comparison
        logger.info("Performing image comparison...")
        comparison_result = comparison_service.compare_with_current(
            current_img,
            "What differences do you see between the reference and current images?"
        )

        # Log results
        logger.info("Comparison Results:")
        logger.info(f"- Scene similarity: {comparison_result.scene_comparison.scene_similarity:.1%}")
        logger.info(f"- Objects added: {comparison_result.scene_comparison.objects_added}")
        logger.info(f"- Objects removed: {comparison_result.scene_comparison.objects_removed}")
        logger.info(f"- Objects moved: {comparison_result.scene_comparison.objects_moved}")
        logger.info(f"- Analysis duration: {comparison_result.analysis_duration_ms:.1f}ms")
        logger.info(f"- Summary: {comparison_result.chatbot_summary}")

        # Format for chatbot
        chatbot_prompt = comparison_service.format_for_chatbot(
            comparison_result,
            "What changes do you notice?"
        )
        logger.info(f"Chatbot prompt generated ({len(chatbot_prompt)} characters)")

        return comparison_result

    except Exception as e:
        logger.error(f"YOLO comparison service test failed: {e}")
        return None

async def test_integrated_analysis_service(yolo_backend, gemini_service, config, reference_img, current_img):
    """Test the integrated analysis service."""
    try:
        from app.services.integrated_analysis_service import IntegratedAnalysisService

        logger.info("Testing integrated analysis service...")

        # Create integrated service
        integrated_service = IntegratedAnalysisService(
            yolo_backend=yolo_backend,
            gemini_service=gemini_service,
            config=config
        )

        # Set progress callback for testing
        def progress_callback(message: str):
            logger.info(f"Progress: {message}")

        integrated_service.set_progress_callback(progress_callback)

        # Set reference image
        logger.info("Setting reference image for integrated analysis...")
        success = integrated_service.set_reference_image(reference_img)
        if not success:
            logger.error("Failed to set reference image for integrated analysis")
            return None

        # Test different user messages
        test_messages = [
            "What differences do you see between the reference and current images?",
            "Can you identify all the objects in this image?",
            "Has anything moved from the original position?",
            "Describe the changes you detect."
        ]

        results = []
        for message in test_messages:
            logger.info(f"Testing with message: '{message}'")

            result = await integrated_service.analyze_with_chatbot(
                current_img, message
            )

            logger.info(f"Analysis Result:")
            logger.info(f"- Success: {result.success}")
            logger.info(f"- Duration: {result.analysis_duration_ms:.1f}ms")
            logger.info(f"- Response length: {len(result.chatbot_response)} characters")

            if result.success:
                logger.info(f"- Chatbot response preview: {result.chatbot_response[:200]}...")

                if result.yolo_comparison:
                    logger.info(f"- YOLO comparison: {result.yolo_comparison.scene_comparison.scene_similarity:.1%} similarity")

                if result.image_analysis:
                    logger.info(f"- Scene analysis: {len(result.image_analysis.objects)} objects detected")
            else:
                logger.error(f"- Error: {result.error_message}")

            results.append(result)

            # Small delay between tests
            await asyncio.sleep(1)

        # Get performance stats
        stats = integrated_service.get_performance_stats()
        logger.info(f"Performance Statistics: {stats}")

        return results

    except Exception as e:
        logger.error(f"Integrated analysis service test failed: {e}")
        return None

async def test_with_real_webcam(integrated_service, config):
    """Test with real webcam if available."""
    try:
        logger.info("Testing with real webcam...")

        # Try to open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.warning("Webcam not available, skipping real webcam test")
            return

        # Capture reference frame
        logger.info("Capturing reference frame in 3 seconds...")
        await asyncio.sleep(3)

        ret, reference_frame = cap.read()
        if not ret:
            logger.error("Failed to capture reference frame")
            cap.release()
            return

        # Set reference
        success = integrated_service.set_reference_image(reference_frame)
        if not success:
            logger.error("Failed to set webcam reference image")
            cap.release()
            return

        logger.info("Move something in the camera view...")
        await asyncio.sleep(5)

        # Capture current frame
        ret, current_frame = cap.read()
        if not ret:
            logger.error("Failed to capture current frame")
            cap.release()
            return

        # Analyze
        result = await integrated_service.analyze_with_chatbot(
            current_frame,
            "What changes do you see in the webcam feed compared to the reference?"
        )

        if result.success:
            logger.info(f"Webcam Analysis Success!")
            logger.info(f"Response: {result.chatbot_response[:300]}...")
        else:
            logger.error(f"Webcam analysis failed: {result.error_message}")

        cap.release()

    except Exception as e:
        logger.error(f"Webcam test failed: {e}")

async def main():
    """Main test function."""
    logger.info("Starting YOLO + Chatbot Integration Tests")
    logger.info("=" * 60)

    # Load configuration
    config = load_config()
    logger.info(f"Config loaded: {len(config)} settings")

    # Create test images
    reference_img, current_img = create_test_images()

    # Setup services
    yolo_backend = setup_yolo_backend(config)
    if not yolo_backend:
        logger.error("Cannot proceed without YOLO backend")
        return

    gemini_service = setup_gemini_service(config)
    if not gemini_service:
        logger.error("Cannot proceed without Gemini service")
        return

    logger.info("All services initialized successfully")
    logger.info("-" * 40)

    # Test 1: YOLO Comparison Service
    logger.info("TEST 1: YOLO Comparison Service")
    comparison_result = await test_yolo_comparison_service(
        yolo_backend, gemini_service, config, reference_img, current_img
    )

    if comparison_result:
        logger.info("✓ YOLO Comparison Service test passed")
    else:
        logger.error("✗ YOLO Comparison Service test failed")

    logger.info("-" * 40)

    # Test 2: Integrated Analysis Service
    logger.info("TEST 2: Integrated Analysis Service")
    analysis_results = await test_integrated_analysis_service(
        yolo_backend, gemini_service, config, reference_img, current_img
    )

    if analysis_results and all(r.success for r in analysis_results):
        logger.info("✓ Integrated Analysis Service test passed")
    else:
        logger.error("✗ Integrated Analysis Service test failed")

    logger.info("-" * 40)

    # Test 3: Real webcam (optional)
    if config.get('test_with_webcam', False):
        logger.info("TEST 3: Real Webcam Integration")
        from app.services.integrated_analysis_service import IntegratedAnalysisService

        integrated_service = IntegratedAnalysisService(
            yolo_backend=yolo_backend,
            gemini_service=gemini_service,
            config=config
        )

        await test_with_real_webcam(integrated_service, config)

    logger.info("=" * 60)
    logger.info("YOLO + Chatbot Integration Tests Completed")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        raise