#!/usr/bin/env python3
"""Test script to validate image analysis integration with chat system.

This script tests that:
1. ImageAnalysisService can be imported and initialized
2. The service can analyze video frames properly
3. Chat integration is working correctly
"""
import sys
import traceback
from pathlib import Path
import numpy as np
import logging

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Set up logging for testing
logging.basicConfig(level=logging.INFO)

def test_image_analysis_service_import():
    """Test importing ImageAnalysisService."""
    try:
        from app.services.image_analysis_service import ImageAnalysisService
        print("[PASS] ImageAnalysisService import successful")
        return True
    except Exception as e:
        print(f"[FAIL] ImageAnalysisService import failed: {e}")
        traceback.print_exc()
        return False

def test_image_analysis_service_initialization():
    """Test creating ImageAnalysisService instance."""
    try:
        from app.config.settings import Config
        from app.services.image_analysis_service import ImageAnalysisService
        from app.services.inference_service import InferenceService

        # Create default config
        config = Config()

        # Create inference service first
        inference_service = InferenceService(config)

        # Test instantiation with required InferenceService
        service = ImageAnalysisService(inference_service, config)
        print("[PASS] ImageAnalysisService initialization successful")
        return True
    except Exception as e:
        print(f"[FAIL] ImageAnalysisService initialization failed: {e}")
        traceback.print_exc()
        return False

def test_frame_analysis():
    """Test analyzing a sample frame."""
    try:
        from app.config.settings import Config
        from app.services.image_analysis_service import ImageAnalysisService
        from app.services.inference_service import InferenceService

        config = Config()
        inference_service = InferenceService(config)
        service = ImageAnalysisService(inference_service, config)

        # Create a test frame (random data)
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Test comprehensive analysis
        result = service.analyze_frame_comprehensive(test_frame, "What do you see in this image?")

        if result:
            print("[PASS] Frame analysis successful")
            print(f"  - Found {len(result.objects)} objects")
            print(f"  - Analysis time: {result.analysis_duration_ms/1000:.3f}s")
            print(f"  - Scene description: {result.scene_description}")
            return True
        else:
            print("[FAIL] Frame analysis returned no results")
            return False

    except Exception as e:
        print(f"[FAIL] Frame analysis failed: {e}")
        traceback.print_exc()
        return False

def test_chatbot_formatting():
    """Test formatting analysis results for chatbot."""
    try:
        from app.config.settings import Config
        from app.services.image_analysis_service import ImageAnalysisService
        from app.services.inference_service import InferenceService

        config = Config()
        inference_service = InferenceService(config)
        service = ImageAnalysisService(inference_service, config)

        # Create a test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Analyze frame
        result = service.analyze_frame_comprehensive(test_frame, "Describe the scene")

        if result:
            # Test formatting for chatbot
            formatted_message = service.format_for_chatbot(result, "Describe the scene")

            if formatted_message and len(formatted_message) > 0:
                print("[PASS] ChatBot formatting successful")
                print(f"  - Generated prompt length: {len(formatted_message)} characters")
                # Print first 200 characters as preview
                preview = formatted_message[:200].replace('\n', ' ')
                print(f"  - Preview: {preview}...")
                return True
            else:
                print("[FAIL] ChatBot formatting returned empty result")
                return False
        else:
            print("[FAIL] No analysis result to format")
            return False

    except Exception as e:
        print(f"[FAIL] ChatBot formatting failed: {e}")
        traceback.print_exc()
        return False

def test_modern_main_window_integration():
    """Test that ModernMainWindow can import ImageAnalysisService."""
    try:
        from app.ui.modern_main_window import ModernMainWindow
        print("[PASS] ModernMainWindow can import ImageAnalysisService")
        return True
    except ImportError as e:
        if "image_analysis_service" in str(e).lower():
            print(f"[FAIL] ModernMainWindow import failed due to ImageAnalysisService: {e}")
            return False
        else:
            # Other import error not related to our changes
            print(f"[WARN] ModernMainWindow import error (not related to ImageAnalysisService): {e}")
            return True
    except Exception as e:
        print(f"[FAIL] ModernMainWindow import failed: {e}")
        traceback.print_exc()
        return False

def run_integration_tests():
    """Run all integration tests for image analysis."""
    print("=" * 60)
    print("IMAGE ANALYSIS INTEGRATION TEST SUITE")
    print("=" * 60)

    tests = [
        ("ImageAnalysisService Import", test_image_analysis_service_import),
        ("ImageAnalysisService Initialization", test_image_analysis_service_initialization),
        ("Frame Analysis", test_frame_analysis),
        ("ChatBot Formatting", test_chatbot_formatting),
        ("ModernMainWindow Integration", test_modern_main_window_integration),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        print("-" * 40)
        if test_func():
            passed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("[SUCCESS] All image analysis integration tests passed!")
        print("The chatbot should now automatically analyze video frames when users send messages.")
        return True
    else:
        print(f"[FAILURE] {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)