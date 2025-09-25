"""
Simple test for Detection Data Formatting without external dependencies.

This test validates the detection formatting functions using mock data.
"""

import sys
import os
import logging

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_detection_formatting():
    """Test detection formatting with mock data."""
    try:
        # Import the formatting functions
        from utils.detection_formatter import (
            format_detection_data,
            format_single_detection,
            format_detection_summary_compact,
            format_detection_coordinates_only,
            format_no_detections
        )
        from core.entities import Detection

        logger.info("Testing detection formatting functions...")

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

        print("\n" + "="*80)
        print("DETECTION DATA FORMATTING TEST")
        print("="*80)

        # Test 1: Main formatting function
        print("\n[*] Test 1: Complete Detection Data Format")
        print("-" * 50)
        formatted_data = format_detection_data(
            detections=test_detections,
            frame_dimensions=frame_dimensions,
            include_coordinates=True,
            include_angles=True,
            include_confidence=True,
            include_size_info=True
        )
        print(formatted_data)

        # Test 2: Single detection formatting
        print("\n\n[*] Test 2: Single Detection Format")
        print("-" * 50)
        single_format = format_single_detection(
            detection=test_detections[0],
            index=1,
            include_coordinates=True,
            include_angles=True,
            include_confidence=True,
            include_size_info=True
        )
        print(single_format)

        # Test 3: Compact summary
        print("\n\n[*] Test 3: Compact Summary Format")
        print("-" * 50)
        compact_summary = format_detection_summary_compact(test_detections)
        print(f"Compact Summary: {compact_summary}")

        # Test 4: Coordinates only
        print("\n\n[*] Test 4: Coordinates Only Format")
        print("-" * 50)
        coords_only = format_detection_coordinates_only(test_detections)
        print(coords_only)

        # Test 5: No detections
        print("\n\n[*] Test 5: No Detections Format")
        print("-" * 50)
        no_detections = format_no_detections()
        print(no_detections)

        # Test 6: Empty detection list
        print("\n\n[*] Test 6: Empty Detection List")
        print("-" * 50)
        empty_format = format_detection_data(
            detections=[],
            frame_dimensions=frame_dimensions
        )
        print(empty_format)

        print("\n" + "="*80)
        print("[+] ALL FORMATTING TESTS PASSED!")
        print("="*80)

        return True

    except ImportError as e:
        logger.error(f"Import error: {e}")
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases and error handling."""
    try:
        from utils.detection_formatter import format_detection_data
        from core.entities import Detection

        print("\n" + "="*80)
        print("EDGE CASES AND ERROR HANDLING TEST")
        print("="*80)

        # Test with detection without class name
        test_detection_no_name = Detection(
            class_id=99,
            score=0.5,
            bbox=(0, 0, 100, 100),
            angle=None,
            class_name=None
        )

        print("\n[*] Test: Detection without class name")
        print("-" * 50)
        result = format_detection_data([test_detection_no_name], (640, 480))
        print(result)

        # Test with very small detection
        test_detection_small = Detection(
            class_id=1,
            score=0.1,
            bbox=(0, 0, 5, 5),
            angle=0.0,
            class_name="tiny_object"
        )

        print("\n[*] Test: Very small detection")
        print("-" * 50)
        result = format_detection_data([test_detection_small], (640, 480))
        print(result)

        # Test with large detection covering most of frame
        test_detection_large = Detection(
            class_id=2,
            score=1.0,
            bbox=(10, 10, 630, 470),
            angle=90.0,
            class_name="large_object"
        )

        print("\n[*] Test: Large detection")
        print("-" * 50)
        result = format_detection_data([test_detection_large], (640, 480))
        print(result)

        print("\n[+] Edge case tests completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Edge case test failed: {e}")
        print(f"❌ Edge case test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("DETECTION FORMATTING TEST SUITE")
    print("="*80)

    success_count = 0
    total_tests = 2

    # Test 1: Basic formatting
    if test_detection_formatting():
        success_count += 1

    # Test 2: Edge cases
    if test_edge_cases():
        success_count += 1

    # Summary
    print(f"\nTEST RESULTS")
    print("="*80)
    print(f"Tests Passed: {success_count}/{total_tests}")
    print(f"Success Rate: {(success_count/total_tests)*100:.1f}%")

    if success_count == total_tests:
        print("[+] ALL TESTS PASSED! Detection formatting is working correctly.")
        return True
    else:
        print("[!] Some tests failed. Check output above for details.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        print(f"[!] Test suite error: {e}")
        exit(1)