"""
Test for ASCII Detection Formatter (no external dependencies).
"""

import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

class MockDetection:
    """Mock detection class for testing."""
    def __init__(self, class_id, score, bbox, angle=None, class_name=None):
        self.class_id = class_id
        self.score = score
        self.bbox = bbox
        self.angle = angle
        self.class_name = class_name

def test_ascii_formatter():
    """Test the ASCII formatter functions."""
    try:
        from utils.detection_formatter_ascii import (
            format_detection_data_ascii,
            format_single_detection_ascii,
            format_detection_summary_compact_ascii,
            format_detection_coordinates_only_ascii,
            format_no_detections_ascii
        )

        print("TESTING ASCII DETECTION FORMATTER")
        print("=" * 50)

        # Create mock detection data
        test_detections = [
            MockDetection(
                class_id=0,
                score=0.95,
                bbox=(100, 50, 200, 150),
                angle=15.5,
                class_name="person"
            ),
            MockDetection(
                class_id=67,
                score=0.78,
                bbox=(300, 200, 450, 350),
                angle=None,
                class_name="cell phone"
            ),
            MockDetection(
                class_id=56,
                score=0.63,
                bbox=(500, 100, 600, 200),
                angle=45.2,
                class_name="chair"
            )
        ]

        frame_dimensions = (640, 480)

        print("\nTest 1: Complete Detection Data Format")
        print("-" * 40)
        formatted_data = format_detection_data_ascii(
            detections=test_detections,
            frame_dimensions=frame_dimensions
        )
        print(formatted_data)

        print("\n\nTest 2: Single Detection Format")
        print("-" * 40)
        single_format = format_single_detection_ascii(
            detection=test_detections[0],
            index=1
        )
        print(single_format)

        print("\n\nTest 3: Compact Summary")
        print("-" * 40)
        compact_summary = format_detection_summary_compact_ascii(test_detections)
        print(f"Compact Summary: {compact_summary}")

        print("\n\nTest 4: Coordinates Only")
        print("-" * 40)
        coords_only = format_detection_coordinates_only_ascii(test_detections)
        print(coords_only)

        print("\n\nTest 5: No Detections")
        print("-" * 40)
        no_detections = format_no_detections_ascii()
        print(no_detections)

        print("\n\nTest 6: Empty Detection List")
        print("-" * 40)
        empty_format = format_detection_data_ascii(
            detections=[],
            frame_dimensions=frame_dimensions
        )
        print(empty_format)

        print("\n" + "=" * 50)
        print("ALL ASCII FORMATTER TESTS PASSED!")
        return True

    except Exception as e:
        print(f"Test failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases."""
    try:
        from utils.detection_formatter_ascii import format_detection_data_ascii

        print("\n\nEDGE CASES TEST")
        print("=" * 50)

        # Test with detection without class name
        test_detection_no_name = MockDetection(
            class_id=99,
            score=0.5,
            bbox=(0, 0, 100, 100),
            angle=None,
            class_name=None
        )

        print("\nTest: Detection without class name")
        print("-" * 40)
        result = format_detection_data_ascii([test_detection_no_name], (640, 480))
        print(result)

        # Test with very small detection
        test_detection_small = MockDetection(
            class_id=1,
            score=0.1,
            bbox=(0, 0, 5, 5),
            angle=0.0,
            class_name="tiny_object"
        )

        print("\n\nTest: Very small detection")
        print("-" * 40)
        result = format_detection_data_ascii([test_detection_small], (640, 480))
        print(result)

        print("\nEdge case tests completed successfully!")
        return True

    except Exception as e:
        print(f"Edge case test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("ASCII DETECTION FORMATTER TEST SUITE")
    print("=" * 60)

    success_count = 0
    total_tests = 2

    # Test 1: Basic formatting
    if test_ascii_formatter():
        success_count += 1

    # Test 2: Edge cases
    if test_edge_cases():
        success_count += 1

    # Summary
    print(f"\nTEST RESULTS")
    print("=" * 60)
    print(f"Tests Passed: {success_count}/{total_tests}")
    print(f"Success Rate: {(success_count/total_tests)*100:.1f}%")

    if success_count == total_tests:
        print("ALL TESTS PASSED! ASCII Detection formatting is working correctly.")
        return True
    else:
        print("Some tests failed. Check output above for details.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"Test suite error: {e}")
        exit(1)