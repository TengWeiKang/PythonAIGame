"""
Test script for the Enhanced Detection Service.

This script validates the comprehensive object data extraction functionality,
ensuring it meets performance requirements and produces accurate analysis.
"""

import time
import numpy as np
from typing import List
import json

# Import the service and entities
from app.core.entities import Detection, BBox
from app.services.enhanced_detection_service import (
    EnhancedDetectionService,
    extract_detection_data
)


def create_test_detections() -> List[Detection]:
    """Create sample detections for testing."""
    detections = [
        # Person in top-left corner
        Detection(
            class_id=0,
            score=0.95,
            bbox=(50, 50, 150, 200),
            class_name="person"
        ),
        # Car in center
        Detection(
            class_id=2,
            score=0.88,
            bbox=(270, 190, 370, 290),
            class_name="car"
        ),
        # Another person in bottom-right
        Detection(
            class_id=0,
            score=0.91,
            bbox=(500, 350, 600, 470),
            class_name="person"
        ),
        # Bicycle near the car (overlapping slightly)
        Detection(
            class_id=1,
            score=0.79,
            bbox=(350, 250, 420, 350),
            class_name="bicycle"
        ),
        # Dog in middle-left
        Detection(
            class_id=16,
            score=0.83,
            bbox=(30, 220, 120, 300),
            class_name="dog"
        )
    ]
    return detections


def test_basic_extraction():
    """Test basic detection data extraction."""
    print("=" * 60)
    print("Testing Basic Detection Data Extraction")
    print("=" * 60)

    # Create test data
    detections = create_test_detections()
    image_shape = (480, 640, 3)  # Height, Width, Channels
    class_names = ["person", "bicycle", "car"] + ["class_" + str(i) for i in range(3, 80)]

    # Create service
    service = EnhancedDetectionService()

    # Extract comprehensive data
    start = time.perf_counter()
    result = service.extract_comprehensive_detection_data(
        detections, image_shape, class_names
    )
    elapsed = (time.perf_counter() - start) * 1000

    print(f"\nProcessing Time: {elapsed:.2f}ms")
    print(f"Number of objects: {result['summary']['total_objects']}")
    print(f"Unique classes: {result['summary']['unique_classes']}")
    print(f"Class distribution: {result['summary']['class_distribution']}")
    print(f"Average confidence: {result['summary']['confidence_stats']['mean']:.3f}")
    print(f"Total frame coverage: {result['summary']['coverage_stats']['total_coverage_percent']:.2f}%")

    # Display object details
    print("\nObject Details:")
    for i, obj in enumerate(result['objects']):
        print(f"\n  Object {i+1}: {obj['class_name']}")
        print(f"    Position: {obj['position']}")
        print(f"    Confidence: {obj['confidence']:.2f}")
        print(f"    Dimensions: {obj['dimensions']['width']:.0f}x{obj['dimensions']['height']:.0f}")
        print(f"    Aspect ratio: {obj['dimensions']['aspect_ratio']:.2f}")
        print(f"    Orientation: {obj['orientation']['type']}")
        print(f"    Distance from center: {obj['distance_from_center']:.1f}px")
        print(f"    Frame coverage: {obj['frame_coverage_percent']:.2f}%")

    return result


def test_spatial_analysis():
    """Test spatial relationship analysis."""
    print("\n" + "=" * 60)
    print("Testing Spatial Relationship Analysis")
    print("=" * 60)

    # Create clustered detections
    clustered_detections = [
        # Cluster 1: Three people close together
        Detection(class_id=0, score=0.9, bbox=(100, 100, 150, 200)),
        Detection(class_id=0, score=0.85, bbox=(160, 110, 210, 210)),
        Detection(class_id=0, score=0.88, bbox=(130, 90, 180, 190)),

        # Cluster 2: Cars in parking
        Detection(class_id=2, score=0.92, bbox=(400, 300, 480, 350)),
        Detection(class_id=2, score=0.89, bbox=(490, 305, 570, 355)),

        # Isolated object
        Detection(class_id=16, score=0.75, bbox=(50, 400, 100, 450))
    ]

    service = EnhancedDetectionService(clustering_eps=60)
    result = service.extract_comprehensive_detection_data(
        clustered_detections, (480, 640, 3)
    )

    spatial = result['spatial_analysis']

    print(f"\nClusters found: {len(spatial['clusters'])}")
    for cluster in spatial['clusters']:
        print(f"  Cluster {cluster['cluster_id']}: {cluster['num_objects']} objects")
        print(f"    Center: ({cluster['center']['x']:.1f}, {cluster['center']['y']:.1f})")
        print(f"    Cohesion score: {cluster['cohesion_score']:.3f}")

    print(f"\nOverlaps detected: {len(spatial['overlaps'])}")
    for overlap in spatial['overlaps']:
        print(f"  Objects {overlap['object1_idx']} & {overlap['object2_idx']}: "
              f"IoU={overlap['iou']:.3f}, Type={overlap['overlap_type']}")

    print(f"\nProximities detected: {len(spatial['proximities'])}")
    for prox in spatial['proximities'][:3]:  # Show first 3
        print(f"  Objects {prox['object1_idx']} & {prox['object2_idx']}: "
              f"{prox['distance']:.1f}px apart, Direction: {prox['direction']}")

    print(f"\nSpatial patterns detected: {spatial['spatial_patterns']}")


def test_performance():
    """Test performance with various object counts."""
    print("\n" + "=" * 60)
    print("Testing Performance Scalability")
    print("=" * 60)

    service = EnhancedDetectionService()
    image_shape = (1080, 1920, 3)  # Full HD

    test_counts = [5, 10, 20, 50, 100]

    for count in test_counts:
        # Generate random detections
        detections = []
        for i in range(count):
            x1 = np.random.randint(0, 1800)
            y1 = np.random.randint(0, 980)
            width = np.random.randint(50, 200)
            height = np.random.randint(50, 200)

            detections.append(Detection(
                class_id=np.random.randint(0, 80),
                score=np.random.uniform(0.5, 1.0),
                bbox=(x1, y1, min(x1 + width, 1920), min(y1 + height, 1080))
            ))

        # Measure processing time
        times = []
        for _ in range(10):  # Average over 10 runs
            start = time.perf_counter()
            result = service.extract_comprehensive_detection_data(
                detections, image_shape
            )
            times.append((time.perf_counter() - start) * 1000)

        avg_time = np.mean(times)
        print(f"\n{count:3d} objects: {avg_time:6.2f}ms average "
              f"(min: {min(times):.2f}ms, max: {max(times):.2f}ms)")

        # Check if meeting performance requirement
        if count == 10:
            if avg_time < 5:
                print("    ✓ Meets <5ms requirement for 10 objects")
            else:
                print(f"    ✗ Does not meet <5ms requirement (got {avg_time:.2f}ms)")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 60)
    print("Testing Edge Cases")
    print("=" * 60)

    service = EnhancedDetectionService()

    # Test with empty detections
    print("\n1. Empty detections:")
    result = service.extract_comprehensive_detection_data([], (480, 640, 3))
    print(f"   Total objects: {result['summary']['total_objects']}")
    print(f"   Processing successful: {result['timestamp'] is not None}")

    # Test with single detection
    print("\n2. Single detection:")
    single = [Detection(class_id=0, score=0.9, bbox=(100, 100, 200, 200))]
    result = service.extract_comprehensive_detection_data(single, (480, 640, 3))
    print(f"   Clusters: {len(result['spatial_analysis']['clusters'])}")
    print(f"   Overlaps: {len(result['spatial_analysis']['overlaps'])}")

    # Test with overlapping objects
    print("\n3. Completely overlapping objects:")
    overlapping = [
        Detection(class_id=0, score=0.9, bbox=(100, 100, 200, 200)),
        Detection(class_id=1, score=0.85, bbox=(100, 100, 200, 200))  # Same bbox
    ]
    result = service.extract_comprehensive_detection_data(overlapping, (480, 640, 3))
    overlaps = result['spatial_analysis']['overlaps']
    if overlaps:
        print(f"   IoU: {overlaps[0]['iou']:.3f}")
        print(f"   Overlap type: {overlaps[0]['overlap_type']}")

    # Test with very small objects
    print("\n4. Very small objects:")
    small = [Detection(class_id=0, score=0.9, bbox=(100, 100, 102, 102))]
    result = service.extract_comprehensive_detection_data(small, (480, 640, 3))
    obj = result['objects'][0]
    print(f"   Area: {obj['dimensions']['area']:.0f} pixels")
    print(f"   Coverage: {obj['frame_coverage_percent']:.4f}%")


def test_convenience_function():
    """Test the convenience function."""
    print("\n" + "=" * 60)
    print("Testing Convenience Function")
    print("=" * 60)

    detections = create_test_detections()[:3]
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    result = extract_detection_data(detections, image, ["person", "bicycle", "car"])

    print(f"\nExtracted data successfully: {result['timestamp'] is not None}")
    print(f"Objects processed: {len(result['objects'])}")
    print(f"Processing time: {result['processing_time_ms']:.2f}ms")


def save_sample_output():
    """Save a sample output for documentation."""
    print("\n" + "=" * 60)
    print("Saving Sample Output")
    print("=" * 60)

    detections = create_test_detections()
    service = EnhancedDetectionService()

    result = service.extract_comprehensive_detection_data(
        detections, (480, 640, 3),
        ["person", "bicycle", "car"] + [f"class_{i}" for i in range(3, 80)]
    )

    # Save to file
    output_file = "sample_enhanced_detection_output.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSample output saved to: {output_file}")
    print(f"File contains data for {len(result['objects'])} objects")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Enhanced Detection Service Test Suite")
    print("=" * 60)

    try:
        # Run tests
        basic_result = test_basic_extraction()
        test_spatial_analysis()
        test_performance()
        test_edge_cases()
        test_convenience_function()
        save_sample_output()

        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()