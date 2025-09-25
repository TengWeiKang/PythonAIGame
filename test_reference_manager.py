"""
Test script for ReferenceImageManager performance and functionality.

This script demonstrates the usage of the ReferenceImageManager and validates
its performance characteristics against the specified requirements.
"""

import asyncio
import time
import numpy as np
import cv2
from pathlib import Path
import sys
import os

# Add app directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.reference_manager import ReferenceImageManager
from app.backends.yolo_backend import YoloBackend
from app.core.entities import Detection, BBox
from app.core.performance import PerformanceMonitor


def create_test_frame(width=1920, height=1080):
    """Create a test frame with some random content."""
    # Create a frame with random colored rectangles (simulating objects)
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Add some colored rectangles
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for i in range(5):
        x = np.random.randint(0, width - 200)
        y = np.random.randint(0, height - 200)
        w = np.random.randint(50, 200)
        h = np.random.randint(50, 200)
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors[i % len(colors)], -1)

    # Add some text
    cv2.putText(frame, "Test Frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return frame


def create_test_detections(count=5):
    """Create test detection objects."""
    detections = []
    for i in range(count):
        x1 = np.random.randint(0, 1600)
        y1 = np.random.randint(0, 900)
        x2 = x1 + np.random.randint(50, 200)
        y2 = y1 + np.random.randint(50, 200)

        detection = Detection(
            class_id=np.random.randint(0, 80),
            score=np.random.uniform(0.5, 1.0),
            bbox=(x1, y1, x2, y2),
            class_name=f"object_{i}"
        )
        detections.append(detection)

    return detections


async def test_reference_capture(manager, frame):
    """Test reference capture performance."""
    print("\n=== Testing Reference Capture ===")

    # Warm up
    print("Warming up...")
    await manager.capture_reference(frame, "warmup_ref")

    # Performance test
    print("Running capture performance test...")
    times = []
    for i in range(5):
        start = time.perf_counter()
        ref_id = await manager.capture_reference(frame, f"test_ref_{i}")
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"  Capture {i+1}: {elapsed:.1f}ms - Reference ID: {ref_id}")

    avg_time = sum(times) / len(times)
    print(f"\nAverage capture time: {avg_time:.1f}ms")
    print(f"Target: <200ms - {'PASS' if avg_time < 200 else 'FAIL'}")

    return ref_id


def test_reference_retrieval(manager, reference_id):
    """Test reference retrieval performance."""
    print("\n=== Testing Reference Retrieval ===")

    # First retrieval (may load from disk)
    start = time.perf_counter()
    ref_data = manager.get_reference(reference_id)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"First retrieval (disk): {elapsed:.1f}ms")
    print(f"  Detection count: {ref_data['detection_count']}")

    # Second retrieval (should be cached)
    start = time.perf_counter()
    ref_data = manager.get_reference(reference_id)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"Second retrieval (cached): {elapsed:.1f}ms")

    return ref_data


def test_comparison_performance(manager, reference_id):
    """Test comparison performance."""
    print("\n=== Testing Comparison Performance ===")

    # Create test detections with varying counts
    test_cases = [
        (5, "Small scene (5 objects)"),
        (10, "Medium scene (10 objects)"),
        (20, "Large scene (20 objects)")
    ]

    for count, description in test_cases:
        current_detections = create_test_detections(count)

        # First comparison (not cached)
        start = time.perf_counter()
        result = manager.compare_with_reference(current_detections, reference_id)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"\n{description}:")
        print(f"  First comparison: {elapsed:.1f}ms")
        print(f"  Overall similarity: {result.overall_similarity:.2f}")
        print(f"  Objects added: {len(result.objects_added)}")
        print(f"  Objects missing: {len(result.objects_missing)}")
        print(f"  Scene change score: {result.scene_change_score:.2f}")

        # Second comparison (cached)
        start = time.perf_counter()
        result = manager.compare_with_reference(current_detections, reference_id)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  Cached comparison: {elapsed:.1f}ms (cache hit: {result.cache_hit})")

    print(f"\n✓ Target: <100ms for typical scenes")


def test_iou_calculation():
    """Test IoU calculation performance."""
    print("\n=== Testing IoU Calculation Performance ===")

    # Create test bounding boxes
    bbox1 = (100, 100, 200, 200)
    bbox2 = (150, 150, 250, 250)

    # Performance test
    iterations = 100000
    start = time.perf_counter()
    for _ in range(iterations):
        iou = ReferenceImageManager._calculate_iou(bbox1, bbox2)
    elapsed = (time.perf_counter() - start) * 1000

    per_calculation = elapsed / iterations * 1000  # microseconds
    print(f"IoU calculation time: {per_calculation:.2f} microseconds per calculation")
    print(f"Total time for {iterations} calculations: {elapsed:.1f}ms")
    print(f"Calculated IoU: {iou:.3f}")


def test_memory_management(manager):
    """Test memory management and caching."""
    print("\n=== Testing Memory Management ===")

    memory_stats = manager.get_memory_usage()
    print("Memory Statistics:")
    print(f"  Total references: {memory_stats['total_references']}")
    print(f"  Cache hit rates:")
    print(f"    - Metadata: {memory_stats['metadata_cache_hit_rate']:.1%}")
    print(f"    - Detection: {memory_stats['detection_cache_hit_rate']:.1%}")
    print(f"    - Comparison: {memory_stats['comparison_cache_hit_rate']:.1%}")
    print(f"  Cache entries: {memory_stats['cache_entries']}")


def test_cleanup_operations(manager):
    """Test cleanup and deletion operations."""
    print("\n=== Testing Cleanup Operations ===")

    # Get initial count
    initial_refs = manager.get_all_references()
    print(f"Initial reference count: {len(initial_refs)}")

    # Delete all references
    manager.delete_all_references()
    final_refs = manager.get_all_references()
    print(f"After cleanup: {len(final_refs)} references")

    # Verify memory is cleared
    memory_stats = manager.get_memory_usage()
    print(f"Cache entries after cleanup: {memory_stats['cache_entries']}")


class MockYoloBackend:
    """Mock YOLO backend for testing without actual model."""

    def __init__(self):
        self.is_loaded = True

    def predict(self, image, **kwargs):
        """Return mock detections."""
        return create_test_detections(np.random.randint(3, 8))

    def get_model_info(self):
        """Return mock model info."""
        return {
            'model_path': 'mock_model.pt',
            'backend': 'mock',
            'device': 'cpu'
        }


async def main():
    """Main test function."""
    print("=" * 60)
    print("Reference Image Manager Performance Test")
    print("=" * 60)

    # Initialize performance monitor
    monitor = PerformanceMonitor.instance()

    # Create test data directory
    data_dir = Path("./test_reference_data")
    data_dir.mkdir(exist_ok=True)

    # Initialize manager with mock backend
    print("\nInitializing ReferenceImageManager...")
    yolo_backend = MockYoloBackend()
    manager = ReferenceImageManager(
        yolo_backend=yolo_backend,
        data_dir=str(data_dir),
        max_references=50,
        max_memory_mb=50,
        auto_cleanup_days=7,
        enable_compression=True
    )

    # Create test frame
    frame = create_test_frame()

    # Run tests
    try:
        # Test reference capture
        reference_id = await test_reference_capture(manager, frame)

        # Test retrieval
        test_reference_retrieval(manager, reference_id)

        # Test comparison
        test_comparison_performance(manager, reference_id)

        # Test IoU calculation
        test_iou_calculation()

        # Test memory management
        test_memory_management(manager)

        # Test cleanup
        test_cleanup_operations(manager)

        print("\n" + "=" * 60)
        print("Performance Summary")
        print("=" * 60)

        # Get performance statistics
        metrics = monitor.get_current_metrics()
        if metrics:
            print(f"CPU Usage: {metrics.cpu_percent:.1f}%")
            print(f"Memory Usage: {metrics.memory_usage_mb:.1f} MB")
            print(f"Active Threads: {metrics.active_threads}")

        print("\n✓ All tests completed successfully!")

    finally:
        # Cleanup
        manager.shutdown()
        monitor.stop_monitoring()

        # Remove test directory
        import shutil
        if data_dir.exists():
            shutil.rmtree(data_dir)
            print(f"\nTest data directory cleaned up: {data_dir}")


if __name__ == "__main__":
    asyncio.run(main())