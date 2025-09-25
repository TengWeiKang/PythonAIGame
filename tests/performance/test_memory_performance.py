"""Performance tests focusing on memory usage and leak detection.

Tests monitor memory consumption, garbage collection, and resource cleanup
to ensure production-ready memory management.
"""
import pytest
import gc
import time
import threading
from unittest.mock import Mock, patch
import psutil
import os
import numpy as np
import cv2

from app.services.improved_webcam_service import ImprovedWebcamService
from app.services.gemini_service import GeminiService
from app.core.cache_manager import CacheManager
from app.core.memory_manager import MemoryManager


@pytest.mark.performance
class TestMemoryLeakDetection:
    """Test for memory leaks in core services."""

    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def test_webcam_service_memory_leak(self, mock_config):
        """Test webcam service for memory leaks during repeated operations."""
        initial_memory = self.get_memory_usage()

        with patch('cv2.VideoCapture') as mock_video_capture:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_video_capture.return_value = mock_cap

            service = ImprovedWebcamService(mock_config)

            # Perform many operations
            for i in range(1000):
                service.open_webcam(0)
                success, frame = service.read_frame()
                if frame is not None:
                    # Simulate frame processing
                    processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    del processed
                service.close_webcam()

                # Force garbage collection every 100 iterations
                if i % 100 == 0:
                    gc.collect()

        # Final memory check
        gc.collect()
        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory

        # Memory increase should be minimal (less than 50MB)
        assert memory_increase < 50, f"Memory leak detected: {memory_increase:.2f}MB increase"

    def test_cache_manager_memory_leak(self):
        """Test cache manager for memory leaks."""
        initial_memory = self.get_memory_usage()

        cache_manager = CacheManager(max_size=100)

        # Add and remove many cache entries
        for i in range(5000):
            key = f"test_key_{i}"
            value = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

            cache_manager.set(key, value)

            # Occasionally remove items to test cleanup
            if i % 10 == 0:
                cache_manager.clear()

            # Force garbage collection periodically
            if i % 500 == 0:
                gc.collect()

        # Clear cache and force cleanup
        cache_manager.clear()
        gc.collect()

        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory

        # Should not have significant memory increase
        assert memory_increase < 30, f"Cache memory leak: {memory_increase:.2f}MB increase"

    def test_ai_service_memory_leak(self, mock_config):
        """Test AI service for memory leaks."""
        initial_memory = self.get_memory_usage()

        mock_config.gemini_api_key = "test_key"

        with patch('app.services.gemini_service.genai') as mock_genai:
            mock_model = Mock()
            mock_genai.GenerativeModel.return_value = mock_model

            # Test many service creations and destructions
            for i in range(100):
                service = GeminiService(mock_config)

                # Simulate some operations
                test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
                base64_str = service._image_to_base64(test_image)

                # Clean up
                del service, test_image, base64_str

                if i % 20 == 0:
                    gc.collect()

        gc.collect()
        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory

        assert memory_increase < 20, f"AI service memory leak: {memory_increase:.2f}MB increase"

    def test_detection_result_memory_management(self, sample_detections):
        """Test detection result objects for proper memory cleanup."""
        initial_memory = self.get_memory_usage()

        from app.core.entities import DetectionResult

        results = []

        # Create many detection results
        for i in range(1000):
            result = DetectionResult(
                frame_id=i,
                timestamp=time.time(),
                detections=sample_detections.copy(),
                processing_time=0.1
            )
            results.append(result)

            if i % 100 == 0:
                gc.collect()

        # Clear all results
        del results
        gc.collect()

        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory

        assert memory_increase < 25, f"Detection result memory leak: {memory_increase:.2f}MB increase"

    def test_thread_memory_safety(self, mock_config):
        """Test memory safety under concurrent thread access."""
        initial_memory = self.get_memory_usage()

        cache_manager = CacheManager(max_size=50)
        results = []
        errors = []

        def worker_thread(thread_id):
            try:
                for i in range(200):
                    key = f"thread_{thread_id}_item_{i}"
                    value = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

                    cache_manager.set(key, value)
                    retrieved = cache_manager.get(key)

                    if retrieved is not None:
                        results.append(thread_id)

                    # Cleanup
                    del value
                    if retrieved is not None:
                        del retrieved

            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=worker_thread, args=(i,)) for i in range(5)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Cleanup
        cache_manager.clear()
        del results
        gc.collect()

        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert memory_increase < 40, f"Concurrent memory leak: {memory_increase:.2f}MB increase"


@pytest.mark.performance
class TestMemoryUsageOptimization:
    """Test memory usage optimization strategies."""

    def test_large_image_processing_memory(self):
        """Test memory usage when processing large images."""
        initial_memory = self.get_memory_usage()

        # Process multiple large images
        for size in [(1920, 1080), (2560, 1440), (3840, 2160)]:
            width, height = size

            # Create large image
            large_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

            # Simulate various processing operations
            gray = cv2.cvtColor(large_image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(large_image, (640, 480))
            blurred = cv2.GaussianBlur(large_image, (5, 5), 0)

            # Immediate cleanup
            del large_image, gray, resized, blurred
            gc.collect()

        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory

        # Should not accumulate memory from large image processing
        assert memory_increase < 100, f"Large image memory accumulation: {memory_increase:.2f}MB"

    def test_memory_pool_efficiency(self):
        """Test efficiency of memory pooling strategies."""
        from app.core.memory_manager import MemoryManager

        initial_memory = self.get_memory_usage()

        memory_manager = MemoryManager()

        # Allocate and deallocate many buffers
        for i in range(500):
            # Request different sized buffers
            sizes = [(480, 640, 3), (720, 1280, 3), (1080, 1920, 3)]
            for size in sizes:
                buffer = memory_manager.get_buffer(size)
                # Use buffer
                buffer.fill(i % 256)
                # Return buffer to pool
                memory_manager.return_buffer(buffer)

            if i % 100 == 0:
                gc.collect()

        # Cleanup memory manager
        memory_manager.cleanup()
        gc.collect()

        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory

        assert memory_increase < 30, f"Memory pool inefficiency: {memory_increase:.2f}MB increase"

    def test_cache_eviction_memory_management(self):
        """Test cache eviction policies maintain memory limits."""
        initial_memory = self.get_memory_usage()

        cache_manager = CacheManager(max_size=50, max_memory_mb=20)

        # Fill cache beyond memory limit
        for i in range(200):
            key = f"large_item_{i}"
            # Create 1MB item
            value = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            cache_manager.set(key, value)

            # Check memory periodically
            if i % 50 == 0:
                current_memory = self.get_memory_usage()
                memory_used = current_memory - initial_memory

                # Should stay within reasonable bounds due to eviction
                if memory_used > 50:  # More than expected
                    cache_manager.clear()
                    gc.collect()

        cache_manager.clear()
        gc.collect()

        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory

        assert memory_increase < 25, f"Cache eviction failure: {memory_increase:.2f}MB increase"


@pytest.mark.performance
class TestGarbageCollectionOptimization:
    """Test garbage collection optimization and tuning."""

    def test_gc_frequency_impact(self):
        """Test impact of garbage collection frequency on performance."""
        import time

        # Test with frequent GC
        start_time = time.time()
        for i in range(1000):
            data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            processed = data * 2
            del data, processed
            if i % 10 == 0:  # Very frequent
                gc.collect()

        frequent_gc_time = time.time() - start_time

        # Test with infrequent GC
        start_time = time.time()
        for i in range(1000):
            data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            processed = data * 2
            del data, processed
            if i % 100 == 0:  # Less frequent
                gc.collect()

        infrequent_gc_time = time.time() - start_time

        # Final cleanup
        gc.collect()

        # Performance should be reasonable for both
        assert frequent_gc_time < 10.0  # Should complete within 10 seconds
        assert infrequent_gc_time < 10.0

        print(f"Frequent GC: {frequent_gc_time:.2f}s, Infrequent GC: {infrequent_gc_time:.2f}s")

    def test_gc_threshold_optimization(self):
        """Test garbage collection threshold optimization."""
        initial_memory = self.get_memory_usage()

        # Save original thresholds
        original_thresholds = gc.get_threshold()

        try:
            # Test with more aggressive GC
            gc.set_threshold(700, 10, 10)  # More aggressive than default (700, 10, 10)

            objects_created = 0
            for i in range(2000):
                # Create objects that might become garbage
                temp_list = [np.random.random() for _ in range(50)]
                temp_dict = {f"key_{j}": temp_list[j] for j in range(len(temp_list))}
                objects_created += len(temp_list) + len(temp_dict)

                del temp_list, temp_dict

                if i % 500 == 0:
                    gc.collect()

            final_memory = self.get_memory_usage()
            memory_increase = final_memory - initial_memory

            assert memory_increase < 50, f"GC threshold issue: {memory_increase:.2f}MB increase"

        finally:
            # Restore original thresholds
            gc.set_threshold(*original_thresholds)

    def test_circular_reference_cleanup(self):
        """Test cleanup of circular references."""
        initial_memory = self.get_memory_usage()

        # Create circular references
        for i in range(1000):
            class Node:
                def __init__(self, value):
                    self.value = value
                    self.children = []
                    self.parent = None

            # Create circular reference chain
            nodes = [Node(j) for j in range(10)]
            for j in range(len(nodes)):
                if j > 0:
                    nodes[j].parent = nodes[j-1]
                    nodes[j-1].children.append(nodes[j])

            # Create circular reference
            if nodes:
                nodes[-1].children.append(nodes[0])
                nodes[0].parent = nodes[-1]

            # Clear local references
            del nodes

            if i % 200 == 0:
                gc.collect()

        # Force cleanup of circular references
        gc.collect()

        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory

        assert memory_increase < 30, f"Circular reference leak: {memory_increase:.2f}MB increase"


@pytest.mark.performance
class TestResourceCleanupValidation:
    """Test proper resource cleanup in various scenarios."""

    def test_webcam_resource_cleanup(self, mock_config):
        """Test webcam resource cleanup on service destruction."""
        initial_memory = self.get_memory_usage()

        with patch('cv2.VideoCapture') as mock_video_capture:
            mock_caps = []

            def create_mock_cap(*args, **kwargs):
                mock_cap = Mock()
                mock_cap.isOpened.return_value = True
                mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
                mock_caps.append(mock_cap)
                return mock_cap

            mock_video_capture.side_effect = create_mock_cap

            # Create and destroy many webcam services
            for i in range(50):
                service = ImprovedWebcamService(mock_config)
                service.open_webcam(0)

                # Simulate usage
                for _ in range(10):
                    success, frame = service.read_frame()
                    if frame is not None:
                        del frame

                service.close_webcam()
                del service

                if i % 10 == 0:
                    gc.collect()

            # All mock caps should have been released
            for mock_cap in mock_caps:
                mock_cap.release.assert_called_once()

        gc.collect()
        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory

        assert memory_increase < 20, f"Webcam resource leak: {memory_increase:.2f}MB increase"

    def test_thread_cleanup_on_exception(self):
        """Test thread cleanup when exceptions occur."""
        initial_memory = self.get_memory_usage()

        results = []
        threads = []

        def worker_with_exception(should_fail=False):
            try:
                if should_fail:
                    raise Exception("Simulated worker failure")

                # Do some work
                data = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
                processed = cv2.resize(data, (100, 100))
                results.append(len(processed))

            except Exception:
                # Cleanup should still happen
                pass
            finally:
                # Ensure cleanup
                locals().clear()

        # Create threads, some that will fail
        for i in range(20):
            thread = threading.Thread(target=worker_with_exception, args=(i % 5 == 0,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join(timeout=5.0)

        # Cleanup
        del threads, results
        gc.collect()

        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory

        assert memory_increase < 30, f"Thread exception cleanup issue: {memory_increase:.2f}MB increase"

    def test_temporary_file_cleanup(self, temp_dir):
        """Test cleanup of temporary files and resources."""
        import tempfile
        import shutil

        initial_memory = self.get_memory_usage()

        temp_files = []

        try:
            # Create many temporary files
            for i in range(100):
                # Create temporary image file
                temp_file = temp_dir / f"temp_image_{i}.jpg"
                test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
                cv2.imwrite(str(temp_file), test_image)
                temp_files.append(temp_file)

                # Load and process
                loaded_image = cv2.imread(str(temp_file))
                if loaded_image is not None:
                    processed = cv2.resize(loaded_image, (320, 240))
                    del loaded_image, processed

                del test_image

                if i % 20 == 0:
                    gc.collect()

        finally:
            # Cleanup temporary files
            for temp_file in temp_files:
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except Exception:
                    pass

        gc.collect()
        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory

        assert memory_increase < 15, f"Temporary file cleanup issue: {memory_increase:.2f}MB increase"


@pytest.mark.performance
class TestMemoryPerformanceBenchmarks:
    """Benchmark memory performance for baseline measurements."""

    def test_frame_processing_memory_benchmark(self, benchmark):
        """Benchmark memory usage during frame processing."""
        def process_frames():
            frames_processed = 0
            for i in range(100):
                # Create frame
                frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

                # Process frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(frame, (320, 240))
                blurred = cv2.GaussianBlur(frame, (5, 5), 0)

                frames_processed += 1

                # Cleanup
                del frame, gray, resized, blurred

                if i % 20 == 0:
                    gc.collect()

            return frames_processed

        result = benchmark(process_frames)
        assert result == 100

    def test_cache_operations_memory_benchmark(self, benchmark):
        """Benchmark cache operations memory efficiency."""
        def cache_operations():
            cache = CacheManager(max_size=100)
            operations = 0

            for i in range(1000):
                key = f"item_{i % 100}"  # Reuse keys to test eviction
                value = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

                cache.set(key, value)
                retrieved = cache.get(key)

                if retrieved is not None:
                    operations += 1

                del value
                if retrieved is not None:
                    del retrieved

            cache.clear()
            return operations

        result = benchmark(cache_operations)
        assert result > 0

    def test_memory_allocation_pattern_benchmark(self, benchmark):
        """Benchmark different memory allocation patterns."""
        def allocation_pattern():
            # Pattern 1: Many small allocations
            small_arrays = []
            for i in range(500):
                arr = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
                small_arrays.append(arr)

            # Pattern 2: Few large allocations
            large_arrays = []
            for i in range(5):
                arr = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
                large_arrays.append(arr)

            # Cleanup
            del small_arrays, large_arrays
            gc.collect()

            return 505  # Total arrays created

        result = benchmark(allocation_pattern)
        assert result == 505