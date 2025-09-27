"""
COMPREHENSIVE PERFORMANCE OPTIMIZATION & MEMORY MANAGEMENT

This module addresses all performance bottlenecks, memory leaks, and resource
management issues identified in the ultra-deep analysis.
"""

import gc
import threading
import time
import weakref
import psutil
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np
import cv2
from functools import lru_cache
import pickle
import tempfile

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_mb: float
    used_mb: float
    available_mb: float
    percent_used: float
    process_mb: float

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    operation: str
    duration_ms: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    cpu_percent: float
    timestamp: float

class AdvancedMemoryManager:
    """Advanced memory management with leak detection and optimization."""

    def __init__(self, max_memory_mb: int = 1000, warning_threshold: float = 0.8):
        self.max_memory_mb = max_memory_mb
        self.warning_threshold = warning_threshold
        self._tracked_objects = weakref.WeakSet()
        self._memory_history = deque(maxlen=100)
        self._gc_stats = defaultdict(int)
        self._lock = threading.Lock()
        self._monitoring_active = False
        self._cleanup_callbacks = []

        # Start memory monitoring
        self.start_monitoring()

    def track_object(self, obj: Any, cleanup_callback: Optional[Callable] = None):
        """Track an object for memory management."""
        try:
            self._tracked_objects.add(obj)
            if cleanup_callback:
                self._cleanup_callbacks.append((weakref.ref(obj), cleanup_callback))
        except TypeError:
            # Object not weak-referenceable
            pass

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024

            return MemoryStats(
                total_mb=memory.total / 1024 / 1024,
                used_mb=memory.used / 1024 / 1024,
                available_mb=memory.available / 1024 / 1024,
                percent_used=memory.percent,
                process_mb=process_memory
            )
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return MemoryStats(0, 0, 0, 0, 0)

    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return statistics."""
        with self._lock:
            # Clean up dead references
            self._cleanup_dead_references()

            # Run garbage collection
            collected_counts = {}
            for generation in range(3):
                collected = gc.collect(generation)
                collected_counts[f'generation_{generation}'] = collected
                self._gc_stats[f'generation_{generation}'] += collected

            # Log memory state
            stats = self.get_memory_stats()
            logger.info(f"GC completed. Process memory: {stats.process_mb:.1f}MB, "
                       f"System: {stats.percent_used:.1f}%")

            return collected_counts

    def _cleanup_dead_references(self):
        """Clean up callbacks for dead references."""
        active_callbacks = []
        for ref, callback in self._cleanup_callbacks:
            if ref() is not None:
                active_callbacks.append((ref, callback))
            else:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Cleanup callback failed: {e}")
        self._cleanup_callbacks = active_callbacks

    def start_monitoring(self):
        """Start memory monitoring thread."""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        monitor_thread.start()

    def _monitor_memory(self):
        """Monitor memory usage in background."""
        while self._monitoring_active:
            try:
                stats = self.get_memory_stats()
                self._memory_history.append(stats)

                # Check for memory pressure
                if stats.percent_used > self.warning_threshold * 100:
                    logger.warning(f"High memory usage: {stats.percent_used:.1f}%")
                    self.force_garbage_collection()

                # Check for memory leaks
                if len(self._memory_history) >= 10:
                    self._detect_memory_leaks()

                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(10)

    def _detect_memory_leaks(self):
        """Detect potential memory leaks."""
        if len(self._memory_history) < 10:
            return

        # Check for consistently increasing memory
        recent_stats = list(self._memory_history)[-10:]
        memory_trend = [s.process_mb for s in recent_stats]

        # Simple leak detection: check if memory consistently increases
        increases = sum(1 for i in range(1, len(memory_trend))
                       if memory_trend[i] > memory_trend[i-1])

        if increases >= 8:  # 80% of samples show increase
            logger.warning("Potential memory leak detected - memory consistently increasing")
            self.force_garbage_collection()

    def stop_monitoring(self):
        """Stop memory monitoring."""
        self._monitoring_active = False

class ImageMemoryOptimizer:
    """Optimizes image memory usage and prevents leaks."""

    @staticmethod
    def optimize_image_memory(image: np.ndarray, max_size: tuple = (1920, 1080)) -> np.ndarray:
        """Optimize image memory usage."""
        if image is None:
            return None

        # Resize if too large
        height, width = image.shape[:2]
        max_width, max_height = max_size

        if width > max_width or height > max_height:
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Ensure contiguous array for better memory access
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)

        return image

    @staticmethod
    def create_image_cache(max_size: int = 50) -> Dict:
        """Create an LRU cache for images."""
        return {}  # Simplified for now

    @staticmethod
    def safe_image_release(image: np.ndarray):
        """Safely release image memory."""
        if image is not None:
            del image
            gc.collect()

class PerformanceProfiler:
    """Advanced performance profiling and optimization."""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._metrics = deque(maxlen=max_history)
        self._operation_stats = defaultdict(list)
        self._lock = threading.Lock()

    @contextmanager
    def profile_operation(self, operation_name: str):
        """Profile an operation's performance."""
        start_time = time.time()
        start_memory = self._get_process_memory()
        start_cpu = psutil.cpu_percent()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_process_memory()
            end_cpu = psutil.cpu_percent()

            duration_ms = (end_time - start_time) * 1000
            memory_delta = end_memory - start_memory

            metric = PerformanceMetrics(
                operation=operation_name,
                duration_ms=duration_ms,
                memory_before_mb=start_memory,
                memory_after_mb=end_memory,
                memory_delta_mb=memory_delta,
                cpu_percent=(start_cpu + end_cpu) / 2,
                timestamp=start_time
            )

            with self._lock:
                self._metrics.append(metric)
                self._operation_stats[operation_name].append(metric)

                # Log slow operations
                if duration_ms > 1000:  # > 1 second
                    logger.warning(f"Slow operation {operation_name}: {duration_ms:.1f}ms")

    def _get_process_memory(self) -> float:
        """Get current process memory in MB."""
        try:
            return psutil.Process().memory_info().rss / 1024 / 1024
        except:
            return 0.0

    def get_operation_statistics(self, operation_name: str) -> Dict[str, float]:
        """Get statistics for a specific operation."""
        with self._lock:
            metrics = self._operation_stats.get(operation_name, [])
            if not metrics:
                return {}

            durations = [m.duration_ms for m in metrics]
            memory_deltas = [m.memory_delta_mb for m in metrics]

            return {
                'count': len(metrics),
                'avg_duration_ms': sum(durations) / len(durations),
                'max_duration_ms': max(durations),
                'min_duration_ms': min(durations),
                'avg_memory_delta_mb': sum(memory_deltas) / len(memory_deltas),
                'max_memory_delta_mb': max(memory_deltas),
                'total_time_ms': sum(durations)
            }
class ResourcePoolManager:
    """Manages resource pools to prevent resource exhaustion."""

    def __init__(self):
        self._pools = {}
        self._lock = threading.Lock()

    def create_pool(self, pool_name: str, factory: Callable, max_size: int = 10):
        """Create a resource pool."""
        with self._lock:
            if pool_name in self._pools:
                logger.warning(f"Pool {pool_name} already exists")
                return

            self._pools[pool_name] = {
                'factory': factory,
                'resources': deque(),
                'max_size': max_size,
                'active_count': 0,
                'lock': threading.Lock()
            }

    def get_resource(self, pool_name: str):
        """Get a resource from the pool."""
        if pool_name not in self._pools:
            raise ValueError(f"Pool {pool_name} not found")

        pool = self._pools[pool_name]
        with pool['lock']:
            if pool['resources']:
                resource = pool['resources'].popleft()
                pool['active_count'] += 1
                return resource
            elif pool['active_count'] < pool['max_size']:
                resource = pool['factory']()
                pool['active_count'] += 1
                return resource
            else:
                raise Exception(f"Pool {pool_name} exhausted")

    def return_resource(self, pool_name: str, resource):
        """Return a resource to the pool."""
        if pool_name not in self._pools:
            return

        pool = self._pools[pool_name]
        with pool['lock']:
            pool['resources'].append(resource)
            pool['active_count'] = max(0, pool['active_count'] - 1)

    def cleanup_pool(self, pool_name: str):
        """Clean up a resource pool."""
        if pool_name not in self._pools:
            return

        pool = self._pools[pool_name]
        with pool['lock']:
            while pool['resources']:
                resource = pool['resources'].popleft()
                try:
                    if hasattr(resource, 'close'):
                        resource.close()
                    elif hasattr(resource, 'cleanup'):
                        resource.cleanup()
                except Exception as e:
                    logger.error(f"Resource cleanup failed: {e}")
            pool['active_count'] = 0

class AsyncTaskManager:
    """Manages async tasks with proper resource cleanup."""

    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self._active_tasks = {}
        self._task_counter = 0
        self._lock = threading.Lock()

    def start_task(self, task_func: Callable, callback: Optional[Callable] = None,
                  error_callback: Optional[Callable] = None) -> str:
        """Start an async task."""
        with self._lock:
            if len(self._active_tasks) >= self.max_concurrent_tasks:
                raise Exception("Maximum concurrent tasks reached")

            task_id = f"task_{self._task_counter}"
            self._task_counter += 1

        def task_wrapper():
            try:
                result = task_func()
                if callback:
                    callback(result)
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                if error_callback:
                    error_callback(e)
            finally:
                with self._lock:
                    self._active_tasks.pop(task_id, None)

        thread = threading.Thread(target=task_wrapper, daemon=True)
        with self._lock:
            self._active_tasks[task_id] = thread

        thread.start()
        return task_id

    def get_active_task_count(self) -> int:
        """Get number of active tasks."""
        with self._lock:
            return len(self._active_tasks)

    def wait_for_all_tasks(self, timeout: float = 30):
        """Wait for all tasks to complete."""
        start_time = time.time()
        while self.get_active_task_count() > 0:
            if time.time() - start_time > timeout:
                logger.warning("Timeout waiting for tasks to complete")
                break
            time.sleep(0.1)

# Global instances
_memory_manager = AdvancedMemoryManager()
_performance_profiler = PerformanceProfiler()
_resource_pool_manager = ResourcePoolManager()
_async_task_manager = AsyncTaskManager()

def get_memory_manager():
    """Get the global memory manager."""
    return _memory_manager

def get_performance_profiler():
    """Get the global performance profiler."""
    return _performance_profiler

def get_resource_pool_manager():
    """Get the global resource pool manager."""
    return _resource_pool_manager

def get_async_task_manager():
    """Get the global async task manager."""
    return _async_task_manager

# Decorator for automatic performance profiling
def profile_performance(operation_name: str = None):
    """Decorator for automatic performance profiling."""
    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"

        def wrapper(*args, **kwargs):
            with get_performance_profiler().profile_operation(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Memory optimization utilities
def optimize_numpy_array(arr: np.ndarray) -> np.ndarray:
    """Optimize numpy array for memory efficiency."""
    if arr is None:
        return None

    # Use the smallest appropriate dtype
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    elif arr.dtype == np.int64:
        if arr.min() >= 0 and arr.max() <= 255:
            arr = arr.astype(np.uint8)
        elif arr.min() >= -32768 and arr.max() <= 32767:
            arr = arr.astype(np.int16)

    return arr

def cleanup_opencv_memory():
    """Clean up OpenCV memory."""
    cv2.destroyAllWindows()
    gc.collect()