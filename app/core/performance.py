"""Performance monitoring and optimization system for the Vision Analysis System."""

import time
import psutil
import threading
import gc
import sys
import cProfile
import pstats
import io
from typing import Dict, List, Optional, Any, Callable, Tuple
from functools import wraps, lru_cache
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import weakref
import tracemalloc
import logging

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    cpu_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_percent: float = 0.0
    fps: float = 0.0
    frame_processing_time: float = 0.0
    ui_update_time: float = 0.0
    api_response_time: float = 0.0
    cache_hit_rate: float = 0.0
    active_threads: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        duration = self.duration
        PerformanceMonitor.instance().record_operation_time(self.operation_name, duration)
    
    @property
    def duration(self) -> float:
        """Get operation duration in seconds."""
        if self.end_time is None or self.start_time is None:
            return 0.0
        return self.end_time - self.start_time

def performance_timer(operation_name: str = None):
    """Decorator for timing function execution."""
    def decorator(func: Callable) -> Callable:
        name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with PerformanceTimer(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

class LRUCache:
    """High-performance LRU cache implementation with statistics."""
    
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self._cache = {}
        self._access_order = deque()
        self._stats = CacheStats(max_size=max_size)
        self._lock = threading.RLock()
    
    def get(self, key: Any) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._access_order.remove(key)
                self._access_order.append(key)
                self._stats.hits += 1
                return self._cache[key]
            else:
                self._stats.misses += 1
                return None
    
    def put(self, key: Any, value: Any) -> None:
        """Put item in cache."""
        with self._lock:
            if key in self._cache:
                # Update existing
                self._cache[key] = value
                self._access_order.remove(key)
                self._access_order.append(key)
            else:
                # Add new item
                if len(self._cache) >= self.max_size:
                    # Evict least recently used
                    oldest_key = self._access_order.popleft()
                    del self._cache[oldest_key]
                    self._stats.evictions += 1
                
                self._cache[key] = value
                self._access_order.append(key)
                self._stats.size = len(self._cache)
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._stats.size = 0
    
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

class ImageCache(LRUCache):
    """Specialized cache for image data with memory management."""
    
    def __init__(self, max_size: int = 32, max_memory_mb: int = 512):
        super().__init__(max_size)
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._memory_usage = 0
        self._weak_refs = weakref.WeakSet()
    
    def put(self, key: Any, image_data: Any) -> None:
        """Put image in cache with memory management."""
        # Estimate memory usage (rough calculation for numpy arrays)
        if hasattr(image_data, 'nbytes'):
            data_size = image_data.nbytes
        else:
            data_size = sys.getsizeof(image_data)
        
        # Check memory limit
        while (self._memory_usage + data_size > self.max_memory_bytes and 
               len(self._cache) > 0):
            self._evict_oldest()
        
        super().put(key, image_data)
        self._memory_usage += data_size
        
        # Add to weak reference set for memory tracking
        if hasattr(image_data, '__weakref__'):
            self._weak_refs.add(image_data)
    
    def _evict_oldest(self) -> None:
        """Evict oldest item and update memory usage."""
        if not self._access_order:
            return
        
        oldest_key = self._access_order[0]
        oldest_value = self._cache.get(oldest_key)
        
        if oldest_value is not None:
            if hasattr(oldest_value, 'nbytes'):
                self._memory_usage -= oldest_value.nbytes
            else:
                self._memory_usage -= sys.getsizeof(oldest_value)
        
        # Remove from cache
        self._access_order.popleft()
        del self._cache[oldest_key]
        self._stats.evictions += 1
        self._stats.size = len(self._cache)

class ThreadPool:
    """High-performance thread pool with monitoring."""
    
    def __init__(self, max_workers: int = 4, thread_name_prefix: str = "Worker"):
        self.max_workers = max_workers
        self.thread_name_prefix = thread_name_prefix
        self._workers = []
        self._task_queue = deque()
        self._lock = threading.Lock()
        self._shutdown = False
        self._active_tasks = 0
        
        # Start worker threads
        self._start_workers()
    
    def _start_workers(self):
        """Start worker threads."""
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker,
                name=f"{self.thread_name_prefix}-{i}",
                daemon=True
            )
            worker.start()
            self._workers.append(worker)
    
    def _worker(self):
        """Worker thread main loop."""
        while not self._shutdown:
            task = None
            
            with self._lock:
                if self._task_queue:
                    task = self._task_queue.popleft()
                    self._active_tasks += 1
            
            if task:
                try:
                    func, args, kwargs, callback = task
                    result = func(*args, **kwargs)
                    if callback:
                        callback(result, None)
                except Exception as e:
                    if len(task) > 3 and task[3]:
                        task[3](None, e)
                    logger.exception(f"Task execution failed: {e}")
                finally:
                    with self._lock:
                        self._active_tasks -= 1
            else:
                time.sleep(0.01)  # Small delay when no tasks
    
    def submit(self, func: Callable, *args, callback: Callable = None, **kwargs):
        """Submit task to thread pool."""
        if self._shutdown:
            raise RuntimeError("ThreadPool is shutdown")
        
        with self._lock:
            self._task_queue.append((func, args, kwargs, callback))
    
    def shutdown(self, wait: bool = True):
        """Shutdown thread pool."""
        self._shutdown = True
        if wait:
            for worker in self._workers:
                worker.join(timeout=1.0)
    
    @property
    def active_tasks(self) -> int:
        """Get number of active tasks."""
        return self._active_tasks

class PerformanceMonitor:
    """Centralized performance monitoring system."""
    
    _instance = None
    _instance_lock = threading.Lock()
    
    def __init__(self):
        if PerformanceMonitor._instance is not None:
            raise RuntimeError("Use PerformanceMonitor.instance() to get singleton")
        
        self._metrics_history = deque(maxlen=1000)  # Keep last 1000 metrics
        self._operation_times = defaultdict(list)
        self._monitoring_thread = None
        self._monitoring_active = False
        self._monitoring_interval = 1.0  # seconds
        self._caches = {}  # Registry of caches
        self._thread_pools = {}  # Registry of thread pools
        
        # Memory tracking
        self._memory_tracker_active = False
        self._memory_snapshots = deque(maxlen=100)
        
        # Start monitoring
        self.start_monitoring()
    
    @classmethod
    def instance(cls) -> 'PerformanceMonitor':
        """Get singleton instance."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def start_monitoring(self):
        """Start performance monitoring thread."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._monitoring_active = True
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="PerformanceMonitor",
                daemon=True
            )
            self._monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=2.0)
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                metrics = self._collect_metrics()
                self._metrics_history.append(metrics)
                
                # Cleanup old operation times
                self._cleanup_operation_times()
                
                time.sleep(self._monitoring_interval)
            except Exception as e:
                logger.exception(f"Monitoring error: {e}")
                time.sleep(1.0)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        process = psutil.Process()
        
        # System metrics
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = process.memory_percent()
        
        # Thread count
        active_threads = threading.active_count()
        
        # Cache hit rate (average across all registered caches)
        cache_hit_rates = []
        for cache in self._caches.values():
            if hasattr(cache, 'stats'):
                cache_hit_rates.append(cache.stats().hit_rate)
        
        avg_cache_hit_rate = sum(cache_hit_rates) / len(cache_hit_rates) if cache_hit_rates else 0.0
        
        return PerformanceMetrics(
            cpu_percent=cpu_percent,
            memory_usage_mb=memory_mb,
            memory_percent=memory_percent,
            cache_hit_rate=avg_cache_hit_rate,
            active_threads=active_threads
        )
    
    def _cleanup_operation_times(self):
        """Clean up old operation times."""
        cutoff_time = time.time() - 300  # Keep last 5 minutes
        for operation, times in self._operation_times.items():
            self._operation_times[operation] = [
                (timestamp, duration) for timestamp, duration in times
                if timestamp > cutoff_time
            ]
    
    def record_operation_time(self, operation: str, duration: float):
        """Record operation execution time."""
        timestamp = time.time()
        self._operation_times[operation].append((timestamp, duration))
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get current performance metrics."""
        if self._metrics_history:
            return self._metrics_history[-1]
        return None
    
    def get_metrics_history(self, last_n: int = 60) -> List[PerformanceMetrics]:
        """Get recent metrics history."""
        return list(self._metrics_history)[-last_n:]
    
    def get_operation_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for specific operation."""
        times = self._operation_times.get(operation, [])
        if not times:
            return {}
        
        durations = [duration for _, duration in times]
        return {
            'count': len(durations),
            'min': min(durations),
            'max': max(durations),
            'avg': sum(durations) / len(durations),
            'total': sum(durations)
        }
    
    def register_cache(self, name: str, cache: LRUCache):
        """Register cache for monitoring."""
        self._caches[name] = cache
    
    def register_thread_pool(self, name: str, thread_pool: ThreadPool):
        """Register thread pool for monitoring."""
        self._thread_pools[name] = thread_pool
    
    def start_memory_tracking(self):
        """Start memory tracking."""
        if not self._memory_tracker_active:
            tracemalloc.start()
            self._memory_tracker_active = True
    
    def take_memory_snapshot(self, label: str = ""):
        """Take memory snapshot."""
        if self._memory_tracker_active:
            snapshot = tracemalloc.take_snapshot()
            self._memory_snapshots.append({
                'timestamp': datetime.now(),
                'label': label,
                'snapshot': snapshot
            })
    
    def get_memory_report(self) -> str:
        """Get memory usage report."""
        if not self._memory_tracker_active or not self._memory_snapshots:
            return "Memory tracking not active"
        
        latest = self._memory_snapshots[-1]['snapshot']
        top_stats = latest.statistics('lineno')[:10]
        
        report = ["Top 10 Memory Usage by Line:"]
        for stat in top_stats:
            report.append(f"{stat}")
        
        return "\n".join(report)
    
    def force_garbage_collection(self):
        """Force garbage collection and return collected count."""
        return gc.collect()
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, str]:
        """Profile function execution and return result with stats."""
        profiler = cProfile.Profile()
        
        profiler.enable()
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
        
        # Get stats
        stats_buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_buffer)
        stats.sort_stats('cumulative').print_stats(20)
        
        return result, stats_buffer.getvalue()

# Decorator for automatic cache management
def cached_result(cache_key_func: Callable = None, cache_name: str = "default", max_size: int = 128):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        # Get or create cache
        monitor = PerformanceMonitor.instance()
        if cache_name not in monitor._caches:
            cache = LRUCache(max_size=max_size)
            monitor.register_cache(cache_name, cache)
        else:
            cache = monitor._caches[cache_name]
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                key = cache_key_func(*args, **kwargs)
            else:
                key = str(args) + str(sorted(kwargs.items()))
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result
        
        return wrapper
    return decorator

# Memory optimization utilities
def optimize_memory():
    """Perform memory optimization operations."""
    # Force garbage collection
    collected = gc.collect()
    
    # Clear weak references
    gc.collect()
    
    return collected

def get_object_size(obj) -> int:
    """Get deep size of object."""
    seen = set()
    
    def sizeof(obj):
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        
        size = sys.getsizeof(obj)
        
        if isinstance(obj, dict):
            size += sum(sizeof(k) + sizeof(v) for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set)):
            size += sum(sizeof(item) for item in obj)
        
        return size
    
    return sizeof(obj)