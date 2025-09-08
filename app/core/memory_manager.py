"""Advanced memory management system for the Vision Analysis System."""

import gc
import sys
import threading
import time
import tracemalloc
import psutil
import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Callable, Set
from collections import defaultdict, deque
from datetime import datetime, timedelta
import weakref
from dataclasses import dataclass
import logging

from .performance import PerformanceMonitor, optimize_memory

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Memory statistics data structure."""
    total_memory_mb: float
    used_memory_mb: float
    available_memory_mb: float
    memory_percent: float
    gc_collections: int
    tracked_objects: int
    leaked_objects: int
    timestamp: datetime

class MemoryTracker:
    """Track memory usage of specific objects and detect leaks."""
    
    def __init__(self):
        self._tracked_objects = weakref.WeakSet()
        self._object_counts = defaultdict(int)
        self._creation_times = {}
        self._leak_threshold_hours = 24
        self._max_objects_per_type = 1000
    
    def track_object(self, obj: Any, obj_type: str = None):
        """Track an object for memory monitoring."""
        if obj_type is None:
            obj_type = type(obj).__name__
        
        try:
            self._tracked_objects.add(obj)
            self._object_counts[obj_type] += 1
            self._creation_times[id(obj)] = datetime.now()
        except TypeError:
            # Object doesn't support weak references
            pass
    
    def get_tracked_count(self, obj_type: str = None) -> int:
        """Get count of tracked objects."""
        if obj_type:
            return self._object_counts.get(obj_type, 0)
        return len(self._tracked_objects)
    
    def detect_leaks(self) -> Dict[str, int]:
        """Detect potential memory leaks."""
        leaks = {}
        cutoff_time = datetime.now() - timedelta(hours=self._leak_threshold_hours)
        
        for obj_id, creation_time in list(self._creation_times.items()):
            if creation_time < cutoff_time:
                # Object has been alive for too long, potential leak
                obj_type = "unknown"
                for tracked_obj in self._tracked_objects:
                    if id(tracked_obj) == obj_id:
                        obj_type = type(tracked_obj).__name__
                        break
                
                leaks[obj_type] = leaks.get(obj_type, 0) + 1
        
        return leaks
    
    def cleanup_stale_references(self):
        """Clean up stale object references."""
        # Clean up creation times for objects that no longer exist
        active_ids = {id(obj) for obj in self._tracked_objects}
        stale_ids = set(self._creation_times.keys()) - active_ids
        
        for stale_id in stale_ids:
            del self._creation_times[stale_id]
        
        # Update object counts
        self._object_counts.clear()
        for obj in self._tracked_objects:
            obj_type = type(obj).__name__
            self._object_counts[obj_type] += 1

class ImageMemoryManager:
    """Specialized memory manager for image data."""
    
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cached_images = {}
        self._image_sizes = {}
        self._access_times = {}
        self._lock = threading.RLock()
        
        # Image processing optimizations
        self._image_pool = deque(maxlen=10)  # Reusable image arrays
        self._processing_cache = {}  # Cache for processed images
    
    def add_image(self, key: str, image: np.ndarray, metadata: Dict = None) -> bool:
        """Add image to memory management."""
        if image is None:
            return False
        
        image_size = image.nbytes
        
        with self._lock:
            # Check if adding this image would exceed memory limit
            current_memory = sum(self._image_sizes.values())
            if current_memory + image_size > self.max_memory_bytes:
                # Free some memory by removing least recently used images
                self._free_memory(image_size)
            
            # Store image and metadata
            self._cached_images[key] = image.copy() if isinstance(image, np.ndarray) else image
            self._image_sizes[key] = image_size
            self._access_times[key] = time.time()
            
            return True
    
    def get_image(self, key: str) -> Optional[np.ndarray]:
        """Get image from memory management."""
        with self._lock:
            if key in self._cached_images:
                self._access_times[key] = time.time()
                return self._cached_images[key]
            return None
    
    def remove_image(self, key: str) -> bool:
        """Remove image from memory management."""
        with self._lock:
            if key in self._cached_images:
                del self._cached_images[key]
                del self._image_sizes[key]
                del self._access_times[key]
                return True
            return False
    
    def _free_memory(self, needed_bytes: int):
        """Free memory by removing least recently used images."""
        # Sort by access time (oldest first)
        sorted_images = sorted(self._access_times.items(), key=lambda x: x[1])
        
        freed_bytes = 0
        for key, _ in sorted_images:
            if freed_bytes >= needed_bytes:
                break
            
            freed_bytes += self._image_sizes[key]
            del self._cached_images[key]
            del self._image_sizes[key]
            del self._access_times[key]
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        with self._lock:
            total_size = sum(self._image_sizes.values())
            return {
                'total_images': len(self._cached_images),
                'total_memory_mb': total_size / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'memory_utilization': total_size / self.max_memory_bytes if self.max_memory_bytes > 0 else 0
            }
    
    def get_reusable_array(self, shape: tuple, dtype=np.uint8) -> np.ndarray:
        """Get reusable array from pool to avoid allocations."""
        required_size = np.prod(shape) * np.dtype(dtype).itemsize
        
        # Try to find suitable array in pool
        for i, arr in enumerate(self._image_pool):
            if arr.nbytes >= required_size and arr.dtype == dtype:
                # Remove from pool and return
                reused_array = self._image_pool[i]
                del self._image_pool[i]
                
                # Reshape if needed
                if reused_array.shape != shape:
                    return reused_array[:np.prod(shape)].reshape(shape)
                return reused_array
        
        # No suitable array found, create new one
        return np.empty(shape, dtype=dtype)
    
    def return_array(self, arr: np.ndarray):
        """Return array to pool for reuse."""
        if arr is not None and arr.nbytes > 0:
            self._image_pool.append(arr.copy())
    
    def optimize_image_processing(self, image: np.ndarray, operation: str) -> np.ndarray:
        """Optimize image processing operations with caching and pooling."""
        # Generate cache key
        image_hash = hash(image.tobytes()[:1000])  # Use sample for speed
        cache_key = f"{operation}:{image_hash}:{image.shape}"
        
        # Check cache
        if cache_key in self._processing_cache:
            return self._processing_cache[cache_key]
        
        # Perform operation with memory-efficient approach
        if operation == 'resize':
            # Use memory pool for temporary arrays
            temp_array = self.get_reusable_array(image.shape)
            try:
                # Processing logic would go here
                result = image.copy()  # Placeholder
                
                # Cache result
                self._processing_cache[cache_key] = result
                return result
            finally:
                self.return_array(temp_array)
        
        return image

class MemoryManager:
    """Central memory management system."""
    
    def __init__(self, max_memory_mb: int = 2048, enable_tracking: bool = True):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.enable_tracking = enable_tracking
        
        # Memory components
        self.image_manager = ImageMemoryManager(max_memory_mb // 2)  # Half for images
        self.tracker = MemoryTracker()
        
        # Memory monitoring
        self._memory_stats = deque(maxlen=100)
        self._monitoring_thread = None
        self._monitoring_active = False
        self._monitoring_interval = 5.0  # seconds
        
        # Garbage collection optimization
        self._gc_thresholds = (700, 10, 10)  # Optimized thresholds
        self._last_gc_time = time.time()
        self._gc_interval = 30.0  # Force GC every 30 seconds
        
        # Memory pressure management
        self._memory_pressure_callbacks = []
        self._memory_warning_threshold = 0.85  # 85% memory usage
        self._memory_critical_threshold = 0.95  # 95% memory usage
        
        # Start memory tracking if enabled
        if enable_tracking:
            tracemalloc.start()
        
        # Configure garbage collection
        gc.set_threshold(*self._gc_thresholds)
        
        # Start monitoring
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start memory monitoring thread."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._monitoring_active = True
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="MemoryMonitor",
                daemon=True
            )
            self._monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=2.0)
    
    def _monitoring_loop(self):
        """Memory monitoring main loop."""
        while self._monitoring_active:
            try:
                stats = self._collect_memory_stats()
                self._memory_stats.append(stats)
                
                # Check for memory pressure
                self._check_memory_pressure(stats)
                
                # Perform periodic cleanup
                self._perform_periodic_cleanup()
                
                time.sleep(self._monitoring_interval)
                
            except Exception as e:
                logger.exception(f"Memory monitoring error: {e}")
                time.sleep(1.0)
    
    def _collect_memory_stats(self) -> MemoryStats:
        """Collect current memory statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        # Get GC stats
        gc_stats = gc.get_stats()
        total_collections = sum(gen['collections'] for gen in gc_stats)
        
        return MemoryStats(
            total_memory_mb=system_memory.total / (1024 * 1024),
            used_memory_mb=memory_info.rss / (1024 * 1024),
            available_memory_mb=system_memory.available / (1024 * 1024),
            memory_percent=process.memory_percent(),
            gc_collections=total_collections,
            tracked_objects=self.tracker.get_tracked_count(),
            leaked_objects=len(self.tracker.detect_leaks()),
            timestamp=datetime.now()
        )
    
    def _check_memory_pressure(self, stats: MemoryStats):
        """Check for memory pressure and take action."""
        memory_usage_ratio = stats.memory_percent / 100.0
        
        if memory_usage_ratio > self._memory_critical_threshold:
            # Critical memory pressure - aggressive cleanup
            logger.warning(f"Critical memory pressure: {stats.memory_percent:.1f}%")
            self.emergency_cleanup()
            
            # Notify callbacks
            for callback in self._memory_pressure_callbacks:
                try:
                    callback('critical', stats)
                except Exception as e:
                    logger.error(f"Memory pressure callback error: {e}")
        
        elif memory_usage_ratio > self._memory_warning_threshold:
            # Warning level - gentle cleanup
            logger.info(f"Memory warning: {stats.memory_percent:.1f}%")
            self.optimize_memory_usage()
            
            # Notify callbacks
            for callback in self._memory_pressure_callbacks:
                try:
                    callback('warning', stats)
                except Exception as e:
                    logger.error(f"Memory pressure callback error: {e}")
    
    def _perform_periodic_cleanup(self):
        """Perform periodic memory cleanup."""
        current_time = time.time()
        
        # Force garbage collection periodically
        if current_time - self._last_gc_time >= self._gc_interval:
            collected = gc.collect()
            self._last_gc_time = current_time
            
            if collected > 0:
                logger.debug(f"Garbage collection freed {collected} objects")
        
        # Clean up tracker
        self.tracker.cleanup_stale_references()
    
    def register_memory_pressure_callback(self, callback: Callable[[str, MemoryStats], None]):
        """Register callback for memory pressure events."""
        self._memory_pressure_callbacks.append(callback)
    
    def track_object(self, obj: Any, obj_type: str = None):
        """Track object for memory monitoring."""
        if self.enable_tracking:
            self.tracker.track_object(obj, obj_type)
    
    def optimize_memory_usage(self):
        """Optimize memory usage with gentle cleanup."""
        # Force garbage collection
        collected = optimize_memory()
        
        # Clear some caches
        monitor = PerformanceMonitor.instance()
        if hasattr(monitor, '_caches'):
            for cache_name, cache in monitor._caches.items():
                if hasattr(cache, 'clear') and cache_name.endswith('_temp'):
                    cache.clear()
        
        logger.debug(f"Memory optimization freed {collected} objects")
    
    def emergency_cleanup(self):
        """Emergency memory cleanup - aggressive."""
        # Multiple garbage collection passes
        total_collected = 0
        for _ in range(3):
            total_collected += gc.collect()
        
        # Clear all temporary caches
        monitor = PerformanceMonitor.instance()
        if hasattr(monitor, '_caches'):
            for cache in monitor._caches.values():
                if hasattr(cache, 'clear'):
                    cache.clear()
        
        # Clear image manager cache
        self.image_manager._cached_images.clear()
        self.image_manager._image_sizes.clear()
        self.image_manager._access_times.clear()
        
        logger.warning(f"Emergency cleanup freed {total_collected} objects")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report."""
        if not self._memory_stats:
            return {'error': 'No memory statistics available'}
        
        latest = self._memory_stats[-1]
        
        # Calculate trends
        if len(self._memory_stats) > 1:
            previous = self._memory_stats[-2]
            memory_trend = latest.used_memory_mb - previous.used_memory_mb
        else:
            memory_trend = 0
        
        # Detect leaks
        leaks = self.tracker.detect_leaks()
        
        return {
            'current': {
                'used_memory_mb': latest.used_memory_mb,
                'memory_percent': latest.memory_percent,
                'available_memory_mb': latest.available_memory_mb,
                'tracked_objects': latest.tracked_objects
            },
            'trends': {
                'memory_trend_mb': memory_trend,
                'gc_collections': latest.gc_collections
            },
            'image_manager': self.image_manager.get_memory_usage(),
            'potential_leaks': leaks,
            'gc_stats': gc.get_stats()
        }
    
    def get_top_memory_consumers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top memory consuming objects."""
        if not tracemalloc.is_tracing():
            return []
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')[:limit]
        
        consumers = []
        for stat in top_stats:
            consumers.append({
                'file': stat.traceback.format()[-1] if stat.traceback else 'Unknown',
                'size_mb': stat.size / (1024 * 1024),
                'count': stat.count
            })
        
        return consumers
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest memory optimizations based on current usage."""
        suggestions = []
        
        if not self._memory_stats:
            return ['Enable memory monitoring for suggestions']
        
        latest = self._memory_stats[-1]
        
        # High memory usage
        if latest.memory_percent > 80:
            suggestions.append("High memory usage detected - consider increasing available RAM")
        
        # Too many tracked objects
        if latest.tracked_objects > 10000:
            suggestions.append("Large number of tracked objects - review object lifecycle")
        
        # Potential memory leaks
        leaks = self.tracker.detect_leaks()
        if leaks:
            suggestions.append(f"Potential memory leaks detected: {', '.join(leaks.keys())}")
        
        # GC performance
        gc_stats = gc.get_stats()
        if any(gen['collections'] > 1000 for gen in gc_stats):
            suggestions.append("High garbage collection activity - review object creation patterns")
        
        # Image memory usage
        image_stats = self.image_manager.get_memory_usage()
        if image_stats['memory_utilization'] > 0.9:
            suggestions.append("Image cache nearly full - consider reducing cache size or image resolution")
        
        if not suggestions:
            suggestions.append("Memory usage appears optimal")
        
        return suggestions

# Decorators for automatic memory management
def memory_managed(obj_type: str = None):
    """Decorator to automatically track objects for memory management."""
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Track this object
            MemoryManager.instance().track_object(self, obj_type or cls.__name__)
        
        cls.__init__ = new_init
        return cls
    return decorator

def memory_optimized(func):
    """Decorator for memory-optimized function execution."""
    def wrapper(*args, **kwargs):
        # Force GC before expensive operation
        gc.collect()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Optional GC after operation (commented out as it might be too aggressive)
            # gc.collect()
            pass
    
    return wrapper

# Global memory manager instance
_memory_manager = None
_manager_lock = threading.Lock()

def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        with _manager_lock:
            if _memory_manager is None:
                _memory_manager = MemoryManager()
    return _memory_manager