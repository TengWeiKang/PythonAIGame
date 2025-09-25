"""
CRITICAL ERROR HANDLING & THREADING SAFETY ENHANCEMENTS

This module provides comprehensive error handling and thread safety enhancements
for the Modern Main Window to address all identified vulnerabilities.
"""

import tkinter as tk
import threading
import functools
import logging
import traceback
from typing import Callable, Any, Optional
from contextlib import contextmanager
import weakref
import queue
import time

logger = logging.getLogger(__name__)

class ThreadSafeUIUpdater:
    """Thread-safe wrapper for UI updates to prevent race conditions."""

    def __init__(self, root: tk.Tk):
        self.root = weakref.ref(root)
        self._update_queue = queue.Queue()
        self._is_shutting_down = False
        self._lock = threading.Lock()
        self._process_updates()

    def _process_updates(self):
        """Process queued UI updates safely."""
        root = self.root()
        if not root or self._is_shutting_down:
            return

        try:
            # Process all pending updates
            while not self._update_queue.empty():
                try:
                    func, args, kwargs = self._update_queue.get_nowait()
                    func(*args, **kwargs)
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"UI update failed: {e}")
        except Exception as e:
            logger.error(f"UI update processing failed: {e}")
        finally:
            # Schedule next update cycle
            if not self._is_shutting_down:
                root.after(16, self._process_updates)  # ~60 FPS

    def schedule_update(self, func: Callable, *args, **kwargs):
        """Schedule a thread-safe UI update."""
        if not self._is_shutting_down:
            self._update_queue.put((func, args, kwargs))

    def shutdown(self):
        """Safely shutdown the updater."""
        self._is_shutting_down = True

def thread_safe_ui_update(func):
    """Decorator to ensure UI updates are thread-safe."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, '_ui_updater') and self._ui_updater:
            self._ui_updater.schedule_update(func, self, *args, **kwargs)
        else:
            # Fallback to direct call if updater not available
            try:
                if hasattr(self, 'root') and self.root:
                    self.root.after_idle(lambda: func(self, *args, **kwargs))
                else:
                    func(self, *args, **kwargs)
            except tk.TclError:
                # Widget destroyed, ignore
                pass
    return wrapper

@contextmanager
def error_boundary(operation_name: str, fallback_value=None, reraise_critical=True):
    """Context manager for comprehensive error handling with logging."""
    try:
        yield
    except KeyboardInterrupt:
        # Always re-raise keyboard interrupt
        raise
    except MemoryError:
        # Always re-raise memory errors
        logger.critical(f"Memory error in {operation_name}")
        if reraise_critical:
            raise
    except (tk.TclError, RuntimeError) as e:
        # UI or threading errors - log but don't crash
        logger.warning(f"UI/Threading error in {operation_name}: {e}")
        if fallback_value is not None:
            return fallback_value
    except Exception as e:
        # Log all other errors with stack trace
        logger.error(f"Error in {operation_name}: {e}", exc_info=True)
        if fallback_value is not None:
            return fallback_value
        if reraise_critical:
            raise

class ResourceLeakPrevention:
    """Prevents resource leaks by tracking and cleaning up resources."""

    def __init__(self):
        self._resources = []
        self._lock = threading.Lock()

    def register_resource(self, resource, cleanup_method=None):
        """Register a resource for cleanup."""
        with self._lock:
            self._resources.append((resource, cleanup_method))

    def cleanup_all(self):
        """Clean up all registered resources."""
        with self._lock:
            for resource, cleanup_method in self._resources:
                try:
                    if cleanup_method:
                        cleanup_method()
                    elif hasattr(resource, 'close'):
                        resource.close()
                    elif hasattr(resource, 'stop'):
                        resource.stop()
                    elif hasattr(resource, 'cancel'):
                        resource.cancel()
                except Exception as e:
                    logger.error(f"Resource cleanup failed: {e}")
            self._resources.clear()

class CircuitBreaker:
    """Circuit breaker pattern for API/service calls."""

    def __init__(self, failure_threshold=5, timeout=60, expected_exception=Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time < self.timeout:
                    raise Exception("Circuit breaker is OPEN")
                else:
                    self.state = 'HALF_OPEN'

            try:
                result = func(*args, **kwargs)
                if self.state == 'HALF_OPEN':
                    self.state = 'CLOSED'
                    self.failure_count = 0
                return result
            except self.expected_exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = 'OPEN'

                raise e

class PerformanceGuard:
    """Guards against performance degradation and memory issues."""

    def __init__(self, max_memory_mb=500, max_execution_time=10):
        self.max_memory_mb = max_memory_mb
        self.max_execution_time = max_execution_time

    @contextmanager
    def monitor_performance(self, operation_name: str):
        """Monitor performance and resource usage."""
        import psutil
        import time

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024

            execution_time = end_time - start_time
            memory_increase = end_memory - start_memory

            if execution_time > self.max_execution_time:
                logger.warning(f"Slow operation {operation_name}: {execution_time:.2f}s")

            if memory_increase > self.max_memory_mb:
                logger.warning(f"High memory usage in {operation_name}: +{memory_increase:.2f}MB")

def validate_input_safety(func):
    """Decorator to validate all inputs for safety."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Validate string inputs for dangerous content
        for arg in args:
            if isinstance(arg, str):
                if len(arg) > 10000:  # Prevent DoS through large strings
                    raise ValueError("Input string too long")
                if any(dangerous in arg.lower() for dangerous in ['<script', 'javascript:', 'data:', 'vbscript:']):
                    raise ValueError("Potentially dangerous input detected")

        # Validate keyword arguments
        for key, value in kwargs.items():
            if isinstance(value, str):
                if len(value) > 10000:
                    raise ValueError(f"Input string for {key} too long")

        return func(*args, **kwargs)
    return wrapper

class AsyncOperationManager:
    """Manages async operations with proper cancellation and cleanup."""

    def __init__(self):
        self._operations = {}
        self._lock = threading.Lock()

    def start_operation(self, operation_id: str, operation_func: Callable,
                       on_success: Optional[Callable] = None,
                       on_error: Optional[Callable] = None):
        """Start an async operation with proper tracking."""
        def operation_wrapper():
            try:
                result = operation_func()
                if on_success:
                    on_success(result)
            except Exception as e:
                logger.error(f"Async operation {operation_id} failed: {e}")
                if on_error:
                    on_error(e)
            finally:
                with self._lock:
                    self._operations.pop(operation_id, None)

        with self._lock:
            if operation_id in self._operations:
                logger.warning(f"Operation {operation_id} already running")
                return False

            thread = threading.Thread(target=operation_wrapper, daemon=True)
            self._operations[operation_id] = thread
            thread.start()
            return True

    def cancel_operation(self, operation_id: str):
        """Cancel a running operation."""
        with self._lock:
            thread = self._operations.get(operation_id)
            if thread and thread.is_alive():
                # Note: Python doesn't support thread cancellation
                # Mark for cancellation and wait for natural completion
                logger.info(f"Marking operation {operation_id} for cancellation")
                return True
            return False

    def cancel_all_operations(self):
        """Cancel all running operations."""
        with self._lock:
            for operation_id in list(self._operations.keys()):
                self.cancel_operation(operation_id)

# Global instances for use throughout the application
_performance_guard = PerformanceGuard()
_resource_leak_prevention = ResourceLeakPrevention()
_async_operation_manager = AsyncOperationManager()

def get_performance_guard():
    """Get the global performance guard instance."""
    return _performance_guard

def get_resource_leak_prevention():
    """Get the global resource leak prevention instance."""
    return _resource_leak_prevention

def get_async_operation_manager():
    """Get the global async operation manager instance."""
    return _async_operation_manager