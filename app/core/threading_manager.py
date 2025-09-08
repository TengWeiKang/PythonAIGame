"""Advanced threading and concurrency management for optimal performance."""

import threading
import time
import queue
import concurrent.futures
import asyncio
import weakref
from typing import Callable, Any, Optional, Dict, List, Union, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum
import traceback

from .performance import PerformanceMonitor

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4

@dataclass
class Task:
    """Task data structure for thread pool."""
    func: Callable
    args: tuple
    kwargs: dict
    callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 0
    created_at: datetime = None
    task_id: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.task_id is None:
            self.task_id = f"task_{id(self)}"

class ThreadPoolManager:
    """High-performance thread pool with priority queues and load balancing."""
    
    def __init__(self, 
                 min_workers: int = 2,
                 max_workers: int = 8,
                 thread_name_prefix: str = "Worker",
                 queue_size: int = 1000):
        
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.thread_name_prefix = thread_name_prefix
        self.queue_size = queue_size
        
        # Priority queues for different task types
        self._priority_queues = {
            TaskPriority.CRITICAL: queue.PriorityQueue(maxsize=queue_size // 5),
            TaskPriority.HIGH: queue.PriorityQueue(maxsize=queue_size // 4),
            TaskPriority.NORMAL: queue.PriorityQueue(maxsize=queue_size // 2),
            TaskPriority.LOW: queue.PriorityQueue(maxsize=queue_size // 4),
            TaskPriority.BACKGROUND: queue.PriorityQueue(maxsize=queue_size // 10)
        }
        
        # Worker management
        self._workers = []
        self._worker_stats = defaultdict(dict)
        self._shutdown_event = threading.Event()
        self._workers_lock = threading.Lock()
        
        # Task tracking
        self._active_tasks = {}
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._task_history = deque(maxlen=1000)
        
        # Load balancing
        self._worker_load = defaultdict(int)
        self._load_balance_enabled = True
        
        # Performance monitoring
        self._start_time = time.time()
        self._last_stats_time = time.time()
        
        # Start initial workers
        self._start_workers(self.min_workers)
        
        # Register with performance monitor
        monitor = PerformanceMonitor.instance()
        monitor.register_thread_pool("main", self)
    
    def _start_workers(self, count: int):
        """Start worker threads."""
        with self._workers_lock:
            for i in range(count):
                worker_id = len(self._workers)
                worker = threading.Thread(
                    target=self._worker_loop,
                    args=(worker_id,),
                    name=f"{self.thread_name_prefix}-{worker_id}",
                    daemon=True
                )
                worker.start()
                self._workers.append(worker)
                self._worker_stats[worker_id] = {
                    'tasks_completed': 0,
                    'tasks_failed': 0,
                    'start_time': time.time(),
                    'last_task_time': 0,
                    'total_processing_time': 0
                }
    
    def _worker_loop(self, worker_id: int):
        """Main worker loop."""
        stats = self._worker_stats[worker_id]
        
        while not self._shutdown_event.is_set():
            try:
                task = self._get_next_task(timeout=1.0)
                
                if task is None:
                    # No task available, continue
                    continue
                
                # Update worker load
                self._worker_load[worker_id] += 1
                
                # Execute task
                start_time = time.time()
                try:
                    self._execute_task(task, worker_id)
                    stats['tasks_completed'] += 1
                    self._completed_tasks += 1
                except Exception as e:
                    stats['tasks_failed'] += 1
                    self._failed_tasks += 1
                    logger.exception(f"Task execution failed in worker {worker_id}: {e}")
                finally:
                    # Update statistics
                    end_time = time.time()
                    processing_time = end_time - start_time
                    stats['last_task_time'] = end_time
                    stats['total_processing_time'] += processing_time
                    
                    # Remove from active tasks
                    if task.task_id in self._active_tasks:
                        del self._active_tasks[task.task_id]
                    
                    # Update worker load
                    self._worker_load[worker_id] -= 1
                    
                    # Record task completion
                    self._task_history.append({
                        'task_id': task.task_id,
                        'worker_id': worker_id,
                        'priority': task.priority,
                        'processing_time': processing_time,
                        'completed_at': datetime.now()
                    })
                
            except Exception as e:
                logger.exception(f"Worker {worker_id} error: {e}")
                time.sleep(0.1)  # Brief pause on error
    
    def _get_next_task(self, timeout: float = None) -> Optional[Task]:
        """Get next task from priority queues."""
        # Check queues in priority order
        for priority in TaskPriority:
            try:
                priority_queue = self._priority_queues[priority]
                _, task = priority_queue.get(timeout=timeout/5 if timeout else 0.2)
                priority_queue.task_done()
                return task
            except queue.Empty:
                continue
        
        return None
    
    def _execute_task(self, task: Task, worker_id: int):
        """Execute a single task."""
        retries = 0
        last_exception = None
        
        while retries <= task.max_retries:
            try:
                # Apply timeout if specified
                if task.timeout:
                    def timeout_handler():
                        raise TimeoutError(f"Task {task.task_id} timed out after {task.timeout}s")
                    
                    timer = threading.Timer(task.timeout, timeout_handler)
                    timer.start()
                
                try:
                    # Execute the task
                    result = task.func(*task.args, **task.kwargs)
                    
                    # Call success callback
                    if task.callback:
                        try:
                            task.callback(result)
                        except Exception as e:
                            logger.exception(f"Task callback error: {e}")
                    
                    return result
                    
                finally:
                    if task.timeout:
                        timer.cancel()
                
            except Exception as e:
                last_exception = e
                retries += 1
                
                if retries <= task.max_retries:
                    # Wait before retry with exponential backoff
                    wait_time = min(2 ** retries, 10)  # Max 10 seconds
                    time.sleep(wait_time)
                    logger.warning(f"Task {task.task_id} failed, retrying ({retries}/{task.max_retries})")
                else:
                    # All retries exhausted
                    logger.error(f"Task {task.task_id} failed after {task.max_retries} retries")
                    
                    # Call error callback
                    if task.error_callback:
                        try:
                            task.error_callback(last_exception)
                        except Exception as cb_e:
                            logger.exception(f"Task error callback error: {cb_e}")
                    
                    raise last_exception
    
    def submit_task(self, 
                   func: Callable,
                   *args,
                   callback: Optional[Callable] = None,
                   error_callback: Optional[Callable] = None,
                   priority: TaskPriority = TaskPriority.NORMAL,
                   timeout: Optional[float] = None,
                   max_retries: int = 0,
                   **kwargs) -> str:
        """Submit task to thread pool."""
        
        if self._shutdown_event.is_set():
            raise RuntimeError("Thread pool is shutting down")
        
        # Create task
        task = Task(
            func=func,
            args=args,
            kwargs=kwargs,
            callback=callback,
            error_callback=error_callback,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries
        )
        
        # Add to appropriate priority queue
        try:
            priority_queue = self._priority_queues[priority]
            
            # Use priority value as queue priority (lower number = higher priority)
            priority_queue.put((priority.value, task), timeout=5.0)
            
            # Track active task
            self._active_tasks[task.task_id] = task
            
            # Dynamic worker scaling
            self._maybe_scale_workers()
            
            return task.task_id
            
        except queue.Full:
            raise RuntimeError(f"Task queue for priority {priority.name} is full")
    
    def _maybe_scale_workers(self):
        """Dynamically scale worker count based on load."""
        if not self._load_balance_enabled:
            return
        
        total_queued_tasks = sum(q.qsize() for q in self._priority_queues.values())
        active_workers = len([w for w in self._workers if w.is_alive()])
        
        # Scale up if queues are getting full
        if (total_queued_tasks > active_workers * 3 and 
            active_workers < self.max_workers):
            self._start_workers(1)
            logger.debug(f"Scaled up to {len(self._workers)} workers")
        
        # Scale down if workers are idle (implementation would go here)
        # Note: Scaling down is more complex and omitted for brevity
    
    def get_stats(self) -> Dict[str, Any]:
        """Get thread pool statistics."""
        current_time = time.time()
        uptime = current_time - self._start_time
        
        # Calculate queue sizes
        queue_sizes = {
            priority.name: queue_obj.qsize() 
            for priority, queue_obj in self._priority_queues.items()
        }
        
        # Worker statistics
        active_workers = len([w for w in self._workers if w.is_alive()])
        
        # Task throughput
        time_since_last_stats = current_time - self._last_stats_time
        if time_since_last_stats > 0:
            tasks_per_second = len(self._task_history) / uptime
        else:
            tasks_per_second = 0
        
        self._last_stats_time = current_time
        
        return {
            'active_workers': active_workers,
            'total_workers': len(self._workers),
            'active_tasks': len(self._active_tasks),
            'completed_tasks': self._completed_tasks,
            'failed_tasks': self._failed_tasks,
            'tasks_per_second': tasks_per_second,
            'queue_sizes': queue_sizes,
            'uptime_seconds': uptime,
            'worker_stats': dict(self._worker_stats)
        }
    
    def shutdown(self, wait: bool = True, timeout: float = 30.0):
        """Shutdown thread pool."""
        logger.info("Shutting down thread pool...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        if wait:
            # Wait for workers to finish
            end_time = time.time() + timeout
            
            for worker in self._workers:
                remaining_time = end_time - time.time()
                if remaining_time > 0:
                    worker.join(timeout=remaining_time)
                
                if worker.is_alive():
                    logger.warning(f"Worker {worker.name} did not shutdown gracefully")
        
        logger.info("Thread pool shutdown complete")

class AsyncTaskManager:
    """Manager for async operations and coroutines."""
    
    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self._running_tasks = set()
        self._task_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._loop = None
        self._loop_thread = None
        
        # Start event loop in separate thread
        self._start_event_loop()
    
    def _start_event_loop(self):
        """Start async event loop in separate thread."""
        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()
        
        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()
        
        # Wait for loop to be ready
        while self._loop is None:
            time.sleep(0.01)
    
    def submit_async_task(self, coro, callback: Optional[Callable] = None) -> asyncio.Task:
        """Submit async task to event loop."""
        async def wrapped_coro():
            async with self._task_semaphore:
                try:
                    result = await coro
                    if callback:
                        callback(result, None)
                    return result
                except Exception as e:
                    if callback:
                        callback(None, e)
                    raise
        
        if self._loop:
            task = asyncio.run_coroutine_threadsafe(wrapped_coro(), self._loop)
            self._running_tasks.add(task)
            
            # Clean up completed tasks
            self._cleanup_completed_tasks()
            
            return task
        else:
            raise RuntimeError("Event loop not available")
    
    def _cleanup_completed_tasks(self):
        """Remove completed tasks from tracking."""
        completed_tasks = [task for task in self._running_tasks if task.done()]
        for task in completed_tasks:
            self._running_tasks.discard(task)
    
    def shutdown(self):
        """Shutdown async task manager."""
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        
        if self._loop_thread:
            self._loop_thread.join(timeout=5.0)

class ThreadingManager:
    """Central threading and concurrency management system."""
    
    def __init__(self):
        # Thread pools for different purposes
        self.main_pool = ThreadPoolManager(
            min_workers=2, 
            max_workers=8, 
            thread_name_prefix="Main"
        )
        
        self.io_pool = ThreadPoolManager(
            min_workers=1,
            max_workers=4,
            thread_name_prefix="IO"
        )
        
        self.compute_pool = ThreadPoolManager(
            min_workers=1,
            max_workers=4,
            thread_name_prefix="Compute"
        )
        
        # Async manager
        self.async_manager = AsyncTaskManager()
        
        # Background tasks
        self.background_pool = ThreadPoolManager(
            min_workers=1,
            max_workers=2,
            thread_name_prefix="Background"
        )
        
        # Thread safety utilities
        self._locks = {}
        self._conditions = {}
        self._events = {}
        
        # Resource monitoring
        self._resource_monitor_thread = None
        self._monitoring_active = False
        
        # Start resource monitoring
        self._start_resource_monitoring()
    
    def _start_resource_monitoring(self):
        """Start thread resource monitoring."""
        self._monitoring_active = True
        self._resource_monitor_thread = threading.Thread(
            target=self._resource_monitor_loop,
            name="ThreadResourceMonitor",
            daemon=True
        )
        self._resource_monitor_thread.start()
    
    def _resource_monitor_loop(self):
        """Monitor thread resources and performance."""
        while self._monitoring_active:
            try:
                # Get statistics from all pools
                main_stats = self.main_pool.get_stats()
                io_stats = self.io_pool.get_stats()
                compute_stats = self.compute_pool.get_stats()
                background_stats = self.background_pool.get_stats()
                
                # Record performance metrics
                monitor = PerformanceMonitor.instance()
                monitor.record_operation_time("thread_pool_main_active", main_stats['active_tasks'])
                monitor.record_operation_time("thread_pool_io_active", io_stats['active_tasks'])
                monitor.record_operation_time("thread_pool_compute_active", compute_stats['active_tasks'])
                
                # Check for potential issues
                total_active_tasks = (main_stats['active_tasks'] + 
                                    io_stats['active_tasks'] + 
                                    compute_stats['active_tasks'])
                
                if total_active_tasks > 50:
                    logger.warning(f"High thread pool usage: {total_active_tasks} active tasks")
                
                time.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                logger.exception(f"Thread resource monitoring error: {e}")
                time.sleep(1.0)
    
    def submit_task(self, func: Callable, *args, 
                   pool_type: str = 'main',
                   priority: TaskPriority = TaskPriority.NORMAL,
                   **kwargs) -> str:
        """Submit task to appropriate thread pool."""
        
        pool_map = {
            'main': self.main_pool,
            'io': self.io_pool,
            'compute': self.compute_pool,
            'background': self.background_pool
        }
        
        if pool_type not in pool_map:
            raise ValueError(f"Invalid pool type: {pool_type}")
        
        pool = pool_map[pool_type]
        return pool.submit_task(func, *args, priority=priority, **kwargs)
    
    def submit_async_task(self, coro, callback: Optional[Callable] = None):
        """Submit async task."""
        return self.async_manager.submit_async_task(coro, callback)
    
    def get_thread_safe_lock(self, name: str) -> threading.Lock:
        """Get or create a named thread-safe lock."""
        if name not in self._locks:
            self._locks[name] = threading.Lock()
        return self._locks[name]
    
    def get_condition(self, name: str) -> threading.Condition:
        """Get or create a named condition variable."""
        if name not in self._conditions:
            self._conditions[name] = threading.Condition()
        return self._conditions[name]
    
    def get_event(self, name: str) -> threading.Event:
        """Get or create a named event."""
        if name not in self._events:
            self._events[name] = threading.Event()
        return self._events[name]
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive threading statistics."""
        return {
            'main_pool': self.main_pool.get_stats(),
            'io_pool': self.io_pool.get_stats(),
            'compute_pool': self.compute_pool.get_stats(),
            'background_pool': self.background_pool.get_stats(),
            'async_manager': {
                'running_tasks': len(self.async_manager._running_tasks),
                'max_concurrent': self.async_manager.max_concurrent_tasks
            },
            'system': {
                'total_threads': threading.active_count(),
                'main_thread_alive': threading.main_thread().is_alive()
            }
        }
    
    def shutdown_all(self, timeout: float = 30.0):
        """Shutdown all thread pools and managers."""
        logger.info("Shutting down all thread pools...")
        
        # Stop monitoring
        self._monitoring_active = False
        
        # Shutdown pools
        self.main_pool.shutdown(wait=True, timeout=timeout/4)
        self.io_pool.shutdown(wait=True, timeout=timeout/4)
        self.compute_pool.shutdown(wait=True, timeout=timeout/4)
        self.background_pool.shutdown(wait=True, timeout=timeout/4)
        
        # Shutdown async manager
        self.async_manager.shutdown()
        
        logger.info("All thread pools shut down")

# Global threading manager
_threading_manager = None
_manager_lock = threading.Lock()

def get_threading_manager() -> ThreadingManager:
    """Get global threading manager instance."""
    global _threading_manager
    if _threading_manager is None:
        with _manager_lock:
            if _threading_manager is None:
                _threading_manager = ThreadingManager()
    return _threading_manager

# Decorators for threading optimization
def run_in_thread(pool_type: str = 'main', priority: TaskPriority = TaskPriority.NORMAL):
    """Decorator to run function in thread pool."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = get_threading_manager()
            return manager.submit_task(func, *args, pool_type=pool_type, priority=priority, **kwargs)
        return wrapper
    return decorator

def thread_safe(lock_name: str = None):
    """Decorator to make function thread-safe."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = get_threading_manager()
            lock_name_actual = lock_name or f"{func.__module__}.{func.__name__}"
            lock = manager.get_thread_safe_lock(lock_name_actual)
            
            with lock:
                return func(*args, **kwargs)
        return wrapper
    return decorator