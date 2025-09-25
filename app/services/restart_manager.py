"""Service Restart Manager for graceful service restarts with minimal disruption.

This module implements a high-performance service restart system that handles:
- Hot-swapping of critical services without interruption
- Graceful shutdowns with operation completion
- Service dependencies and restart ordering
- Resource management and cleanup
- Progress tracking and feedback
- Performance optimization for real-time processing

Performance Targets:
- Hot-swap in <50ms for UI services
- Webcam restart in <200ms
- Model reload in <500ms
- Zero frame drops during swap
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import gc
import logging
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union,
    Protocol, runtime_checkable
)

import psutil

from ..core.exceptions import ConfigurationError
from ..core.performance import PerformanceTimer, performance_timer

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RestartStrategy(Enum):
    """Service restart strategies with different performance characteristics."""
    HOT_SWAP = "hot_swap"      # Swap services without interruption (<50ms)
    GRACEFUL = "graceful"      # Wait for operations to complete
    IMMEDIATE = "immediate"     # Force stop and restart
    ROLLING = "rolling"        # Restart services one by one
    PARALLEL = "parallel"      # Restart independent services simultaneously


class ServiceState(Enum):
    """Service lifecycle states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    SUSPENDED = "suspended"  # Temporarily paused for hot-swap


@runtime_checkable
class RestartableService(Protocol):
    """Protocol for services that support restart operations."""

    def get_state(self) -> ServiceState:
        """Get current service state."""
        ...

    def prepare_shutdown(self) -> None:
        """Prepare for shutdown (save state, finish operations)."""
        ...

    def shutdown(self) -> None:
        """Shutdown the service."""
        ...

    def startup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Start the service with optional configuration."""
        ...

    def health_check(self) -> bool:
        """Check if service is healthy."""
        ...

    def get_resources(self) -> Dict[str, Any]:
        """Get current resource usage."""
        ...


@dataclass
class ServiceMetadata:
    """Metadata about a service for restart operations."""
    name: str
    service_ref: Optional[Any] = None
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    restart_strategy: RestartStrategy = RestartStrategy.GRACEFUL
    priority: int = 50  # Lower = higher priority
    max_restart_time: float = 5.0  # seconds
    requires_gpu: bool = False
    requires_camera: bool = False
    supports_hot_swap: bool = False
    warm_up_time: float = 0.0  # seconds after startup
    resource_pool_size: int = 0  # For service pooling


@dataclass
class RestartProgress:
    """Progress tracking for restart operations."""
    total_services: int
    completed: int
    current_service: str
    current_operation: str
    estimated_time_remaining: float
    started_at: datetime = field(default_factory=datetime.now)
    errors: List[str] = field(default_factory=list)

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_services == 0:
            return 100.0
        return (self.completed / self.total_services) * 100

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return (datetime.now() - self.started_at).total_seconds()


@dataclass
class RestartPlan:
    """Execution plan for service restarts."""
    strategy: RestartStrategy
    services: List[ServiceMetadata]
    phases: List[List[str]]  # Services grouped by restart phase
    estimated_duration: float
    requires_gpu_reset: bool = False
    requires_camera_reset: bool = False
    parallel_execution: bool = False
    rollback_on_failure: bool = True


@dataclass
class ServiceSnapshot:
    """Snapshot of service state for rollback."""
    service_name: str
    timestamp: datetime
    state: ServiceState
    config: Dict[str, Any]
    resources: Dict[str, Any]
    metrics: Dict[str, Any]


class ServiceHealthCheck:
    """Health check implementation for services."""

    def __init__(self, service_name: str, service: RestartableService):
        self.service_name = service_name
        self.service = service
        self.max_retries = 3
        self.retry_delay = 0.5

    def check_pre_restart(self) -> bool:
        """Verify service is in a state that allows restart."""
        try:
            state = self.service.get_state()
            return state in [ServiceState.RUNNING, ServiceState.STOPPED, ServiceState.ERROR]
        except Exception as e:
            logger.error(f"Pre-restart check failed for {self.service_name}: {e}")
            return False

    @performance_timer("health_check_wait")
    def wait_for_ready(self, timeout: float = 5.0) -> bool:
        """Wait for service to become ready after startup."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                if self.service.health_check():
                    return True
            except Exception:
                pass

            time.sleep(0.1)

        return False

    def verify_functionality(self) -> bool:
        """Verify service is functioning correctly."""
        try:
            # Basic health check
            if not self.service.health_check():
                return False

            # Check resource availability
            resources = self.service.get_resources()
            if resources.get('error'):
                return False

            return True

        except Exception as e:
            logger.error(f"Functionality verification failed for {self.service_name}: {e}")
            return False


class ServicePool:
    """Pool of service instances for hot-swapping."""

    def __init__(self, service_factory: Callable, pool_size: int = 2):
        self.service_factory = service_factory
        self.pool_size = pool_size
        self._pool: deque = deque(maxlen=pool_size)
        self._active: Optional[Any] = None
        self._lock = threading.RLock()
        self._warm_up_thread: Optional[threading.Thread] = None

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the service pool."""
        with self._lock:
            # Create initial service
            self._active = self.service_factory(config)

            # Pre-warm additional instances in background
            if self.pool_size > 1:
                self._warm_up_thread = threading.Thread(
                    target=self._warm_up_instances,
                    args=(config,),
                    daemon=True
                )
                self._warm_up_thread.start()

    def _warm_up_instances(self, config: Dict[str, Any]) -> None:
        """Warm up pool instances in background."""
        for _ in range(self.pool_size - 1):
            try:
                instance = self.service_factory(config)
                with self._lock:
                    self._pool.append(instance)
            except Exception as e:
                logger.warning(f"Failed to warm up service instance: {e}")

    def swap(self, new_config: Dict[str, Any]) -> Tuple[Any, Any]:
        """Hot-swap to a new service instance."""
        with self._lock:
            # Get or create new instance
            if self._pool:
                new_instance = self._pool.popleft()
                # Reconfigure if needed
                if hasattr(new_instance, 'reconfigure'):
                    new_instance.reconfigure(new_config)
            else:
                new_instance = self.service_factory(new_config)

            # Swap instances
            old_instance = self._active
            self._active = new_instance

            # Schedule old instance cleanup
            if old_instance:
                threading.Thread(
                    target=self._cleanup_instance,
                    args=(old_instance,),
                    daemon=True
                ).start()

            return new_instance, old_instance

    def _cleanup_instance(self, instance: Any) -> None:
        """Clean up an old service instance."""
        try:
            if hasattr(instance, 'cleanup'):
                instance.cleanup()
            elif hasattr(instance, 'close'):
                instance.close()
        except Exception as e:
            logger.warning(f"Error cleaning up service instance: {e}")

    @property
    def active(self) -> Optional[Any]:
        """Get the active service instance."""
        return self._active


class ProgressCallback:
    """Callback system for restart progress updates."""

    def __init__(self):
        self._callbacks: List[Callable] = []
        self._lock = threading.Lock()

    def register(self, callback: Callable) -> None:
        """Register a progress callback."""
        with self._lock:
            self._callbacks.append(callback)

    def unregister(self, callback: Callable) -> None:
        """Unregister a progress callback."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def on_service_stopping(self, service: str) -> None:
        """Notify that a service is stopping."""
        self._notify('stopping', service)

    def on_service_starting(self, service: str) -> None:
        """Notify that a service is starting."""
        self._notify('starting', service)

    def on_service_ready(self, service: str) -> None:
        """Notify that a service is ready."""
        self._notify('ready', service)

    def on_restart_complete(self, success: bool) -> None:
        """Notify that restart is complete."""
        self._notify('complete', None, success=success)

    def _notify(self, event: str, service: Optional[str], **kwargs) -> None:
        """Send notification to all callbacks."""
        with self._lock:
            for callback in self._callbacks:
                try:
                    callback(event, service, **kwargs)
                except Exception as e:
                    logger.error(f"Error in progress callback: {e}")


class ServiceRestarter:
    """Core service restart orchestration engine."""

    def __init__(self):
        self._services: Dict[str, ServiceMetadata] = {}
        self._service_pools: Dict[str, ServicePool] = {}
        self._snapshots: Dict[str, ServiceSnapshot] = {}
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._progress = None
        self._progress_callback = ProgressCallback()
        self._resource_monitor = ResourceMonitor()
        self._lock = threading.RLock()

    def register_service(self, metadata: ServiceMetadata) -> None:
        """Register a service for restart management."""
        with self._lock:
            self._services[metadata.name] = metadata

            # Create service pool if hot-swap is supported
            if metadata.supports_hot_swap and metadata.resource_pool_size > 0:
                if metadata.service_ref and hasattr(metadata.service_ref, '__class__'):
                    factory = lambda cfg: metadata.service_ref.__class__(cfg)
                    self._service_pools[metadata.name] = ServicePool(
                        factory, metadata.resource_pool_size
                    )

    @performance_timer("restart_plan")
    def plan_restart(self, services: List[str]) -> RestartPlan:
        """Create an optimized restart plan for services."""
        with self._lock:
            # Validate services exist
            for service in services:
                if service not in self._services:
                    raise ValueError(f"Unknown service: {service}")

            # Build dependency graph
            dependency_graph = self._build_dependency_graph(services)

            # Determine restart strategy
            strategy = self._determine_strategy(services)

            # Create restart phases (topological sort)
            phases = self._create_restart_phases(dependency_graph)

            # Estimate duration
            duration = self._estimate_duration(phases, strategy)

            # Check resource requirements
            requires_gpu = any(self._services[s].requires_gpu for s in services)
            requires_camera = any(self._services[s].requires_camera for s in services)

            return RestartPlan(
                strategy=strategy,
                services=[self._services[s] for s in services],
                phases=phases,
                estimated_duration=duration,
                requires_gpu_reset=requires_gpu,
                requires_camera_reset=requires_camera,
                parallel_execution=(strategy == RestartStrategy.PARALLEL),
                rollback_on_failure=True
            )

    @performance_timer("restart_execution")
    def execute_restart(self, plan: RestartPlan) -> bool:
        """Execute a restart plan with progress tracking."""
        self._progress = RestartProgress(
            total_services=len(plan.services),
            completed=0,
            current_service="",
            current_operation="Initializing"
        )

        try:
            # Take snapshots for rollback
            self._take_snapshots(plan.services)

            # Pre-restart resource cleanup
            if plan.requires_gpu_reset:
                self._resource_monitor.release_gpu_memory()

            # Execute restart phases
            for phase_services in plan.phases:
                if plan.parallel_execution:
                    success = self._execute_parallel(phase_services, plan.strategy)
                else:
                    success = self._execute_sequential(phase_services, plan.strategy)

                if not success and plan.rollback_on_failure:
                    self._rollback()
                    return False

            # Post-restart validation
            if not self._validate_restart(plan.services):
                if plan.rollback_on_failure:
                    self._rollback()
                return False

            self._progress_callback.on_restart_complete(True)
            return True

        except Exception as e:
            logger.error(f"Restart execution failed: {e}")
            if plan.rollback_on_failure:
                self._rollback()
            self._progress_callback.on_restart_complete(False)
            return False

    def _execute_sequential(self, services: List[str], strategy: RestartStrategy) -> bool:
        """Execute services restart sequentially."""
        for service_name in services:
            self._progress.current_service = service_name

            metadata = self._services[service_name]

            if strategy == RestartStrategy.HOT_SWAP and metadata.supports_hot_swap:
                success = self._hot_swap_service(service_name)
            elif strategy == RestartStrategy.GRACEFUL:
                success = self._graceful_restart_service(service_name)
            else:
                success = self._immediate_restart_service(service_name)

            if not success:
                return False

            self._progress.completed += 1

        return True

    def _execute_parallel(self, services: List[str], strategy: RestartStrategy) -> bool:
        """Execute services restart in parallel."""
        futures = []

        for service_name in services:
            metadata = self._services[service_name]

            if strategy == RestartStrategy.HOT_SWAP and metadata.supports_hot_swap:
                future = self._executor.submit(self._hot_swap_service, service_name)
            elif strategy == RestartStrategy.GRACEFUL:
                future = self._executor.submit(self._graceful_restart_service, service_name)
            else:
                future = self._executor.submit(self._immediate_restart_service, service_name)

            futures.append((service_name, future))

        # Wait for all futures to complete
        success = True
        for service_name, future in futures:
            try:
                if not future.result(timeout=10.0):
                    success = False
                    break
                self._progress.completed += 1
            except Exception as e:
                logger.error(f"Parallel restart failed for {service_name}: {e}")
                success = False
                break

        return success

    @performance_timer("hot_swap")
    def _hot_swap_service(self, service_name: str) -> bool:
        """Perform hot-swap of a service without interruption."""
        try:
            metadata = self._services[service_name]
            self._progress_callback.on_service_stopping(service_name)

            if service_name in self._service_pools:
                # Use service pool for instant swap
                pool = self._service_pools[service_name]
                new_instance, old_instance = pool.swap({})

                # Verify new instance
                health_check = ServiceHealthCheck(service_name, new_instance)
                if not health_check.verify_functionality():
                    # Swap back on failure
                    pool._active = old_instance
                    return False
            else:
                # Manual hot-swap
                service = metadata.service_ref
                if not service:
                    return False

                # Create new instance in parallel
                new_service = self._create_parallel_instance(service)

                # Atomic swap
                self._atomic_swap(service_name, service, new_service)

                # Cleanup old instance asynchronously
                self._executor.submit(self._cleanup_old_instance, service)

            self._progress_callback.on_service_ready(service_name)
            return True

        except Exception as e:
            logger.error(f"Hot-swap failed for {service_name}: {e}")
            return False

    def _graceful_restart_service(self, service_name: str) -> bool:
        """Perform graceful restart with operation completion."""
        try:
            metadata = self._services[service_name]
            service = metadata.service_ref

            if not service:
                return False

            self._progress_callback.on_service_stopping(service_name)

            # Prepare for shutdown
            if hasattr(service, 'prepare_shutdown'):
                service.prepare_shutdown()

            # Wait for operations to complete
            time.sleep(0.5)

            # Shutdown
            if hasattr(service, 'shutdown'):
                service.shutdown()

            # Clear resources
            self._resource_monitor.clear_service_resources(service_name)

            self._progress_callback.on_service_starting(service_name)

            # Startup with new config
            if hasattr(service, 'startup'):
                service.startup()

            # Wait for warm-up
            if metadata.warm_up_time > 0:
                time.sleep(metadata.warm_up_time)

            # Verify health
            health_check = ServiceHealthCheck(service_name, service)
            if not health_check.wait_for_ready(metadata.max_restart_time):
                return False

            self._progress_callback.on_service_ready(service_name)
            return True

        except Exception as e:
            logger.error(f"Graceful restart failed for {service_name}: {e}")
            return False

    def _immediate_restart_service(self, service_name: str) -> bool:
        """Perform immediate restart without waiting."""
        try:
            metadata = self._services[service_name]
            service = metadata.service_ref

            if not service:
                return False

            self._progress_callback.on_service_stopping(service_name)

            # Force shutdown
            if hasattr(service, 'shutdown'):
                service.shutdown()

            # Clear all resources immediately
            self._resource_monitor.force_clear_resources(service_name)

            self._progress_callback.on_service_starting(service_name)

            # Immediate startup
            if hasattr(service, 'startup'):
                service.startup()

            self._progress_callback.on_service_ready(service_name)
            return True

        except Exception as e:
            logger.error(f"Immediate restart failed for {service_name}: {e}")
            return False

    def _build_dependency_graph(self, services: List[str]) -> Dict[str, Set[str]]:
        """Build dependency graph for services."""
        graph = defaultdict(set)

        for service in services:
            metadata = self._services[service]
            for dep in metadata.dependencies:
                if dep in services:
                    graph[service].add(dep)

        return graph

    def _determine_strategy(self, services: List[str]) -> RestartStrategy:
        """Determine optimal restart strategy."""
        # Check if all support hot-swap
        all_hot_swap = all(
            self._services[s].supports_hot_swap
            for s in services
        )

        if all_hot_swap:
            return RestartStrategy.HOT_SWAP

        # Check for dependencies
        has_dependencies = any(
            self._services[s].dependencies
            for s in services
        )

        if not has_dependencies:
            return RestartStrategy.PARALLEL

        return RestartStrategy.ROLLING

    def _create_restart_phases(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Create restart phases using topological sort."""
        # Kahn's algorithm for topological sorting
        in_degree = defaultdict(int)
        for node in graph:
            for dep in graph[node]:
                in_degree[dep] += 1

        queue = deque([node for node in graph if in_degree[node] == 0])
        phases = []

        while queue:
            phase = list(queue)
            phases.append(phase)
            queue.clear()

            for node in phase:
                for dep in graph[node]:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        queue.append(dep)

        return phases if phases else [list(graph.keys())]

    def _estimate_duration(self, phases: List[List[str]],
                          strategy: RestartStrategy) -> float:
        """Estimate total restart duration."""
        total_time = 0.0

        for phase in phases:
            phase_time = 0.0

            if strategy == RestartStrategy.PARALLEL:
                # Maximum time in parallel execution
                phase_time = max(
                    self._services[s].max_restart_time
                    for s in phase
                )
            else:
                # Sum of sequential execution
                phase_time = sum(
                    self._services[s].max_restart_time
                    for s in phase
                )

            total_time += phase_time

        # Add buffer for coordination
        return total_time * 1.2

    def estimate_downtime(self, plan: RestartPlan) -> float:
        """Estimate service downtime for a restart plan."""
        if plan.strategy == RestartStrategy.HOT_SWAP:
            return 0.05  # 50ms for pointer swap
        elif plan.strategy == RestartStrategy.GRACEFUL:
            return max(s.max_restart_time for s in plan.services)
        else:
            return sum(s.max_restart_time for s in plan.services)

    def _take_snapshots(self, services: List[ServiceMetadata]) -> None:
        """Take snapshots of service states for rollback."""
        for metadata in services:
            if metadata.service_ref:
                snapshot = ServiceSnapshot(
                    service_name=metadata.name,
                    timestamp=datetime.now(),
                    state=metadata.service_ref.get_state() if hasattr(
                        metadata.service_ref, 'get_state'
                    ) else ServiceState.UNKNOWN,
                    config={},
                    resources=metadata.service_ref.get_resources() if hasattr(
                        metadata.service_ref, 'get_resources'
                    ) else {},
                    metrics={}
                )
                self._snapshots[metadata.name] = snapshot

    def _rollback(self) -> None:
        """Rollback services to previous state."""
        logger.info("Rolling back services to previous state")

        for service_name, snapshot in self._snapshots.items():
            try:
                metadata = self._services[service_name]
                if metadata.service_ref and hasattr(metadata.service_ref, 'restore'):
                    metadata.service_ref.restore(snapshot)
            except Exception as e:
                logger.error(f"Failed to rollback {service_name}: {e}")

    def _validate_restart(self, services: List[ServiceMetadata]) -> bool:
        """Validate all services are healthy after restart."""
        for metadata in services:
            if metadata.service_ref:
                health_check = ServiceHealthCheck(metadata.name, metadata.service_ref)
                if not health_check.verify_functionality():
                    logger.error(f"Service {metadata.name} failed validation")
                    return False
        return True

    def _create_parallel_instance(self, service: Any) -> Any:
        """Create a new service instance in parallel."""
        # Implementation depends on specific service type
        if hasattr(service, '__class__'):
            return service.__class__()
        return None

    def _atomic_swap(self, service_name: str, old_service: Any,
                     new_service: Any) -> None:
        """Perform atomic swap of service instances."""
        with self._lock:
            # Update service reference
            self._services[service_name].service_ref = new_service

    def _cleanup_old_instance(self, service: Any) -> None:
        """Clean up old service instance after swap."""
        try:
            if hasattr(service, 'cleanup'):
                service.cleanup()
            elif hasattr(service, 'close'):
                service.close()

            # Force garbage collection
            gc.collect()
        except Exception as e:
            logger.warning(f"Error cleaning up old instance: {e}")


class ResourceMonitor:
    """Monitor and manage system resources during restarts."""

    def __init__(self):
        self._process = psutil.Process()
        self._initial_memory = self._process.memory_info().rss
        self._resource_locks = defaultdict(threading.Lock)

    def get_current_resources(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        return {
            'cpu_percent': self._process.cpu_percent(),
            'memory_mb': self._process.memory_info().rss / 1024 / 1024,
            'threads': len(self._process.threads()),
            'open_files': len(self._process.open_files()),
            'gpu_memory_mb': self._get_gpu_memory()
        }

    def release_gpu_memory(self) -> None:
        """Release GPU memory before restart."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    def clear_service_resources(self, service_name: str) -> None:
        """Clear resources associated with a service."""
        with self._resource_locks[service_name]:
            # Trigger garbage collection
            gc.collect()

    def force_clear_resources(self, service_name: str) -> None:
        """Force clear all resources for a service."""
        with self._resource_locks[service_name]:
            # Aggressive garbage collection
            gc.collect(2)

    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
        except ImportError:
            pass
        return 0.0


# Singleton instance
_restart_manager = None
_lock = threading.Lock()


def get_restart_manager() -> ServiceRestarter:
    """Get the singleton restart manager instance."""
    global _restart_manager
    if _restart_manager is None:
        with _lock:
            if _restart_manager is None:
                _restart_manager = ServiceRestarter()
    return _restart_manager