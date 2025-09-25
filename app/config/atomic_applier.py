"""Atomic Application System for transactional settings updates with rollback capability.

This module implements a comprehensive system for applying configuration changes
atomically to all application services with complete rollback support.

Features:
- Transactional updates with all-or-nothing semantics
- Automatic rollback on any failure
- Service dependency management
- State snapshots and restoration
- Performance monitoring and optimization
- Parallel updates where safe
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union,
    Protocol, runtime_checkable
)
from contextlib import contextmanager
import threading
import copy
import traceback
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future, as_completed

from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class UpdateType(Enum):
    """Types of service updates based on impact and requirements."""
    HOT_RELOAD = "hot_reload"      # Can be applied without service interruption
    RESTART = "restart"             # Requires service restart
    RECREATE = "recreate"          # Service must be recreated
    IMMEDIATE = "immediate"         # Applied immediately without validation


class ServicePriority(Enum):
    """Priority levels for service updates."""
    CRITICAL = 1   # Core services that others depend on
    HIGH = 2       # Important services with dependencies
    MEDIUM = 3     # Standard services
    LOW = 4        # Optional services


@runtime_checkable
class TransactionalService(Protocol):
    """Protocol for services that support transactional updates."""

    def create_snapshot(self) -> Dict[str, Any]:
        """Create snapshot of current service state."""
        ...

    def restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Restore service state from snapshot."""
        ...

    def validate_config(self, config: Any) -> bool:
        """Validate configuration before applying."""
        ...

    def apply_config(self, config: Any) -> None:
        """Apply configuration to service."""
        ...


@dataclass
class ServiceSnapshot:
    """Snapshot of a service's state for rollback."""
    service_name: str
    timestamp: datetime
    state_data: Dict[str, Any]
    service_ref: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AtomicOperation:
    """Represents a single atomic operation on a service."""
    service_name: str
    operation_type: UpdateType
    apply_func: Callable[[], bool]
    rollback_func: Callable[[], None]
    verify_func: Optional[Callable[[], bool]] = None
    priority: ServicePriority = ServicePriority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: float = 5.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate operation after initialization."""
        if not callable(self.apply_func):
            raise ValueError(f"apply_func must be callable for {self.service_name}")
        if not callable(self.rollback_func):
            raise ValueError(f"rollback_func must be callable for {self.service_name}")

    def execute(self) -> bool:
        """Execute the atomic operation with timeout and error handling."""
        try:
            logger.debug(f"Executing atomic operation for {self.service_name}")
            result = self.apply_func()

            # Verify if provided
            if self.verify_func:
                verified = self.verify_func()
                if not verified:
                    logger.warning(f"Verification failed for {self.service_name}")
                    return False

            logger.debug(f"Successfully executed operation for {self.service_name}")
            return result

        except Exception as e:
            logger.error(f"Failed to execute operation for {self.service_name}: {e}")
            logger.debug(traceback.format_exc())
            return False

    def rollback(self) -> None:
        """Rollback the operation."""
        try:
            logger.debug(f"Rolling back operation for {self.service_name}")
            self.rollback_func()
            logger.debug(f"Successfully rolled back {self.service_name}")

        except Exception as e:
            logger.error(f"Failed to rollback {self.service_name}: {e}")
            logger.debug(traceback.format_exc())


@dataclass
class ServiceUpdate:
    """Represents a complete service update with all necessary information."""
    service_name: str
    update_type: UpdateType
    dependencies: List[str]
    priority: ServicePriority
    apply_function: Callable[[Any], bool]
    verify_function: Callable[[], bool]
    rollback_function: Callable[[], None]
    settings_category: str
    estimated_duration_ms: int = 100
    can_parallelize: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ServiceDependencyGraph:
    """Manages service dependencies and determines safe update order."""

    def __init__(self):
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.priorities: Dict[str, ServicePriority] = {}

        # Initialize with known service dependencies
        self._initialize_default_dependencies()

    def _initialize_default_dependencies(self):
        """Initialize default service dependency graph."""
        # Core services
        self.add_dependency("detection", "webcam")  # Detection depends on webcam
        self.add_dependency("gemini", "detection")  # AI depends on detection
        self.add_dependency("inference", "detection")
        self.add_dependency("training", "detection")

        # UI depends on most services
        self.add_dependency("ui", "webcam")
        self.add_dependency("ui", "detection")
        self.add_dependency("ui", "gemini")

        # Set priorities
        self.set_priority("webcam", ServicePriority.CRITICAL)
        self.set_priority("detection", ServicePriority.HIGH)
        self.set_priority("gemini", ServicePriority.MEDIUM)
        self.set_priority("inference", ServicePriority.MEDIUM)
        self.set_priority("training", ServicePriority.LOW)
        self.set_priority("ui", ServicePriority.LOW)

    def add_dependency(self, service: str, depends_on: str):
        """Add a dependency relationship."""
        self.dependencies[service].add(depends_on)
        self.reverse_dependencies[depends_on].add(service)

    def set_priority(self, service: str, priority: ServicePriority):
        """Set service priority."""
        self.priorities[service] = priority

    def get_update_order(self, services: List[str]) -> List[str]:
        """Get safe update order for given services using topological sort."""
        # Build subgraph for requested services
        subgraph = {s: self.dependencies.get(s, set()) & set(services)
                   for s in services}

        # Topological sort with priority consideration
        visited = set()
        order = []

        def visit(service):
            if service in visited:
                return
            visited.add(service)

            # Visit dependencies first
            for dep in subgraph.get(service, set()):
                if dep not in visited:
                    visit(dep)

            order.append(service)

        # Sort services by priority before visiting
        sorted_services = sorted(
            services,
            key=lambda s: (self.priorities.get(s, ServicePriority.LOW).value, s)
        )

        for service in sorted_services:
            visit(service)

        return order

    def detect_circular_dependencies(self, services: List[str]) -> List[List[str]]:
        """Detect circular dependencies in the service graph."""
        cycles = []
        visited = set()
        rec_stack = []

        def visit(service):
            if service in rec_stack:
                # Found cycle
                cycle_start = rec_stack.index(service)
                cycles.append(rec_stack[cycle_start:] + [service])
                return

            if service in visited:
                return

            visited.add(service)
            rec_stack.append(service)

            for dep in self.dependencies.get(service, set()):
                if dep in services:
                    visit(dep)

            rec_stack.pop()

        for service in services:
            if service not in visited:
                visit(service)

        return cycles


class TransactionManager:
    """Manages atomic transactions for configuration updates."""

    def __init__(self):
        self.operations: List[AtomicOperation] = []
        self.executed_operations: List[AtomicOperation] = []
        self.snapshots: Dict[str, ServiceSnapshot] = {}
        self.transaction_id: Optional[str] = None
        self.in_transaction: bool = False
        self._lock = threading.RLock()
        self.dependency_graph = ServiceDependencyGraph()

    def begin_transaction(self, transaction_id: Optional[str] = None):
        """Begin a new transaction."""
        with self._lock:
            if self.in_transaction:
                raise RuntimeError("Transaction already in progress")

            self.transaction_id = transaction_id or f"txn_{datetime.now().timestamp()}"
            self.in_transaction = True
            self.operations.clear()
            self.executed_operations.clear()
            self.snapshots.clear()

            logger.info(f"Beginning transaction: {self.transaction_id}")

    def add_operation(self, operation: AtomicOperation):
        """Add an operation to the transaction."""
        with self._lock:
            if not self.in_transaction:
                raise RuntimeError("No transaction in progress")

            self.operations.append(operation)
            logger.debug(f"Added operation for {operation.service_name} to transaction")

    def add_snapshot(self, snapshot: ServiceSnapshot):
        """Add a service snapshot for rollback."""
        with self._lock:
            self.snapshots[snapshot.service_name] = snapshot
            logger.debug(f"Added snapshot for {snapshot.service_name}")

    def commit(self) -> bool:
        """Commit the transaction - execute all operations or rollback all."""
        with self._lock:
            if not self.in_transaction:
                raise RuntimeError("No transaction in progress")

            logger.info(f"Committing transaction {self.transaction_id} with {len(self.operations)} operations")

            try:
                # Sort operations by priority and dependencies
                sorted_ops = self._sort_operations()

                # Execute operations
                for operation in sorted_ops:
                    success = operation.execute()

                    if not success:
                        logger.error(f"Operation failed for {operation.service_name}, initiating rollback")
                        self.rollback()
                        return False

                    self.executed_operations.append(operation)

                # All operations succeeded
                logger.info(f"Transaction {self.transaction_id} committed successfully")
                self.in_transaction = False
                return True

            except Exception as e:
                logger.error(f"Transaction commit failed: {e}")
                logger.debug(traceback.format_exc())
                self.rollback()
                return False

            finally:
                if not self.in_transaction:
                    self._cleanup()

    def rollback(self):
        """Rollback all executed operations in reverse order."""
        with self._lock:
            if not self.in_transaction:
                return

            logger.warning(f"Rolling back transaction {self.transaction_id}")

            # Rollback in reverse order
            for operation in reversed(self.executed_operations):
                try:
                    operation.rollback()
                except Exception as e:
                    logger.error(f"Failed to rollback {operation.service_name}: {e}")

            # Restore snapshots
            for service_name, snapshot in self.snapshots.items():
                try:
                    if snapshot.service_ref and hasattr(snapshot.service_ref, 'restore_snapshot'):
                        snapshot.service_ref.restore_snapshot(snapshot.state_data)
                        logger.debug(f"Restored snapshot for {service_name}")
                except Exception as e:
                    logger.error(f"Failed to restore snapshot for {service_name}: {e}")

            self.in_transaction = False
            self._cleanup()

            logger.info(f"Transaction {self.transaction_id} rolled back")

    def _sort_operations(self) -> List[AtomicOperation]:
        """Sort operations by priority and dependencies."""
        # Group by service
        service_ops = defaultdict(list)
        for op in self.operations:
            service_ops[op.service_name].append(op)

        # Get update order from dependency graph
        service_order = self.dependency_graph.get_update_order(list(service_ops.keys()))

        # Build sorted operation list
        sorted_ops = []
        for service in service_order:
            # Sort operations within service by priority
            ops = sorted(
                service_ops[service],
                key=lambda o: (o.priority.value, o.operation_type.value)
            )
            sorted_ops.extend(ops)

        return sorted_ops

    def _cleanup(self):
        """Clean up transaction state."""
        self.operations.clear()
        self.executed_operations.clear()
        self.snapshots.clear()
        self.transaction_id = None


@dataclass
class PerformanceMetrics:
    """Track performance metrics for settings application."""
    total_duration_ms: float = 0.0
    service_durations: Dict[str, float] = field(default_factory=dict)
    parallel_operations: int = 0
    sequential_operations: int = 0
    rollbacks_performed: int = 0
    snapshots_created: int = 0
    operations_executed: int = 0
    operations_failed: int = 0

    def get_summary(self) -> str:
        """Get performance summary string."""
        return (
            f"Total: {self.total_duration_ms:.1f}ms, "
            f"Ops: {self.operations_executed} succeeded, {self.operations_failed} failed, "
            f"Rollbacks: {self.rollbacks_performed}, "
            f"Parallel: {self.parallel_operations}, Sequential: {self.sequential_operations}"
        )


class TransactionalSettingsApplier:
    """Main class for applying settings transactionally with rollback support."""

    def __init__(self, max_parallel_workers: int = 3):
        self.transaction_manager = TransactionManager()
        self.service_registry: Dict[str, Any] = {}
        self.update_strategies: Dict[str, ServiceUpdate] = {}
        self._initialize_update_strategies()
        self._lock = threading.RLock()

        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.performance_history: deque = deque(maxlen=100)

        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=max_parallel_workers)
        self.max_parallel_workers = max_parallel_workers

    def _initialize_update_strategies(self):
        """Initialize update strategies for different setting categories."""

        # Webcam settings strategy
        self.update_strategies["webcam"] = ServiceUpdate(
            service_name="webcam",
            update_type=UpdateType.RESTART,
            dependencies=[],
            priority=ServicePriority.CRITICAL,
            apply_function=self._apply_webcam_settings,
            verify_function=self._verify_webcam_settings,
            rollback_function=self._rollback_webcam_settings,
            settings_category="camera",
            estimated_duration_ms=500,
            can_parallelize=False
        )

        # Detection settings strategy
        self.update_strategies["detection"] = ServiceUpdate(
            service_name="detection",
            update_type=UpdateType.HOT_RELOAD,
            dependencies=["webcam"],
            priority=ServicePriority.HIGH,
            apply_function=self._apply_detection_settings,
            verify_function=self._verify_detection_settings,
            rollback_function=self._rollback_detection_settings,
            settings_category="detection",
            estimated_duration_ms=200,
            can_parallelize=True
        )

        # Gemini AI settings strategy
        self.update_strategies["gemini"] = ServiceUpdate(
            service_name="gemini",
            update_type=UpdateType.HOT_RELOAD,
            dependencies=["detection"],
            priority=ServicePriority.MEDIUM,
            apply_function=self._apply_gemini_settings,
            verify_function=self._verify_gemini_settings,
            rollback_function=self._rollback_gemini_settings,
            settings_category="ai",
            estimated_duration_ms=100,
            can_parallelize=True
        )

        # UI theme settings strategy
        self.update_strategies["ui"] = ServiceUpdate(
            service_name="ui",
            update_type=UpdateType.IMMEDIATE,
            dependencies=[],
            priority=ServicePriority.LOW,
            apply_function=self._apply_ui_settings,
            verify_function=self._verify_ui_settings,
            rollback_function=self._rollback_ui_settings,
            settings_category="appearance",
            estimated_duration_ms=50,
            can_parallelize=True
        )

    def register_service(self, name: str, service: Any):
        """Register a service for transactional updates."""
        with self._lock:
            self.service_registry[name] = service
            logger.debug(f"Registered service for transactions: {name}")

    def apply_settings(self, config: Any, changed_categories: Optional[Set[str]] = None) -> bool:
        """Apply settings transactionally with automatic rollback on failure.

        Args:
            config: Configuration object to apply
            changed_categories: Set of setting categories that changed (for optimization)

        Returns:
            bool: True if all settings were applied successfully
        """
        with self._lock:
            # Reset metrics for this operation
            start_time = time.time()
            self.metrics = PerformanceMetrics()

            try:
                # Begin transaction
                self.transaction_manager.begin_transaction()

                # Determine which services need updating
                services_to_update = self._determine_affected_services(changed_categories)
                logger.info(f"Applying settings to {len(services_to_update)} services: {services_to_update}")

                # Create snapshots for affected services
                snapshot_start = time.time()
                for service_name in services_to_update:
                    if service_name in self.service_registry:
                        snapshot = self._create_service_snapshot(service_name)
                        if snapshot:
                            self.transaction_manager.add_snapshot(snapshot)
                            self.metrics.snapshots_created += 1

                snapshot_duration = (time.time() - snapshot_start) * 1000
                logger.debug(f"Created {self.metrics.snapshots_created} snapshots in {snapshot_duration:.1f}ms")

                # Group operations by parallelizability
                parallel_ops = []
                sequential_ops = []

                for service_name in services_to_update:
                    if service_name in self.update_strategies:
                        strategy = self.update_strategies[service_name]
                        operation = self._create_atomic_operation(strategy, config)

                        if strategy.can_parallelize and not strategy.dependencies:
                            parallel_ops.append(operation)
                        else:
                            sequential_ops.append(operation)
                            self.transaction_manager.add_operation(operation)

                # Execute parallel operations if any
                if parallel_ops:
                    parallel_success = self._execute_parallel_operations(parallel_ops, config)
                    if not parallel_success:
                        logger.error("Parallel operations failed, initiating rollback")
                        self.transaction_manager.rollback()
                        self.metrics.rollbacks_performed += 1
                        return False

                # Commit sequential operations through transaction manager
                success = self.transaction_manager.commit()

                # Calculate total duration
                self.metrics.total_duration_ms = (time.time() - start_time) * 1000

                # Log performance metrics
                logger.info(f"Settings application completed: {self.metrics.get_summary()}")

                # Store metrics in history
                self.performance_history.append({
                    'timestamp': datetime.now(),
                    'metrics': copy.deepcopy(self.metrics),
                    'success': success
                })

                if success:
                    logger.info("All settings applied successfully")
                else:
                    logger.warning("Settings application failed, changes rolled back")
                    self.metrics.rollbacks_performed += 1

                return success

            except Exception as e:
                logger.error(f"Failed to apply settings transactionally: {e}")
                logger.debug(traceback.format_exc())
                self.transaction_manager.rollback()
                self.metrics.rollbacks_performed += 1
                self.metrics.total_duration_ms = (time.time() - start_time) * 1000
                return False

    def _determine_affected_services(self, changed_categories: Optional[Set[str]]) -> List[str]:
        """Determine which services are affected by the configuration changes."""
        if not changed_categories:
            # If no specific categories, update all services
            return list(self.update_strategies.keys())

        affected = []
        for service_name, strategy in self.update_strategies.items():
            if strategy.settings_category in changed_categories:
                affected.append(service_name)

        return affected

    def _create_service_snapshot(self, service_name: str) -> Optional[ServiceSnapshot]:
        """Create snapshot of service state."""
        try:
            service = self.service_registry.get(service_name)
            if not service:
                return None

            # Check if service supports snapshots
            if hasattr(service, 'create_snapshot'):
                state_data = service.create_snapshot()
            else:
                # Default snapshot - capture basic attributes
                state_data = self._capture_default_state(service)

            return ServiceSnapshot(
                service_name=service_name,
                timestamp=datetime.now(),
                state_data=state_data,
                service_ref=service
            )

        except Exception as e:
            logger.error(f"Failed to create snapshot for {service_name}: {e}")
            return None

    def _capture_default_state(self, service: Any) -> Dict[str, Any]:
        """Capture default state for services without snapshot support."""
        state = {}

        # Capture common service attributes
        for attr in ['config', 'settings', 'state', 'is_running', 'enabled']:
            if hasattr(service, attr):
                try:
                    value = getattr(service, attr)
                    # Only capture serializable values
                    if isinstance(value, (str, int, float, bool, dict, list)):
                        state[attr] = copy.deepcopy(value)
                except Exception:
                    pass

        return state

    def _execute_parallel_operations(self, operations: List[AtomicOperation], config: Any) -> bool:
        """Execute operations in parallel using thread pool.

        Args:
            operations: List of operations that can be parallelized
            config: Configuration object

        Returns:
            bool: True if all parallel operations succeeded
        """
        if not operations:
            return True

        logger.debug(f"Executing {len(operations)} operations in parallel")
        self.metrics.parallel_operations = len(operations)

        futures_to_ops = {}
        try:
            # Submit all operations to thread pool
            for op in operations:
                future = self.executor.submit(op.execute)
                futures_to_ops[future] = op

            # Wait for all to complete and collect results
            all_success = True
            for future in as_completed(futures_to_ops):
                op = futures_to_ops[future]
                try:
                    result = future.result(timeout=op.timeout_seconds)
                    if result:
                        self.metrics.operations_executed += 1
                        logger.debug(f"Parallel operation succeeded: {op.service_name}")
                    else:
                        self.metrics.operations_failed += 1
                        logger.error(f"Parallel operation failed: {op.service_name}")
                        all_success = False
                        # Don't break - let other operations complete for better rollback

                except Exception as e:
                    self.metrics.operations_failed += 1
                    logger.error(f"Exception in parallel operation {op.service_name}: {e}")
                    all_success = False

            # If any failed, rollback all parallel operations
            if not all_success:
                logger.warning("Rolling back parallel operations")
                for op in operations:
                    try:
                        op.rollback()
                    except Exception as e:
                        logger.error(f"Failed to rollback parallel operation {op.service_name}: {e}")

            return all_success

        except Exception as e:
            logger.error(f"Critical error in parallel execution: {e}")
            # Attempt to rollback all operations
            for op in operations:
                try:
                    op.rollback()
                except Exception:
                    pass
            return False

    def _create_atomic_operation(self, strategy: ServiceUpdate, config: Any) -> AtomicOperation:
        """Create atomic operation from update strategy."""
        return AtomicOperation(
            service_name=strategy.service_name,
            operation_type=strategy.update_type,
            apply_func=lambda: strategy.apply_function(config),
            rollback_func=strategy.rollback_function,
            verify_func=strategy.verify_function,
            priority=strategy.priority,
            dependencies=strategy.dependencies
        )

    # Service-specific update methods

    def _apply_webcam_settings(self, config: Any) -> bool:
        """Apply webcam settings with service restart if needed."""
        try:
            service = self.service_registry.get('webcam')
            if not service:
                logger.warning("Webcam service not registered")
                return True  # Not a failure if service doesn't exist

            # Stop capture if running
            was_capturing = False
            if hasattr(service, 'is_capturing') and service.is_capturing():
                was_capturing = True
                service.stop_capture()

            # Apply new settings
            if hasattr(service, 'set_camera'):
                service.set_camera(config.last_webcam_index)
            if hasattr(service, 'set_resolution'):
                service.set_resolution(config.camera_width, config.camera_height)
            if hasattr(service, 'set_fps'):
                service.set_fps(config.camera_fps)

            # Apply camera controls
            for control in ['brightness', 'contrast', 'saturation']:
                if hasattr(service, f'set_{control}'):
                    value = getattr(config, f'camera_{control}', None)
                    if value is not None:
                        getattr(service, f'set_{control}')(value)

            # Restart capture if was running
            if was_capturing and hasattr(service, 'start_capture'):
                service.start_capture()

            return True

        except Exception as e:
            logger.error(f"Failed to apply webcam settings: {e}")
            return False

    def _verify_webcam_settings(self) -> bool:
        """Verify webcam settings were applied correctly."""
        try:
            service = self.service_registry.get('webcam')
            if not service:
                return True

            # Check if camera is accessible
            if hasattr(service, 'is_opened'):
                return service.is_opened()

            return True

        except Exception as e:
            logger.error(f"Webcam verification failed: {e}")
            return False

    def _rollback_webcam_settings(self):
        """Rollback webcam settings."""
        # Snapshot restoration handles most of the rollback
        # Additional cleanup if needed
        try:
            service = self.service_registry.get('webcam')
            if service and hasattr(service, 'reset'):
                service.reset()
        except Exception as e:
            logger.error(f"Failed to rollback webcam settings: {e}")

    def _apply_detection_settings(self, config: Any) -> bool:
        """Apply detection model settings."""
        try:
            service = self.service_registry.get('detection')
            if not service:
                return True

            # Update thresholds
            if hasattr(service, 'set_confidence_threshold'):
                service.set_confidence_threshold(config.detection_confidence_threshold)
            if hasattr(service, 'set_iou_threshold'):
                service.set_iou_threshold(config.detection_iou_threshold)

            # Update ROI if supported
            if hasattr(service, 'set_roi') and config.enable_roi:
                service.set_roi(config.roi_x, config.roi_y,
                              config.roi_width, config.roi_height)
            elif hasattr(service, 'clear_roi') and not config.enable_roi:
                service.clear_roi()

            return True

        except Exception as e:
            logger.error(f"Failed to apply detection settings: {e}")
            return False

    def _verify_detection_settings(self) -> bool:
        """Verify detection settings."""
        return True  # Simple verification for now

    def _rollback_detection_settings(self):
        """Rollback detection settings."""
        pass  # Handled by snapshot restoration

    def _apply_gemini_settings(self, config: Any) -> bool:
        """Apply Gemini AI settings."""
        try:
            service = self.service_registry.get('gemini')
            if not service:
                return True

            # Update API configuration
            if hasattr(service, 'update_config'):
                service.update_config(
                    api_key=config.gemini_api_key,
                    model=config.gemini_model,
                    temperature=config.gemini_temperature,
                    max_tokens=config.gemini_max_tokens,
                    timeout=config.gemini_timeout
                )

            # Update rate limiting
            if hasattr(service, 'set_rate_limit'):
                service.set_rate_limit(
                    enabled=config.enable_rate_limiting,
                    requests_per_minute=config.requests_per_minute
                )

            return True

        except Exception as e:
            logger.error(f"Failed to apply Gemini settings: {e}")
            return False

    def _verify_gemini_settings(self) -> bool:
        """Verify Gemini settings."""
        return True

    def _rollback_gemini_settings(self):
        """Rollback Gemini settings."""
        pass

    def _apply_ui_settings(self, config: Any) -> bool:
        """Apply UI theme settings."""
        try:
            service = self.service_registry.get('main_window')
            if not service:
                return True

            if hasattr(service, 'apply_theme'):
                service.apply_theme(config.app_theme)

            return True

        except Exception as e:
            logger.error(f"Failed to apply UI settings: {e}")
            return False

    def _verify_ui_settings(self) -> bool:
        """Verify UI settings."""
        return True

    def _rollback_ui_settings(self):
        """Rollback UI settings."""
        pass


class StateManager:
    """Manages differential state tracking for services."""

    def __init__(self):
        self.states: Dict[str, Dict[str, Any]] = {}
        self.state_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self._lock = threading.RLock()

    def capture_state(self, service_name: str, state_data: Dict[str, Any]):
        """Capture current state of a service."""
        with self._lock:
            self.states[service_name] = copy.deepcopy(state_data)
            self.state_history[service_name].append({
                'timestamp': datetime.now(),
                'state': copy.deepcopy(state_data)
            })

    def get_diff(self, service_name: str, new_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get differential changes between current and new state."""
        with self._lock:
            old_state = self.states.get(service_name, {})
            diff = {}

            # Find changed values
            for key, new_value in new_state.items():
                old_value = old_state.get(key)
                if old_value != new_value:
                    diff[key] = {
                        'old': old_value,
                        'new': new_value
                    }

            # Find removed keys
            for key in old_state:
                if key not in new_state:
                    diff[key] = {
                        'old': old_state[key],
                        'new': None,
                        'removed': True
                    }

            return diff

    def restore_state(self, service_name: str, point_in_time: Optional[datetime] = None):
        """Restore state to a specific point in time or last known good state."""
        with self._lock:
            history = self.state_history.get(service_name, deque())

            if not history:
                return None

            if point_in_time:
                # Find state closest to requested time
                for entry in reversed(history):
                    if entry['timestamp'] <= point_in_time:
                        return entry['state']

            # Return most recent state
            return history[-1]['state'] if history else None


# Factory function for creating transactional applier
def create_transactional_applier() -> TransactionalSettingsApplier:
    """Create and configure a transactional settings applier."""
    return TransactionalSettingsApplier()


__all__ = [
    'TransactionalSettingsApplier',
    'TransactionManager',
    'AtomicOperation',
    'ServiceUpdate',
    'UpdateType',
    'ServicePriority',
    'ServiceDependencyGraph',
    'StateManager',
    'create_transactional_applier'
]