"""Base service classes to provide common functionality and reduce code duplication.

This module defines abstract base classes that implement common service patterns
used throughout the application, including logging, configuration management,
validation, error handling, and lifecycle management.
"""
from __future__ import annotations

import abc
import logging
import threading
import time
from typing import Any, Dict, Optional, Callable, Protocol, TypeVar, Generic, List
from dataclasses import dataclass, field
from contextlib import contextmanager

from .exceptions import ServiceError, ValidationError, ConfigError
from ..config.settings import Config

# Type variables for generic service implementations
ServiceType = TypeVar('ServiceType', bound='BaseService')
ConfigType = TypeVar('ConfigType')


@dataclass
class ServiceHealth:
    """Health check result for a service."""
    is_healthy: bool
    status_message: str
    last_check: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


class ServiceLifecycle(Protocol):
    """Protocol for service lifecycle management."""

    def start(self) -> None:
        """Start the service."""
        ...

    def stop(self) -> None:
        """Stop the service."""
        ...

    def is_running(self) -> bool:
        """Check if service is running."""
        ...


class ConfigurableService(Protocol[ConfigType]):
    """Protocol for services that can be configured."""

    def configure(self, config: ConfigType) -> None:
        """Configure the service."""
        ...

    def update_configuration(self, **kwargs: Any) -> None:
        """Update service configuration."""
        ...


class BaseService(abc.ABC):
    """Abstract base class for all services.

    Provides common functionality including:
    - Structured logging with service context
    - Configuration management and validation
    - Error handling with proper exception mapping
    - Health checks and status monitoring
    - Performance monitoring integration
    - Thread-safe state management
    """

    def __init__(self, config: Optional[Config] = None, service_name: Optional[str] = None):
        """Initialize base service.

        Args:
            config: Optional configuration object
            service_name: Optional service name for logging (defaults to class name)
        """
        self._service_name = service_name or self.__class__.__name__
        self._config = config
        self._logger = logging.getLogger(f"{__name__}.{self._service_name}")

        # Service state management
        self._is_initialized = False
        self._is_running = False
        self._state_lock = threading.RLock()
        self._last_error: Optional[Exception] = None

        # Health monitoring
        self._health_status = ServiceHealth(is_healthy=False, status_message="Not initialized")
        self._health_check_interval = 30.0  # seconds
        self._last_health_check = 0.0

        # Service metrics
        self._operation_counters: Dict[str, int] = {}
        self._error_counters: Dict[str, int] = {}
        self._start_time: Optional[float] = None
        self._total_operations = 0
        self._successful_operations = 0

        self._logger.debug(f"Initializing {self._service_name}")

    @property
    def service_name(self) -> str:
        """Get the service name."""
        return self._service_name

    @property
    def config(self) -> Optional[Config]:
        """Get the service configuration."""
        return self._config

    @property
    def logger(self) -> logging.Logger:
        """Get the service logger."""
        return self._logger

    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        with self._state_lock:
            return self._is_initialized

    @property
    def is_running(self) -> bool:
        """Check if service is running."""
        with self._state_lock:
            return self._is_running

    @property
    def last_error(self) -> Optional[Exception]:
        """Get the last error that occurred."""
        return self._last_error

    def initialize(self) -> None:
        """Initialize the service.

        This method should be called before using the service.
        Subclasses should override _initialize() to provide specific initialization logic.

        Raises:
            ServiceError: If initialization fails
            ConfigError: If configuration is invalid
        """
        with self._state_lock:
            if self._is_initialized:
                self._logger.debug("Service already initialized")
                return

            self._logger.info(f"Initializing service {self._service_name}")

            try:
                # Validate configuration
                if self._config:
                    self._validate_configuration()

                # Call subclass initialization
                self._initialize()

                self._is_initialized = True
                self._start_time = time.time()
                self._update_health_status(True, "Initialized successfully")
                self._logger.info(f"Service {self._service_name} initialized successfully")

            except Exception as e:
                self._last_error = e
                self._update_health_status(False, f"Initialization failed: {e}")
                self._logger.error(f"Failed to initialize service {self._service_name}: {e}")
                raise ServiceError(f"Failed to initialize {self._service_name}: {e}") from e

    def shutdown(self) -> None:
        """Shutdown the service gracefully.

        This method ensures proper cleanup of resources.
        Subclasses should override _shutdown() to provide specific cleanup logic.
        """
        with self._state_lock:
            if not self._is_initialized:
                self._logger.debug("Service not initialized, nothing to shutdown")
                return

            self._logger.info(f"Shutting down service {self._service_name}")

            try:
                # Stop the service if running
                if self._is_running:
                    self._stop()

                # Call subclass shutdown
                self._shutdown()

                self._is_initialized = False
                self._is_running = False
                self._update_health_status(False, "Service shutdown")
                self._logger.info(f"Service {self._service_name} shutdown completed")

            except Exception as e:
                self._last_error = e
                self._logger.error(f"Error during service shutdown: {e}")
                # Don't raise exception during shutdown to allow cleanup to continue

    def start(self) -> None:
        """Start the service.

        Raises:
            ServiceError: If service is not initialized or start fails
        """
        with self._state_lock:
            if not self._is_initialized:
                raise ServiceError(f"Service {self._service_name} not initialized")

            if self._is_running:
                self._logger.debug("Service already running")
                return

            self._logger.info(f"Starting service {self._service_name}")

            try:
                self._start()
                self._is_running = True
                self._update_health_status(True, "Service running")
                self._logger.info(f"Service {self._service_name} started successfully")

            except Exception as e:
                self._last_error = e
                self._update_health_status(False, f"Start failed: {e}")
                self._logger.error(f"Failed to start service {self._service_name}: {e}")
                raise ServiceError(f"Failed to start {self._service_name}: {e}") from e

    def stop(self) -> None:
        """Stop the service.

        Raises:
            ServiceError: If stop fails
        """
        with self._state_lock:
            if not self._is_running:
                self._logger.debug("Service not running")
                return

            self._logger.info(f"Stopping service {self._service_name}")

            try:
                self._stop()
                self._is_running = False
                self._update_health_status(True, "Service stopped")
                self._logger.info(f"Service {self._service_name} stopped successfully")

            except Exception as e:
                self._last_error = e
                self._update_health_status(False, f"Stop failed: {e}")
                self._logger.error(f"Failed to stop service {self._service_name}: {e}")
                raise ServiceError(f"Failed to stop {self._service_name}: {e}") from e

    def get_health_status(self, force_check: bool = False) -> ServiceHealth:
        """Get current health status of the service.

        Args:
            force_check: Whether to force a new health check

        Returns:
            ServiceHealth object with current status
        """
        current_time = time.time()

        # Check if we need to refresh health status
        if (force_check or
            current_time - self._last_health_check > self._health_check_interval):

            try:
                # Perform health check
                is_healthy = self._health_check()
                status_msg = "Service healthy" if is_healthy else "Service unhealthy"

                # Update health details
                details = self._get_health_details()

                self._health_status = ServiceHealth(
                    is_healthy=is_healthy,
                    status_message=status_msg,
                    last_check=current_time,
                    details=details
                )

                self._last_health_check = current_time

            except Exception as e:
                self._health_status = ServiceHealth(
                    is_healthy=False,
                    status_message=f"Health check failed: {e}",
                    last_check=current_time,
                    details={'error': str(e)}
                )

        return self._health_status

    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics and statistics.

        Returns:
            Dictionary containing service metrics
        """
        with self._state_lock:
            uptime = time.time() - self._start_time if self._start_time else 0.0
            success_rate = (self._successful_operations / self._total_operations
                          if self._total_operations > 0 else 0.0)

            return {
                'service_name': self._service_name,
                'is_initialized': self._is_initialized,
                'is_running': self._is_running,
                'uptime_seconds': uptime,
                'total_operations': self._total_operations,
                'successful_operations': self._successful_operations,
                'success_rate': success_rate,
                'operation_counters': self._operation_counters.copy(),
                'error_counters': self._error_counters.copy(),
                'last_error': str(self._last_error) if self._last_error else None,
                'health_status': self.get_health_status()
            }

    @contextmanager
    def operation_context(self, operation_name: str):
        """Context manager for tracking operations.

        Args:
            operation_name: Name of the operation being performed

        Usage:
            with self.operation_context("process_data"):
                # perform operation
                pass
        """
        start_time = time.time()
        self._total_operations += 1
        self._operation_counters[operation_name] = self._operation_counters.get(operation_name, 0) + 1

        try:
            yield
            self._successful_operations += 1

        except Exception as e:
            error_type = type(e).__name__
            self._error_counters[error_type] = self._error_counters.get(error_type, 0) + 1
            self._last_error = e
            raise

    # Abstract methods that subclasses must implement
    @abc.abstractmethod
    def _initialize(self) -> None:
        """Initialize service-specific resources."""
        pass

    @abc.abstractmethod
    def _shutdown(self) -> None:
        """Cleanup service-specific resources."""
        pass

    # Optional methods that subclasses can override
    def _start(self) -> None:
        """Start service-specific operations."""
        pass

    def _stop(self) -> None:
        """Stop service-specific operations."""
        pass

    def _validate_configuration(self) -> None:
        """Validate service configuration.

        Override this method to provide configuration validation.

        Raises:
            ConfigError: If configuration is invalid
        """
        pass

    def _health_check(self) -> bool:
        """Perform service-specific health check.

        Returns:
            True if service is healthy, False otherwise
        """
        return self._is_initialized and not self._last_error

    def _get_health_details(self) -> Dict[str, Any]:
        """Get detailed health information.

        Returns:
            Dictionary with detailed health information
        """
        return {
            'initialized': self._is_initialized,
            'running': self._is_running,
            'total_operations': self._total_operations,
            'error_count': len(self._error_counters)
        }

    def _update_health_status(self, is_healthy: bool, message: str) -> None:
        """Update the health status."""
        self._health_status = ServiceHealth(
            is_healthy=is_healthy,
            status_message=message,
            last_check=time.time()
        )


class AsyncServiceMixin:
    """Mixin for services that need async/threaded operations.

    Provides common functionality for managing background threads and
    async operations safely.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._background_threads: List[threading.Thread] = []
        self._shutdown_event = threading.Event()
        self._thread_pool_size = 4

    def _start_background_thread(self,
                                target: Callable,
                                name: Optional[str] = None,
                                daemon: bool = True,
                                **kwargs) -> threading.Thread:
        """Start a background thread and track it.

        Args:
            target: Function to run in thread
            name: Thread name
            daemon: Whether thread should be daemon
            **kwargs: Additional arguments for target function

        Returns:
            The started thread
        """
        thread_name = name or f"{self.service_name}Worker"
        thread = threading.Thread(
            target=target,
            name=thread_name,
            daemon=daemon,
            kwargs=kwargs
        )

        self._background_threads.append(thread)
        thread.start()

        self.logger.debug(f"Started background thread: {thread_name}")
        return thread

    def _stop_background_threads(self, timeout: float = 5.0) -> None:
        """Stop all background threads.

        Args:
            timeout: Maximum time to wait for threads to stop
        """
        if not self._background_threads:
            return

        self.logger.debug("Stopping background threads...")

        # Signal shutdown
        self._shutdown_event.set()

        # Wait for threads to stop
        for thread in self._background_threads:
            if thread.is_alive():
                thread.join(timeout=timeout)

                if thread.is_alive():
                    self.logger.warning(f"Thread {thread.name} did not stop gracefully")

        # Clean up finished threads
        self._background_threads = [t for t in self._background_threads if t.is_alive()]

        if not self._background_threads:
            self.logger.debug("All background threads stopped")
        else:
            self.logger.warning(f"{len(self._background_threads)} threads still running")

    def _shutdown(self) -> None:
        """Shutdown async service components."""
        self._stop_background_threads()
        super()._shutdown()


class CacheableServiceMixin:
    """Mixin for services that need simple caching functionality."""

    def __init__(self, cache_size: int = 100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: Dict[str, Any] = {}
        self._cache_max_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        value = self._cache.get(key)
        if value is not None:
            self._cache_hits += 1
        else:
            self._cache_misses += 1
        return value

    def _put_in_cache(self, key: str, value: Any) -> None:
        """Put item in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        # Simple size-limited cache
        if len(self._cache) >= self._cache_max_size:
            # Remove first item (simplest eviction strategy)
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        self._cache[key] = value

    def _clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self.logger.debug("Cache cleared")

    def _get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            'cache_size': len(self._cache),
            'max_size': self._cache_max_size,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate
        }

    def _get_health_details(self) -> Dict[str, Any]:
        """Include cache stats in health details."""
        details = super()._get_health_details()
        details.update(self._get_cache_stats())
        return details


class RetryableServiceMixin:
    """Mixin for services that need retry functionality."""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._retry_backoff_factor = 2.0

    def _retry_operation(self,
                        operation: Callable,
                        *args,
                        max_retries: Optional[int] = None,
                        retry_delay: Optional[float] = None,
                        **kwargs) -> Any:
        """Retry an operation with exponential backoff.

        Args:
            operation: Function to retry
            max_retries: Maximum number of retries (uses default if None)
            retry_delay: Initial delay between retries (uses default if None)
            *args, **kwargs: Arguments for the operation

        Returns:
            Result of the successful operation

        Raises:
            The last exception if all retries fail
        """
        max_retries = max_retries or self._max_retries
        retry_delay = retry_delay or self._retry_delay

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return operation(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if attempt < max_retries:
                    delay = retry_delay * (self._retry_backoff_factor ** attempt)
                    self.logger.debug(
                        f"Operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(f"Operation failed after {max_retries + 1} attempts: {e}")

        # Re-raise the last exception
        raise last_exception


# Composite base classes combining mixins
class AsyncCacheableService(AsyncServiceMixin, CacheableServiceMixin, BaseService):
    """Service with async and caching capabilities."""
    pass


class RetryableAsyncService(RetryableServiceMixin, AsyncServiceMixin, BaseService):
    """Service with retry and async capabilities."""
    pass


class FullFeaturedService(RetryableServiceMixin, CacheableServiceMixin, AsyncServiceMixin, BaseService):
    """Service with all available capabilities."""
    pass