"""Health monitoring and metrics collection for production deployment.

This module provides health checks, basic metrics collection, and monitoring
endpoints for the Python Game Detection System.
"""
import time
import threading
import psutil
import os
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

from .logging_config import get_logger, log_security_event


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    name: str
    status: str  # 'healthy', 'unhealthy', 'degraded'
    message: str
    timestamp: datetime
    duration_ms: float
    metadata: Dict[str, Any]


@dataclass
class SystemMetrics:
    """System-level metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    process_count: int
    thread_count: int


@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    timestamp: datetime
    uptime_seconds: float
    total_requests: int
    successful_operations: int
    failed_operations: int
    webcam_frames_processed: int
    ai_requests_made: int
    ai_requests_successful: int
    detection_operations: int
    cache_hits: int
    cache_misses: int
    active_threads: int


class HealthCheck:
    """Base class for health checks."""

    def __init__(self, name: str, timeout_seconds: float = 5.0):
        self.name = name
        self.timeout_seconds = timeout_seconds

    def execute(self) -> HealthCheckResult:
        """Execute the health check."""
        start_time = time.time()

        try:
            status, message, metadata = self._check()
            duration_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                duration_ms=duration_ms,
                metadata=metadata
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                name=self.name,
                status='unhealthy',
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                duration_ms=duration_ms,
                metadata={'error': str(e)}
            )

    def _check(self) -> tuple[str, str, Dict[str, Any]]:
        """Override this method to implement specific health check logic."""
        raise NotImplementedError


class SystemResourcesHealthCheck(HealthCheck):
    """Health check for system resources."""

    def __init__(self, cpu_threshold: float = 90.0, memory_threshold: float = 90.0, disk_threshold: float = 95.0):
        super().__init__("system_resources")
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold

    def _check(self) -> tuple[str, str, Dict[str, Any]]:
        """Check system resource usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        metadata = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / (1024**3)
        }

        issues = []

        if cpu_percent > self.cpu_threshold:
            issues.append(f"High CPU usage: {cpu_percent:.1f}%")

        if memory.percent > self.memory_threshold:
            issues.append(f"High memory usage: {memory.percent:.1f}%")

        if disk.percent > self.disk_threshold:
            issues.append(f"Low disk space: {disk.percent:.1f}% used")

        if issues:
            status = 'degraded' if len(issues) == 1 and cpu_percent < 95 else 'unhealthy'
            message = "; ".join(issues)
        else:
            status = 'healthy'
            message = "System resources within normal limits"

        return status, message, metadata


class ApplicationServicesHealthCheck(HealthCheck):
    """Health check for application services."""

    def __init__(self, services: Dict[str, Callable[[], bool]]):
        super().__init__("application_services")
        self.services = services

    def _check(self) -> tuple[str, str, Dict[str, Any]]:
        """Check application services status."""
        service_statuses = {}
        failed_services = []

        for service_name, check_func in self.services.items():
            try:
                is_healthy = check_func()
                service_statuses[service_name] = 'healthy' if is_healthy else 'unhealthy'

                if not is_healthy:
                    failed_services.append(service_name)

            except Exception as e:
                service_statuses[service_name] = 'error'
                failed_services.append(f"{service_name} (error: {str(e)})")

        metadata = {'services': service_statuses}

        if failed_services:
            status = 'unhealthy'
            message = f"Failed services: {', '.join(failed_services)}"
        else:
            status = 'healthy'
            message = "All services operational"

        return status, message, metadata


class ConfigurationHealthCheck(HealthCheck):
    """Health check for configuration validity."""

    def __init__(self, config_validator: Callable[[], tuple[bool, str]]):
        super().__init__("configuration")
        self.config_validator = config_validator

    def _check(self) -> tuple[str, str, Dict[str, Any]]:
        """Check configuration validity."""
        is_valid, validation_message = self.config_validator()

        metadata = {
            'configuration_valid': is_valid,
            'validation_details': validation_message
        }

        if is_valid:
            status = 'healthy'
            message = "Configuration is valid"
        else:
            status = 'unhealthy'
            message = f"Configuration issues: {validation_message}"

        return status, message, metadata


class HealthMonitor:
    """Central health monitoring system."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.health_checks: List[HealthCheck] = []
        self.metrics_collectors: List[Callable[[], Dict[str, Any]]] = []

        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._metrics_thread: Optional[threading.Thread] = None

        self._application_start_time = time.time()
        self._last_health_results: Dict[str, HealthCheckResult] = {}
        self._metrics_history: List[Dict[str, Any]] = []
        self._max_history_size = 1000

        # Application metrics
        self.total_requests = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.webcam_frames_processed = 0
        self.ai_requests_made = 0
        self.ai_requests_successful = 0
        self.detection_operations = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def add_health_check(self, health_check: HealthCheck) -> None:
        """Add a health check to the monitoring system."""
        self.health_checks.append(health_check)
        self.logger.info(f"Added health check: {health_check.name}")

    def add_metrics_collector(self, collector: Callable[[], Dict[str, Any]]) -> None:
        """Add a metrics collector function."""
        self.metrics_collectors.append(collector)
        self.logger.info("Added metrics collector")

    def start(self, health_check_interval: float = 30.0, metrics_interval: float = 60.0) -> None:
        """Start the health monitoring system."""
        if self._running:
            self.logger.warning("Health monitor already running")
            return

        self._running = True

        # Start health check thread
        self._monitor_thread = threading.Thread(
            target=self._health_check_loop,
            args=(health_check_interval,),
            daemon=True
        )
        self._monitor_thread.start()

        # Start metrics collection thread
        self._metrics_thread = threading.Thread(
            target=self._metrics_collection_loop,
            args=(metrics_interval,),
            daemon=True
        )
        self._metrics_thread.start()

        self.logger.info("Health monitoring started")

    def stop(self) -> None:
        """Stop the health monitoring system."""
        if not self._running:
            return

        self._running = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

        if self._metrics_thread:
            self._metrics_thread.join(timeout=5.0)

        self.logger.info("Health monitoring stopped")

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        if not self._last_health_results:
            return {
                'status': 'unknown',
                'message': 'No health checks have been performed yet',
                'timestamp': datetime.utcnow().isoformat(),
                'checks': []
            }

        overall_status = 'healthy'
        unhealthy_checks = []
        degraded_checks = []

        checks = []
        for result in self._last_health_results.values():
            checks.append(asdict(result))

            if result.status == 'unhealthy':
                overall_status = 'unhealthy'
                unhealthy_checks.append(result.name)
            elif result.status == 'degraded' and overall_status != 'unhealthy':
                overall_status = 'degraded'
                degraded_checks.append(result.name)

        if overall_status == 'unhealthy':
            message = f"System unhealthy: {', '.join(unhealthy_checks)}"
        elif overall_status == 'degraded':
            message = f"System degraded: {', '.join(degraded_checks)}"
        else:
            message = "All systems operational"

        return {
            'status': overall_status,
            'message': message,
            'timestamp': datetime.utcnow().isoformat(),
            'uptime_seconds': time.time() - self._application_start_time,
            'checks': checks
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get current application metrics."""
        system_metrics = self._collect_system_metrics()
        app_metrics = self._collect_application_metrics()

        # Collect custom metrics
        custom_metrics = {}
        for collector in self.metrics_collectors:
            try:
                custom_metrics.update(collector())
            except Exception as e:
                self.logger.warning(f"Metrics collector failed: {e}")

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'system': asdict(system_metrics),
            'application': asdict(app_metrics),
            'custom': custom_metrics
        }

    def _health_check_loop(self, interval: float) -> None:
        """Main health check loop."""
        self.logger.info("Health check loop started")

        while self._running:
            try:
                for health_check in self.health_checks:
                    result = health_check.execute()
                    self._last_health_results[health_check.name] = result

                    # Log health check results
                    log_level = logging.INFO
                    if result.status == 'unhealthy':
                        log_level = logging.ERROR
                    elif result.status == 'degraded':
                        log_level = logging.WARNING

                    self.logger.log(
                        log_level,
                        f"Health check completed: {result.name}",
                        extra={
                            'health_check': result.name,
                            'status': result.status,
                            'duration_ms': result.duration_ms,
                            'health_message': result.message  # Renamed to avoid LogRecord conflict
                        }
                    )

                    # Log security events for unhealthy systems
                    if result.status == 'unhealthy':
                        log_security_event(
                            event_type='system_health',
                            description=f"System component unhealthy: {result.name}",
                            severity='WARNING',
                            additional_data={
                                'health_check': result.name,
                                'health_message': result.message,  # Renamed to avoid LogRecord conflict
                                'metadata': result.metadata
                            }
                        )

            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}", exc_info=True)

            # Wait for next interval
            time.sleep(interval)

        self.logger.info("Health check loop stopped")

    def _metrics_collection_loop(self, interval: float) -> None:
        """Main metrics collection loop."""
        self.logger.info("Metrics collection loop started")

        while self._running:
            try:
                metrics = self.get_metrics()

                # Add to history
                self._metrics_history.append(metrics)

                # Trim history if too large
                if len(self._metrics_history) > self._max_history_size:
                    self._metrics_history = self._metrics_history[-self._max_history_size:]

                # Log metrics periodically
                self.logger.debug("Metrics collected", extra={'metrics': metrics})

            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}", exc_info=True)

            # Wait for next interval
            time.sleep(interval)

        self.logger.info("Metrics collection loop stopped")

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics."""
        process = psutil.Process()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')

        return SystemMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=psutil.cpu_percent(),
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024**2),
            memory_available_mb=memory.available / (1024**2),
            disk_usage_percent=disk.percent,
            disk_free_gb=disk.free / (1024**3),
            process_count=len(psutil.pids()),
            thread_count=process.num_threads()
        )

    def _collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-level metrics."""
        return ApplicationMetrics(
            timestamp=datetime.utcnow(),
            uptime_seconds=time.time() - self._application_start_time,
            total_requests=self.total_requests,
            successful_operations=self.successful_operations,
            failed_operations=self.failed_operations,
            webcam_frames_processed=self.webcam_frames_processed,
            ai_requests_made=self.ai_requests_made,
            ai_requests_successful=self.ai_requests_successful,
            detection_operations=self.detection_operations,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            active_threads=threading.active_count()
        )

    def increment_counter(self, counter_name: str, value: int = 1) -> None:
        """Increment a metrics counter."""
        if hasattr(self, counter_name):
            current_value = getattr(self, counter_name)
            setattr(self, counter_name, current_value + value)

    def export_metrics(self, file_path: str) -> None:
        """Export metrics history to file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self._metrics_history, f, indent=2, default=str)
            self.logger.info(f"Metrics exported to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor


def setup_default_health_checks(config: Any) -> None:
    """Set up default health checks for the application."""
    monitor = get_health_monitor()

    # System resources health check
    monitor.add_health_check(SystemResourcesHealthCheck())

    # Configuration health check
    def validate_config() -> tuple[bool, str]:
        """Validate application configuration."""
        try:
            required_dirs = [
                getattr(config, 'data_dir', 'data'),
                getattr(config, 'models_dir', 'models'),
                getattr(config, 'master_dir', 'data/master')
            ]

            for directory in required_dirs:
                if not Path(directory).exists():
                    return False, f"Required directory missing: {directory}"

            return True, "Configuration valid"
        except Exception as e:
            return False, f"Configuration validation error: {e}"

    monitor.add_health_check(ConfigurationHealthCheck(validate_config))

    # Application services health check
    services = {
        'logging': lambda: True,  # Logging is working if we can execute this
        'performance_monitor': lambda: True,  # Add specific checks as needed
    }

    monitor.add_health_check(ApplicationServicesHealthCheck(services))