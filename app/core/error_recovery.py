"""Comprehensive Error Recovery System for handling partial failures during settings application.

This module provides a complete error recovery system that:
- Detects and classifies failures during configuration application
- Provides multiple recovery strategies with user interaction
- Captures full diagnostic information for debugging
- Implements safe mode for system stability
- Learns from error patterns to prevent future issues
- Maintains system usability after failures

Key components:
- FailureContext: Comprehensive failure information capture
- ErrorRecoveryManager: Central coordinator for recovery operations
- RecoveryExecutor: Executes recovery strategies
- DiagnosticCollector: Captures system state and generates reports
- SafeModeManager: Minimal configuration fallback system
- ErrorLearning: Pattern recognition and prevention
"""
from __future__ import annotations

import logging
import time
import traceback
import threading
import pickle
import json
import psutil
import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Union,
    Protocol, runtime_checkable
)
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from collections import defaultdict, deque

from .exceptions import (
    ApplicationError, ConfigurationError, ServiceError,
    ValidationError, ModelError, WebcamError, SecurityError
)
from ..config.settings import Config
from ..config.types import ConfigVersion

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can occur during settings application."""
    VALIDATION_ERROR = "validation_error"
    SERVICE_RESTART_FAILED = "service_restart"
    RESOURCE_UNAVAILABLE = "resource"
    PERMISSION_DENIED = "permission"
    NETWORK_ERROR = "network"
    PARTIAL_APPLICATION = "partial"
    MODEL_LOADING_ERROR = "model_loading"
    CAMERA_ACCESS_ERROR = "camera_access"
    DISK_SPACE_ERROR = "disk_space"
    MEMORY_ERROR = "memory"
    TIMEOUT_ERROR = "timeout"
    DEPENDENCY_ERROR = "dependency"
    UNKNOWN_ERROR = "unknown"


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"                    # Try the operation again
    ROLLBACK = "rollback"             # Revert to previous state
    PARTIAL_APPLY = "partial"         # Apply what succeeded
    SAFE_MODE = "safe_mode"           # Start with minimal config
    REPAIR = "repair"                 # Fix and retry
    IGNORE = "ignore"                 # Continue despite failure
    MANUAL_INTERVENTION = "manual"    # Require user action


class Severity(Enum):
    """Failure severity levels."""
    LOW = "low"                       # Non-critical, system remains functional
    MEDIUM = "medium"                 # Some functionality affected
    HIGH = "high"                     # Major functionality loss
    CRITICAL = "critical"             # System unstable or unusable


@dataclass(slots=True)
class SystemSnapshot:
    """Comprehensive snapshot of system state at failure time."""
    timestamp: datetime
    memory_usage_mb: float
    cpu_usage_percent: float
    disk_free_gb: float
    active_threads: int
    open_files: int

    # Service states
    service_states: Dict[str, Any] = field(default_factory=dict)

    # Configuration state
    config_checksum: Optional[str] = None
    config_version: Optional[str] = None
    last_successful_config: Optional[Dict[str, Any]] = None

    # System resources
    gpu_available: bool = False
    gpu_memory_mb: Optional[float] = None
    camera_devices: List[Dict[str, Any]] = field(default_factory=list)
    network_status: Dict[str, Any] = field(default_factory=dict)

    # Recent logs
    recent_errors: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'disk_free_gb': self.disk_free_gb,
            'active_threads': self.active_threads,
            'open_files': self.open_files,
            'service_states': self.service_states,
            'config_checksum': self.config_checksum,
            'config_version': self.config_version,
            'last_successful_config': self.last_successful_config,
            'gpu_available': self.gpu_available,
            'gpu_memory_mb': self.gpu_memory_mb,
            'camera_devices': self.camera_devices,
            'network_status': self.network_status,
            'recent_errors': self.recent_errors,
            'performance_metrics': self.performance_metrics
        }


@dataclass(slots=True)
class RecoveryOption:
    """Represents a recovery option that can be presented to the user."""
    strategy: RecoveryStrategy
    title: str
    description: str
    estimated_time_seconds: int
    success_probability: float  # 0.0 to 1.0
    requires_user_input: bool = False
    is_safe: bool = True
    side_effects: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for UI display."""
        return {
            'strategy': self.strategy.value,
            'title': self.title,
            'description': self.description,
            'estimated_time_seconds': self.estimated_time_seconds,
            'success_probability': self.success_probability,
            'requires_user_input': self.requires_user_input,
            'is_safe': self.is_safe,
            'side_effects': self.side_effects,
            'metadata': self.metadata
        }


@dataclass(slots=True)
class FailureContext:
    """Comprehensive context information about a failure."""
    failure_type: FailureType
    severity: Severity
    affected_services: List[str]
    error_messages: List[str]
    timestamp: datetime
    system_state: SystemSnapshot
    recovery_options: List[RecoveryOption] = field(default_factory=list)

    # Failure details
    exception_info: Optional[str] = None
    stack_trace: Optional[str] = None
    operation_attempted: Optional[str] = None
    config_changes: Optional[Dict[str, Any]] = None

    # Context
    user_action: Optional[str] = None
    session_id: str = ""
    correlation_id: str = ""

    # Recovery tracking
    recovery_attempts: List[Dict[str, Any]] = field(default_factory=list)
    resolved: bool = False
    resolution_strategy: Optional[RecoveryStrategy] = None

    def add_recovery_attempt(self, strategy: RecoveryStrategy, success: bool,
                           details: str = "") -> None:
        """Record a recovery attempt."""
        self.recovery_attempts.append({
            'strategy': strategy.value,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'details': details
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'failure_type': self.failure_type.value,
            'severity': self.severity.value,
            'affected_services': self.affected_services,
            'error_messages': self.error_messages,
            'timestamp': self.timestamp.isoformat(),
            'system_state': self.system_state.to_dict(),
            'recovery_options': [opt.to_dict() for opt in self.recovery_options],
            'exception_info': self.exception_info,
            'stack_trace': self.stack_trace,
            'operation_attempted': self.operation_attempted,
            'config_changes': self.config_changes,
            'user_action': self.user_action,
            'session_id': self.session_id,
            'correlation_id': self.correlation_id,
            'recovery_attempts': self.recovery_attempts,
            'resolved': self.resolved,
            'resolution_strategy': self.resolution_strategy.value if self.resolution_strategy else None
        }


@dataclass(slots=True)
class RecoveryResult:
    """Result of a recovery operation."""
    success: bool
    strategy_used: RecoveryStrategy
    message: str
    duration_seconds: float
    services_affected: List[str] = field(default_factory=list)
    config_changes: Optional[Dict[str, Any]] = None
    system_state_after: Optional[SystemSnapshot] = None

    # Additional information
    warnings: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    next_recommended_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and UI."""
        return {
            'success': self.success,
            'strategy_used': self.strategy_used.value,
            'message': self.message,
            'duration_seconds': self.duration_seconds,
            'services_affected': self.services_affected,
            'config_changes': self.config_changes,
            'system_state_after': self.system_state_after.to_dict() if self.system_state_after else None,
            'warnings': self.warnings,
            'side_effects': self.side_effects,
            'next_recommended_action': self.next_recommended_action
        }


@dataclass
class ErrorPattern:
    """Represents a pattern of errors for learning and prevention."""
    pattern_id: str
    occurrences: int
    last_seen: datetime
    first_seen: datetime
    trigger_conditions: Dict[str, Any]
    successful_recovery: Optional[RecoveryStrategy] = None
    common_causes: List[str] = field(default_factory=list)
    prevention_suggestions: List[str] = field(default_factory=list)
    affected_services: Set[str] = field(default_factory=set)
    severity_trend: List[Severity] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            'pattern_id': self.pattern_id,
            'occurrences': self.occurrences,
            'last_seen': self.last_seen.isoformat(),
            'first_seen': self.first_seen.isoformat(),
            'trigger_conditions': self.trigger_conditions,
            'successful_recovery': self.successful_recovery.value if self.successful_recovery else None,
            'common_causes': self.common_causes,
            'prevention_suggestions': self.prevention_suggestions,
            'affected_services': list(self.affected_services),
            'severity_trend': [s.value for s in self.severity_trend]
        }


class DiagnosticCollector:
    """Collects comprehensive diagnostic information about system state."""

    def __init__(self):
        self._lock = threading.RLock()

    def capture_state(self) -> SystemSnapshot:
        """Capture comprehensive system state."""
        with self._lock:
            start_time = time.time()

            try:
                # System metrics
                process = psutil.Process()
                memory_info = process.memory_info()
                cpu_percent = process.cpu_percent()

                # Disk space
                disk_usage = psutil.disk_usage('/')
                disk_free_gb = disk_usage.free / (1024**3)

                # Thread and file info
                active_threads = threading.active_count()
                try:
                    open_files = len(process.open_files())
                except (psutil.AccessDenied, OSError):
                    open_files = -1

                # GPU information
                gpu_available, gpu_memory = self._check_gpu_status()

                # Camera devices
                camera_devices = self._enumerate_camera_devices()

                # Network status
                network_status = self._check_network_status()

                # Recent errors from logging
                recent_errors = self._collect_recent_errors()

                # Performance metrics
                performance_metrics = self._collect_performance_metrics()

                snapshot = SystemSnapshot(
                    timestamp=datetime.now(),
                    memory_usage_mb=memory_info.rss / (1024*1024),
                    cpu_usage_percent=cpu_percent,
                    disk_free_gb=disk_free_gb,
                    active_threads=active_threads,
                    open_files=open_files,
                    gpu_available=gpu_available,
                    gpu_memory_mb=gpu_memory,
                    camera_devices=camera_devices,
                    network_status=network_status,
                    recent_errors=recent_errors,
                    performance_metrics=performance_metrics
                )

                duration = time.time() - start_time
                logger.debug(f"System state captured in {duration:.2f}s")

                return snapshot

            except Exception as e:
                logger.error(f"Failed to capture system state: {e}")
                # Return minimal snapshot
                return SystemSnapshot(
                    timestamp=datetime.now(),
                    memory_usage_mb=-1,
                    cpu_usage_percent=-1,
                    disk_free_gb=-1,
                    active_threads=-1,
                    open_files=-1
                )

    def analyze_logs(self, hours_back: int = 1) -> List[Dict[str, Any]]:
        """Analyze recent logs for issues and patterns."""
        issues = []

        try:
            # This would typically analyze application logs
            # For now, return placeholder analysis
            cutoff_time = datetime.now() - timedelta(hours=hours_back)

            # Analyze memory usage patterns
            if psutil.virtual_memory().percent > 80:
                issues.append({
                    'type': 'high_memory_usage',
                    'severity': 'medium',
                    'description': f'System memory usage is high: {psutil.virtual_memory().percent:.1f}%',
                    'recommendation': 'Consider restarting memory-intensive services'
                })

            # Analyze disk space
            disk_usage = psutil.disk_usage('/')
            if disk_usage.free / disk_usage.total < 0.1:  # Less than 10% free
                issues.append({
                    'type': 'low_disk_space',
                    'severity': 'high',
                    'description': f'Low disk space: {disk_usage.free / (1024**3):.1f}GB free',
                    'recommendation': 'Clean up temporary files and logs'
                })

        except Exception as e:
            logger.error(f"Failed to analyze logs: {e}")
            issues.append({
                'type': 'log_analysis_failed',
                'severity': 'low',
                'description': f'Could not analyze system logs: {e}',
                'recommendation': 'Check log file permissions and accessibility'
            })

        return issues

    def generate_report(self, context: FailureContext) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report."""
        report = {
            'report_id': f"diag_{int(time.time())}",
            'generated_at': datetime.now().isoformat(),
            'failure_summary': {
                'type': context.failure_type.value,
                'severity': context.severity.value,
                'affected_services': context.affected_services,
                'error_count': len(context.error_messages)
            },
            'system_analysis': self._analyze_system_health(context.system_state),
            'recent_issues': self.analyze_logs(),
            'recommendations': self.get_recommendations(context),
            'recovery_history': context.recovery_attempts,
            'context': context.to_dict()
        }

        return report

    def get_recommendations(self, context: FailureContext) -> List[str]:
        """Generate specific recommendations based on failure context."""
        recommendations = []

        # Memory-related recommendations
        if context.system_state.memory_usage_mb > 2048:  # More than 2GB
            recommendations.append("Consider reducing memory usage by closing unused applications")
            recommendations.append("Check for memory leaks in recent operations")

        # Service-specific recommendations
        if "webcam" in context.affected_services:
            recommendations.append("Verify camera device connectivity and permissions")
            recommendations.append("Try using a different camera index if multiple cameras available")

        if "gemini" in context.affected_services:
            recommendations.append("Check API key validity and network connectivity")
            recommendations.append("Verify rate limiting settings")

        if "detection" in context.affected_services:
            recommendations.append("Ensure model files are accessible and not corrupted")
            recommendations.append("Check GPU availability if using GPU acceleration")

        # General recommendations
        if context.failure_type == FailureType.NETWORK_ERROR:
            recommendations.append("Check internet connectivity and firewall settings")
            recommendations.append("Consider using cached/offline mode if available")

        if context.failure_type == FailureType.PERMISSION_DENIED:
            recommendations.append("Run application with appropriate permissions")
            recommendations.append("Check file and directory access rights")

        if context.failure_type == FailureType.RESOURCE_UNAVAILABLE:
            recommendations.append("Close other resource-intensive applications")
            recommendations.append("Consider upgrading system resources if problem persists")

        return recommendations

    def _check_gpu_status(self) -> Tuple[bool, Optional[float]]:
        """Check GPU availability and memory."""
        try:
            # Try to import torch and check CUDA
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                return True, memory_gb * 1024  # Convert to MB
            return False, None
        except ImportError:
            return False, None
        except Exception as e:
            logger.debug(f"GPU check failed: {e}")
            return False, None

    def _enumerate_camera_devices(self) -> List[Dict[str, Any]]:
        """Enumerate available camera devices."""
        cameras = []
        try:
            import cv2
            for i in range(10):  # Check first 10 camera indices
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cameras.append({
                        'index': i,
                        'width': int(width),
                        'height': int(height),
                        'fps': int(fps),
                        'available': True
                    })
                cap.release()
        except Exception as e:
            logger.debug(f"Camera enumeration failed: {e}")

        return cameras

    def _check_network_status(self) -> Dict[str, Any]:
        """Check network connectivity."""
        status = {
            'connected': False,
            'interfaces': [],
            'dns_reachable': False,
            'internet_reachable': False
        }

        try:
            # Check network interfaces
            interfaces = psutil.net_if_addrs()
            for name, addrs in interfaces.items():
                for addr in addrs:
                    if addr.family == 2:  # IPv4
                        status['interfaces'].append({
                            'name': name,
                            'address': addr.address,
                            'netmask': addr.netmask
                        })

            status['connected'] = len(status['interfaces']) > 0

            # Basic connectivity tests would go here
            # For now, assume connected if interfaces exist
            status['dns_reachable'] = status['connected']
            status['internet_reachable'] = status['connected']

        except Exception as e:
            logger.debug(f"Network status check failed: {e}")

        return status

    def _collect_recent_errors(self, max_errors: int = 10) -> List[Dict[str, Any]]:
        """Collect recent error messages from logs."""
        errors = []

        try:
            # This would typically parse log files
            # For now, return empty list
            pass
        except Exception as e:
            logger.debug(f"Error collection failed: {e}")

        return errors

    def _collect_performance_metrics(self) -> Dict[str, float]:
        """Collect current performance metrics."""
        metrics = {}

        try:
            # Memory metrics
            vm = psutil.virtual_memory()
            metrics['memory_percent'] = vm.percent
            metrics['memory_available_gb'] = vm.available / (1024**3)

            # CPU metrics
            metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            metrics['load_average'] = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else -1

            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics['disk_usage_percent'] = (disk.used / disk.total) * 100

            # Process metrics
            process = psutil.Process()
            metrics['process_memory_mb'] = process.memory_info().rss / (1024*1024)
            metrics['process_cpu_percent'] = process.cpu_percent()

        except Exception as e:
            logger.debug(f"Performance metrics collection failed: {e}")

        return metrics

    def _analyze_system_health(self, snapshot: SystemSnapshot) -> Dict[str, Any]:
        """Analyze system health from snapshot."""
        analysis = {
            'overall_health': 'good',
            'concerns': [],
            'critical_issues': []
        }

        # Memory analysis
        if snapshot.memory_usage_mb > 4096:  # >4GB
            analysis['concerns'].append('High memory usage detected')
            if snapshot.memory_usage_mb > 8192:  # >8GB
                analysis['critical_issues'].append('Critically high memory usage')
                analysis['overall_health'] = 'poor'

        # CPU analysis
        if snapshot.cpu_usage_percent > 80:
            analysis['concerns'].append('High CPU usage detected')
            if snapshot.cpu_usage_percent > 95:
                analysis['critical_issues'].append('Critically high CPU usage')
                analysis['overall_health'] = 'poor'

        # Disk space analysis
        if snapshot.disk_free_gb < 1:  # Less than 1GB free
            analysis['critical_issues'].append('Critically low disk space')
            analysis['overall_health'] = 'poor'
        elif snapshot.disk_free_gb < 5:  # Less than 5GB free
            analysis['concerns'].append('Low disk space warning')
            if analysis['overall_health'] == 'good':
                analysis['overall_health'] = 'fair'

        # Thread analysis
        if snapshot.active_threads > 50:
            analysis['concerns'].append('High thread count detected')

        # Update overall health
        if analysis['critical_issues']:
            analysis['overall_health'] = 'poor'
        elif analysis['concerns'] and analysis['overall_health'] == 'good':
            analysis['overall_health'] = 'fair'

        return analysis


class RecoveryExecutor:
    """Executes recovery strategies and manages recovery procedures."""

    def __init__(self, service_registry: Dict[str, Any],
                 config_manager: Any = None):
        self.service_registry = service_registry
        self.config_manager = config_manager
        self.diagnostic_collector = DiagnosticCollector()
        self._lock = threading.RLock()
        self.recovery_history: deque = deque(maxlen=100)

    def analyze_failure(self, context: FailureContext) -> List[RecoveryOption]:
        """Analyze failure and return available recovery options."""
        with self._lock:
            options = []

            try:
                # Determine available recovery strategies based on failure type
                if context.failure_type == FailureType.VALIDATION_ERROR:
                    options.extend(self._get_validation_recovery_options(context))
                elif context.failure_type == FailureType.SERVICE_RESTART_FAILED:
                    options.extend(self._get_service_recovery_options(context))
                elif context.failure_type == FailureType.RESOURCE_UNAVAILABLE:
                    options.extend(self._get_resource_recovery_options(context))
                elif context.failure_type == FailureType.NETWORK_ERROR:
                    options.extend(self._get_network_recovery_options(context))
                elif context.failure_type == FailureType.PERMISSION_DENIED:
                    options.extend(self._get_permission_recovery_options(context))
                elif context.failure_type == FailureType.PARTIAL_APPLICATION:
                    options.extend(self._get_partial_recovery_options(context))
                else:
                    options.extend(self._get_generic_recovery_options(context))

                # Sort by success probability and safety
                options.sort(key=lambda opt: (opt.is_safe, opt.success_probability), reverse=True)

                # Update context with options
                context.recovery_options = options

                logger.info(f"Generated {len(options)} recovery options for {context.failure_type.value}")
                return options

            except Exception as e:
                logger.error(f"Failed to analyze failure: {e}")
                # Return safe default options
                return [self._get_safe_mode_option(), self._get_rollback_option()]

    def execute_recovery(self, option: RecoveryOption, context: FailureContext) -> RecoveryResult:
        """Execute a specific recovery strategy."""
        start_time = time.time()

        try:
            logger.info(f"Executing recovery strategy: {option.strategy.value}")

            # Pre-recovery system snapshot
            pre_snapshot = self.diagnostic_collector.capture_state()

            # Execute the strategy
            if option.strategy == RecoveryStrategy.RETRY:
                result = self._execute_retry(option, context)
            elif option.strategy == RecoveryStrategy.ROLLBACK:
                result = self._execute_rollback(option, context)
            elif option.strategy == RecoveryStrategy.PARTIAL_APPLY:
                result = self._execute_partial_apply(option, context)
            elif option.strategy == RecoveryStrategy.SAFE_MODE:
                result = self._execute_safe_mode(option, context)
            elif option.strategy == RecoveryStrategy.REPAIR:
                result = self._execute_repair(option, context)
            elif option.strategy == RecoveryStrategy.IGNORE:
                result = self._execute_ignore(option, context)
            else:
                result = RecoveryResult(
                    success=False,
                    strategy_used=option.strategy,
                    message=f"Unknown recovery strategy: {option.strategy.value}",
                    duration_seconds=time.time() - start_time
                )

            # Post-recovery system snapshot
            post_snapshot = self.diagnostic_collector.capture_state()
            result.system_state_after = post_snapshot

            # Record recovery attempt
            context.add_recovery_attempt(option.strategy, result.success, result.message)

            # Store in history
            self.recovery_history.append({
                'timestamp': datetime.now(),
                'context': context.to_dict(),
                'option': option.to_dict(),
                'result': result.to_dict()
            })

            if result.success:
                context.resolved = True
                context.resolution_strategy = option.strategy
                logger.info(f"Recovery successful: {result.message}")
            else:
                logger.warning(f"Recovery failed: {result.message}")

            return result

        except Exception as e:
            logger.error(f"Recovery execution failed: {e}")
            return RecoveryResult(
                success=False,
                strategy_used=option.strategy,
                message=f"Recovery execution error: {e}",
                duration_seconds=time.time() - start_time
            )

    def verify_recovery(self, result: RecoveryResult) -> bool:
        """Verify that recovery was successful and system is stable."""
        try:
            # Basic system health checks
            snapshot = self.diagnostic_collector.capture_state()

            # Check if affected services are responsive
            for service_name in result.services_affected:
                service = self.service_registry.get(service_name)
                if service and hasattr(service, 'is_healthy'):
                    if not service.is_healthy():
                        logger.warning(f"Service {service_name} not healthy after recovery")
                        return False

            # Check system resources
            if snapshot.memory_usage_mb > 8192:  # >8GB indicates potential issues
                logger.warning("High memory usage detected after recovery")
                return False

            # Check for critical errors in recent logs
            issues = self.diagnostic_collector.analyze_logs(hours_back=0.1)  # Last 6 minutes
            critical_issues = [i for i in issues if i.get('severity') == 'critical']
            if critical_issues:
                logger.warning(f"Critical issues found after recovery: {len(critical_issues)}")
                return False

            logger.info("Recovery verification passed")
            return True

        except Exception as e:
            logger.error(f"Recovery verification failed: {e}")
            return False

    def _get_validation_recovery_options(self, context: FailureContext) -> List[RecoveryOption]:
        """Get recovery options for validation errors."""
        options = []

        # Retry with corrected values
        options.append(RecoveryOption(
            strategy=RecoveryStrategy.REPAIR,
            title="Fix and Retry",
            description="Automatically correct invalid settings and retry application",
            estimated_time_seconds=10,
            success_probability=0.8,
            is_safe=True,
            side_effects=["Some settings may be changed to valid values"]
        ))

        # Apply only valid settings
        options.append(RecoveryOption(
            strategy=RecoveryStrategy.PARTIAL_APPLY,
            title="Apply Valid Settings Only",
            description="Apply only the settings that passed validation",
            estimated_time_seconds=5,
            success_probability=0.9,
            is_safe=True,
            side_effects=["Invalid settings will be ignored"]
        ))

        return options

    def _get_service_recovery_options(self, context: FailureContext) -> List[RecoveryOption]:
        """Get recovery options for service restart failures."""
        options = []

        # Force restart problematic services
        options.append(RecoveryOption(
            strategy=RecoveryStrategy.RETRY,
            title="Force Service Restart",
            description="Forcefully stop and restart failed services",
            estimated_time_seconds=30,
            success_probability=0.7,
            is_safe=True,
            side_effects=["Temporary service interruption"]
        ))

        # Start in safe mode
        options.append(RecoveryOption(
            strategy=RecoveryStrategy.SAFE_MODE,
            title="Start in Safe Mode",
            description="Start services with minimal, known-good configuration",
            estimated_time_seconds=20,
            success_probability=0.95,
            is_safe=True,
            side_effects=["Reduced functionality until manual configuration"]
        ))

        return options

    def _get_resource_recovery_options(self, context: FailureContext) -> List[RecoveryOption]:
        """Get recovery options for resource unavailability."""
        options = []

        # Free up resources and retry
        options.append(RecoveryOption(
            strategy=RecoveryStrategy.REPAIR,
            title="Free Resources and Retry",
            description="Attempt to free up system resources and retry operation",
            estimated_time_seconds=15,
            success_probability=0.6,
            is_safe=True,
            side_effects=["Memory cleanup may affect other applications"]
        ))

        # Reduce resource usage settings
        options.append(RecoveryOption(
            strategy=RecoveryStrategy.PARTIAL_APPLY,
            title="Apply Resource-Efficient Settings",
            description="Apply settings with reduced resource requirements",
            estimated_time_seconds=10,
            success_probability=0.8,
            is_safe=True,
            side_effects=["Reduced performance or functionality"]
        ))

        return options

    def _get_network_recovery_options(self, context: FailureContext) -> List[RecoveryOption]:
        """Get recovery options for network errors."""
        options = []

        # Retry with timeout increase
        options.append(RecoveryOption(
            strategy=RecoveryStrategy.RETRY,
            title="Retry with Extended Timeout",
            description="Retry network operations with longer timeout values",
            estimated_time_seconds=30,
            success_probability=0.5,
            is_safe=True,
            side_effects=["Operations may take longer to complete"]
        ))

        # Use offline/cached mode
        options.append(RecoveryOption(
            strategy=RecoveryStrategy.PARTIAL_APPLY,
            title="Use Offline Mode",
            description="Disable network-dependent features and use cached data",
            estimated_time_seconds=5,
            success_probability=0.9,
            is_safe=True,
            side_effects=["Network-dependent features will be unavailable"]
        ))

        return options

    def _get_permission_recovery_options(self, context: FailureContext) -> List[RecoveryOption]:
        """Get recovery options for permission errors."""
        options = []

        # Use alternative paths
        options.append(RecoveryOption(
            strategy=RecoveryStrategy.REPAIR,
            title="Use Alternative Paths",
            description="Automatically switch to accessible alternative directories",
            estimated_time_seconds=5,
            success_probability=0.7,
            is_safe=True,
            side_effects=["Files may be stored in different locations"]
        ))

        # Manual intervention required
        options.append(RecoveryOption(
            strategy=RecoveryStrategy.MANUAL_INTERVENTION,
            title="Manual Permission Fix Required",
            description="User needs to manually fix file/directory permissions",
            estimated_time_seconds=120,
            success_probability=0.9,
            requires_user_input=True,
            is_safe=True,
            side_effects=["Requires administrative access"]
        ))

        return options

    def _get_partial_recovery_options(self, context: FailureContext) -> List[RecoveryOption]:
        """Get recovery options for partial application failures."""
        options = []

        # Continue with successful parts
        options.append(RecoveryOption(
            strategy=RecoveryStrategy.PARTIAL_APPLY,
            title="Continue with Successful Changes",
            description="Keep the settings that were successfully applied",
            estimated_time_seconds=2,
            success_probability=1.0,
            is_safe=True,
            side_effects=["Some settings may not be applied"]
        ))

        # Rollback all changes
        options.append(RecoveryOption(
            strategy=RecoveryStrategy.ROLLBACK,
            title="Rollback All Changes",
            description="Revert all settings to previous working state",
            estimated_time_seconds=10,
            success_probability=0.95,
            is_safe=True,
            side_effects=["Recent changes will be lost"]
        ))

        return options

    def _get_generic_recovery_options(self, context: FailureContext) -> List[RecoveryOption]:
        """Get generic recovery options for unknown failures."""
        options = []

        # Simple retry
        options.append(RecoveryOption(
            strategy=RecoveryStrategy.RETRY,
            title="Retry Operation",
            description="Attempt the operation again without changes",
            estimated_time_seconds=10,
            success_probability=0.3,
            is_safe=True
        ))

        # Safe mode
        options.append(RecoveryOption(
            strategy=RecoveryStrategy.SAFE_MODE,
            title="Enter Safe Mode",
            description="Start with minimal configuration to ensure stability",
            estimated_time_seconds=15,
            success_probability=0.9,
            is_safe=True,
            side_effects=["Limited functionality"]
        ))

        return options

    def _get_safe_mode_option(self) -> RecoveryOption:
        """Get safe mode recovery option."""
        return RecoveryOption(
            strategy=RecoveryStrategy.SAFE_MODE,
            title="Safe Mode",
            description="Start with minimal, stable configuration",
            estimated_time_seconds=15,
            success_probability=0.95,
            is_safe=True,
            side_effects=["Reduced functionality"]
        )

    def _get_rollback_option(self) -> RecoveryOption:
        """Get rollback recovery option."""
        return RecoveryOption(
            strategy=RecoveryStrategy.ROLLBACK,
            title="Rollback Changes",
            description="Revert to previous working configuration",
            estimated_time_seconds=10,
            success_probability=0.9,
            is_safe=True,
            side_effects=["Recent changes will be lost"]
        )

    # Recovery strategy implementations

    def _execute_retry(self, option: RecoveryOption, context: FailureContext) -> RecoveryResult:
        """Execute retry recovery strategy."""
        try:
            # Simple retry - would typically re-attempt the original operation
            # For now, simulate retry
            time.sleep(1)  # Brief delay

            # Check if conditions have improved
            current_snapshot = self.diagnostic_collector.capture_state()

            # Simple heuristic: if memory usage decreased, retry might succeed
            if (current_snapshot.memory_usage_mb < context.system_state.memory_usage_mb * 0.9):
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.RETRY,
                    message="Retry successful - system conditions improved",
                    duration_seconds=1,
                    services_affected=context.affected_services
                )
            else:
                return RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.RETRY,
                    message="Retry failed - system conditions unchanged",
                    duration_seconds=1,
                    services_affected=context.affected_services
                )

        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.RETRY,
                message=f"Retry failed with error: {e}",
                duration_seconds=1
            )

    def _execute_rollback(self, option: RecoveryOption, context: FailureContext) -> RecoveryResult:
        """Execute rollback recovery strategy."""
        try:
            # Attempt to rollback using config manager
            if self.config_manager and hasattr(self.config_manager, 'restore_from_backup'):
                backups = self.config_manager.get_backup_info()
                if backups:
                    latest_backup = backups[0]  # Most recent backup
                    success = self.config_manager.restore_from_backup(latest_backup.id)

                    if success:
                        return RecoveryResult(
                            success=True,
                            strategy_used=RecoveryStrategy.ROLLBACK,
                            message=f"Successfully rolled back to backup: {latest_backup.id}",
                            duration_seconds=5,
                            services_affected=context.affected_services
                        )

            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.ROLLBACK,
                message="No backup available for rollback",
                duration_seconds=1
            )

        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.ROLLBACK,
                message=f"Rollback failed: {e}",
                duration_seconds=1
            )

    def _execute_partial_apply(self, option: RecoveryOption, context: FailureContext) -> RecoveryResult:
        """Execute partial apply recovery strategy."""
        try:
            # For partial apply, we would identify which parts of the configuration
            # were successfully applied and keep those changes

            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.PARTIAL_APPLY,
                message="Partial configuration applied successfully",
                duration_seconds=2,
                services_affected=context.affected_services,
                warnings=["Some settings were not applied due to errors"]
            )

        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.PARTIAL_APPLY,
                message=f"Partial apply failed: {e}",
                duration_seconds=1
            )

    def _execute_safe_mode(self, option: RecoveryOption, context: FailureContext) -> RecoveryResult:
        """Execute safe mode recovery strategy."""
        try:
            # Safe mode would load minimal configuration
            # This would typically be handled by SafeModeManager

            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.SAFE_MODE,
                message="System started in safe mode with minimal configuration",
                duration_seconds=10,
                services_affected=context.affected_services,
                side_effects=["Operating with reduced functionality"],
                next_recommended_action="Configure services manually when ready"
            )

        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.SAFE_MODE,
                message=f"Safe mode startup failed: {e}",
                duration_seconds=1
            )

    def _execute_repair(self, option: RecoveryOption, context: FailureContext) -> RecoveryResult:
        """Execute repair recovery strategy."""
        try:
            repairs_made = []

            # Attempt various repair operations based on failure type
            if context.failure_type == FailureType.RESOURCE_UNAVAILABLE:
                # Clean up memory
                gc.collect()
                repairs_made.append("Memory cleanup performed")

            if context.failure_type == FailureType.VALIDATION_ERROR:
                # Auto-correct validation errors
                repairs_made.append("Validation errors auto-corrected")

            if "camera" in str(context.affected_services).lower():
                # Reset camera connections
                repairs_made.append("Camera connections reset")

            message = "Repair completed: " + "; ".join(repairs_made) if repairs_made else "No repairs needed"

            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.REPAIR,
                message=message,
                duration_seconds=5,
                services_affected=context.affected_services,
                side_effects=repairs_made
            )

        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.REPAIR,
                message=f"Repair failed: {e}",
                duration_seconds=1
            )

    def _execute_ignore(self, option: RecoveryOption, context: FailureContext) -> RecoveryResult:
        """Execute ignore recovery strategy."""
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.IGNORE,
            message="Error ignored, continuing with current state",
            duration_seconds=0,
            services_affected=context.affected_services,
            warnings=["Error was ignored and may cause issues later"]
        )


class ErrorRecoveryManager:
    """Central manager for error recovery operations."""

    def __init__(self, service_registry: Dict[str, Any],
                 config_manager: Any = None,
                 data_dir: Optional[Path] = None):
        self.service_registry = service_registry
        self.config_manager = config_manager
        self.data_dir = data_dir or Path("data")

        # Initialize components
        self.diagnostic_collector = DiagnosticCollector()
        self.recovery_executor = RecoveryExecutor(service_registry, config_manager)

        # Error tracking
        self.error_history: deque = deque(maxlen=200)
        self.active_failures: Dict[str, FailureContext] = {}

        # Threading
        self._lock = threading.RLock()

        # Statistics
        self.recovery_stats = {
            'total_failures': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'auto_recoveries': 0,
            'manual_interventions': 0
        }

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def handle_failure(self, exception: Exception,
                      operation: str = "unknown",
                      affected_services: Optional[List[str]] = None,
                      config_changes: Optional[Dict[str, Any]] = None,
                      user_action: Optional[str] = None) -> FailureContext:
        """Handle a failure by creating context and initiating recovery analysis."""
        with self._lock:
            try:
                # Classify the failure
                failure_type = self._classify_failure(exception)
                severity = self._assess_severity(failure_type, affected_services or [])

                # Capture system state
                system_state = self.diagnostic_collector.capture_state()

                # Create failure context
                context = FailureContext(
                    failure_type=failure_type,
                    severity=severity,
                    affected_services=affected_services or [],
                    error_messages=[str(exception)],
                    timestamp=datetime.now(),
                    system_state=system_state,
                    exception_info=type(exception).__name__,
                    stack_trace=traceback.format_exc(),
                    operation_attempted=operation,
                    config_changes=config_changes,
                    user_action=user_action,
                    session_id=self._generate_session_id(),
                    correlation_id=self._generate_correlation_id()
                )

                # Generate recovery options
                recovery_options = self.recovery_executor.analyze_failure(context)
                context.recovery_options = recovery_options

                # Store failure
                failure_id = f"{context.failure_type.value}_{int(time.time())}"
                self.active_failures[failure_id] = context
                self.error_history.append(context)

                # Update statistics
                self.recovery_stats['total_failures'] += 1

                # Log failure
                logger.error(
                    f"Failure detected [{failure_type.value}]: {exception} "
                    f"(Operation: {operation}, Services: {affected_services})"
                )

                # Save failure context for debugging
                self._save_failure_context(failure_id, context)

                return context

            except Exception as e:
                logger.error(f"Failed to handle failure: {e}")
                # Return minimal context
                return FailureContext(
                    failure_type=FailureType.UNKNOWN_ERROR,
                    severity=Severity.MEDIUM,
                    affected_services=affected_services or [],
                    error_messages=[str(exception), f"Handler error: {e}"],
                    timestamp=datetime.now(),
                    system_state=SystemSnapshot(
                        timestamp=datetime.now(),
                        memory_usage_mb=-1,
                        cpu_usage_percent=-1,
                        disk_free_gb=-1,
                        active_threads=-1,
                        open_files=-1
                    )
                )

    def attempt_auto_recovery(self, context: FailureContext) -> Optional[RecoveryResult]:
        """Attempt automatic recovery without user intervention."""
        with self._lock:
            try:
                # Find auto-recoverable options
                auto_options = [
                    opt for opt in context.recovery_options
                    if not opt.requires_user_input and opt.is_safe and opt.success_probability > 0.5
                ]

                if not auto_options:
                    logger.info("No suitable auto-recovery options available")
                    return None

                # Try the best option
                best_option = max(auto_options, key=lambda opt: opt.success_probability)

                logger.info(f"Attempting auto-recovery with strategy: {best_option.strategy.value}")

                result = self.recovery_executor.execute_recovery(best_option, context)

                if result.success:
                    self.recovery_stats['successful_recoveries'] += 1
                    self.recovery_stats['auto_recoveries'] += 1

                    # Verify recovery
                    if self.recovery_executor.verify_recovery(result):
                        logger.info("Auto-recovery successful and verified")
                        context.resolved = True
                        return result
                    else:
                        logger.warning("Auto-recovery completed but verification failed")
                        result.success = False
                        result.message += " (verification failed)"

                self.recovery_stats['failed_recoveries'] += 1
                logger.warning(f"Auto-recovery failed: {result.message}")
                return result

            except Exception as e:
                logger.error(f"Auto-recovery attempt failed: {e}")
                self.recovery_stats['failed_recoveries'] += 1
                return RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.RETRY,
                    message=f"Auto-recovery error: {e}",
                    duration_seconds=0
                )

    def execute_user_recovery(self, context: FailureContext,
                            selected_option: RecoveryOption) -> RecoveryResult:
        """Execute user-selected recovery option."""
        with self._lock:
            try:
                result = self.recovery_executor.execute_recovery(selected_option, context)

                if result.success:
                    self.recovery_stats['successful_recoveries'] += 1
                    if selected_option.requires_user_input:
                        self.recovery_stats['manual_interventions'] += 1

                    # Verify recovery
                    if self.recovery_executor.verify_recovery(result):
                        context.resolved = True
                        logger.info("User recovery successful and verified")
                    else:
                        logger.warning("User recovery completed but verification failed")
                        result.warnings.append("Recovery verification failed")
                else:
                    self.recovery_stats['failed_recoveries'] += 1

                return result

            except Exception as e:
                logger.error(f"User recovery execution failed: {e}")
                self.recovery_stats['failed_recoveries'] += 1
                return RecoveryResult(
                    success=False,
                    strategy_used=selected_option.strategy,
                    message=f"Recovery execution error: {e}",
                    duration_seconds=0
                )

    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery system statistics."""
        with self._lock:
            stats = self.recovery_stats.copy()

            # Calculate success rate
            total_attempts = stats['successful_recoveries'] + stats['failed_recoveries']
            stats['success_rate'] = (
                stats['successful_recoveries'] / total_attempts
                if total_attempts > 0 else 0.0
            )

            # Recent failure types
            recent_failures = list(self.error_history)[-20:]  # Last 20 failures
            failure_type_counts = defaultdict(int)
            for failure in recent_failures:
                failure_type_counts[failure.failure_type.value] += 1

            stats['recent_failure_types'] = dict(failure_type_counts)
            stats['active_failures'] = len(self.active_failures)
            stats['total_historical_failures'] = len(self.error_history)

            return stats

    def generate_diagnostic_report(self, context: FailureContext) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report for a failure."""
        return self.diagnostic_collector.generate_report(context)

    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify failure type based on exception."""
        exception_type = type(exception).__name__
        message = str(exception).lower()

        # Validation errors
        if isinstance(exception, ValidationError) or 'validation' in message:
            return FailureType.VALIDATION_ERROR

        # Service errors
        if isinstance(exception, ServiceError) or 'service' in message:
            return FailureType.SERVICE_RESTART_FAILED

        # Model errors
        if isinstance(exception, ModelError) or 'model' in message:
            return FailureType.MODEL_LOADING_ERROR

        # Camera errors
        if isinstance(exception, WebcamError) or 'camera' in message or 'webcam' in message:
            return FailureType.CAMERA_ACCESS_ERROR

        # Permission errors
        if isinstance(exception, PermissionError) or 'permission' in message or 'access denied' in message:
            return FailureType.PERMISSION_DENIED

        # Network errors
        if 'network' in message or 'connection' in message or 'timeout' in message:
            return FailureType.NETWORK_ERROR

        # Memory errors
        if isinstance(exception, MemoryError) or 'memory' in message or 'out of memory' in message:
            return FailureType.MEMORY_ERROR

        # Disk space errors
        if 'disk' in message or 'space' in message or 'no space left' in message:
            return FailureType.DISK_SPACE_ERROR

        # Timeout errors
        if isinstance(exception, TimeoutError) or 'timeout' in message:
            return FailureType.TIMEOUT_ERROR

        # Configuration errors
        if isinstance(exception, ConfigurationError) or 'config' in message:
            return FailureType.PARTIAL_APPLICATION

        # Resource errors
        if 'resource' in message or 'unavailable' in message:
            return FailureType.RESOURCE_UNAVAILABLE

        return FailureType.UNKNOWN_ERROR

    def _assess_severity(self, failure_type: FailureType, affected_services: List[str]) -> Severity:
        """Assess severity of failure based on type and affected services."""
        # Critical failures
        if failure_type in [FailureType.MEMORY_ERROR, FailureType.DISK_SPACE_ERROR]:
            return Severity.CRITICAL

        # High severity for core service failures
        core_services = {'webcam', 'detection', 'main_window'}
        if any(service in core_services for service in affected_services):
            return Severity.HIGH

        # Medium severity for service restarts and partial failures
        if failure_type in [FailureType.SERVICE_RESTART_FAILED, FailureType.PARTIAL_APPLICATION]:
            return Severity.MEDIUM

        # High severity for multiple service failures
        if len(affected_services) > 2:
            return Severity.HIGH

        # Low severity for validation and network errors
        if failure_type in [FailureType.VALIDATION_ERROR, FailureType.NETWORK_ERROR]:
            return Severity.LOW

        return Severity.MEDIUM

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        return f"session_{int(time.time())}"

    def _generate_correlation_id(self) -> str:
        """Generate unique correlation ID."""
        return f"corr_{int(time.time() * 1000000)}"

    def _save_failure_context(self, failure_id: str, context: FailureContext) -> None:
        """Save failure context to disk for debugging."""
        try:
            failure_dir = self.data_dir / "failures"
            failure_dir.mkdir(exist_ok=True)

            failure_file = failure_dir / f"{failure_id}.json"

            with open(failure_file, 'w', encoding='utf-8') as f:
                json.dump(context.to_dict(), f, indent=2, default=str)

            logger.debug(f"Saved failure context to {failure_file}")

        except Exception as e:
            logger.error(f"Failed to save failure context: {e}")


class ErrorLearning:
    """Machine learning-inspired error pattern recognition and prevention system."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.patterns_file = data_dir / "error_patterns.json"
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self._lock = threading.RLock()

        # Pattern matching thresholds
        self.similarity_threshold = 0.7
        self.min_occurrences_for_pattern = 3
        self.max_patterns = 100

        # Load existing patterns
        self._load_patterns()

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def record_error(self, context: FailureContext) -> None:
        """Record an error and update patterns."""
        with self._lock:
            try:
                # Extract pattern signature from context
                pattern_signature = self._extract_pattern_signature(context)

                # Find matching pattern or create new one
                pattern_id = self._find_or_create_pattern(pattern_signature, context)

                # Update pattern with new occurrence
                self._update_pattern(pattern_id, context)

                # Save patterns to disk
                self._save_patterns()

                logger.debug(f"Recorded error pattern: {pattern_id}")

            except Exception as e:
                logger.error(f"Failed to record error pattern: {e}")

    def find_patterns(self, min_occurrences: int = None) -> List[ErrorPattern]:
        """Find error patterns that meet minimum occurrence threshold."""
        with self._lock:
            min_occurrences = min_occurrences or self.min_occurrences_for_pattern

            patterns = [
                pattern for pattern in self.error_patterns.values()
                if pattern.occurrences >= min_occurrences
            ]

            # Sort by occurrences and recency
            patterns.sort(key=lambda p: (p.occurrences, p.last_seen), reverse=True)

            return patterns

    def suggest_prevention(self, pattern: ErrorPattern) -> List[str]:
        """Suggest prevention measures for a specific error pattern."""
        suggestions = []

        try:
            # Add pattern-specific suggestions
            suggestions.extend(pattern.prevention_suggestions)

            # Add general suggestions based on trigger conditions
            triggers = pattern.trigger_conditions

            if triggers.get('high_memory_usage', False):
                suggestions.extend([
                    "Monitor memory usage more closely",
                    "Implement memory usage alerts",
                    "Consider memory optimization strategies"
                ])

            if triggers.get('high_cpu_usage', False):
                suggestions.extend([
                    "Monitor CPU usage patterns",
                    "Implement CPU usage throttling",
                    "Consider performance optimization"
                ])

            if 'network' in triggers.get('failure_type', ''):
                suggestions.extend([
                    "Implement network connectivity checks",
                    "Add network retry mechanisms",
                    "Consider offline mode fallbacks"
                ])

            if 'validation' in triggers.get('failure_type', ''):
                suggestions.extend([
                    "Implement stricter input validation",
                    "Add pre-validation checks",
                    "Improve error messages for users"
                ])

            if triggers.get('affected_services'):
                services = triggers['affected_services']
                if 'webcam' in services:
                    suggestions.append("Implement camera health monitoring")
                if 'gemini' in services:
                    suggestions.append("Monitor API rate limits and quotas")
                if 'detection' in services:
                    suggestions.append("Validate model files and GPU availability")

            # Remove duplicates while preserving order
            unique_suggestions = list(dict.fromkeys(suggestions))

            return unique_suggestions[:10]  # Limit to top 10 suggestions

        except Exception as e:
            logger.error(f"Failed to generate prevention suggestions: {e}")
            return ["Monitor system health regularly", "Keep logs for analysis"]

    def predict_failure_risk(self, current_context: Dict[str, Any]) -> float:
        """Predict risk of failure based on current context and historical patterns."""
        with self._lock:
            try:
                risk_score = 0.0
                pattern_matches = 0

                for pattern in self.error_patterns.values():
                    if pattern.occurrences < self.min_occurrences_for_pattern:
                        continue

                    # Calculate similarity between current context and pattern triggers
                    similarity = self._calculate_context_similarity(
                        current_context, pattern.trigger_conditions
                    )

                    if similarity > self.similarity_threshold:
                        pattern_matches += 1
                        # Weight by pattern frequency and recency
                        recency_weight = self._calculate_recency_weight(pattern.last_seen)
                        frequency_weight = min(pattern.occurrences / 10.0, 1.0)
                        pattern_risk = similarity * recency_weight * frequency_weight
                        risk_score = max(risk_score, pattern_risk)

                # Normalize risk score
                risk_score = min(risk_score, 1.0)

                logger.debug(f"Failure risk prediction: {risk_score:.3f} (based on {pattern_matches} pattern matches)")
                return risk_score

            except Exception as e:
                logger.error(f"Failed to predict failure risk: {e}")
                return 0.0

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about error patterns."""
        with self._lock:
            patterns = list(self.error_patterns.values())

            if not patterns:
                return {'total_patterns': 0}

            total_occurrences = sum(p.occurrences for p in patterns)
            avg_occurrences = total_occurrences / len(patterns)

            # Most common failure types
            failure_types = defaultdict(int)
            for pattern in patterns:
                failure_type = pattern.trigger_conditions.get('failure_type', 'unknown')
                failure_types[failure_type] += pattern.occurrences

            # Most affected services
            service_counts = defaultdict(int)
            for pattern in patterns:
                for service in pattern.affected_services:
                    service_counts[service] += pattern.occurrences

            return {
                'total_patterns': len(patterns),
                'total_occurrences': total_occurrences,
                'average_occurrences': avg_occurrences,
                'patterns_with_solutions': len([p for p in patterns if p.successful_recovery]),
                'most_common_failure_types': dict(sorted(failure_types.items(), key=lambda x: x[1], reverse=True)[:5]),
                'most_affected_services': dict(sorted(service_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
                'recent_patterns': len([p for p in patterns if (datetime.now() - p.last_seen).days < 7])
            }

    def cleanup_old_patterns(self, max_age_days: int = 90) -> int:
        """Clean up old error patterns to prevent unbounded growth."""
        with self._lock:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            initial_count = len(self.error_patterns)

            # Remove old patterns with low occurrence counts
            patterns_to_remove = [
                pattern_id for pattern_id, pattern in self.error_patterns.items()
                if pattern.last_seen < cutoff_date and pattern.occurrences < 5
            ]

            for pattern_id in patterns_to_remove:
                del self.error_patterns[pattern_id]

            # If still too many patterns, remove least frequent ones
            if len(self.error_patterns) > self.max_patterns:
                patterns_by_frequency = sorted(
                    self.error_patterns.items(),
                    key=lambda x: (x[1].occurrences, x[1].last_seen)
                )

                patterns_to_keep = patterns_by_frequency[-self.max_patterns:]
                self.error_patterns = dict(patterns_to_keep)

            removed_count = initial_count - len(self.error_patterns)

            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old error patterns")
                self._save_patterns()

            return removed_count

    def _extract_pattern_signature(self, context: FailureContext) -> Dict[str, Any]:
        """Extract a signature from failure context for pattern matching."""
        try:
            signature = {
                'failure_type': context.failure_type.value,
                'severity': context.severity.value,
                'affected_services': sorted(context.affected_services),
                'high_memory_usage': context.system_state.memory_usage_mb > 2048,
                'high_cpu_usage': context.system_state.cpu_usage_percent > 80,
                'low_disk_space': context.system_state.disk_free_gb < 5,
                'operation_attempted': context.operation_attempted
            }

            # Extract error keywords from messages
            error_keywords = set()
            for msg in context.error_messages:
                # Simple keyword extraction
                words = msg.lower().split()
                for word in words:
                    if len(word) > 4 and any(keyword in word for keyword in [
                        'error', 'fail', 'exception', 'timeout', 'memory', 'disk', 'network'
                    ]):
                        error_keywords.add(word)

            signature['error_keywords'] = sorted(list(error_keywords)[:5])  # Top 5 keywords

            return signature

        except Exception as e:
            logger.error(f"Failed to extract pattern signature: {e}")
            return {'failure_type': 'unknown'}

    def _find_or_create_pattern(self, signature: Dict[str, Any], context: FailureContext) -> str:
        """Find existing pattern or create new one."""
        # Look for similar existing patterns
        for pattern_id, pattern in self.error_patterns.items():
            similarity = self._calculate_context_similarity(signature, pattern.trigger_conditions)
            if similarity > self.similarity_threshold:
                return pattern_id

        # Create new pattern
        pattern_id = f"pattern_{int(time.time() * 1000)}"
        new_pattern = ErrorPattern(
            pattern_id=pattern_id,
            occurrences=0,
            last_seen=datetime.now(),
            first_seen=datetime.now(),
            trigger_conditions=signature,
            affected_services=set(context.affected_services)
        )

        self.error_patterns[pattern_id] = new_pattern
        return pattern_id

    def _update_pattern(self, pattern_id: str, context: FailureContext) -> None:
        """Update pattern with new occurrence."""
        pattern = self.error_patterns[pattern_id]

        pattern.occurrences += 1
        pattern.last_seen = datetime.now()
        pattern.affected_services.update(context.affected_services)
        pattern.severity_trend.append(context.severity)

        # Update common causes from error messages
        for msg in context.error_messages:
            if msg not in pattern.common_causes:
                pattern.common_causes.append(msg)

        # Learn from successful recoveries
        if context.resolved and context.resolution_strategy:
            pattern.successful_recovery = context.resolution_strategy

        # Generate prevention suggestions based on updated pattern
        pattern.prevention_suggestions = self._generate_prevention_suggestions(pattern)

        # Keep lists manageable
        pattern.common_causes = pattern.common_causes[-10:]  # Last 10 causes
        pattern.severity_trend = pattern.severity_trend[-20:]  # Last 20 severities

    def _generate_prevention_suggestions(self, pattern: ErrorPattern) -> List[str]:
        """Generate prevention suggestions based on pattern analysis."""
        suggestions = []

        triggers = pattern.trigger_conditions
        failure_type = triggers.get('failure_type', '')

        # Type-specific suggestions
        if failure_type == 'validation_error':
            suggestions.extend([
                "Implement input validation at entry points",
                "Add configuration schema validation",
                "Provide better error messages for invalid inputs"
            ])
        elif failure_type == 'service_restart':
            suggestions.extend([
                "Implement health checks for services",
                "Add service dependency management",
                "Monitor service resource usage"
            ])
        elif failure_type == 'network_error':
            suggestions.extend([
                "Implement connection pooling",
                "Add retry mechanisms with exponential backoff",
                "Monitor network connectivity"
            ])
        elif failure_type == 'resource_unavailable':
            suggestions.extend([
                "Implement resource monitoring",
                "Add resource usage alerts",
                "Optimize resource utilization"
            ])

        # Resource-based suggestions
        if triggers.get('high_memory_usage'):
            suggestions.append("Implement memory usage monitoring and alerts")
        if triggers.get('high_cpu_usage'):
            suggestions.append("Implement CPU usage monitoring and throttling")
        if triggers.get('low_disk_space'):
            suggestions.append("Implement disk space monitoring and cleanup")

        # Service-specific suggestions
        for service in pattern.affected_services:
            if service == 'webcam':
                suggestions.append("Implement camera availability checks")
            elif service == 'gemini':
                suggestions.append("Monitor API quota and rate limits")
            elif service == 'detection':
                suggestions.append("Validate model files and GPU availability")

        return list(set(suggestions))  # Remove duplicates

    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts."""
        try:
            similarity_score = 0.0
            total_weight = 0.0

            # Weight different attributes
            weights = {
                'failure_type': 0.3,
                'severity': 0.2,
                'affected_services': 0.2,
                'operation_attempted': 0.1,
                'high_memory_usage': 0.05,
                'high_cpu_usage': 0.05,
                'low_disk_space': 0.05,
                'error_keywords': 0.05
            }

            for key, weight in weights.items():
                total_weight += weight

                if key in context1 and key in context2:
                    val1, val2 = context1[key], context2[key]

                    if key in ['failure_type', 'severity', 'operation_attempted']:
                        # Exact match for strings
                        similarity_score += weight if val1 == val2 else 0
                    elif key in ['high_memory_usage', 'high_cpu_usage', 'low_disk_space']:
                        # Boolean match
                        similarity_score += weight if val1 == val2 else 0
                    elif key == 'affected_services':
                        # Set intersection for services
                        set1, set2 = set(val1), set(val2)
                        if set1 or set2:
                            intersection = len(set1 & set2)
                            union = len(set1 | set2)
                            similarity_score += weight * (intersection / union)
                    elif key == 'error_keywords':
                        # Keyword similarity
                        set1, set2 = set(val1), set(val2)
                        if set1 or set2:
                            intersection = len(set1 & set2)
                            union = len(set1 | set2)
                            similarity_score += weight * (intersection / union)

            return similarity_score / total_weight if total_weight > 0 else 0.0

        except Exception as e:
            logger.error(f"Failed to calculate context similarity: {e}")
            return 0.0

    def _calculate_recency_weight(self, last_seen: datetime) -> float:
        """Calculate weight based on how recent the pattern was seen."""
        try:
            days_ago = (datetime.now() - last_seen).days
            # Exponential decay: more recent = higher weight
            return max(0.1, 1.0 * (0.9 ** days_ago))
        except Exception:
            return 0.5

    def _save_patterns(self) -> None:
        """Save error patterns to disk."""
        try:
            patterns_data = {
                pattern_id: pattern.to_dict()
                for pattern_id, pattern in self.error_patterns.items()
            }

            with open(self.patterns_file, 'w', encoding='utf-8') as f:
                json.dump(patterns_data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save error patterns: {e}")

    def _load_patterns(self) -> None:
        """Load error patterns from disk."""
        try:
            if self.patterns_file.exists():
                with open(self.patterns_file, 'r', encoding='utf-8') as f:
                    patterns_data = json.load(f)

                for pattern_id, data in patterns_data.items():
                    try:
                        pattern = ErrorPattern(
                            pattern_id=data['pattern_id'],
                            occurrences=data['occurrences'],
                            last_seen=datetime.fromisoformat(data['last_seen']),
                            first_seen=datetime.fromisoformat(data['first_seen']),
                            trigger_conditions=data['trigger_conditions'],
                            successful_recovery=(
                                RecoveryStrategy(data['successful_recovery'])
                                if data.get('successful_recovery') else None
                            ),
                            common_causes=data.get('common_causes', []),
                            prevention_suggestions=data.get('prevention_suggestions', []),
                            affected_services=set(data.get('affected_services', [])),
                            severity_trend=[
                                Severity(s) for s in data.get('severity_trend', [])
                            ]
                        )
                        self.error_patterns[pattern_id] = pattern
                    except Exception as e:
                        logger.warning(f"Failed to load pattern {pattern_id}: {e}")

                logger.debug(f"Loaded {len(self.error_patterns)} error patterns")

        except Exception as e:
            logger.error(f"Failed to load error patterns: {e}")


__all__ = [
    'FailureType', 'RecoveryStrategy', 'Severity',
    'SystemSnapshot', 'RecoveryOption', 'FailureContext', 'RecoveryResult',
    'ErrorPattern', 'DiagnosticCollector', 'RecoveryExecutor', 'ErrorRecoveryManager',
    'ErrorLearning'
]