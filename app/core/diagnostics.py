"""Advanced Diagnostics System for comprehensive system analysis and troubleshooting.

This module provides detailed diagnostic capabilities including:
- System health monitoring and analysis
- Performance bottleneck detection
- Resource usage tracking
- Service dependency analysis
- Configuration validation
- Predictive issue detection
- Automated troubleshooting suggestions
"""
from __future__ import annotations

import logging
import time
import psutil
import threading
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class DiagnosticCategory(Enum):
    """Categories of diagnostic checks."""
    SYSTEM_HEALTH = "system_health"
    PERFORMANCE = "performance"
    CONFIGURATION = "configuration"
    SERVICES = "services"
    RESOURCES = "resources"
    SECURITY = "security"
    NETWORK = "network"
    STORAGE = "storage"


class DiagnosticSeverity(Enum):
    """Severity levels for diagnostic findings."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(slots=True)
class DiagnosticFinding:
    """Represents a single diagnostic finding."""
    category: DiagnosticCategory
    severity: DiagnosticSeverity
    title: str
    description: str
    timestamp: datetime

    # Additional details
    affected_components: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    # Resolution tracking
    auto_fixable: bool = False
    fix_command: Optional[str] = None
    resolved: bool = False
    resolution_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'category': self.category.value,
            'severity': self.severity.value,
            'title': self.title,
            'description': self.description,
            'timestamp': self.timestamp.isoformat(),
            'affected_components': self.affected_components,
            'metrics': self.metrics,
            'recommendations': self.recommendations,
            'auto_fixable': self.auto_fixable,
            'fix_command': self.fix_command,
            'resolved': self.resolved,
            'resolution_notes': self.resolution_notes
        }


@dataclass(slots=True)
class DiagnosticReport:
    """Comprehensive diagnostic report."""
    report_id: str
    generated_at: datetime
    system_overview: Dict[str, Any]
    findings: List[DiagnosticFinding]

    # Summary statistics
    total_findings: int = 0
    critical_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0

    # Performance metrics
    overall_health_score: float = 0.0  # 0-100
    performance_score: float = 0.0     # 0-100
    stability_score: float = 0.0       # 0-100

    # Recommendations
    priority_actions: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)
    preventive_measures: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate summary statistics after initialization."""
        self.total_findings = len(self.findings)

        for finding in self.findings:
            if finding.severity == DiagnosticSeverity.CRITICAL:
                self.critical_count += 1
            elif finding.severity == DiagnosticSeverity.ERROR:
                self.error_count += 1
            elif finding.severity == DiagnosticSeverity.WARNING:
                self.warning_count += 1
            elif finding.severity == DiagnosticSeverity.INFO:
                self.info_count += 1

        # Calculate health scores
        self._calculate_health_scores()

    def _calculate_health_scores(self):
        """Calculate overall health scores based on findings."""
        # Base score starts at 100
        base_score = 100.0

        # Deduct points for issues
        penalty = (
            self.critical_count * 25 +  # Critical issues: -25 points each
            self.error_count * 10 +     # Errors: -10 points each
            self.warning_count * 5 +    # Warnings: -5 points each
            self.info_count * 1         # Info: -1 point each
        )

        self.overall_health_score = max(0.0, base_score - penalty)

        # Performance score based on system metrics
        system_metrics = self.system_overview.get('performance_metrics', {})
        memory_usage = system_metrics.get('memory_percent', 0)
        cpu_usage = system_metrics.get('cpu_percent', 0)
        disk_usage = system_metrics.get('disk_usage_percent', 0)

        # Performance penalties
        perf_penalty = 0
        if memory_usage > 80: perf_penalty += 20
        elif memory_usage > 60: perf_penalty += 10

        if cpu_usage > 80: perf_penalty += 20
        elif cpu_usage > 60: perf_penalty += 10

        if disk_usage > 90: perf_penalty += 15
        elif disk_usage > 80: perf_penalty += 10

        self.performance_score = max(0.0, 100.0 - perf_penalty)

        # Stability score based on error patterns
        stability_penalty = self.critical_count * 30 + self.error_count * 15
        self.stability_score = max(0.0, 100.0 - stability_penalty)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'report_id': self.report_id,
            'generated_at': self.generated_at.isoformat(),
            'system_overview': self.system_overview,
            'findings': [f.to_dict() for f in self.findings],
            'summary': {
                'total_findings': self.total_findings,
                'critical_count': self.critical_count,
                'error_count': self.error_count,
                'warning_count': self.warning_count,
                'info_count': self.info_count
            },
            'scores': {
                'overall_health_score': self.overall_health_score,
                'performance_score': self.performance_score,
                'stability_score': self.stability_score
            },
            'recommendations': {
                'priority_actions': self.priority_actions,
                'optimization_suggestions': self.optimization_suggestions,
                'preventive_measures': self.preventive_measures
            }
        }


class SystemHealthMonitor:
    """Monitors real-time system health metrics."""

    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.metrics_history: deque = deque(maxlen=300)  # 5 minutes at 1s intervals
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

    def start_monitoring(self):
        """Start continuous system monitoring."""
        with self._lock:
            if self._monitoring:
                return

            self._monitoring = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="SystemHealthMonitor"
            )
            self._monitor_thread.start()
            logger.info("System health monitoring started")

    def stop_monitoring(self):
        """Stop system monitoring."""
        with self._lock:
            self._monitoring = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=2.0)
            logger.info("System health monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                metrics = self._collect_metrics()
                with self._lock:
                    self.metrics_history.append(metrics)

                time.sleep(self.sampling_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.sampling_interval)

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)

            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()

            # Network metrics
            network_io = psutil.net_io_counters()

            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()

            return {
                'timestamp': datetime.now(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'load_avg': load_avg
                },
                'memory': {
                    'percent': memory.percent,
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3)
                },
                'swap': {
                    'percent': swap.percent,
                    'total_gb': swap.total / (1024**3),
                    'used_gb': swap.used / (1024**3)
                },
                'disk': {
                    'percent': (disk.used / disk.total) * 100,
                    'total_gb': disk.total / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'io_read_mb': disk_io.read_bytes / (1024**2) if disk_io else 0,
                    'io_write_mb': disk_io.write_bytes / (1024**2) if disk_io else 0
                },
                'network': {
                    'bytes_sent_mb': network_io.bytes_sent / (1024**2),
                    'bytes_recv_mb': network_io.bytes_recv / (1024**2),
                    'packets_sent': network_io.packets_sent,
                    'packets_recv': network_io.packets_recv,
                    'errors_in': network_io.errin,
                    'errors_out': network_io.errout
                },
                'process': {
                    'memory_mb': process_memory.rss / (1024**2),
                    'cpu_percent': process.cpu_percent(),
                    'num_threads': process.num_threads()
                }
            }

        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return {'timestamp': datetime.now(), 'error': str(e)}

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        return self._collect_metrics()

    def get_metrics_summary(self, minutes: int = 5) -> Dict[str, Any]:
        """Get summary of metrics over specified time period."""
        with self._lock:
            if not self.metrics_history:
                return {}

            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            recent_metrics = [
                m for m in self.metrics_history
                if m.get('timestamp', datetime.min) >= cutoff_time
            ]

            if not recent_metrics:
                return {}

            # Calculate averages
            summary = {
                'period_minutes': minutes,
                'sample_count': len(recent_metrics),
                'cpu_avg': sum(m.get('cpu', {}).get('percent', 0) for m in recent_metrics) / len(recent_metrics),
                'memory_avg': sum(m.get('memory', {}).get('percent', 0) for m in recent_metrics) / len(recent_metrics),
                'disk_usage_avg': sum(m.get('disk', {}).get('percent', 0) for m in recent_metrics) / len(recent_metrics),
                'process_memory_avg': sum(m.get('process', {}).get('memory_mb', 0) for m in recent_metrics) / len(recent_metrics)
            }

            # Calculate peaks
            summary.update({
                'cpu_peak': max(m.get('cpu', {}).get('percent', 0) for m in recent_metrics),
                'memory_peak': max(m.get('memory', {}).get('percent', 0) for m in recent_metrics),
                'process_memory_peak': max(m.get('process', {}).get('memory_mb', 0) for m in recent_metrics)
            })

            return summary


class AdvancedDiagnosticEngine:
    """Advanced diagnostic engine with comprehensive system analysis."""

    def __init__(self, service_registry: Dict[str, Any] = None):
        self.service_registry = service_registry or {}
        self.health_monitor = SystemHealthMonitor()
        self.findings_cache: Dict[str, DiagnosticFinding] = {}
        self._lock = threading.RLock()

        # Diagnostic rules and thresholds
        self.thresholds = {
            'cpu_warning': 70,
            'cpu_critical': 90,
            'memory_warning': 80,
            'memory_critical': 95,
            'disk_warning': 85,
            'disk_critical': 95,
            'load_warning': 2.0,
            'load_critical': 5.0
        }

    def run_full_diagnostic(self) -> DiagnosticReport:
        """Run comprehensive diagnostic analysis."""
        start_time = time.time()
        report_id = f"diag_full_{int(time.time())}"

        logger.info("Starting full diagnostic analysis")

        # Collect findings from all categories
        findings = []
        findings.extend(self._check_system_health())
        findings.extend(self._check_performance())
        findings.extend(self._check_resources())
        findings.extend(self._check_services())
        findings.extend(self._check_configuration())
        findings.extend(self._check_storage())
        findings.extend(self._check_network())

        # System overview
        system_overview = self._build_system_overview()

        # Create report
        report = DiagnosticReport(
            report_id=report_id,
            generated_at=datetime.now(),
            system_overview=system_overview,
            findings=findings
        )

        # Generate recommendations
        self._generate_recommendations(report)

        duration = time.time() - start_time
        logger.info(f"Full diagnostic completed in {duration:.2f}s - {len(findings)} findings")

        return report

    def run_quick_diagnostic(self) -> DiagnosticReport:
        """Run quick diagnostic for immediate issues."""
        start_time = time.time()
        report_id = f"diag_quick_{int(time.time())}"

        logger.info("Starting quick diagnostic analysis")

        # Quick checks only
        findings = []
        findings.extend(self._check_critical_resources())
        findings.extend(self._check_service_availability())
        findings.extend(self._check_immediate_threats())

        # Minimal system overview
        system_overview = {
            'diagnostic_type': 'quick',
            'current_metrics': self.health_monitor.get_current_metrics(),
            'timestamp': datetime.now().isoformat()
        }

        # Create report
        report = DiagnosticReport(
            report_id=report_id,
            generated_at=datetime.now(),
            system_overview=system_overview,
            findings=findings
        )

        duration = time.time() - start_time
        logger.info(f"Quick diagnostic completed in {duration:.2f}s - {len(findings)} findings")

        return report

    def _check_system_health(self) -> List[DiagnosticFinding]:
        """Check overall system health."""
        findings = []

        try:
            metrics = self.health_monitor.get_current_metrics()

            # CPU health
            cpu_percent = metrics.get('cpu', {}).get('percent', 0)
            if cpu_percent >= self.thresholds['cpu_critical']:
                findings.append(DiagnosticFinding(
                    category=DiagnosticCategory.SYSTEM_HEALTH,
                    severity=DiagnosticSeverity.CRITICAL,
                    title="Critical CPU Usage",
                    description=f"CPU usage is critically high at {cpu_percent:.1f}%",
                    timestamp=datetime.now(),
                    metrics={'cpu_percent': cpu_percent},
                    recommendations=[
                        "Close unnecessary applications",
                        "Check for runaway processes",
                        "Consider system restart if problem persists"
                    ]
                ))
            elif cpu_percent >= self.thresholds['cpu_warning']:
                findings.append(DiagnosticFinding(
                    category=DiagnosticCategory.SYSTEM_HEALTH,
                    severity=DiagnosticSeverity.WARNING,
                    title="High CPU Usage",
                    description=f"CPU usage is high at {cpu_percent:.1f}%",
                    timestamp=datetime.now(),
                    metrics={'cpu_percent': cpu_percent},
                    recommendations=[
                        "Monitor CPU usage trends",
                        "Consider closing non-essential applications"
                    ]
                ))

            # Memory health
            memory_percent = metrics.get('memory', {}).get('percent', 0)
            if memory_percent >= self.thresholds['memory_critical']:
                findings.append(DiagnosticFinding(
                    category=DiagnosticCategory.SYSTEM_HEALTH,
                    severity=DiagnosticSeverity.CRITICAL,
                    title="Critical Memory Usage",
                    description=f"Memory usage is critically high at {memory_percent:.1f}%",
                    timestamp=datetime.now(),
                    metrics={'memory_percent': memory_percent},
                    recommendations=[
                        "Close memory-intensive applications immediately",
                        "Restart the application to free memory",
                        "Consider adding more RAM if problem persists"
                    ]
                ))
            elif memory_percent >= self.thresholds['memory_warning']:
                findings.append(DiagnosticFinding(
                    category=DiagnosticCategory.SYSTEM_HEALTH,
                    severity=DiagnosticSeverity.WARNING,
                    title="High Memory Usage",
                    description=f"Memory usage is high at {memory_percent:.1f}%",
                    timestamp=datetime.now(),
                    metrics={'memory_percent': memory_percent},
                    recommendations=[
                        "Monitor memory usage patterns",
                        "Close unnecessary browser tabs and applications"
                    ]
                ))

            # Load average (Unix-like systems)
            load_avg = metrics.get('cpu', {}).get('load_avg', (0, 0, 0))
            if load_avg[0] >= self.thresholds['load_critical']:
                findings.append(DiagnosticFinding(
                    category=DiagnosticCategory.SYSTEM_HEALTH,
                    severity=DiagnosticSeverity.ERROR,
                    title="High System Load",
                    description=f"System load average is high: {load_avg[0]:.2f}",
                    timestamp=datetime.now(),
                    metrics={'load_avg_1min': load_avg[0]},
                    recommendations=[
                        "Check for CPU-intensive processes",
                        "Consider load balancing if running servers"
                    ]
                ))

        except Exception as e:
            logger.error(f"System health check failed: {e}")
            findings.append(DiagnosticFinding(
                category=DiagnosticCategory.SYSTEM_HEALTH,
                severity=DiagnosticSeverity.ERROR,
                title="Health Check Failed",
                description=f"Unable to assess system health: {e}",
                timestamp=datetime.now(),
                recommendations=["Check system monitoring capabilities"]
            ))

        return findings

    def _check_performance(self) -> List[DiagnosticFinding]:
        """Check system performance metrics."""
        findings = []

        try:
            # Get recent performance summary
            summary = self.health_monitor.get_metrics_summary(minutes=5)

            if not summary:
                return findings

            # Sustained high CPU usage
            if summary.get('cpu_avg', 0) > 80:
                findings.append(DiagnosticFinding(
                    category=DiagnosticCategory.PERFORMANCE,
                    severity=DiagnosticSeverity.WARNING,
                    title="Sustained High CPU Usage",
                    description=f"Average CPU usage over 5 minutes: {summary['cpu_avg']:.1f}%",
                    timestamp=datetime.now(),
                    metrics=summary,
                    recommendations=[
                        "Identify CPU-intensive processes",
                        "Consider performance optimization",
                        "Check for background tasks"
                    ]
                ))

            # Memory pressure
            if summary.get('memory_avg', 0) > 85:
                findings.append(DiagnosticFinding(
                    category=DiagnosticCategory.PERFORMANCE,
                    severity=DiagnosticSeverity.WARNING,
                    title="Memory Pressure",
                    description=f"Average memory usage over 5 minutes: {summary['memory_avg']:.1f}%",
                    timestamp=datetime.now(),
                    metrics=summary,
                    recommendations=[
                        "Check for memory leaks",
                        "Consider increasing available memory",
                        "Review memory-intensive operations"
                    ]
                ))

            # Process memory growth
            process_memory_avg = summary.get('process_memory_avg', 0)
            process_memory_peak = summary.get('process_memory_peak', 0)

            if process_memory_peak > process_memory_avg * 1.5:
                findings.append(DiagnosticFinding(
                    category=DiagnosticCategory.PERFORMANCE,
                    severity=DiagnosticSeverity.INFO,
                    title="Variable Memory Usage",
                    description=f"Process memory varies significantly (avg: {process_memory_avg:.1f}MB, peak: {process_memory_peak:.1f}MB)",
                    timestamp=datetime.now(),
                    metrics={'memory_variance': process_memory_peak - process_memory_avg},
                    recommendations=[
                        "Monitor for memory leaks",
                        "Check for periodic memory spikes"
                    ]
                ))

        except Exception as e:
            logger.error(f"Performance check failed: {e}")

        return findings

    def _check_resources(self) -> List[DiagnosticFinding]:
        """Check system resource availability."""
        findings = []

        try:
            # Disk space
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100

            if disk_percent >= self.thresholds['disk_critical']:
                findings.append(DiagnosticFinding(
                    category=DiagnosticCategory.RESOURCES,
                    severity=DiagnosticSeverity.CRITICAL,
                    title="Critical Disk Space",
                    description=f"Disk usage is critically high at {disk_percent:.1f}%",
                    timestamp=datetime.now(),
                    metrics={'disk_percent': disk_percent, 'free_gb': disk.free / (1024**3)},
                    recommendations=[
                        "Free up disk space immediately",
                        "Delete temporary files and logs",
                        "Move large files to external storage"
                    ]
                ))
            elif disk_percent >= self.thresholds['disk_warning']:
                findings.append(DiagnosticFinding(
                    category=DiagnosticCategory.RESOURCES,
                    severity=DiagnosticSeverity.WARNING,
                    title="Low Disk Space",
                    description=f"Disk usage is high at {disk_percent:.1f}%",
                    timestamp=datetime.now(),
                    metrics={'disk_percent': disk_percent, 'free_gb': disk.free / (1024**3)},
                    recommendations=[
                        "Clean up unnecessary files",
                        "Monitor disk usage trends"
                    ]
                ))

            # Check for GPU availability if needed
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    for i in range(gpu_count):
                        memory_total = torch.cuda.get_device_properties(i).total_memory
                        memory_allocated = torch.cuda.memory_allocated(i)
                        memory_percent = (memory_allocated / memory_total) * 100

                        if memory_percent > 90:
                            findings.append(DiagnosticFinding(
                                category=DiagnosticCategory.RESOURCES,
                                severity=DiagnosticSeverity.WARNING,
                                title=f"High GPU Memory Usage (GPU {i})",
                                description=f"GPU {i} memory usage: {memory_percent:.1f}%",
                                timestamp=datetime.now(),
                                metrics={'gpu_memory_percent': memory_percent},
                                recommendations=[
                                    "Clear GPU memory cache",
                                    "Reduce batch sizes if applicable"
                                ]
                            ))
            except ImportError:
                pass  # GPU monitoring not available

        except Exception as e:
            logger.error(f"Resource check failed: {e}")

        return findings

    def _check_services(self) -> List[DiagnosticFinding]:
        """Check service health and availability."""
        findings = []

        try:
            for service_name, service in self.service_registry.items():
                if hasattr(service, 'is_healthy'):
                    try:
                        if not service.is_healthy():
                            findings.append(DiagnosticFinding(
                                category=DiagnosticCategory.SERVICES,
                                severity=DiagnosticSeverity.ERROR,
                                title=f"Service Health Issue: {service_name}",
                                description=f"Service {service_name} is not healthy",
                                timestamp=datetime.now(),
                                affected_components=[service_name],
                                recommendations=[
                                    f"Restart {service_name} service",
                                    "Check service configuration",
                                    "Review service logs for errors"
                                ]
                            ))
                    except Exception as e:
                        findings.append(DiagnosticFinding(
                            category=DiagnosticCategory.SERVICES,
                            severity=DiagnosticSeverity.WARNING,
                            title=f"Service Check Failed: {service_name}",
                            description=f"Unable to check health of {service_name}: {e}",
                            timestamp=datetime.now(),
                            affected_components=[service_name],
                            recommendations=[
                                "Verify service implementation",
                                "Check service interface compatibility"
                            ]
                        ))

                # Check if service is running (basic check)
                if hasattr(service, 'is_running'):
                    try:
                        if not service.is_running():
                            findings.append(DiagnosticFinding(
                                category=DiagnosticCategory.SERVICES,
                                severity=DiagnosticSeverity.WARNING,
                                title=f"Service Not Running: {service_name}",
                                description=f"Service {service_name} is not currently running",
                                timestamp=datetime.now(),
                                affected_components=[service_name],
                                recommendations=[
                                    f"Start {service_name} service",
                                    "Check service startup dependencies"
                                ]
                            ))
                    except Exception as e:
                        logger.debug(f"Could not check running status for {service_name}: {e}")

        except Exception as e:
            logger.error(f"Service check failed: {e}")

        return findings

    def _check_configuration(self) -> List[DiagnosticFinding]:
        """Check configuration validity and consistency."""
        findings = []

        try:
            # This would integrate with the configuration validation system
            # For now, provide basic configuration checks

            # Check for common configuration issues
            config_files = [
                Path('config.json'),
                Path('app/config/settings.py'),
                Path('.env')
            ]

            for config_file in config_files:
                if config_file.exists():
                    try:
                        if config_file.suffix == '.json':
                            with open(config_file, 'r') as f:
                                json.load(f)  # Validate JSON
                    except json.JSONDecodeError as e:
                        findings.append(DiagnosticFinding(
                            category=DiagnosticCategory.CONFIGURATION,
                            severity=DiagnosticSeverity.ERROR,
                            title=f"Invalid Configuration File: {config_file.name}",
                            description=f"JSON syntax error in {config_file}: {e}",
                            timestamp=datetime.now(),
                            recommendations=[
                                "Fix JSON syntax errors",
                                "Restore from backup if available",
                                "Validate configuration format"
                            ]
                        ))
                    except Exception as e:
                        findings.append(DiagnosticFinding(
                            category=DiagnosticCategory.CONFIGURATION,
                            severity=DiagnosticSeverity.WARNING,
                            title=f"Configuration File Issue: {config_file.name}",
                            description=f"Could not validate {config_file}: {e}",
                            timestamp=datetime.now(),
                            recommendations=[
                                "Check file permissions",
                                "Verify file format"
                            ]
                        ))

        except Exception as e:
            logger.error(f"Configuration check failed: {e}")

        return findings

    def _check_storage(self) -> List[DiagnosticFinding]:
        """Check storage health and accessibility."""
        findings = []

        try:
            # Check important directories
            important_dirs = [
                Path('data'),
                Path('logs'),
                Path('config_backups'),
                Path('models')
            ]

            for directory in important_dirs:
                if not directory.exists():
                    findings.append(DiagnosticFinding(
                        category=DiagnosticCategory.STORAGE,
                        severity=DiagnosticSeverity.WARNING,
                        title=f"Missing Directory: {directory.name}",
                        description=f"Important directory {directory} does not exist",
                        timestamp=datetime.now(),
                        recommendations=[
                            f"Create {directory} directory",
                            "Check application setup"
                        ],
                        auto_fixable=True,
                        fix_command=f"mkdir -p {directory}"
                    ))
                else:
                    # Check write permissions
                    try:
                        test_file = directory / '.write_test'
                        test_file.touch()
                        test_file.unlink()
                    except Exception as e:
                        findings.append(DiagnosticFinding(
                            category=DiagnosticCategory.STORAGE,
                            severity=DiagnosticSeverity.ERROR,
                            title=f"Write Permission Error: {directory.name}",
                            description=f"Cannot write to {directory}: {e}",
                            timestamp=datetime.now(),
                            recommendations=[
                                "Check directory permissions",
                                "Ensure user has write access",
                                "Check disk space availability"
                            ]
                        ))

        except Exception as e:
            logger.error(f"Storage check failed: {e}")

        return findings

    def _check_network(self) -> List[DiagnosticFinding]:
        """Check network connectivity and health."""
        findings = []

        try:
            # Check network interfaces
            interfaces = psutil.net_if_addrs()
            active_interfaces = 0

            for interface_name, addresses in interfaces.items():
                if any(addr.family == 2 for addr in addresses):  # IPv4
                    active_interfaces += 1

            if active_interfaces == 0:
                findings.append(DiagnosticFinding(
                    category=DiagnosticCategory.NETWORK,
                    severity=DiagnosticSeverity.CRITICAL,
                    title="No Network Connectivity",
                    description="No active network interfaces found",
                    timestamp=datetime.now(),
                    recommendations=[
                        "Check network cable connections",
                        "Restart network adapter",
                        "Check network configuration"
                    ]
                ))

            # Check for network errors
            net_io = psutil.net_io_counters()
            if net_io and (net_io.errin > 100 or net_io.errout > 100):
                findings.append(DiagnosticFinding(
                    category=DiagnosticCategory.NETWORK,
                    severity=DiagnosticSeverity.WARNING,
                    title="Network Errors Detected",
                    description=f"Network errors detected (in: {net_io.errin}, out: {net_io.errout})",
                    timestamp=datetime.now(),
                    metrics={'errors_in': net_io.errin, 'errors_out': net_io.errout},
                    recommendations=[
                        "Check network stability",
                        "Monitor network error patterns",
                        "Consider network hardware issues"
                    ]
                ))

        except Exception as e:
            logger.error(f"Network check failed: {e}")

        return findings

    def _check_critical_resources(self) -> List[DiagnosticFinding]:
        """Quick check for critical resource issues."""
        findings = []

        try:
            # Memory check
            memory = psutil.virtual_memory()
            if memory.percent > 95:
                findings.append(DiagnosticFinding(
                    category=DiagnosticCategory.RESOURCES,
                    severity=DiagnosticSeverity.CRITICAL,
                    title="Critical Memory Usage",
                    description=f"Memory usage critically high: {memory.percent:.1f}%",
                    timestamp=datetime.now(),
                    recommendations=["Close applications immediately"]
                ))

            # Disk check
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > 95:
                findings.append(DiagnosticFinding(
                    category=DiagnosticCategory.RESOURCES,
                    severity=DiagnosticSeverity.CRITICAL,
                    title="Critical Disk Space",
                    description=f"Disk space critically low: {disk_percent:.1f}%",
                    timestamp=datetime.now(),
                    recommendations=["Free disk space immediately"]
                ))

        except Exception as e:
            logger.error(f"Critical resource check failed: {e}")

        return findings

    def _check_service_availability(self) -> List[DiagnosticFinding]:
        """Quick check for service availability."""
        findings = []

        try:
            critical_services = ['webcam', 'detection', 'main_window']

            for service_name in critical_services:
                service = self.service_registry.get(service_name)
                if not service:
                    findings.append(DiagnosticFinding(
                        category=DiagnosticCategory.SERVICES,
                        severity=DiagnosticSeverity.ERROR,
                        title=f"Critical Service Missing: {service_name}",
                        description=f"Critical service {service_name} is not registered",
                        timestamp=datetime.now(),
                        affected_components=[service_name],
                        recommendations=[f"Initialize {service_name} service"]
                    ))
                elif hasattr(service, 'is_healthy') and not service.is_healthy():
                    findings.append(DiagnosticFinding(
                        category=DiagnosticCategory.SERVICES,
                        severity=DiagnosticSeverity.ERROR,
                        title=f"Critical Service Unhealthy: {service_name}",
                        description=f"Critical service {service_name} is not healthy",
                        timestamp=datetime.now(),
                        affected_components=[service_name],
                        recommendations=[f"Restart {service_name} service"]
                    ))

        except Exception as e:
            logger.error(f"Service availability check failed: {e}")

        return findings

    def _check_immediate_threats(self) -> List[DiagnosticFinding]:
        """Check for immediate system threats."""
        findings = []

        try:
            # Check for very high resource usage
            current_metrics = self.health_monitor.get_current_metrics()

            cpu_percent = current_metrics.get('cpu', {}).get('percent', 0)
            if cpu_percent > 95:
                findings.append(DiagnosticFinding(
                    category=DiagnosticCategory.SYSTEM_HEALTH,
                    severity=DiagnosticSeverity.CRITICAL,
                    title="System Unresponsive Risk",
                    description=f"CPU usage extremely high: {cpu_percent:.1f}%",
                    timestamp=datetime.now(),
                    recommendations=["System may become unresponsive"]
                ))

            # Check for process issues
            process = psutil.Process()
            if process.num_threads() > 100:
                findings.append(DiagnosticFinding(
                    category=DiagnosticCategory.SYSTEM_HEALTH,
                    severity=DiagnosticSeverity.WARNING,
                    title="High Thread Count",
                    description=f"Process has {process.num_threads()} threads",
                    timestamp=datetime.now(),
                    recommendations=["Monitor for thread leaks"]
                ))

        except Exception as e:
            logger.error(f"Immediate threat check failed: {e}")

        return findings

    def _build_system_overview(self) -> Dict[str, Any]:
        """Build comprehensive system overview."""
        try:
            current_metrics = self.health_monitor.get_current_metrics()

            return {
                'diagnostic_type': 'full',
                'system_info': {
                    'cpu_count': psutil.cpu_count(),
                    'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                    'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
                    'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat()
                },
                'current_metrics': current_metrics,
                'performance_summary': self.health_monitor.get_metrics_summary(),
                'registered_services': list(self.service_registry.keys()),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to build system overview: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def _generate_recommendations(self, report: DiagnosticReport):
        """Generate prioritized recommendations based on findings."""
        try:
            # Priority actions (critical/error findings)
            critical_findings = [f for f in report.findings if f.severity in [DiagnosticSeverity.CRITICAL, DiagnosticSeverity.ERROR]]
            for finding in critical_findings:
                if finding.recommendations:
                    report.priority_actions.extend(finding.recommendations[:2])  # Top 2 recommendations

            # Remove duplicates while preserving order
            report.priority_actions = list(dict.fromkeys(report.priority_actions))

            # Optimization suggestions (warning findings)
            warning_findings = [f for f in report.findings if f.severity == DiagnosticSeverity.WARNING]
            for finding in warning_findings:
                if finding.recommendations:
                    report.optimization_suggestions.extend(finding.recommendations[:1])  # Top recommendation

            report.optimization_suggestions = list(dict.fromkeys(report.optimization_suggestions))

            # Preventive measures (general recommendations)
            report.preventive_measures = [
                "Regularly monitor system resources",
                "Keep system and applications updated",
                "Maintain adequate free disk space (>10%)",
                "Monitor for memory leaks in long-running processes",
                "Backup important configurations regularly",
                "Review and clean up temporary files periodically"
            ]

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")


__all__ = [
    'DiagnosticCategory', 'DiagnosticSeverity', 'DiagnosticFinding',
    'DiagnosticReport', 'SystemHealthMonitor', 'AdvancedDiagnosticEngine'
]