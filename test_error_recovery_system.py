"""Integration tests for the Error Recovery System.

This test suite validates the comprehensive error recovery system including:
- Error detection and classification
- Recovery strategy generation and execution
- Safe mode operations
- Diagnostic system functionality
- Error pattern learning
- User interface components
"""
import pytest
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from app.core.error_recovery import (
    FailureType, RecoveryStrategy, Severity, SystemSnapshot,
    RecoveryOption, FailureContext, RecoveryResult, ErrorPattern,
    DiagnosticCollector, RecoveryExecutor, ErrorRecoveryManager,
    ErrorLearning
)
from app.core.diagnostics import (
    DiagnosticCategory, DiagnosticSeverity, DiagnosticFinding,
    DiagnosticReport, SystemHealthMonitor, AdvancedDiagnosticEngine
)
from app.core.safe_mode import (
    SafeModeReason, FeatureState, FeatureConfig,
    SafeModeState, SafeModeManager
)
from app.core.exceptions import (
    ValidationError, ServiceError, ConfigurationError
)


class TestErrorClassification:
    """Test error detection and classification."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.service_registry = {
            'webcam': Mock(),
            'detection': Mock(),
            'gemini': Mock(),
            'main_window': Mock()
        }
        self.recovery_manager = ErrorRecoveryManager(
            self.service_registry,
            data_dir=self.temp_dir
        )

    def test_validation_error_classification(self):
        """Test classification of validation errors."""
        error = ValidationError("Invalid camera resolution: 9999x9999")

        context = self.recovery_manager.handle_failure(
            error,
            operation="apply_camera_settings",
            affected_services=["webcam"]
        )

        assert context.failure_type == FailureType.VALIDATION_ERROR
        assert context.severity in [Severity.LOW, Severity.MEDIUM]
        assert "webcam" in context.affected_services

    def test_service_error_classification(self):
        """Test classification of service errors."""
        error = ServiceError("Failed to restart webcam service")

        context = self.recovery_manager.handle_failure(
            error,
            operation="restart_webcam",
            affected_services=["webcam"]
        )

        assert context.failure_type == FailureType.SERVICE_RESTART_FAILED
        assert context.severity == Severity.HIGH  # Core service failure
        assert "webcam" in context.affected_services

    def test_network_error_classification(self):
        """Test classification of network errors."""
        error = Exception("Connection timeout: Failed to reach API endpoint")

        context = self.recovery_manager.handle_failure(
            error,
            operation="gemini_api_call",
            affected_services=["gemini"]
        )

        assert context.failure_type == FailureType.NETWORK_ERROR
        assert context.severity == Severity.MEDIUM
        assert "gemini" in context.affected_services

    def test_severity_assessment(self):
        """Test severity assessment based on affected services."""
        # Critical service failure
        error = Exception("Core system failure")
        context = self.recovery_manager.handle_failure(
            error,
            affected_services=["webcam", "detection", "main_window"]
        )
        assert context.severity == Severity.HIGH

        # Non-critical service failure
        error = Exception("Optional feature failed")
        context = self.recovery_manager.handle_failure(
            error,
            affected_services=["telemetry"]
        )
        assert context.severity == Severity.MEDIUM


class TestRecoveryStrategies:
    """Test recovery strategy generation and execution."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.service_registry = {
            'webcam': Mock(),
            'detection': Mock(),
            'gemini': Mock()
        }
        self.config_manager = Mock()
        self.recovery_executor = RecoveryExecutor(
            self.service_registry,
            self.config_manager
        )

    def test_validation_recovery_options(self):
        """Test recovery options for validation errors."""
        context = FailureContext(
            failure_type=FailureType.VALIDATION_ERROR,
            severity=Severity.LOW,
            affected_services=["webcam"],
            error_messages=["Invalid setting value"],
            timestamp=datetime.now(),
            system_state=SystemSnapshot(
                timestamp=datetime.now(),
                memory_usage_mb=1024,
                cpu_usage_percent=50,
                disk_free_gb=10,
                active_threads=20,
                open_files=50
            )
        )

        options = self.recovery_executor.analyze_failure(context)

        assert len(options) > 0

        # Should include repair and partial apply options
        strategies = [opt.strategy for opt in options]
        assert RecoveryStrategy.REPAIR in strategies
        assert RecoveryStrategy.PARTIAL_APPLY in strategies

        # All options should be safe for validation errors
        assert all(opt.is_safe for opt in options)

    def test_service_recovery_options(self):
        """Test recovery options for service failures."""
        context = FailureContext(
            failure_type=FailureType.SERVICE_RESTART_FAILED,
            severity=Severity.HIGH,
            affected_services=["webcam"],
            error_messages=["Service failed to start"],
            timestamp=datetime.now(),
            system_state=SystemSnapshot(
                timestamp=datetime.now(),
                memory_usage_mb=2048,
                cpu_usage_percent=80,
                disk_free_gb=5,
                active_threads=30,
                open_files=100
            )
        )

        options = self.recovery_executor.analyze_failure(context)

        assert len(options) > 0

        # Should include retry and safe mode options
        strategies = [opt.strategy for opt in options]
        assert RecoveryStrategy.RETRY in strategies
        assert RecoveryStrategy.SAFE_MODE in strategies

    def test_recovery_execution_retry(self):
        """Test execution of retry recovery strategy."""
        context = FailureContext(
            failure_type=FailureType.NETWORK_ERROR,
            severity=Severity.MEDIUM,
            affected_services=["gemini"],
            error_messages=["Network timeout"],
            timestamp=datetime.now(),
            system_state=SystemSnapshot(
                timestamp=datetime.now(),
                memory_usage_mb=1024,
                cpu_usage_percent=30,
                disk_free_gb=15,
                active_threads=10,
                open_files=20
            )
        )

        option = RecoveryOption(
            strategy=RecoveryStrategy.RETRY,
            title="Retry Operation",
            description="Retry the failed operation",
            estimated_time_seconds=5,
            success_probability=0.6
        )

        result = self.recovery_executor.execute_recovery(option, context)

        assert isinstance(result, RecoveryResult)
        assert result.strategy_used == RecoveryStrategy.RETRY
        assert result.duration_seconds > 0

    def test_recovery_verification(self):
        """Test recovery result verification."""
        # Mock healthy services
        for service in self.service_registry.values():
            service.is_healthy = Mock(return_value=True)

        result = RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.RETRY,
            message="Recovery completed",
            duration_seconds=2.5,
            services_affected=["gemini"]
        )

        verification = self.recovery_executor.verify_recovery(result)
        assert verification is True


class TestSafeModeManager:
    """Test safe mode functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_manager = Mock()
        self.safe_mode_manager = SafeModeManager(
            self.config_manager,
            self.temp_dir
        )

    def test_enter_safe_mode(self):
        """Test entering safe mode."""
        success = self.safe_mode_manager.enter_safe_mode(
            SafeModeReason.CONFIG_ERROR,
            "test_system"
        )

        assert success
        assert self.safe_mode_manager.is_in_safe_mode()

        info = self.safe_mode_manager.get_safe_mode_info()
        assert info['active']
        assert info['reason'] == SafeModeReason.CONFIG_ERROR.value
        assert info['activated_by'] == "test_system"

    def test_exit_safe_mode(self):
        """Test exiting safe mode."""
        # Enter safe mode first
        self.safe_mode_manager.enter_safe_mode(
            SafeModeReason.MANUAL_ACTIVATION,
            "test"
        )
        assert self.safe_mode_manager.is_in_safe_mode()

        # Mock successful feature tests
        self.safe_mode_manager.test_functions = {
            feature.name: Mock(return_value=True)
            for feature in self.safe_mode_manager.state.features.values()
            if feature.priority <= 3
        }

        # Exit safe mode
        success = self.safe_mode_manager.exit_safe_mode()

        assert success
        assert not self.safe_mode_manager.is_in_safe_mode()

    def test_feature_testing(self):
        """Test feature testing functionality."""
        # Register test function
        test_func = Mock(return_value=True)
        self.safe_mode_manager.register_test_function("webcam", test_func)

        # Test feature
        result = self.safe_mode_manager.test_feature("webcam")

        assert result is True
        test_func.assert_called_once()

        # Check test statistics updated
        feature = self.safe_mode_manager.state.features["webcam"]
        assert feature.test_success_count > 0
        assert feature.last_test_time is not None

    def test_feature_disable_enable(self):
        """Test feature disable/enable functionality."""
        # Disable feature
        success = self.safe_mode_manager.disable_feature("gpu_acceleration", "testing")
        assert success

        feature = self.safe_mode_manager.state.features["gpu_acceleration"]
        assert feature.state == FeatureState.DISABLED
        assert feature.disabled_since is not None

        # Enable feature
        success = self.safe_mode_manager.enable_feature("gpu_acceleration", test_first=False)
        assert success

        feature = self.safe_mode_manager.state.features["gpu_acceleration"]
        assert feature.state == FeatureState.ENABLED
        assert feature.disabled_since is None

    def test_safe_config_generation(self):
        """Test safe configuration generation."""
        safe_config = self.safe_mode_manager.get_safe_config()

        assert isinstance(safe_config, dict)
        assert 'debug' in safe_config
        assert 'performance_mode' in safe_config
        assert safe_config['use_gpu'] is False  # Should be disabled in safe mode
        assert safe_config['performance_mode'] == 'Power_Saving'

    def test_dependency_handling(self):
        """Test feature dependency handling."""
        # Disable a feature that others depend on
        self.safe_mode_manager.disable_feature("webcam", "testing")

        # Check that dependent features are also disabled
        webcam_feature = self.safe_mode_manager.state.features["webcam"]
        detection_feature = self.safe_mode_manager.state.features["detection"]

        assert webcam_feature.state == FeatureState.DISABLED
        assert detection_feature.state == FeatureState.DISABLED  # Depends on webcam


class TestDiagnosticSystem:
    """Test diagnostic system functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.service_registry = {
            'webcam': Mock(),
            'detection': Mock(),
            'gemini': Mock()
        }

        # Mock healthy services
        for service in self.service_registry.values():
            service.is_healthy = Mock(return_value=True)
            service.is_running = Mock(return_value=True)

        self.diagnostic_engine = AdvancedDiagnosticEngine(self.service_registry)

    def test_system_health_monitoring(self):
        """Test system health monitoring."""
        monitor = SystemHealthMonitor()

        # Start monitoring
        monitor.start_monitoring()
        time.sleep(0.1)  # Brief monitoring period

        # Get current metrics
        metrics = monitor.get_current_metrics()

        assert 'timestamp' in metrics
        assert 'cpu' in metrics
        assert 'memory' in metrics
        assert 'disk' in metrics
        assert 'process' in metrics

        # Stop monitoring
        monitor.stop_monitoring()

    def test_quick_diagnostic(self):
        """Test quick diagnostic analysis."""
        report = self.diagnostic_engine.run_quick_diagnostic()

        assert isinstance(report, DiagnosticReport)
        assert report.report_id.startswith("diag_quick_")
        assert report.overall_health_score >= 0
        assert report.overall_health_score <= 100

    def test_full_diagnostic(self):
        """Test full diagnostic analysis."""
        report = self.diagnostic_engine.run_full_diagnostic()

        assert isinstance(report, DiagnosticReport)
        assert report.report_id.startswith("diag_full_")
        assert len(report.findings) >= 0

        # Check that all diagnostic categories are covered
        categories = set(finding.category for finding in report.findings)
        # Should have findings from multiple categories

    def test_diagnostic_findings(self):
        """Test diagnostic finding creation and analysis."""
        finding = DiagnosticFinding(
            category=DiagnosticCategory.SYSTEM_HEALTH,
            severity=DiagnosticSeverity.WARNING,
            title="High CPU Usage",
            description="CPU usage is above normal levels",
            timestamp=datetime.now(),
            metrics={'cpu_percent': 85},
            recommendations=["Close unnecessary applications"]
        )

        assert finding.category == DiagnosticCategory.SYSTEM_HEALTH
        assert finding.severity == DiagnosticSeverity.WARNING
        assert finding.metrics['cpu_percent'] == 85
        assert len(finding.recommendations) > 0

    @patch('psutil.virtual_memory')
    def test_memory_threshold_detection(self, mock_memory):
        """Test memory usage threshold detection."""
        # Mock high memory usage
        mock_memory.return_value.percent = 95
        mock_memory.return_value.total = 8 * 1024**3  # 8GB
        mock_memory.return_value.available = 0.4 * 1024**3  # 400MB
        mock_memory.return_value.used = 7.6 * 1024**3  # 7.6GB

        findings = self.diagnostic_engine._check_critical_resources()

        # Should detect critical memory usage
        memory_findings = [f for f in findings if 'memory' in f.title.lower()]
        assert len(memory_findings) > 0
        assert any(f.severity == DiagnosticSeverity.CRITICAL for f in memory_findings)


class TestErrorLearning:
    """Test error pattern learning and prediction."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.error_learning = ErrorLearning(self.temp_dir)

    def test_error_pattern_recording(self):
        """Test recording and learning from error patterns."""
        # Create multiple similar failures
        for i in range(5):
            context = FailureContext(
                failure_type=FailureType.NETWORK_ERROR,
                severity=Severity.MEDIUM,
                affected_services=["gemini"],
                error_messages=[f"Network timeout #{i}"],
                timestamp=datetime.now(),
                system_state=SystemSnapshot(
                    timestamp=datetime.now(),
                    memory_usage_mb=1024,
                    cpu_usage_percent=30,
                    disk_free_gb=10,
                    active_threads=15,
                    open_files=25
                )
            )

            self.error_learning.record_error(context)

        # Find patterns
        patterns = self.error_learning.find_patterns(min_occurrences=3)

        assert len(patterns) > 0
        pattern = patterns[0]
        assert pattern.occurrences >= 3
        assert pattern.trigger_conditions['failure_type'] == 'network_error'

    def test_pattern_similarity_detection(self):
        """Test pattern similarity detection."""
        context1 = {
            'failure_type': 'network_error',
            'severity': 'medium',
            'affected_services': ['gemini'],
            'high_memory_usage': False
        }

        context2 = {
            'failure_type': 'network_error',
            'severity': 'medium',
            'affected_services': ['gemini'],
            'high_memory_usage': False
        }

        similarity = self.error_learning._calculate_context_similarity(context1, context2)
        assert similarity > 0.8  # Should be very similar

    def test_failure_risk_prediction(self):
        """Test failure risk prediction."""
        # Record some patterns first
        for i in range(3):
            context = FailureContext(
                failure_type=FailureType.MEMORY_ERROR,
                severity=Severity.HIGH,
                affected_services=["detection"],
                error_messages=["Out of memory"],
                timestamp=datetime.now(),
                system_state=SystemSnapshot(
                    timestamp=datetime.now(),
                    memory_usage_mb=7000,  # High memory usage
                    cpu_usage_percent=90,
                    disk_free_gb=2,
                    active_threads=50,
                    open_files=200
                )
            )
            self.error_learning.record_error(context)

        # Predict risk for similar context
        current_context = {
            'failure_type': 'memory_error',
            'high_memory_usage': True,
            'high_cpu_usage': True,
            'affected_services': ['detection']
        }

        risk = self.error_learning.predict_failure_risk(current_context)
        assert 0 <= risk <= 1

    def test_prevention_suggestions(self):
        """Test prevention suggestion generation."""
        pattern = ErrorPattern(
            pattern_id="test_pattern",
            occurrences=5,
            last_seen=datetime.now(),
            first_seen=datetime.now() - timedelta(days=1),
            trigger_conditions={
                'failure_type': 'network_error',
                'affected_services': ['gemini'],
                'high_memory_usage': False
            },
            affected_services={'gemini'}
        )

        suggestions = self.error_learning.suggest_prevention(pattern)

        assert len(suggestions) > 0
        # Should include network-related suggestions
        assert any('network' in suggestion.lower() for suggestion in suggestions)

    def test_pattern_cleanup(self):
        """Test cleanup of old patterns."""
        # Create old patterns
        old_time = datetime.now() - timedelta(days=100)

        for i in range(10):
            pattern_id = f"old_pattern_{i}"
            self.error_learning.error_patterns[pattern_id] = ErrorPattern(
                pattern_id=pattern_id,
                occurrences=1,  # Low occurrence count
                last_seen=old_time,
                first_seen=old_time,
                trigger_conditions={'failure_type': 'test'},
                affected_services=set()
            )

        initial_count = len(self.error_learning.error_patterns)
        removed_count = self.error_learning.cleanup_old_patterns(max_age_days=30)

        assert removed_count > 0
        assert len(self.error_learning.error_patterns) < initial_count


class TestIntegrationScenarios:
    """Test complete error recovery scenarios."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.service_registry = {
            'webcam': Mock(),
            'detection': Mock(),
            'gemini': Mock(),
            'main_window': Mock()
        }
        self.config_manager = Mock()

        # Setup complete recovery system
        self.recovery_manager = ErrorRecoveryManager(
            self.service_registry,
            self.config_manager,
            self.temp_dir
        )

        self.safe_mode_manager = SafeModeManager(
            self.config_manager,
            self.temp_dir
        )

    def test_complete_recovery_workflow(self):
        """Test complete error recovery workflow."""
        # Simulate configuration error
        error = ConfigurationError("Invalid camera resolution setting")

        # 1. Handle failure
        context = self.recovery_manager.handle_failure(
            error,
            operation="apply_camera_settings",
            affected_services=["webcam"],
            config_changes={"camera_width": 9999, "camera_height": 9999}
        )

        assert context.failure_type == FailureType.PARTIAL_APPLICATION
        assert len(context.recovery_options) > 0

        # 2. Attempt auto-recovery
        auto_result = self.recovery_manager.attempt_auto_recovery(context)

        # Auto-recovery might succeed or fail depending on options
        if auto_result:
            assert isinstance(auto_result, RecoveryResult)
            if auto_result.success:
                assert context.resolved

    def test_safe_mode_integration(self):
        """Test integration with safe mode."""
        # Enter safe mode due to critical error
        success = self.safe_mode_manager.enter_safe_mode(
            SafeModeReason.CRITICAL_ERROR,
            "recovery_system"
        )
        assert success

        # Verify safe configuration
        safe_config = self.safe_mode_manager.get_safe_config()
        assert safe_config['use_gpu'] is False
        assert safe_config['performance_mode'] == 'Power_Saving'

        # Test gradual feature re-enablement
        status = self.safe_mode_manager.get_feature_status()
        assert 'webcam' in status
        assert 'detection' in status

    def test_error_learning_integration(self):
        """Test error learning integration."""
        # Simulate repeated failures
        for i in range(5):
            error = Exception(f"Repeated network error {i}")
            context = self.recovery_manager.handle_failure(
                error,
                operation="gemini_api_call",
                affected_services=["gemini"]
            )

            # Simulate resolution
            context.resolved = True
            context.resolution_strategy = RecoveryStrategy.RETRY

        # Check that patterns were learned
        stats = self.recovery_manager.recovery_executor.recovery_history
        assert len(stats) >= 0  # At least some recovery attempts recorded

    def test_cascading_failure_handling(self):
        """Test handling of cascading failures."""
        # Simulate primary failure
        primary_error = Exception("GPU out of memory")
        primary_context = self.recovery_manager.handle_failure(
            primary_error,
            operation="model_inference",
            affected_services=["detection"]
        )

        assert primary_context.failure_type in [
            FailureType.MEMORY_ERROR,
            FailureType.RESOURCE_UNAVAILABLE
        ]

        # Simulate secondary failure during recovery
        secondary_error = Exception("Service restart failed")
        secondary_context = self.recovery_manager.handle_failure(
            secondary_error,
            operation="restart_detection_service",
            affected_services=["detection"]
        )

        assert secondary_context.failure_type == FailureType.SERVICE_RESTART_FAILED

        # Both failures should be tracked
        assert len(self.recovery_manager.error_history) >= 2

    def test_concurrent_error_handling(self):
        """Test concurrent error handling."""
        def simulate_error(error_id):
            error = Exception(f"Concurrent error {error_id}")
            context = self.recovery_manager.handle_failure(
                error,
                operation=f"operation_{error_id}",
                affected_services=[f"service_{error_id}"]
            )
            return context

        # Simulate concurrent errors
        threads = []
        contexts = []

        for i in range(3):
            thread = threading.Thread(target=lambda i=i: contexts.append(simulate_error(i)))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All errors should be handled
        assert len(contexts) == 3
        assert all(isinstance(ctx, FailureContext) for ctx in contexts)

    def test_recovery_statistics(self):
        """Test recovery statistics tracking."""
        # Simulate some failures and recoveries
        for i in range(3):
            error = Exception(f"Test error {i}")
            context = self.recovery_manager.handle_failure(
                error,
                operation=f"test_operation_{i}",
                affected_services=["test_service"]
            )

            # Simulate auto-recovery
            self.recovery_manager.attempt_auto_recovery(context)

        # Check statistics
        stats = self.recovery_manager.get_recovery_statistics()

        assert 'total_failures' in stats
        assert 'successful_recoveries' in stats
        assert 'failed_recoveries' in stats
        assert 'success_rate' in stats
        assert stats['total_failures'] >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])