# Comprehensive Error Recovery System Documentation

## Overview

The Error Recovery System is a sophisticated solution for handling partial failures during settings application and providing comprehensive recovery options to users. This system ensures the application remains stable and usable even when errors occur during configuration changes or service operations.

## Architecture

### Core Components

1. **Error Recovery Manager** (`app/core/error_recovery.py`)
   - Central coordinator for all recovery operations
   - Handles failure detection, classification, and recovery orchestration
   - Manages error history and statistics

2. **Recovery Executor** (`app/core/error_recovery.py`)
   - Executes specific recovery strategies
   - Analyzes failures and generates recovery options
   - Verifies recovery success

3. **Safe Mode Manager** (`app/core/safe_mode.py`)
   - Provides minimal configuration fallback
   - Manages feature enabling/disabling
   - Tracks feature stability over time

4. **Diagnostic System** (`app/core/diagnostics.py`)
   - Comprehensive system health monitoring
   - Performance analysis and bottleneck detection
   - Automated troubleshooting suggestions

5. **Error Learning System** (`app/core/error_recovery.py`)
   - Pattern recognition from historical errors
   - Predictive failure risk assessment
   - Prevention recommendations

6. **Recovery UI Dialogs** (`app/ui/dialogs/recovery_dialog.py`)
   - User interface for recovery interactions
   - Progress monitoring and result display
   - Diagnostic report visualization

## Key Features

### Failure Classification

The system automatically classifies failures into specific types:

- **Validation Errors**: Invalid configuration values
- **Service Restart Failures**: Services that fail to restart
- **Resource Unavailable**: Memory, disk, or network resources exhausted
- **Permission Denied**: File system or security access issues
- **Network Errors**: API connectivity or timeout issues
- **Partial Application**: Some settings applied, others failed

### Recovery Strategies

Multiple recovery strategies are available:

- **Retry**: Attempt the operation again
- **Rollback**: Revert to previous working state
- **Partial Apply**: Keep successful changes, ignore failures
- **Safe Mode**: Start with minimal, stable configuration
- **Repair**: Fix issues and retry
- **Ignore**: Continue despite the error

### Safe Mode Operations

Safe mode provides a fallback mechanism when normal operation fails:

- Disables non-essential features
- Uses conservative resource settings
- Allows gradual feature re-enablement
- Tracks feature stability scores

### Predictive Analysis

The error learning system provides:

- Pattern recognition from historical failures
- Risk assessment for current system state
- Prevention recommendations
- Automated configuration suggestions

## Usage Guide

### Basic Integration

```python
from app.core.error_recovery import ErrorRecoveryManager
from app.core.safe_mode import SafeModeManager

# Initialize recovery system
recovery_manager = ErrorRecoveryManager(
    service_registry=your_services,
    config_manager=your_config_manager,
    data_dir=Path("data")
)

# Initialize safe mode
safe_mode_manager = SafeModeManager(
    config_manager=your_config_manager,
    data_dir=Path("data")
)
```

### Handling Failures

```python
try:
    # Attempt configuration change
    apply_settings(new_settings)
except Exception as e:
    # Handle with recovery system
    context = recovery_manager.handle_failure(
        e,
        operation="apply_settings",
        affected_services=["webcam", "detection"],
        config_changes=new_settings
    )

    # Attempt automatic recovery
    auto_result = recovery_manager.attempt_auto_recovery(context)

    if not auto_result or not auto_result.success:
        # Show user recovery dialog
        show_recovery_dialog(context)
```

### Safe Mode Management

```python
# Enter safe mode manually
safe_mode_manager.enter_safe_mode(
    SafeModeReason.MANUAL_ACTIVATION,
    triggered_by="user"
)

# Test feature stability
success = safe_mode_manager.test_feature("webcam")

# Exit safe mode when stable
if safe_mode_manager.exit_safe_mode():
    print("Successfully exited safe mode")
```

### Diagnostic Analysis

```python
from app.core.diagnostics import AdvancedDiagnosticEngine

diagnostic_engine = AdvancedDiagnosticEngine(service_registry)

# Run quick diagnostic
quick_report = diagnostic_engine.run_quick_diagnostic()

# Run full diagnostic
full_report = diagnostic_engine.run_full_diagnostic()

print(f"Health Score: {full_report.overall_health_score}/100")
```

## Configuration

### Default Safe Configuration

The system provides sensible defaults for safe mode:

```python
{
    # Application settings
    'debug': True,
    'performance_mode': 'Power_Saving',
    'max_memory_usage_mb': 1024,

    # Camera settings (minimal)
    'camera_width': 640,
    'camera_height': 480,
    'camera_fps': 15,
    'use_gpu': False,

    # Detection settings (conservative)
    'detection_confidence_threshold': 0.5,
    'requests_per_minute': 5,
    'enable_rate_limiting': True
}
```

### Feature Configuration

Features are configured with priorities and dependencies:

```python
features = [
    # Core features (priority 1-3)
    FeatureConfig(
        name="webcam",
        priority=1,
        test_function="test_webcam"
    ),
    FeatureConfig(
        name="detection",
        priority=2,
        dependencies=["webcam"]
    ),

    # Optional features (priority 4+)
    FeatureConfig(
        name="gpu_acceleration",
        priority=5,
        safe_config={'use_gpu': False}
    )
]
```

## Error Scenarios

### Common Recovery Scenarios

1. **Camera Disconnected During Restart**
   - Detection: Hardware error during service restart
   - Recovery: Try alternative camera indices, fallback to mock camera
   - Safe Mode: Disable camera-dependent features

2. **GPU Out of Memory**
   - Detection: CUDA out of memory error
   - Recovery: Clear GPU cache, reduce batch sizes, fallback to CPU
   - Safe Mode: Disable GPU acceleration globally

3. **API Key Invalidated**
   - Detection: Authentication error from API
   - Recovery: Prompt for new key, use cached responses, disable AI features
   - Safe Mode: Disable network-dependent AI features

4. **Disk Full During Save**
   - Detection: Disk space error during configuration save
   - Recovery: Clean temporary files, use alternative paths
   - Safe Mode: Disable auto-save, reduce logging

5. **Network Timeout on API Calls**
   - Detection: Network timeout exceptions
   - Recovery: Retry with longer timeouts, use offline mode
   - Safe Mode: Disable network features

6. **Model File Corrupted**
   - Detection: Model loading errors
   - Recovery: Re-download models, use alternative models
   - Safe Mode: Use basic/lightweight models

7. **Permission Denied on Paths**
   - Detection: File access permission errors
   - Recovery: Use alternative paths, prompt for admin rights
   - Safe Mode: Use temporary directories

## Monitoring and Analytics

### Recovery Statistics

The system tracks comprehensive statistics:

```python
stats = recovery_manager.get_recovery_statistics()
{
    'total_failures': 42,
    'successful_recoveries': 38,
    'failed_recoveries': 4,
    'success_rate': 0.90,
    'auto_recoveries': 30,
    'manual_interventions': 8,
    'recent_failure_types': {
        'network_error': 15,
        'validation_error': 12,
        'service_restart': 8
    }
}
```

### Error Pattern Learning

The system learns from errors to prevent future issues:

```python
patterns = error_learning.find_patterns(min_occurrences=3)
for pattern in patterns:
    print(f"Pattern: {pattern.pattern_id}")
    print(f"Occurrences: {pattern.occurrences}")
    print(f"Success Rate: {pattern.successful_recovery}")
    print(f"Prevention: {pattern.prevention_suggestions}")
```

### Diagnostic Reports

Comprehensive diagnostic reports provide system insights:

```python
report = diagnostic_engine.run_full_diagnostic()
print(f"Health Score: {report.overall_health_score}/100")
print(f"Performance Score: {report.performance_score}/100")
print(f"Findings: {len(report.findings)} issues detected")

for finding in report.findings:
    print(f"- {finding.title} ({finding.severity.value})")
    print(f"  {finding.description}")
    print(f"  Recommendations: {finding.recommendations}")
```

## Testing

### Running Tests

```bash
# Run all error recovery tests
pytest test_error_recovery_system.py -v

# Run specific test classes
pytest test_error_recovery_system.py::TestErrorClassification -v
pytest test_error_recovery_system.py::TestRecoveryStrategies -v
pytest test_error_recovery_system.py::TestSafeModeManager -v
```

### Test Categories

1. **Error Classification Tests**
   - Validate failure type detection
   - Test severity assessment
   - Verify affected service identification

2. **Recovery Strategy Tests**
   - Test recovery option generation
   - Validate recovery execution
   - Verify recovery verification

3. **Safe Mode Tests**
   - Test safe mode activation/deactivation
   - Validate feature management
   - Test configuration generation

4. **Diagnostic Tests**
   - Test system health monitoring
   - Validate diagnostic analysis
   - Test finding generation

5. **Integration Tests**
   - Test complete recovery workflows
   - Validate cross-component interaction
   - Test concurrent error handling

### Example Integration Test

```python
def test_complete_recovery_workflow(self):
    # Simulate configuration error
    error = ConfigurationError("Invalid camera resolution setting")

    # Handle failure
    context = recovery_manager.handle_failure(
        error,
        operation="apply_camera_settings",
        affected_services=["webcam"]
    )

    # Verify failure classification
    assert context.failure_type == FailureType.VALIDATION_ERROR
    assert len(context.recovery_options) > 0

    # Attempt auto-recovery
    auto_result = recovery_manager.attempt_auto_recovery(context)

    # Verify recovery result
    if auto_result:
        assert isinstance(auto_result, RecoveryResult)
```

## Best Practices

### Implementation Guidelines

1. **Fail Fast**: Detect errors early and provide clear feedback
2. **Graceful Degradation**: Maintain functionality even with failures
3. **User Communication**: Provide clear, actionable error messages
4. **Recovery Verification**: Always verify recovery success
5. **Learning Integration**: Use error patterns to improve system reliability

### Error Handling Pattern

```python
def robust_operation():
    try:
        # Attempt operation
        return perform_operation()
    except Exception as e:
        # Log error
        logger.error(f"Operation failed: {e}")

        # Handle with recovery system
        context = recovery_manager.handle_failure(
            e,
            operation="operation_name",
            affected_services=["service1", "service2"]
        )

        # Attempt recovery
        result = recovery_manager.attempt_auto_recovery(context)

        if result and result.success:
            return result
        else:
            # Escalate to user if needed
            return handle_user_recovery(context)
```

### Safe Mode Best Practices

1. **Conservative Defaults**: Use safe, known-working configurations
2. **Gradual Re-enablement**: Test features before full activation
3. **Dependency Management**: Respect feature dependencies
4. **Stability Tracking**: Monitor feature stability over time

### Diagnostic Best Practices

1. **Regular Monitoring**: Run diagnostics periodically
2. **Threshold Management**: Set appropriate warning/error thresholds
3. **Trend Analysis**: Monitor metrics over time
4. **Proactive Alerts**: Warn before critical thresholds

## Troubleshooting

### Common Issues

1. **Recovery System Not Starting**
   - Check data directory permissions
   - Verify service registry is properly populated
   - Ensure configuration manager is available

2. **Safe Mode Not Activating**
   - Check for file system permissions
   - Verify feature configurations are valid
   - Review safe mode state persistence

3. **Diagnostics Failing**
   - Check system monitoring permissions
   - Verify psutil library availability
   - Review resource access rights

4. **Recovery Dialogs Not Showing**
   - Verify UI root window is initialized
   - Check for threading issues
   - Ensure proper dialog parent relationships

### Debug Information

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable recovery system debug logs
logger = logging.getLogger('app.core.error_recovery')
logger.setLevel(logging.DEBUG)
```

### Configuration Validation

Validate recovery system configuration:

```python
def validate_recovery_system():
    # Check data directory
    assert data_dir.exists() and data_dir.is_dir()

    # Check service registry
    assert len(service_registry) > 0

    # Check safe mode features
    assert len(safe_mode_manager.state.features) > 0

    # Check test functions
    assert len(safe_mode_manager.test_functions) > 0
```

## API Reference

### Error Recovery Manager

```python
class ErrorRecoveryManager:
    def handle_failure(self, exception: Exception,
                      operation: str = "unknown",
                      affected_services: Optional[List[str]] = None,
                      config_changes: Optional[Dict[str, Any]] = None,
                      user_action: Optional[str] = None) -> FailureContext

    def attempt_auto_recovery(self, context: FailureContext) -> Optional[RecoveryResult]

    def execute_user_recovery(self, context: FailureContext,
                            selected_option: RecoveryOption) -> RecoveryResult

    def get_recovery_statistics(self) -> Dict[str, Any]
```

### Safe Mode Manager

```python
class SafeModeManager:
    def enter_safe_mode(self, reason: SafeModeReason,
                       triggered_by: str = "system") -> bool

    def exit_safe_mode(self, force: bool = False) -> bool

    def test_feature(self, feature_name: str) -> bool

    def disable_feature(self, feature_name: str, reason: str = "") -> bool

    def enable_feature(self, feature_name: str, test_first: bool = True) -> bool

    def get_safe_config(self) -> Dict[str, Any]
```

### Diagnostic Engine

```python
class AdvancedDiagnosticEngine:
    def run_full_diagnostic(self) -> DiagnosticReport

    def run_quick_diagnostic(self) -> DiagnosticReport

    def analyze_logs(self, hours_back: int = 1) -> List[Dict[str, Any]]
```

## Conclusion

The Comprehensive Error Recovery System provides a robust foundation for handling errors and maintaining application stability. By implementing proper error classification, recovery strategies, safe mode fallbacks, and predictive analysis, the system ensures users can continue working even when errors occur.

The system's modular design allows for easy extension and customization while maintaining a consistent approach to error handling across the entire application. Regular monitoring and analysis of error patterns helps improve system reliability over time.

For more information or support, refer to the test suite and example integration code provided with the system.