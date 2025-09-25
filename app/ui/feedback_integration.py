"""Integration layer for the User Feedback System with settings dialog."""

from __future__ import annotations
import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import logging

from .dialogs.progress_dialog import SettingsProgressDialog, ProgressState, progress_manager
from .dialogs.error_recovery_dialog import (
    ErrorRecoveryDialog, ErrorDetails, ErrorSeverity, ActionType,
    create_settings_error, create_webcam_error, create_file_error
)
from .dialogs.success_dialog import (
    SuccessDialog, ChangesSummary, PerformanceMetric, NextStep,
    show_success_dialog, create_settings_success
)
from .components.status_indicator import StatusIndicator, IndicatorState
from .components.step_tracker import StepTracker, StepStatus
from .components.notification_manager import (
    NotificationManager, NotificationType, get_notification_manager
)
from .components.live_status_panel import LiveStatusPanel, ServiceStatus, LogLevel

logger = logging.getLogger(__name__)


@dataclass
class SettingsOperation:
    """Settings operation definition."""
    name: str
    estimated_ms: int
    callback: Callable
    description: str = ""
    substeps: List[str] = None

    def __post_init__(self):
        if self.substeps is None:
            self.substeps = []


class SettingsFeedbackManager:
    """Manager for settings application feedback."""

    def __init__(self, parent_window, notification_manager: NotificationManager = None):
        """Initialize feedback manager."""
        self.parent = parent_window
        self.notification_manager = notification_manager or get_notification_manager(parent_window)

        # State tracking
        self.current_progress_dialog: Optional[SettingsProgressDialog] = None
        self.current_operation_start_time: Optional[float] = None
        self.cancelled = False
        self.rollback_performed = False

        # Performance tracking
        self.performance_metrics = {}
        self.pre_operation_metrics = {}

        # Operation steps
        self.operations: List[SettingsOperation] = []
        self.current_operation_index = 0

    def setup_standard_operations(self) -> List[SettingsOperation]:
        """Setup standard settings application operations."""
        operations = [
            SettingsOperation(
                name="Validating configuration",
                estimated_ms=500,
                callback=self._validate_configuration,
                description="Checking settings for validity and compatibility",
                substeps=["Checking camera settings", "Validating detection parameters", "Verifying file paths"]
            ),
            SettingsOperation(
                name="Creating backup",
                estimated_ms=100,
                callback=self._create_backup,
                description="Creating backup of current configuration",
                substeps=["Saving current settings", "Creating backup file"]
            ),
            SettingsOperation(
                name="Stopping services",
                estimated_ms=200,
                callback=self._stop_services,
                description="Stopping application services for reconfiguration",
                substeps=["Stopping webcam service", "Stopping detection service", "Stopping AI service"]
            ),
            SettingsOperation(
                name="Applying changes",
                estimated_ms=800,
                callback=self._apply_changes,
                description="Applying new settings to application",
                substeps=["Updating configuration", "Saving settings", "Updating service parameters"]
            ),
            SettingsOperation(
                name="Restarting services",
                estimated_ms=400,
                callback=self._restart_services,
                description="Restarting services with new configuration",
                substeps=["Starting webcam service", "Starting detection service", "Starting AI service"]
            ),
            SettingsOperation(
                name="Verifying operation",
                estimated_ms=200,
                callback=self._verify_operation,
                description="Verifying that settings were applied correctly",
                substeps=["Testing camera connection", "Verifying detection pipeline", "Checking service health"]
            )
        ]

        self.operations = operations
        return operations

    def apply_settings_with_feedback(self, config_changes: Dict[str, Any],
                                   on_success: Callable = None,
                                   on_error: Callable = None,
                                   on_cancel: Callable = None) -> bool:
        """Apply settings with comprehensive feedback."""
        # Setup operations
        self.setup_standard_operations()

        # Reset state
        self.cancelled = False
        self.rollback_performed = False
        self.current_operation_index = 0
        self.current_operation_start_time = time.time()

        # Store callbacks
        self.success_callback = on_success
        self.error_callback = on_error
        self.cancel_callback = on_cancel

        # Store config changes for processing
        self.config_changes = config_changes

        # Capture pre-operation metrics
        self._capture_pre_operation_metrics()

        # Show progress dialog
        total_steps = len(self.operations)
        self.current_progress_dialog = progress_manager.show_progress(
            self.parent,
            total_steps,
            "Applying Settings"
        )

        # Setup progress dialog
        self.current_progress_dialog.enable_cancel(self._handle_cancellation)
        self.current_progress_dialog.set_state(ProgressState.VALIDATING)

        # Start operation in background thread
        operation_thread = threading.Thread(
            target=self._execute_operations,
            daemon=True
        )
        operation_thread.start()

        return True

    def _execute_operations(self):
        """Execute all operations sequentially."""
        try:
            for i, operation in enumerate(self.operations):
                if self.cancelled:
                    self._handle_cancellation_cleanup()
                    return

                # Update progress
                self.current_operation_index = i
                self._update_progress_dialog(i, operation.name)

                # Execute substeps
                for substep in operation.substeps:
                    if self.cancelled:
                        self._handle_cancellation_cleanup()
                        return

                    self._update_substep(substep)
                    time.sleep(0.1)  # Brief pause for UI updates

                # Execute main operation
                try:
                    success = operation.callback()
                    if not success:
                        raise Exception(f"Operation '{operation.name}' failed")

                except Exception as e:
                    logger.error(f"Operation failed: {operation.name} - {str(e)}")
                    self._handle_operation_error(operation, str(e))
                    return

                # Brief pause between operations
                time.sleep(0.05)

            # All operations completed successfully
            self._handle_success()

        except Exception as e:
            logger.error(f"Critical error during settings application: {str(e)}")
            self._handle_critical_error(str(e))

    def _update_progress_dialog(self, step_index: int, operation_name: str):
        """Update progress dialog with current operation."""
        if self.current_progress_dialog and not self.cancelled:
            self.current_progress_dialog.update_progress(step_index + 1, operation_name)

            # Update state based on operation
            if "validating" in operation_name.lower():
                self.current_progress_dialog.set_state(ProgressState.VALIDATING)
            else:
                self.current_progress_dialog.set_state(ProgressState.APPLYING)

    def _update_substep(self, substep: str):
        """Update progress dialog with substep."""
        if self.current_progress_dialog and not self.cancelled:
            self.current_progress_dialog.update_substep(substep)

    def _validate_configuration(self) -> bool:
        """Validate configuration changes."""
        time.sleep(0.3)  # Simulate validation time
        # Add actual validation logic here
        return True

    def _create_backup(self) -> bool:
        """Create configuration backup."""
        time.sleep(0.1)  # Simulate backup time
        # Add actual backup logic here
        return True

    def _stop_services(self) -> bool:
        """Stop application services."""
        time.sleep(0.2)  # Simulate stop time
        # Add actual service stopping logic here
        return True

    def _apply_changes(self) -> bool:
        """Apply configuration changes."""
        time.sleep(0.5)  # Simulate application time
        # Add actual configuration application logic here
        return True

    def _restart_services(self) -> bool:
        """Restart application services."""
        time.sleep(0.3)  # Simulate restart time
        # Add actual service restart logic here
        return True

    def _verify_operation(self) -> bool:
        """Verify that operation completed successfully."""
        time.sleep(0.2)  # Simulate verification time
        # Add actual verification logic here
        return True

    def _capture_pre_operation_metrics(self):
        """Capture performance metrics before operation."""
        self.pre_operation_metrics = {
            'fps': 15.0,  # Example values - replace with actual metrics
            'latency': 50.0,
            'memory_usage': 45.0,
            'cpu_usage': 30.0
        }

    def _handle_success(self):
        """Handle successful completion of all operations."""
        if self.current_progress_dialog:
            self.current_progress_dialog.show_completion(
                success=True,
                message="All settings have been applied successfully!",
                details="Configuration updated and services restarted."
            )

        # Show success notification
        self.notification_manager.show_success(
            "Settings applied successfully!",
            "Configuration Updated"
        )

        # Capture post-operation metrics
        post_metrics = {
            'fps': 30.0,  # Example improved values
            'latency': 25.0,
            'memory_usage': 40.0,
            'cpu_usage': 35.0
        }

        # Show detailed success dialog after brief delay
        def show_success_details():
            time.sleep(1)  # Wait for progress dialog to close

            # Create success content
            changes, metrics, next_steps = create_settings_success(
                old_fps=self.pre_operation_metrics.get('fps'),
                new_fps=post_metrics.get('fps'),
                old_latency=self.pre_operation_metrics.get('latency'),
                new_latency=post_metrics.get('latency')
            )

            # Show success dialog
            self.parent.after(0, lambda: show_success_dialog(
                self.parent,
                "Settings Applied Successfully!",
                "Your configuration has been updated and all services are running with the new settings.",
                changes, metrics, next_steps
            ))

        threading.Thread(target=show_success_details, daemon=True).start()

        # Call success callback
        if self.success_callback:
            self.success_callback()

    def _handle_operation_error(self, operation: SettingsOperation, error_message: str):
        """Handle error during operation."""
        if self.current_progress_dialog:
            self.current_progress_dialog.show_completion(
                success=False,
                message=f"Failed during: {operation.name}",
                details=f"Error: {error_message}"
            )

        # Show error notification
        self.notification_manager.show_error(
            f"Settings application failed: {error_message}",
            "Configuration Error"
        )

        # Create error details
        error_details = create_settings_error(
            title="Settings Application Failed",
            message=f"An error occurred while {operation.name.lower()}: {error_message}",
            suggestions=[
                "Check that all services are properly installed",
                "Verify that configuration values are valid",
                "Ensure sufficient system resources are available",
                "Try applying settings again with default values"
            ],
            retry_callback=self._retry_operation,
            defaults_callback=self._apply_defaults
        )

        # Show error recovery dialog
        def show_error_dialog():
            time.sleep(1)  # Wait for progress dialog to close
            self.parent.after(0, lambda: self._show_error_recovery_dialog(error_details))

        threading.Thread(target=show_error_dialog, daemon=True).start()

        # Call error callback
        if self.error_callback:
            self.error_callback(error_message)

    def _handle_critical_error(self, error_message: str):
        """Handle critical error that prevents operation completion."""
        if self.current_progress_dialog:
            self.current_progress_dialog.show_completion(
                success=False,
                message="Critical error occurred",
                details=f"Error: {error_message}"
            )

        # Show critical error notification
        self.notification_manager.show_error(
            f"Critical error: {error_message}",
            "System Error"
        )

        # Call error callback
        if self.error_callback:
            self.error_callback(error_message)

    def _handle_cancellation(self):
        """Handle user cancellation."""
        self.cancelled = True

        if self.current_progress_dialog:
            self.current_progress_dialog.set_state(ProgressState.CANCELLED)

        # Show cancellation notification
        self.notification_manager.show_warning(
            "Settings application cancelled by user",
            "Operation Cancelled"
        )

        # Start rollback process
        self._perform_rollback()

    def _handle_cancellation_cleanup(self):
        """Cleanup after cancellation."""
        if self.current_progress_dialog:
            self.current_progress_dialog.show_completion(
                success=False,
                message="Operation cancelled by user",
                details="Settings were not applied. Previous configuration restored."
            )

        # Call cancel callback
        if self.cancel_callback:
            self.cancel_callback()

    def _perform_rollback(self):
        """Perform rollback to previous configuration."""
        def rollback_operations():
            time.sleep(0.5)  # Brief delay

            if self.current_progress_dialog:
                self.current_progress_dialog.set_state(ProgressState.ROLLED_BACK)
                self.current_progress_dialog.update_progress(
                    0, "Rolling back changes..."
                )

            # Simulate rollback operations
            rollback_steps = [
                "Restoring previous configuration",
                "Restarting services",
                "Verifying rollback"
            ]

            for step in rollback_steps:
                if self.current_progress_dialog:
                    self.current_progress_dialog.update_substep(step)
                time.sleep(0.2)

            self.rollback_performed = True

            # Show rollback notification
            self.notification_manager.show_info(
                "Previous settings have been restored",
                "Rollback Complete"
            )

        threading.Thread(target=rollback_operations, daemon=True).start()

    def _show_error_recovery_dialog(self, error_details: ErrorDetails):
        """Show error recovery dialog."""
        dialog = ErrorRecoveryDialog(self.parent, error_details)
        result = dialog.show()

        # Handle user action
        if result == ActionType.RETRY:
            self._retry_operation()
        elif result == ActionType.USE_DEFAULTS:
            self._apply_defaults()

    def _retry_operation(self, data=None):
        """Retry the failed operation."""
        # Reset state and retry
        self.cancelled = False
        self.apply_settings_with_feedback(
            self.config_changes,
            self.success_callback,
            self.error_callback,
            self.cancel_callback
        )
        return True

    def _apply_defaults(self, data=None):
        """Apply default configuration."""
        # Apply default settings
        default_config = {
            'camera': {'resolution': '640x480', 'fps': 15},
            'detection': {'confidence': 0.5, 'model': 'yolov8n'},
            'ai': {'enabled': False}
        }

        self.apply_settings_with_feedback(
            default_config,
            self.success_callback,
            self.error_callback,
            self.cancel_callback
        )
        return True

    def cleanup(self):
        """Cleanup feedback manager resources."""
        if self.current_progress_dialog:
            self.current_progress_dialog.destroy()
            self.current_progress_dialog = None


class LiveFeedbackIntegration:
    """Integration for live feedback during settings application."""

    def __init__(self, parent_window):
        """Initialize live feedback integration."""
        self.parent = parent_window
        self.live_panel: Optional[LiveStatusPanel] = None
        self.services = ["webcam_service", "detection_service", "ai_service"]

    def create_live_panel(self, parent_frame) -> LiveStatusPanel:
        """Create and return live status panel."""
        self.live_panel = LiveStatusPanel(
            parent_frame,
            services=self.services,
            show_logs=True,
            show_metrics=True
        )

        # Setup service health checks
        self._setup_health_checks()

        return self.live_panel

    def _setup_health_checks(self):
        """Setup health check functions for services."""
        if not self.live_panel:
            return

        # Example health check functions
        def check_webcam_service():
            # Add actual webcam service health check
            return True

        def check_detection_service():
            # Add actual detection service health check
            return True

        def check_ai_service():
            # Add actual AI service health check
            return True

        self.live_panel.set_service_health_check("webcam_service", check_webcam_service)
        self.live_panel.set_service_health_check("detection_service", check_detection_service)
        self.live_panel.set_service_health_check("ai_service", check_ai_service)

    def update_service_status(self, service: str, status: ServiceStatus, error_msg: str = None):
        """Update service status in live panel."""
        if self.live_panel:
            self.live_panel.update_service_status(service, status, error_msg)

    def log_message(self, message: str, level: LogLevel = LogLevel.INFO, source: str = "Settings"):
        """Log message to live panel."""
        if self.live_panel:
            self.live_panel.append_log(message, level, source)

    def update_performance_metrics(self, metrics: Dict[str, float]):
        """Update performance metrics in live panel."""
        if not self.live_panel:
            return

        for metric_name, value in metrics.items():
            unit = ""
            max_value = None

            if "cpu" in metric_name.lower() or "memory" in metric_name.lower():
                unit = "%"
                max_value = 100.0
            elif "fps" in metric_name.lower():
                unit = "FPS"
            elif "latency" in metric_name.lower():
                unit = "ms"

            self.live_panel.update_metric(metric_name, value, unit, max_value)

    def cleanup(self):
        """Cleanup live feedback integration."""
        if self.live_panel:
            self.live_panel.shutdown()


# Global feedback manager instance
_feedback_manager: Optional[SettingsFeedbackManager] = None


def get_feedback_manager(parent_window=None) -> Optional[SettingsFeedbackManager]:
    """Get global feedback manager instance."""
    global _feedback_manager
    if _feedback_manager is None and parent_window:
        _feedback_manager = SettingsFeedbackManager(parent_window)
    return _feedback_manager


def initialize_feedback_system(parent_window, notification_manager=None):
    """Initialize the feedback system."""
    global _feedback_manager
    _feedback_manager = SettingsFeedbackManager(parent_window, notification_manager)
    return _feedback_manager