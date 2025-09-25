"""Demonstration of the User Feedback System integration."""

import tkinter as tk
from tkinter import ttk
import time
import random
from typing import Dict, Any

# Import the feedback system components
from app.ui.feedback_integration import (
    SettingsFeedbackManager, LiveFeedbackIntegration,
    initialize_feedback_system
)
from app.ui.components.notification_manager import (
    NotificationManager, NotificationType, get_notification_manager
)
from app.ui.dialogs.progress_dialog import ProgressState
from app.ui.dialogs.success_dialog import (
    ChangesSummary, PerformanceMetric, NextStep, show_success_dialog
)
from app.ui.dialogs.error_recovery_dialog import (
    ErrorDetails, ErrorSeverity, ActionType, create_settings_error
)
from app.ui.components.live_status_panel import ServiceStatus, LogLevel


class FeedbackSystemDemo:
    """Demo application for the User Feedback System."""

    def __init__(self):
        """Initialize the demo application."""
        self.root = tk.Tk()
        self.root.title("User Feedback System Demo")
        self.root.geometry("800x600")
        self.root.configure(bg='#1e1e1e')

        # Initialize notification system
        self.notification_manager = NotificationManager(self.root)

        # Initialize feedback system
        self.feedback_manager = initialize_feedback_system(
            self.root, self.notification_manager
        )

        # Initialize live feedback
        self.live_feedback = LiveFeedbackIntegration(self.root)

        # Build UI
        self._build_ui()

        # Setup demo data
        self._setup_demo_data()

    def _build_ui(self):
        """Build the demo UI."""
        # Title
        title_label = tk.Label(
            self.root,
            text="User Feedback System Demo",
            font=('Segoe UI', 16, 'bold'),
            fg='#ffffff',
            bg='#1e1e1e'
        )
        title_label.pack(pady=20)

        # Demo buttons frame
        buttons_frame = tk.Frame(self.root, bg='#1e1e1e')
        buttons_frame.pack(pady=20)

        # Progress dialog demo
        progress_btn = tk.Button(
            buttons_frame,
            text="Demo Progress Dialog",
            command=self._demo_progress_dialog,
            bg='#007acc',
            fg='#ffffff',
            font=('Segoe UI', 10),
            padx=20,
            pady=10
        )
        progress_btn.pack(side='left', padx=10)

        # Success dialog demo
        success_btn = tk.Button(
            buttons_frame,
            text="Demo Success Dialog",
            command=self._demo_success_dialog,
            bg='#4caf50',
            fg='#ffffff',
            font=('Segoe UI', 10),
            padx=20,
            pady=10
        )
        success_btn.pack(side='left', padx=10)

        # Error dialog demo
        error_btn = tk.Button(
            buttons_frame,
            text="Demo Error Dialog",
            command=self._demo_error_dialog,
            bg='#f44336',
            fg='#ffffff',
            font=('Segoe UI', 10),
            padx=20,
            pady=10
        )
        error_btn.pack(side='left', padx=10)

        # Settings application demo
        settings_btn = tk.Button(
            buttons_frame,
            text="Demo Settings Application",
            command=self._demo_settings_application,
            bg='#ff9800',
            fg='#ffffff',
            font=('Segoe UI', 10),
            padx=20,
            pady=10
        )
        settings_btn.pack(side='left', padx=10)

        # Notification demos frame
        notifications_frame = tk.Frame(self.root, bg='#1e1e1e')
        notifications_frame.pack(pady=10)

        # Toast notification demos
        toast_label = tk.Label(
            notifications_frame,
            text="Toast Notifications:",
            font=('Segoe UI', 12, 'bold'),
            fg='#ffffff',
            bg='#1e1e1e'
        )
        toast_label.pack()

        toast_buttons_frame = tk.Frame(notifications_frame, bg='#1e1e1e')
        toast_buttons_frame.pack(pady=10)

        # Info toast
        info_btn = tk.Button(
            toast_buttons_frame,
            text="Info",
            command=lambda: self.notification_manager.show_info("This is an info message"),
            bg='#2196f3',
            fg='#ffffff',
            font=('Segoe UI', 9),
            padx=15,
            pady=5
        )
        info_btn.pack(side='left', padx=5)

        # Success toast
        success_toast_btn = tk.Button(
            toast_buttons_frame,
            text="Success",
            command=lambda: self.notification_manager.show_success("Operation completed successfully!"),
            bg='#4caf50',
            fg='#ffffff',
            font=('Segoe UI', 9),
            padx=15,
            pady=5
        )
        success_toast_btn.pack(side='left', padx=5)

        # Warning toast
        warning_btn = tk.Button(
            toast_buttons_frame,
            text="Warning",
            command=lambda: self.notification_manager.show_warning("This is a warning message"),
            bg='#ff9800',
            fg='#ffffff',
            font=('Segoe UI', 9),
            padx=15,
            pady=5
        )
        warning_btn.pack(side='left', padx=5)

        # Error toast
        error_toast_btn = tk.Button(
            toast_buttons_frame,
            text="Error",
            command=lambda: self.notification_manager.show_error("An error occurred"),
            bg='#f44336',
            fg='#ffffff',
            font=('Segoe UI', 9),
            padx=15,
            pady=5
        )
        error_toast_btn.pack(side='left', padx=5)

        # Live status panel
        live_frame = tk.LabelFrame(
            self.root,
            text="Live Status Panel",
            font=('Segoe UI', 10, 'bold'),
            fg='#ffffff',
            bg='#1e1e1e'
        )
        live_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Create live status panel
        self.live_panel = self.live_feedback.create_live_panel(live_frame)
        self.live_panel.pack(fill='both', expand=True)

        # Live demo controls
        live_controls_frame = tk.Frame(live_frame, bg='#1e1e1e')
        live_controls_frame.pack(fill='x', pady=5)

        # Simulate service updates
        service_btn = tk.Button(
            live_controls_frame,
            text="Simulate Service Updates",
            command=self._simulate_service_updates,
            bg='#9c27b0',
            fg='#ffffff',
            font=('Segoe UI', 9),
            padx=15,
            pady=5
        )
        service_btn.pack(side='left', padx=5)

        # Add log entries
        log_btn = tk.Button(
            live_controls_frame,
            text="Add Log Entries",
            command=self._add_demo_logs,
            bg='#607d8b',
            fg='#ffffff',
            font=('Segoe UI', 9),
            padx=15,
            pady=5
        )
        log_btn.pack(side='left', padx=5)

        # Update metrics
        metrics_btn = tk.Button(
            live_controls_frame,
            text="Update Metrics",
            command=self._update_demo_metrics,
            bg='#795548',
            fg='#ffffff',
            font=('Segoe UI', 9),
            padx=15,
            pady=5
        )
        metrics_btn.pack(side='left', padx=5)

    def _setup_demo_data(self):
        """Setup demo data and initial state."""
        # Initialize services in idle state
        services = ["webcam_service", "detection_service", "ai_service"]
        for service in services:
            self.live_feedback.update_service_status(service, ServiceStatus.STOPPED)

        # Add initial log entry
        self.live_feedback.log_message("Demo application started", LogLevel.INFO, "Demo")

        # Set initial metrics
        initial_metrics = {
            "CPU Usage": 25.0,
            "Memory Usage": 40.0,
            "GPU Usage": 15.0,
            "FPS": 15.0,
            "Latency": 50.0
        }
        self.live_feedback.update_performance_metrics(initial_metrics)

    def _demo_progress_dialog(self):
        """Demonstrate progress dialog."""
        # Create sample config changes
        config_changes = {
            'camera': {'resolution': '1920x1080', 'fps': 30},
            'detection': {'confidence': 0.7, 'model': 'yolov8m'},
            'ai': {'enabled': True}
        }

        # Apply settings with feedback
        self.feedback_manager.apply_settings_with_feedback(
            config_changes,
            on_success=lambda: print("Settings applied successfully!"),
            on_error=lambda err: print(f"Settings application failed: {err}"),
            on_cancel=lambda: print("Settings application cancelled")
        )

    def _demo_success_dialog(self):
        """Demonstrate success dialog."""
        # Create sample changes
        changes = [
            ChangesSummary(
                category="Camera Settings",
                changes=[
                    "Resolution changed from 640x480 to 1920x1080",
                    "Frame rate increased from 15 to 30 FPS",
                    "Auto-exposure enabled"
                ],
                icon="📹"
            ),
            ChangesSummary(
                category="Detection Settings",
                changes=[
                    "Confidence threshold increased to 0.7",
                    "Model upgraded to YOLOv8 Medium",
                    "GPU acceleration enabled"
                ],
                icon="🎯"
            ),
            ChangesSummary(
                category="AI Integration",
                changes=[
                    "Gemini AI service enabled",
                    "Real-time analysis activated",
                    "Smart notifications configured"
                ],
                icon="🤖"
            )
        ]

        # Create performance metrics
        metrics = [
            PerformanceMetric(
                name="Frames Per Second",
                old_value=15.0,
                new_value=30.0,
                unit="FPS"
            ),
            PerformanceMetric(
                name="Detection Latency",
                old_value=50.0,
                new_value=25.0,
                unit="ms",
                is_better_higher=False
            ),
            PerformanceMetric(
                name="CPU Usage",
                old_value=45.0,
                new_value=35.0,
                unit="%",
                is_better_higher=False
            )
        ]

        # Create next steps
        next_steps = [
            NextStep(
                title="Test Camera Feed",
                description="Verify that the camera is working with the new settings",
                icon="🧪"
            ),
            NextStep(
                title="Calibrate Detection",
                description="Fine-tune detection parameters for your specific use case",
                icon="🎛"
            ),
            NextStep(
                title="Configure AI Analysis",
                description="Set up AI analysis rules and notification preferences",
                icon="⚙"
            )
        ]

        # Show success dialog
        show_success_dialog(
            self.root,
            "Settings Applied Successfully!",
            "All configuration changes have been applied and services are running optimally.",
            changes,
            metrics,
            next_steps
        )

    def _demo_error_dialog(self):
        """Demonstrate error dialog."""
        error_details = create_settings_error(
            title="Camera Configuration Failed",
            message="Unable to apply camera settings. The selected resolution (4K) is not supported by your camera hardware.",
            suggestions=[
                "Try a lower resolution such as 1920x1080 or 1280x720",
                "Check that your camera supports the selected format",
                "Update camera drivers to the latest version",
                "Verify that no other applications are using the camera"
            ],
            retry_callback=lambda data: self._demo_retry_action(),
            defaults_callback=lambda data: self._demo_defaults_action()
        )

        # Add technical details
        error_details.technical_details = """
Camera Error Details:
====================
Device: USB Camera (VID_1234&PID_5678)
Driver Version: 1.2.3.4
Supported Resolutions: 640x480, 1280x720, 1920x1080
Requested Resolution: 3840x2160 (4K)
Error Code: CAM_RESOLUTION_NOT_SUPPORTED

Stack Trace:
  at CameraService.setResolution(line 123)
  at SettingsManager.applyCameraSettings(line 456)
  at SettingsManager.applyAllSettings(line 789)
"""

        error_details.help_url = "https://docs.example.com/camera-troubleshooting"

        from app.ui.dialogs.error_recovery_dialog import ErrorRecoveryDialog
        dialog = ErrorRecoveryDialog(self.root, error_details)
        result = dialog.show()

        print(f"User selected action: {result}")

    def _demo_retry_action(self):
        """Demo retry action."""
        self.notification_manager.show_info("Retrying camera configuration...")
        return True

    def _demo_defaults_action(self):
        """Demo defaults action."""
        self.notification_manager.show_info("Applying default camera settings...")
        return True

    def _demo_settings_application(self):
        """Demonstrate full settings application process."""
        # Start with some service activity
        self._simulate_service_updates()

        # Wait a moment then start settings application
        self.root.after(1000, self._demo_progress_dialog)

    def _simulate_service_updates(self):
        """Simulate service status updates."""
        services = ["webcam_service", "detection_service", "ai_service"]
        statuses = [ServiceStatus.STARTING, ServiceStatus.RUNNING, ServiceStatus.ERROR, ServiceStatus.RESTARTING]

        for service in services:
            status = random.choice(statuses)
            error_msg = "Simulated error message" if status == ServiceStatus.ERROR else None
            self.live_feedback.update_service_status(service, status, error_msg)

    def _add_demo_logs(self):
        """Add demo log entries."""
        demo_logs = [
            ("Camera service initialized", LogLevel.INFO, "Camera"),
            ("Detection model loaded successfully", LogLevel.INFO, "Detection"),
            ("Warning: High CPU usage detected", LogLevel.WARNING, "Performance"),
            ("AI service connection established", LogLevel.INFO, "AI"),
            ("Debug: Frame processing took 25ms", LogLevel.DEBUG, "Performance"),
            ("Error: Network timeout occurred", LogLevel.ERROR, "Network"),
            ("Configuration backup created", LogLevel.INFO, "Config")
        ]

        for message, level, source in demo_logs:
            self.live_feedback.log_message(message, level, source)
            time.sleep(0.1)

    def _update_demo_metrics(self):
        """Update demo performance metrics."""
        # Generate random metric values
        metrics = {
            "CPU Usage": random.uniform(20, 80),
            "Memory Usage": random.uniform(30, 70),
            "GPU Usage": random.uniform(10, 90),
            "FPS": random.uniform(10, 60),
            "Latency": random.uniform(10, 100)
        }

        self.live_feedback.update_performance_metrics(metrics)

    def run(self):
        """Run the demo application."""
        try:
            self.root.mainloop()
        finally:
            # Cleanup
            self.feedback_manager.cleanup()
            self.live_feedback.cleanup()
            self.notification_manager.shutdown()


if __name__ == "__main__":
    # Run the demo
    demo = FeedbackSystemDemo()
    demo.run()