"""Example integration of the Error Recovery System with the main application.

This script demonstrates how to integrate the comprehensive error recovery system
into the main application, including:
- Setting up the error recovery manager
- Configuring safe mode
- Registering services with the recovery system
- Handling errors during settings application
- Using recovery dialogs
- Implementing automatic recovery
"""
import logging
import tkinter as tk
from pathlib import Path
from typing import Dict, Any, Optional

# Import the error recovery system components
from app.core.error_recovery import (
    ErrorRecoveryManager, FailureType, RecoveryStrategy, Severity
)
from app.core.safe_mode import SafeModeManager, SafeModeReason
from app.core.diagnostics import AdvancedDiagnosticEngine
from app.ui.dialogs.recovery_dialog import ErrorRecoveryDialog, SafeModeDialog
from app.config.settings import Config
from app.core.exceptions import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockService:
    """Mock service for demonstration purposes."""

    def __init__(self, name: str, should_fail: bool = False):
        self.name = name
        self.should_fail = should_fail
        self.is_running_flag = True
        self.is_healthy_flag = True

    def is_running(self) -> bool:
        return self.is_running_flag

    def is_healthy(self) -> bool:
        return self.is_healthy_flag and not self.should_fail

    def restart(self):
        if self.should_fail:
            raise ServiceError(f"Failed to restart {self.name} service")
        self.is_running_flag = True

    def stop(self):
        self.is_running_flag = False

    def apply_config(self, config: dict):
        if self.should_fail:
            raise ConfigurationError(f"Failed to apply config to {self.name}")
        logger.info(f"Applied config to {self.name}: {config}")


class MockConfigManager:
    """Mock configuration manager for demonstration."""

    def __init__(self):
        self.current_config = Config()
        self.backup_configs = []

    def get_current_config(self) -> Config:
        return self.current_config

    def apply_config_dict(self, config_dict: Dict[str, Any]):
        logger.info(f"Applying config: {config_dict}")
        # Simulate config application that might fail
        if config_dict.get('simulate_failure'):
            raise ConfigurationError("Simulated configuration failure")

    def get_backup_info(self):
        return self.backup_configs

    def restore_from_backup(self, backup_id: str) -> bool:
        logger.info(f"Restored from backup: {backup_id}")
        return True


class EnhancedApplication:
    """Enhanced application with integrated error recovery system."""

    def __init__(self):
        # Setup data directory
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

        # Initialize core components
        self.config_manager = MockConfigManager()
        self.service_registry = self._initialize_services()

        # Initialize error recovery system
        self.error_recovery_manager = ErrorRecoveryManager(
            self.service_registry,
            self.config_manager,
            self.data_dir
        )

        # Initialize safe mode manager
        self.safe_mode_manager = SafeModeManager(
            self.config_manager,
            self.data_dir
        )

        # Initialize diagnostic engine
        self.diagnostic_engine = AdvancedDiagnosticEngine(self.service_registry)\n        \n        # Setup UI (for recovery dialogs)\n        self.root = None\n        \n        # Register test functions for safe mode\n        self._register_feature_tests()\n        \n        logger.info(\"Enhanced application initialized with error recovery system\")\n    \n    def _initialize_services(self) -> Dict[str, Any]:\n        \"\"\"Initialize mock services.\"\"\"\n        return {\n            'webcam': MockService('webcam'),\n            'detection': MockService('detection'),\n            'gemini': MockService('gemini'),\n            'main_window': MockService('main_window'),\n            'inference': MockService('inference'),\n            'training': MockService('training')\n        }\n    \n    def _register_feature_tests(self):\n        \"\"\"Register test functions for safe mode features.\"\"\"\n        \n        def test_webcam() -> bool:\n            \"\"\"Test webcam functionality.\"\"\"\n            service = self.service_registry.get('webcam')\n            return service and service.is_healthy()\n            \n        def test_detection() -> bool:\n            \"\"\"Test detection service.\"\"\"\n            service = self.service_registry.get('detection')\n            return service and service.is_healthy()\n            \n        def test_gemini() -> bool:\n            \"\"\"Test Gemini AI service.\"\"\"\n            service = self.service_registry.get('gemini')\n            # Simulate API check\n            return service and service.is_healthy()\n            \n        def test_main_ui() -> bool:\n            \"\"\"Test main UI functionality.\"\"\"\n            service = self.service_registry.get('main_window')\n            return service and service.is_healthy()\n            \n        def test_gpu() -> bool:\n            \"\"\"Test GPU availability.\"\"\"\n            try:\n                import torch\n                return torch.cuda.is_available()\n            except ImportError:\n                return False\n        \n        # Register test functions\n        self.safe_mode_manager.register_test_function('webcam', test_webcam)\n        self.safe_mode_manager.register_test_function('detection', test_detection)\n        self.safe_mode_manager.register_test_function('gemini_ai', test_gemini)\n        self.safe_mode_manager.register_test_function('main_ui', test_main_ui)\n        self.safe_mode_manager.register_test_function('gpu_acceleration', test_gpu)\n    \n    def initialize_ui(self):\n        \"\"\"Initialize UI for recovery dialogs.\"\"\"\n        self.root = tk.Tk()\n        self.root.title(\"Enhanced Application\")\n        self.root.geometry(\"800x600\")\n        \n        # Create main menu\n        menubar = tk.Menu(self.root)\n        self.root.config(menu=menubar)\n        \n        # Tools menu\n        tools_menu = tk.Menu(menubar, tearoff=0)\n        menubar.add_cascade(label=\"Tools\", menu=tools_menu)\n        tools_menu.add_command(label=\"Safe Mode Manager\", command=self.show_safe_mode_dialog)\n        tools_menu.add_command(label=\"Run Diagnostics\", command=self.run_diagnostics)\n        tools_menu.add_separator()\n        tools_menu.add_command(label=\"Simulate Error\", command=self.simulate_error)\n        \n        # Status frame\n        status_frame = tk.Frame(self.root)\n        status_frame.pack(fill=tk.X, padx=10, pady=10)\n        \n        tk.Label(status_frame, text=\"Application Status:\", font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W)\n        \n        # Safe mode status\n        safe_mode_info = self.safe_mode_manager.get_safe_mode_info()\n        status_text = \"Safe Mode: ACTIVE\" if safe_mode_info['active'] else \"Safe Mode: INACTIVE\"\n        self.status_label = tk.Label(status_frame, text=status_text, font=('TkDefaultFont', 10))\n        self.status_label.pack(anchor=tk.W)\n        \n        # Recovery statistics\n        stats = self.error_recovery_manager.get_recovery_statistics()\n        stats_text = f\"Recovery Stats - Total Failures: {stats['total_failures']}, Success Rate: {stats['success_rate']:.1%}\"\n        self.stats_label = tk.Label(status_frame, text=stats_text, font=('TkDefaultFont', 9))\n        self.stats_label.pack(anchor=tk.W)\n        \n        # Update status periodically\n        self.update_status()\n    \n    def update_status(self):\n        \"\"\"Update status display.\"\"\"\n        try:\n            # Update safe mode status\n            safe_mode_info = self.safe_mode_manager.get_safe_mode_info()\n            status_text = \"Safe Mode: ACTIVE\" if safe_mode_info['active'] else \"Safe Mode: INACTIVE\"\n            if safe_mode_info['active']:\n                reason = safe_mode_info.get('reason', 'Unknown')\n                duration = safe_mode_info.get('duration_minutes', 0)\n                status_text += f\" ({reason}, {duration:.1f}min)\"\n            self.status_label.config(text=status_text)\n            \n            # Update recovery statistics\n            stats = self.error_recovery_manager.get_recovery_statistics()\n            stats_text = f\"Recovery Stats - Total Failures: {stats['total_failures']}, Success Rate: {stats['success_rate']:.1%}\"\n            self.stats_label.config(text=stats_text)\n            \n        except Exception as e:\n            logger.error(f\"Failed to update status: {e}\")\n        \n        # Schedule next update\n        self.root.after(5000, self.update_status)  # Update every 5 seconds\n    \n    def apply_settings_with_recovery(self, settings_dict: Dict[str, Any]) -> bool:\n        \"\"\"Apply settings with comprehensive error recovery.\"\"\"\n        try:\n            logger.info(f\"Applying settings: {settings_dict}\")\n            \n            # Determine affected services\n            affected_services = self._determine_affected_services(settings_dict)\n            \n            # Apply settings to each service\n            for service_name in affected_services:\n                service = self.service_registry.get(service_name)\n                if service:\n                    service.apply_config(settings_dict)\n            \n            # Apply to config manager\n            self.config_manager.apply_config_dict(settings_dict)\n            \n            logger.info(\"Settings applied successfully\")\n            return True\n            \n        except Exception as e:\n            logger.error(f\"Failed to apply settings: {e}\")\n            \n            # Handle the failure with recovery system\n            context = self.error_recovery_manager.handle_failure(\n                e,\n                operation=\"apply_settings\",\n                affected_services=affected_services,\n                config_changes=settings_dict,\n                user_action=\"apply_settings\"\n            )\n            \n            # Attempt automatic recovery first\n            auto_result = self.error_recovery_manager.attempt_auto_recovery(context)\n            \n            if auto_result and auto_result.success:\n                logger.info(\"Automatic recovery successful\")\n                return True\n            \n            # If auto-recovery failed, show user dialog\n            if self.root:\n                return self._show_recovery_dialog(context)\n            else:\n                logger.error(\"No UI available for user recovery\")\n                return False\n    \n    def _determine_affected_services(self, settings_dict: Dict[str, Any]) -> list:\n        \"\"\"Determine which services are affected by settings changes.\"\"\"\n        affected = []\n        \n        # Camera settings\n        camera_keys = ['camera_width', 'camera_height', 'camera_fps', 'last_webcam_index']\n        if any(key in settings_dict for key in camera_keys):\n            affected.append('webcam')\n        \n        # Detection settings\n        detection_keys = ['detection_confidence_threshold', 'detection_iou_threshold', 'use_gpu']\n        if any(key in settings_dict for key in detection_keys):\n            affected.append('detection')\n        \n        # Gemini settings\n        gemini_keys = ['gemini_api_key', 'gemini_model', 'gemini_timeout']\n        if any(key in settings_dict for key in gemini_keys):\n            affected.append('gemini')\n        \n        # UI settings\n        ui_keys = ['app_theme', 'window_width', 'window_height']\n        if any(key in settings_dict for key in ui_keys):\n            affected.append('main_window')\n        \n        return affected\n    \n    def _show_recovery_dialog(self, context) -> bool:\n        \"\"\"Show recovery dialog to user and handle their choice.\"\"\"\n        try:\n            dialog = ErrorRecoveryDialog(self.root, context)\n            selected_option, user_choice = dialog.show_and_wait()\n            \n            if user_choice == \"execute\" and selected_option:\n                # Execute user-selected recovery\n                result = self.error_recovery_manager.execute_user_recovery(context, selected_option)\n                \n                if result.success:\n                    tk.messagebox.showinfo(\n                        \"Recovery Successful\",\n                        f\"Recovery completed successfully using '{selected_option.title}'\",\n                        parent=self.root\n                    )\n                    return True\n                else:\n                    tk.messagebox.showerror(\n                        \"Recovery Failed\",\n                        f\"Recovery failed: {result.message}\",\n                        parent=self.root\n                    )\n                    return False\n            \n            return False\n            \n        except Exception as e:\n            logger.error(f\"Error in recovery dialog: {e}\")\n            return False\n    \n    def show_safe_mode_dialog(self):\n        \"\"\"Show safe mode management dialog.\"\"\"\n        if self.root:\n            dialog = SafeModeDialog(self.root, self.safe_mode_manager)\n    \n    def run_diagnostics(self):\n        \"\"\"Run comprehensive diagnostics.\"\"\"\n        try:\n            logger.info(\"Running diagnostics...\")\n            report = self.diagnostic_engine.run_full_diagnostic()\n            \n            # Show results in a simple dialog\n            result_window = tk.Toplevel(self.root)\n            result_window.title(\"Diagnostic Results\")\n            result_window.geometry(\"600x400\")\n            \n            text_widget = tk.Text(result_window, wrap=tk.WORD)\n            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)\n            \n            # Display summary\n            text_widget.insert(tk.END, f\"Diagnostic Report\\n\")\n            text_widget.insert(tk.END, f\"Generated: {report.generated_at}\\n\\n\")\n            text_widget.insert(tk.END, f\"Health Score: {report.overall_health_score:.1f}/100\\n\")\n            text_widget.insert(tk.END, f\"Performance Score: {report.performance_score:.1f}/100\\n\")\n            text_widget.insert(tk.END, f\"Stability Score: {report.stability_score:.1f}/100\\n\\n\")\n            \n            # Display findings\n            if report.findings:\n                text_widget.insert(tk.END, f\"Found {len(report.findings)} issues:\\n\\n\")\n                for finding in report.findings[:10]:  # Show top 10\n                    text_widget.insert(tk.END, f\"• {finding.title} ({finding.severity.value})\\n\")\n                    text_widget.insert(tk.END, f\"  {finding.description}\\n\\n\")\n            else:\n                text_widget.insert(tk.END, \"No issues found.\\n\")\n            \n            text_widget.config(state=tk.DISABLED)\n            \n            tk.Button(\n                result_window,\n                text=\"Close\",\n                command=result_window.destroy\n            ).pack(pady=10)\n            \n        except Exception as e:\n            logger.error(f\"Failed to run diagnostics: {e}\")\n            if self.root:\n                tk.messagebox.showerror(\n                    \"Diagnostic Error\",\n                    f\"Failed to run diagnostics: {e}\",\n                    parent=self.root\n                )\n    \n    def simulate_error(self):\n        \"\"\"Simulate various types of errors for testing.\"\"\"\n        if not self.root:\n            return\n            \n        # Create error simulation dialog\n        sim_window = tk.Toplevel(self.root)\n        sim_window.title(\"Simulate Error\")\n        sim_window.geometry(\"400x300\")\n        sim_window.transient(self.root)\n        \n        tk.Label(sim_window, text=\"Select error type to simulate:\", font=('TkDefaultFont', 12)).pack(pady=10)\n        \n        def simulate_validation_error():\n            try:\n                settings = {'camera_width': 9999, 'camera_height': 9999}  # Invalid resolution\n                self.apply_settings_with_recovery(settings)\n            except Exception as e:\n                logger.error(f\"Simulation error: {e}\")\n            sim_window.destroy()\n        \n        def simulate_service_error():\n            try:\n                # Make webcam service fail\n                self.service_registry['webcam'].should_fail = True\n                settings = {'camera_fps': 30}\n                self.apply_settings_with_recovery(settings)\n                # Reset service\n                self.service_registry['webcam'].should_fail = False\n            except Exception as e:\n                logger.error(f\"Simulation error: {e}\")\n            sim_window.destroy()\n        \n        def simulate_network_error():\n            try:\n                settings = {'gemini_api_key': 'invalid_key', 'simulate_failure': True}\n                self.apply_settings_with_recovery(settings)\n            except Exception as e:\n                logger.error(f\"Simulation error: {e}\")\n            sim_window.destroy()\n        \n        def activate_safe_mode():\n            success = self.safe_mode_manager.enter_safe_mode(\n                SafeModeReason.MANUAL_ACTIVATION,\n                \"user_simulation\"\n            )\n            if success:\n                tk.messagebox.showinfo(\n                    \"Safe Mode\",\n                    \"Safe mode activated successfully\",\n                    parent=sim_window\n                )\n            sim_window.destroy()\n        \n        # Error simulation buttons\n        tk.Button(\n            sim_window,\n            text=\"Validation Error\",\n            command=simulate_validation_error,\n            width=20\n        ).pack(pady=5)\n        \n        tk.Button(\n            sim_window,\n            text=\"Service Error\",\n            command=simulate_service_error,\n            width=20\n        ).pack(pady=5)\n        \n        tk.Button(\n            sim_window,\n            text=\"Network Error\",\n            command=simulate_network_error,\n            width=20\n        ).pack(pady=5)\n        \n        tk.Button(\n            sim_window,\n            text=\"Activate Safe Mode\",\n            command=activate_safe_mode,\n            width=20\n        ).pack(pady=5)\n        \n        tk.Button(\n            sim_window,\n            text=\"Cancel\",\n            command=sim_window.destroy,\n            width=20\n        ).pack(pady=10)\n    \n    def run(self):\n        \"\"\"Run the application.\"\"\"\n        try:\n            logger.info(\"Starting enhanced application...\")\n            \n            # Initialize UI\n            self.initialize_ui()\n            \n            # Check if we should start in safe mode\n            if self.safe_mode_manager.is_in_safe_mode():\n                logger.warning(\"Application starting in safe mode\")\n                tk.messagebox.showwarning(\n                    \"Safe Mode\",\n                    \"Application is running in safe mode with limited functionality.\",\n                    parent=self.root\n                )\n            \n            # Start the main loop\n            logger.info(\"Application ready\")\n            self.root.mainloop()\n            \n        except Exception as e:\n            logger.error(f\"Critical application error: {e}\")\n            \n            # Try to enter safe mode as last resort\n            try:\n                self.safe_mode_manager.enter_safe_mode(\n                    SafeModeReason.CRITICAL_ERROR,\n                    \"application_crash\"\n                )\n                logger.info(\"Entered safe mode due to critical error\")\n            except Exception as safe_error:\n                logger.error(f\"Failed to enter safe mode: {safe_error}\")\n    \n    def shutdown(self):\n        \"\"\"Graceful application shutdown.\"\"\"\n        logger.info(\"Shutting down application...\")\n        \n        # Save any pending data\n        try:\n            # Recovery statistics, error patterns, etc. are automatically saved\n            pass\n        except Exception as e:\n            logger.error(f\"Error during shutdown: {e}\")\n        \n        logger.info(\"Application shutdown complete\")\n\n\ndef main():\n    \"\"\"Main entry point.\"\"\"\n    try:\n        # Create and run the enhanced application\n        app = EnhancedApplication()\n        app.run()\n        \n    except Exception as e:\n        logger.error(f\"Failed to start application: {e}\")\n        print(f\"Critical startup error: {e}\")\n        print(\"Please check the logs for more details.\")\n    \n    finally:\n        logger.info(\"Application terminated\")\n\n\nif __name__ == \"__main__\":\n    main()