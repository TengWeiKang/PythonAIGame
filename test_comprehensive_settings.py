#!/usr/bin/env python3
"""
Comprehensive test script for the enhanced settings dialog functionality.
Tests Apply/OK/Cancel buttons, validation, and all settings categories.
"""

import sys
import os
import tkinter as tk
from pathlib import Path

# Add the current directory to path for app imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from app.config.settings import load_config, Config
    from app.ui.dialogs.comprehensive_settings_dialog import ComprehensiveSettingsDialog
    from app.services.webcam_service import WebcamService
    from app.services.gemini_service import GeminiService
    from app.core.logging_config import configure_logging, get_logger
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def test_settings_dialog():
    """Test the comprehensive settings dialog functionality."""

    # Configure basic logging
    configure_logging(log_level="INFO", enable_console_logging=True)
    logger = get_logger(__name__)

    logger.info("Starting comprehensive settings dialog test")

    # Load configuration
    try:
        config = load_config()
        logger.info("✓ Configuration loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load configuration: {e}")
        return False

    # Create main window
    root = tk.Tk()
    root.title("Settings Dialog Test")
    root.geometry("400x300")
    root.configure(bg='#2d2d2d')

    # Mock services (minimal implementations for testing)
    mock_services = {
        'webcam': MockWebcamService(),
        'gemini': MockGeminiService(),
        'main_window': MockMainWindow()
    }

    def open_settings():
        """Open the comprehensive settings dialog."""
        try:
            logger.info("Opening comprehensive settings dialog")
            dialog = ComprehensiveSettingsDialog(
                parent=root,
                config=config,
                services=mock_services,
                callback=lambda cfg: logger.info(f"Settings callback received: {cfg.app_theme}")
            )
            logger.info("✓ Settings dialog created successfully")
        except Exception as e:
            logger.error(f"✗ Failed to create settings dialog: {e}")
            import traceback
            traceback.print_exc()

    def test_config_values():
        """Display current config values for verification."""
        info_window = tk.Toplevel(root)
        info_window.title("Current Settings")
        info_window.geometry("600x400")
        info_window.configure(bg='#2d2d2d')

        text_widget = tk.Text(info_window, bg='#1e1e1e', fg='white', font=('Consolas', 10))
        scrollbar = tk.Scrollbar(info_window, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)

        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Display key settings
        config_info = [
            "=== CURRENT CONFIGURATION VALUES ===\n",
            f"Theme: {getattr(config, 'app_theme', 'Unknown')}\n",
            f"Language: {getattr(config, 'language', 'Unknown')}\n",
            f"Performance Mode: {getattr(config, 'performance_mode', 'Unknown')}\n",
            f"Memory Limit: {getattr(config, 'max_memory_usage_mb', 'Unknown')} MB\n",
            f"Debug Mode: {getattr(config, 'debug', 'Unknown')}\n",
            f"Camera Resolution: {getattr(config, 'camera_width', '?')}x{getattr(config, 'camera_height', '?')}\n",
            f"Camera FPS: {getattr(config, 'camera_fps', 'Unknown')}\n",
            f"Detection Confidence: {getattr(config, 'detection_confidence_threshold', 'Unknown')}\n",
            f"IoU Threshold: {getattr(config, 'detection_iou_threshold', 'Unknown')}\n",
            f"Gemini Model: {getattr(config, 'gemini_model', 'Unknown')}\n",
            f"Gemini Temperature: {getattr(config, 'gemini_temperature', 'Unknown')}\n",
            f"Auto-save Config: {getattr(config, 'auto_save_config', 'Unknown')}\n",
            f"Startup Fullscreen: {getattr(config, 'startup_fullscreen', 'Unknown')}\n",
            f"Remember Window State: {getattr(config, 'remember_window_state', 'Unknown')}\n",
            f"Auto-save Interval: {getattr(config, 'auto_save_interval_minutes', 'Unknown')} min\n",
            "\n=== EXTRA SETTINGS ===\n",
        ]

        for line in config_info:
            text_widget.insert('end', line)

        # Show extra settings if any
        if hasattr(config, 'extra') and config.extra:
            for key, value in config.extra.items():
                text_widget.insert('end', f"{key}: {value}\n")
        else:
            text_widget.insert('end', "No extra settings found\n")

        text_widget.config(state='disabled')

    # Create test UI
    title_label = tk.Label(root, text="Comprehensive Settings Dialog Test",
                          bg='#2d2d2d', fg='white', font=('Segoe UI', 16, 'bold'))
    title_label.pack(pady=20)

    info_label = tk.Label(root,
                         text="Test the enhanced settings dialog with:\n" +
                              "• Apply/OK/Cancel button functionality\n" +
                              "• Real-time validation and feedback\n" +
                              "• All settings categories\n" +
                              "• Live settings updates",
                         bg='#2d2d2d', fg='#cccccc', font=('Segoe UI', 10),
                         justify='left')
    info_label.pack(pady=10)

    # Test buttons
    button_frame = tk.Frame(root, bg='#2d2d2d')
    button_frame.pack(pady=20)

    open_btn = tk.Button(button_frame, text="🔧 Open Settings Dialog",
                        command=open_settings, bg='#007acc', fg='white',
                        font=('Segoe UI', 12), relief='flat', padx=20, pady=10)
    open_btn.pack(side='left', padx=10)

    info_btn = tk.Button(button_frame, text="📊 Show Current Config",
                        command=test_config_values, bg='#28a745', fg='white',
                        font=('Segoe UI', 12), relief='flat', padx=20, pady=10)
    info_btn.pack(side='left', padx=10)

    # Instructions
    instructions = tk.Label(root,
                           text="Instructions:\n" +
                                "1. Click 'Show Current Config' to see initial values\n" +
                                "2. Click 'Open Settings Dialog' to test functionality\n" +
                                "3. Try changing settings and using Apply/OK/Cancel buttons\n" +
                                "4. Test validation by entering invalid values\n" +
                                "5. Check if changes persist by reopening dialog",
                           bg='#2d2d2d', fg='#999999', font=('Segoe UI', 9),
                           justify='left')
    instructions.pack(pady=20)

    logger.info("✓ Test UI created successfully")

    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")

    # Run the test
    logger.info("Starting test application...")
    root.mainloop()

    logger.info("✓ Settings dialog test completed")
    return True


class MockWebcamService:
    """Mock webcam service for testing."""

    def __init__(self):
        self.is_open = False

    def open(self, camera_index, width, height, fps):
        """Mock camera open."""
        self.is_open = True
        return True

    def close(self):
        """Mock camera close."""
        self.is_open = False

    def apply_settings(self, settings):
        """Mock settings application."""
        print(f"Mock: Applied webcam settings: {settings}")


class MockGeminiService:
    """Mock Gemini service for testing."""

    def __init__(self):
        self.configured = False

    def configure(self, api_key, model, temperature, max_tokens):
        """Mock configuration."""
        self.configured = True
        print(f"Mock: Configured Gemini - Model: {model}, Temp: {temperature}")

    def test_connection(self):
        """Mock connection test."""
        return True


class MockMainWindow:
    """Mock main window for testing."""

    def apply_theme(self, theme):
        """Mock theme application."""
        print(f"Mock: Applied theme: {theme}")


if __name__ == "__main__":
    print("=" * 60)
    print("COMPREHENSIVE SETTINGS DIALOG TEST")
    print("=" * 60)

    success = test_settings_dialog()

    if success:
        print("✓ Test completed successfully")
        sys.exit(0)
    else:
        print("✗ Test failed")
        sys.exit(1)