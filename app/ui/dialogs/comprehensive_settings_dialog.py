"""Comprehensive settings dialog with tabbed interface for all application settings."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import json
import threading
import logging
from typing import Callable, Optional, Dict, Any, List
import cv2
from PIL import Image, ImageTk
import time

logger = logging.getLogger(__name__)


class SettingsValidator:
    """Validator class for settings with predefined valid values."""

    # Valid values for various settings
    VALID_THEMES = ['dark', 'light', 'blue', 'green']
    VALID_LANGUAGES = ['en', 'es', 'fr', 'de', 'zh', 'ja']
    VALID_MODEL_SIZES = [
        'yolo12n', 'yolo12s', 'yolo12m', 'yolo12l', 'yolo12x',
        'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x',
        'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'
    ]
    VALID_GEMINI_MODELS = [
        'gemini-2.5-pro',
        'gemini-2.5-flash',
        'gemini-2.5-flash-lite',
        'gemini-2.0-flash',
        'gemini-2.0-flash-lite',
    ]
    VALID_RESPONSE_FORMATS = ['text', 'markdown', 'json']
    VALID_CHAT_EXPORT_FORMATS = ['txt', 'json', 'csv', 'html']

    @staticmethod
    def validate_confidence(value: float) -> tuple[bool, str]:
        """Validate confidence threshold value."""
        try:
            val = float(value)
            if 0.0 <= val <= 1.0:
                return True, ""
            return False, "Confidence must be between 0.0 and 1.0"
        except (ValueError, TypeError):
            return False, "Confidence must be a number"

    @staticmethod
    def validate_positive_int(value: int, min_val: int = 1, max_val: int = None) -> tuple[bool, str]:
        """Validate positive integer value."""
        try:
            val = int(value)
            if val < min_val:
                return False, f"Value must be at least {min_val}"
            if max_val is not None and val > max_val:
                return False, f"Value must not exceed {max_val}"
            return True, ""
        except (ValueError, TypeError):
            return False, "Value must be an integer"

    @staticmethod
    def validate_api_key(key: str) -> tuple[bool, str]:
        """Validate API key format."""
        if not key or key.strip() == "":
            return False, "API key cannot be empty"
        if len(key.strip()) < 30:
            return False, "API key appears to be invalid (too short)"
        return True, ""

    @staticmethod
    def validate_directory(path: str) -> tuple[bool, str]:
        """Validate directory path."""
        if not path or path.strip() == "":
            return False, "Directory path cannot be empty"
        return True, ""

    @staticmethod
    def validate_persona(text: str) -> tuple[bool, str]:
        """Validate custom persona text."""
        if len(text) > 5000:
            return False, "Persona text cannot exceed 5000 characters"
        return True, ""


class ComprehensiveSettingsDialog:
    """Comprehensive settings dialog with professional tabbed interface."""
    
    # Color scheme matching the main window
    COLORS = {
        'bg_primary': '#1e1e1e',
        'bg_secondary': '#2d2d2d',
        'bg_tertiary': '#3c3c3c',
        'accent_primary': '#007acc',
        'accent_secondary': '#005a9e',
        'text_primary': '#ffffff',
        'text_secondary': '#cccccc',
        'text_muted': '#999999',
        'success': '#4caf50',
        'warning': '#ff9800',
        'error': '#f44336',
        'border': '#404040',
        'button_bg': '#3c3c3c',
        'button_fg': '#ffffff',
    }
    
    def __init__(self, parent: tk.Tk, config: dict, services: Dict[str, Any],
                 callback: Optional[Callable] = None):
        """Initialize the comprehensive settings dialog."""
        self.parent = parent
        self.config = config
        self.services = services
        self.callback = callback

        # Change tracking
        self.has_changes = False
        self.original_values = {}

        # Camera testing state
        self._testing_camera = False
        self._test_thread = None
        self._test_cap = None

        # Create the dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Settings")
        self.dialog.geometry("900x700")
        self.dialog.configure(bg=self.COLORS['bg_primary'])
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)

        # Initialize all settings variables
        self._initialize_variables()

        # Store original values for cancel functionality
        self._store_original_values()

        # Register services (for camera testing, etc.)
        self._register_services()

        # Build the UI
        self._build_ui()

        # Set up change tracking on all variables
        self._setup_change_tracking()

        # Handle dialog close button (X)
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # Center the dialog
        self._center_dialog()

        # Make dialog modal
        self.dialog.grab_set()
        self.dialog.focus_set()

    def _parse_architecture(self, architecture: str) -> tuple[tk.StringVar, tk.StringVar]:
        """Parse architecture string to extract version and size.

        Args:
            architecture: Architecture string like 'yolo11n.pt', 'yolo11n.yaml', 'yolov8s.pt', etc.

        Returns:
            Tuple of (version_var, size_var) StringVars
        """
        try:
            # Remove .pt or .yaml extension
            arch_name = architecture.replace('.pt', '').replace('.yaml', '')

            # Parse version and size
            # Formats: yolo11n, yolov8s, yolo12m, etc.
            if 'yolov8' in arch_name.lower():
                version = 'v8'
                size = arch_name[-1]  # Last character is size
            elif 'yolo11' in arch_name.lower():
                version = 'v11'
                size = arch_name[-1]
            elif 'yolo12' in arch_name.lower():
                version = 'v12'
                size = arch_name[-1]
            else:
                # Default fallback
                version = 'v11'
                size = 'n'

            # Map size to full name with letter
            size_map = {
                'n': 'n', 's': 's', 'm': 'm', 'l': 'l', 'x': 'x'
            }
            size = size_map.get(size.lower(), 'n')

        except Exception as e:
            logger.warning(f"Failed to parse architecture '{architecture}': {e}")
            version = 'v11'
            size = 'n'

        return tk.StringVar(value=version), tk.StringVar(value=size)

    def _get_model_architecture(self) -> str:
        """Construct model architecture filename from version and size.

        Returns:
            Architecture filename like 'yolo11n.pt' (pretrained model for transfer learning)
        """
        try:
            version = self.yolo_version_var.get().replace('v', '')  # v11 -> 11
            size = self.yolo_size_var.get()  # Just the letter (n, s, m, l, x)

            # Construct architecture: yolo + version + size + .pt (default to transfer learning)
            if version == '8':
                # YOLOv8 uses 'yolov8' prefix
                return f"yolov8{size}.pt"
            else:
                # YOLOv11, v12, etc. use 'yolo' prefix
                return f"yolo{version}{size}.pt"

        except Exception as e:
            logger.warning(f"Failed to construct architecture: {e}")
            return "yolo11n.pt"  # Fallback to pretrained model

    def _on_yolo_setting_changed(self, *args):
        """Called when YOLO version or size changes - update architecture display."""
        try:
            # Construct and update architecture display
            architecture = self._get_model_architecture()
            self.architecture_display_var.set(architecture)

            # Also update the model_architecture_var for backward compatibility
            self.model_architecture_var.set(architecture)

            # Trigger general change tracking
            self._on_setting_changed()

        except Exception as e:
            logger.error(f"Error updating YOLO architecture: {e}")

    def _initialize_variables(self):
        """Initialize all Tkinter variables with values from config."""
        # General settings
        self.language_var = tk.StringVar(value=self.config.get('language', 'en'))
        self.data_dir_var = tk.StringVar(value=self.config.get('data_dir', 'data'))
        self.models_dir_var = tk.StringVar(value=self.config.get('models_dir', 'data/models'))
        self.results_dir_var = tk.StringVar(value=self.config.get('results_dir', 'data/results'))

        # Webcam settings
        self.camera_index_var = tk.IntVar(value=self.config.get('last_webcam_index', 0))
        self.camera_device_name_var = tk.StringVar(value=self.config.get('camera_device_name', ''))
        self.video_codec_var = tk.StringVar(value=self.config.get('video_codec', 'Auto'))

        # Analysis settings
        # Detection method: map from config value (yolo/opencv) to display value
        detection_method_config = self.config.get('detection_method', 'yolo').lower()
        detection_method_display = 'YOLO' if detection_method_config == 'yolo' else 'OpenCV Pattern Matching'
        self.detection_method_var = tk.StringVar(value=detection_method_display)

        self.confidence_threshold_var = tk.DoubleVar(
            value=self.config.get('detection_confidence_threshold', 0.8)
        )
        self.iou_threshold_var = tk.DoubleVar(
            value=self.config.get('detection_iou_threshold', 0.5)
        )

        # Training settings
        self.train_epochs_var = tk.IntVar(
            value=self.config.get('train_epochs', 50)
        )
        self.train_batch_size_var = tk.IntVar(
            value=self.config.get('batch_size', 4)  # Default to balanced option (4)
        )
        self.train_device_var = tk.StringVar(
            value=self.config.get('training_device', 'auto')
        )
        self.training_workers_var = tk.IntVar(
            value=self.config.get('training_workers', 2)
        )
        self.training_cache_var = tk.StringVar(
            value=str(self.config.get('training_cache', 'ram'))
        )

        # YOLO Model Selection - New granular controls
        # Extract version and size from existing architecture, or use defaults
        existing_arch = self.config.get('model_architecture', 'yolo11n.pt')
        self.yolo_version_var, self.yolo_size_var = self._parse_architecture(existing_arch)

        # Keep model_architecture_var for backward compatibility
        self.model_architecture_var = tk.StringVar(
            value=self.config.get('model_architecture', 'yolo11n.pt')
        )

        # Chatbot settings
        self.analysis_mode_var = tk.StringVar(
            value=self.config.get('analysis_mode', 'yolo_detection')
        )
        self.api_key_var = tk.StringVar(value=self.config.get('gemini_api_key', ''))
        self.gemini_model_var = tk.StringVar(
            value=self.config.get('gemini_model', 'gemini-2.5-pro')
        )
        self.temperature_var = tk.DoubleVar(
            value=self.config.get('gemini_temperature', 0.7)
        )
        self.max_tokens_var = tk.IntVar(
            value=self.config.get('gemini_max_tokens', 2048)
        )
        self.timeout_var = tk.IntVar(
            value=self.config.get('gemini_timeout', 30)
        )
        self.chatbot_persona_var = tk.StringVar(
            value=self.config.get('chatbot_persona', '')
        )

        # Test result variable
        self.test_result_var = tk.StringVar(value="")

        # Camera info variable
        self.camera_info_var = tk.StringVar(value="Select a camera to test")

        # Architecture display variable (shows constructed architecture)
        self.architecture_display_var = tk.StringVar(value=self._get_model_architecture())

    def _store_original_values(self):
        """Store original values for cancel/revert functionality."""
        self.original_values = {
            'language': self.language_var.get(),
            'data_dir': self.data_dir_var.get(),
            'models_dir': self.models_dir_var.get(),
            'results_dir': self.results_dir_var.get(),
            'camera_index': self.camera_index_var.get(),
            'camera_device_name': self.camera_device_name_var.get(),
            'video_codec': self.video_codec_var.get(),
            'detection_method': self.detection_method_var.get(),
            'confidence_threshold': self.confidence_threshold_var.get(),
            'iou_threshold': self.iou_threshold_var.get(),
            'train_epochs': self.train_epochs_var.get(),
            'train_batch_size': self.train_batch_size_var.get(),
            'train_device': self.train_device_var.get(),
            'model_architecture': self.model_architecture_var.get(),
            'yolo_version': self.yolo_version_var.get(),
            'yolo_size': self.yolo_size_var.get(),
            'analysis_mode': self.analysis_mode_var.get(),
            'api_key': self.api_key_var.get(),
            'gemini_model': self.gemini_model_var.get(),
            'temperature': self.temperature_var.get(),
            'max_tokens': self.max_tokens_var.get(),
            'timeout': self.timeout_var.get(),
            'chatbot_persona': self.chatbot_persona_var.get(),
        }

    def _setup_change_tracking(self):
        """Set up change tracking on all variables."""
        # Track changes on all variables
        variables = [
            self.language_var, self.data_dir_var, self.models_dir_var,
            self.results_dir_var, self.camera_index_var, self.camera_device_name_var,
            self.video_codec_var, self.detection_method_var,
            self.confidence_threshold_var, self.iou_threshold_var,
            self.train_epochs_var, self.train_batch_size_var, self.train_device_var,
            self.model_architecture_var, self.yolo_version_var, self.yolo_size_var,
            self.analysis_mode_var,
            self.api_key_var, self.gemini_model_var, self.temperature_var, self.max_tokens_var,
            self.timeout_var, self.chatbot_persona_var
        ]

        for var in variables:
            var.trace_add('write', self._on_setting_changed)

    def _register_services(self):
        """Register services for use in the dialog."""
        # Store service references if needed
        self.webcam_service = self.services.get('webcam_service')
        self.gemini_service = self.services.get('gemini_service')

    def _center_dialog(self):
        """Center the dialog on the parent window."""
        self.dialog.update_idletasks()
        
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        x = parent_x + (parent_width // 2) - (dialog_width // 2)
        y = parent_y + (parent_height // 2) - (dialog_height // 2)
        
        self.dialog.geometry(f"+{x}+{y}")
    
    def _create_checkbox(self, parent, text, variable, **kwargs):
        """Create a properly configured checkbox with fixed styling."""
        default_config = {
            'bg': self.COLORS['bg_secondary'],
            'fg': self.COLORS['text_primary'],
            'activebackground': self.COLORS['bg_tertiary'],
            'activeforeground': self.COLORS['text_primary'],
            'selectcolor': self.COLORS['bg_tertiary'],  # Background when checked
            'indicatoron': True,  # Ensure checkbox indicator is shown
            'font': ('Segoe UI', 9),
            'anchor': 'w',
            'relief': 'flat',
            'borderwidth': 0,
            'highlightthickness': 0,
            'disabledforeground': self.COLORS['text_muted'],
        }
        
        # Update with any custom kwargs
        default_config.update(kwargs)
        
        checkbox = tk.Checkbutton(
            parent, 
            text=text, 
            variable=variable, 
            **default_config
        )
        
        # Force immediate update of display to ensure proper rendering
        checkbox.update_idletasks()
        
        return checkbox
    
    def _build_ui(self):
        """Build the complete dialog user interface."""
        # Main container
        main_frame = tk.Frame(self.dialog, bg=self.COLORS['bg_primary'])
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="⚙️ Settings",
            bg=self.COLORS['bg_primary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 16, 'bold')
        )
        title_label.pack(pady=(0, 15))
        
        # Settings notebook
        self._build_settings_notebook(main_frame)
        
        # Buttons frame
        self._build_buttons(main_frame)
    
    def _build_settings_notebook(self, parent):
        """Build the tabbed settings notebook."""
        notebook_frame = tk.Frame(parent, bg=self.COLORS['bg_primary'])
        notebook_frame.pack(fill='both', expand=True, pady=(0, 15))
        
        # Create notebook
        self.notebook = ttk.Notebook(notebook_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Configure notebook style
        style = ttk.Style()
        style.configure('Settings.TNotebook', background=self.COLORS['bg_primary'])
        style.configure('Settings.TNotebook.Tab',
                       background=self.COLORS['bg_tertiary'],
                       foreground=self.COLORS['text_primary'],
                       padding=(15, 10))
        style.map('Settings.TNotebook.Tab',
                  background=[('selected', self.COLORS['accent_primary']),
                             ('active', self.COLORS['bg_secondary'])])
        
        # Add tabs
        general_tab = self._build_general_tab()
        self.notebook.add(general_tab, text="🎛️ General")
        
        webcam_tab = self._build_webcam_tab()
        self.notebook.add(webcam_tab, text="📹 Webcam")
        
        analysis_tab = self._build_analysis_tab()
        self.notebook.add(analysis_tab, text="🔍 Analysis")
        
        chatbot_tab = self._build_chatbot_tab()
        self.notebook.add(chatbot_tab, text="🤖 Chatbot")
    
    def _create_section_frame(self, parent, title: str) -> tuple[tk.Frame, tk.Frame]:
        """Create a styled section frame with title and content area."""
        section_frame = tk.Frame(parent, bg=self.COLORS['bg_secondary'], relief='solid', bd=1)
        
        # Section header
        header_frame = tk.Frame(section_frame, bg=self.COLORS['bg_tertiary'], height=35)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame,
            text=title,
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 10, 'bold')
        ).pack(side='left', padx=15, pady=8)
        
        # Content frame
        content_frame = tk.Frame(section_frame, bg=self.COLORS['bg_secondary'])
        content_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        return section_frame, content_frame
    
    def _build_general_tab(self) -> tk.Frame:
        """Build the General settings tab."""
        tab = tk.Frame(self.notebook, bg=self.COLORS['bg_primary'])
        
        # Scrollable frame
        canvas = tk.Canvas(tab, bg=self.COLORS['bg_primary'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.COLORS['bg_primary'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Theme Settings Section
        theme_section, theme_content = self._create_section_frame(scrollable_frame, "🎨 Appearance")
        theme_section.pack(fill='x', pady=(0, 10))
        
        # Language selection
        tk.Label(theme_content, text="Language:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).grid(row=1, column=0, sticky='w', pady=5)
        
        lang_combo = ttk.Combobox(theme_content, textvariable=self.language_var,
                                 values=SettingsValidator.VALID_LANGUAGES, state='readonly')
        lang_combo.grid(row=1, column=1, sticky='w', padx=(10, 0), pady=5)
        
        # Default Directories Section
        dirs_section, dirs_content = self._create_section_frame(scrollable_frame, "📁 Default Directories")
        dirs_section.pack(fill='x', pady=(10, 0))
        
        # Default directories
        dirs_frame = tk.Frame(dirs_content, bg=self.COLORS['bg_secondary'])
        dirs_frame.pack(fill='x', pady=5)
        
        # Data directory
        data_dir_frame = tk.Frame(dirs_frame, bg=self.COLORS['bg_secondary'])
        data_dir_frame.pack(fill='x', pady=2)
        
        tk.Label(data_dir_frame, text="Data Directory:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        data_entry = tk.Entry(data_dir_frame, textvariable=self.data_dir_var,
                             bg=self.COLORS['bg_tertiary'], fg=self.COLORS['text_primary'],
                             font=('Segoe UI', 9), width=30)
        data_entry.pack(side='left', padx=(10, 5))
        
        ttk.Button(data_dir_frame, text="Browse", 
                  command=lambda: self._browse_directory(self.data_dir_var, "Select Data Directory")).pack(side='left')
        
        # Models directory
        models_dir_frame = tk.Frame(dirs_frame, bg=self.COLORS['bg_secondary'])
        models_dir_frame.pack(fill='x', pady=2)
        
        tk.Label(models_dir_frame, text="Models Directory:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        models_entry = tk.Entry(models_dir_frame, textvariable=self.models_dir_var,
                               bg=self.COLORS['bg_tertiary'], fg=self.COLORS['text_primary'],
                               font=('Segoe UI', 9), width=30)
        models_entry.pack(side='left', padx=(10, 5))
        
        ttk.Button(models_dir_frame, text="Browse", 
                  command=lambda: self._browse_directory(self.models_dir_var, "Select Models Directory")).pack(side='left')
        
        # Results directory
        results_dir_frame = tk.Frame(dirs_frame, bg=self.COLORS['bg_secondary'])
        results_dir_frame.pack(fill='x', pady=2)
        
        tk.Label(results_dir_frame, text="Results Directory:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        results_entry = tk.Entry(results_dir_frame, textvariable=self.results_dir_var,
                                bg=self.COLORS['bg_tertiary'], fg=self.COLORS['text_primary'],
                                font=('Segoe UI', 9), width=30)
        results_entry.pack(side='left', padx=(10, 5))
        
        ttk.Button(results_dir_frame, text="Browse", 
                  command=lambda: self._browse_directory(self.results_dir_var, "Select Results Directory")).pack(side='left')
        
        return tab
    
    def _build_webcam_tab(self) -> tk.Frame:
        """Build the Webcam settings tab."""
        tab = tk.Frame(self.notebook, bg=self.COLORS['bg_primary'])
        
        # Scrollable frame
        canvas = tk.Canvas(tab, bg=self.COLORS['bg_primary'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.COLORS['bg_primary'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Device Selection Section
        device_section, device_content = self._create_section_frame(scrollable_frame, "📱 Camera Device")
        device_section.pack(fill='x', pady=(0, 10))

        # Camera selection with dropdown
        cam_frame = tk.Frame(device_content, bg=self.COLORS['bg_secondary'])
        cam_frame.pack(fill='x', pady=5)

        tk.Label(cam_frame, text="Camera Device:", bg=self.COLORS['bg_secondary'],
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')

        # Camera dropdown - will be populated by _detect_cameras
        self.camera_combo = ttk.Combobox(cam_frame, state='readonly', width=40)
        self.camera_combo.pack(side='left', padx=(10, 5))
        self.camera_combo.bind('<<ComboboxSelected>>', self._on_camera_selected)

        # Detect cameras button (also used for refresh)
        self.detect_cameras_button = ttk.Button(cam_frame, text="🔄 Detect Cameras", command=self._detect_cameras)
        self.detect_cameras_button.pack(side='left', padx=(5, 0))

        # Camera info display
        self.camera_name_label = tk.Label(device_content, text="Click 'Detect Cameras' to find available devices",
                                         bg=self.COLORS['bg_secondary'],
                                         fg=self.COLORS['text_muted'], font=('Segoe UI', 8))
        self.camera_name_label.pack(anchor='w', pady=(0, 5))

        # Store camera list for reference
        self._available_cameras = []

        # NOTE: Auto-detection REMOVED for performance optimization
        # Camera detection is now triggered ONLY when user clicks "Detect Cameras" button
        # This significantly speeds up settings dialog opening (no blocking cv2.VideoCapture calls)

        # Video Codec Settings Section
        codec_section, codec_content = self._create_section_frame(scrollable_frame, "🎬 Video Codec")
        codec_section.pack(fill='x', pady=(0, 10))

        # Codec selection
        codec_frame = tk.Frame(codec_content, bg=self.COLORS['bg_secondary'])
        codec_frame.pack(fill='x', pady=5)

        tk.Label(codec_frame, text="Video Codec:", bg=self.COLORS['bg_secondary'],
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')

        # Codec dropdown with common codecs
        codec_options = [
            "Auto",      # Let OpenCV choose automatically (default)
            "MJPG",      # Motion JPEG (most compatible, good quality)
            "YUYV",      # YUV 4:2:2 (uncompressed, high bandwidth)
            "H264",      # H.264/AVC (compressed, modern)
            "VP8",       # VP8 codec (WebM)
            "I420",      # YUV 4:2:0 planar
            "RGB3",      # RGB24 (uncompressed)
            "GREY",      # Grayscale
            "NV12",      # NV12 format
            "UYVY"       # UYVY format
        ]

        self.codec_combo = ttk.Combobox(codec_frame, textvariable=self.video_codec_var,
                                       values=codec_options, state='readonly', width=15)
        self.codec_combo.pack(side='left', padx=(10, 0))

        # Codec description
        codec_desc = tk.Label(
            codec_content,
            text="Video codec format for camera stream. MJPG recommended for best compatibility. Auto lets the system choose.",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8),
            justify='left',
            wraplength=500
        )
        codec_desc.pack(anchor='w', pady=(0, 5), padx=(0, 0))

        # Live Camera Preview Section - moved here per requirements
        preview_section, preview_content = self._create_section_frame(scrollable_frame, "📹 Live Camera Preview")
        preview_section.pack(fill='x', pady=(0, 10))
        
        # Preview canvas
        canvas_frame = tk.Frame(preview_content, bg=self.COLORS['bg_primary'], relief='sunken', bd=1)
        canvas_frame.pack(fill='x', pady=5)
        
        self._preview_canvas = tk.Canvas(
            canvas_frame,
            width=400,
            height=240,
            bg='black',
            highlightthickness=1,
            highlightbackground=self.COLORS['border']
        )
        self._preview_canvas.pack()
        
        # Display placeholder text
        self._preview_canvas.create_text(
            200, 120, 
            text="Camera Preview\nSelect camera and click 'Test' to preview",
            fill='white', 
            font=('Segoe UI', 12),
            justify='center'
        )
        
        # Preview controls
        preview_controls = tk.Frame(preview_content, bg=self.COLORS['bg_secondary'])
        preview_controls.pack(fill='x', pady=(5, 0))
        
        self.camera_test_button = ttk.Button(
            preview_controls,
            text="🎥 Test Camera",
            command=self._test_camera
        )
        self.camera_test_button.pack(side='left', padx=(0, 5))
        
        self.stop_test_button = ttk.Button(
            preview_controls,
            text="⏹ Stop Test",
            command=self._stop_test,
            state='disabled'
        )
        self.stop_test_button.pack(side='left', padx=5)
        
        # Camera info display
        self.camera_info_var = tk.StringVar(value="Select a camera to test")
        info_label = tk.Label(
            preview_content,
            textvariable=self.camera_info_var,
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8)
        )
        info_label.pack(pady=(5, 0))
        
        return tab
    
    def _build_analysis_tab(self) -> tk.Frame:
        """Build the Image Analysis settings tab."""
        tab = tk.Frame(self.notebook, bg=self.COLORS['bg_primary'])
        
        # Scrollable frame
        canvas = tk.Canvas(tab, bg=self.COLORS['bg_primary'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.COLORS['bg_primary'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Detection Settings Section
        detection_section, detection_content = self._create_section_frame(scrollable_frame, "🎯 Detection Settings")
        detection_section.pack(fill='x', pady=(0, 10))

        # Detection Method Selection
        method_frame = tk.Frame(detection_content, bg=self.COLORS['bg_secondary'])
        method_frame.pack(fill='x', pady=5)

        tk.Label(method_frame, text="Detection Method:", bg=self.COLORS['bg_secondary'],
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9), width=20, anchor='w').pack(side='left')

        self.detection_method_combo = ttk.Combobox(
            method_frame,
            textvariable=self.detection_method_var,
            values=['YOLO', 'OpenCV Pattern Matching'],
            state='readonly',
            width=25
        )
        self.detection_method_combo.pack(side='left', padx=(5, 0))

        # Description for detection method
        method_desc = tk.Label(
            detection_content,
            text="Choose detection algorithm: YOLO (deep learning, faster, better accuracy) or OpenCV Pattern Matching (classical CV, no GPU needed, good for exact object matching).",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8),
            justify='left',
            wraplength=500
        )
        method_desc.pack(anchor='w', pady=(0, 10), padx=(0, 0))

        # Confidence threshold
        conf_frame = tk.Frame(detection_content, bg=self.COLORS['bg_secondary'])
        conf_frame.pack(fill='x', pady=5)

        tk.Label(conf_frame, text="Confidence Threshold:", bg=self.COLORS['bg_secondary'],
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9), width=20, anchor='w').pack(side='left')

        conf_scale = tk.Scale(conf_frame, from_=0.0, to=1.0, resolution=0.01, orient='horizontal',
                            variable=self.confidence_threshold_var, bg=self.COLORS['bg_secondary'],
                            fg=self.COLORS['text_primary'], highlightthickness=0, length=200)
        conf_scale.pack(side='left', padx=(5, 0))

        # Description for confidence threshold
        conf_desc = tk.Label(
            detection_content,
            text="Minimum confidence score (0.0-1.0) for an object to be detected.\nHigher values = fewer false positives but may miss objects. Recommended: 0.5 for general use, 0.3 for more detections.",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8),
            justify='left',
            wraplength=500
        )
        conf_desc.pack(anchor='w', pady=(0, 10), padx=(0, 0))

        # IoU threshold
        iou_frame = tk.Frame(detection_content, bg=self.COLORS['bg_secondary'])
        iou_frame.pack(fill='x', pady=5)

        tk.Label(iou_frame, text="IoU Threshold:", bg=self.COLORS['bg_secondary'],
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9), width=20, anchor='w').pack(side='left')

        iou_scale = tk.Scale(iou_frame, from_=0.0, to=1.0, resolution=0.01, orient='horizontal',
                           variable=self.iou_threshold_var, bg=self.COLORS['bg_secondary'],
                           fg=self.COLORS['text_primary'], highlightthickness=0, length=200)
        iou_scale.pack(side='left', padx=(5, 0))

        # Description for IoU threshold
        iou_desc = tk.Label(
            detection_content,
            text="Intersection over Union threshold (0.0-1.0) for removing duplicate detections.\nHigher values allow more overlapping boxes. Recommended: 0.45",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8),
            justify='left',
            wraplength=500
        )
        iou_desc.pack(anchor='w', pady=(0, 10), padx=(0, 0))

        # Training Settings Section
        training_section, training_content = self._create_section_frame(scrollable_frame, "🎓 YOLO Training Settings")
        training_section.pack(fill='x', pady=(0, 10))

        # Epochs slider
        epochs_frame = tk.Frame(training_content, bg=self.COLORS['bg_secondary'])
        epochs_frame.pack(fill='x', pady=5)

        tk.Label(epochs_frame, text="Training Epochs:", bg=self.COLORS['bg_secondary'],
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9), width=20, anchor='w').pack(side='left')

        epochs_scale = tk.Scale(epochs_frame, from_=1, to=500, resolution=1, orient='horizontal',
                               variable=self.train_epochs_var, bg=self.COLORS['bg_secondary'],
                               fg=self.COLORS['text_primary'], highlightthickness=0, length=200)
        epochs_scale.pack(side='left', padx=(5, 0))

        # Description for epochs
        epochs_desc = tk.Label(
            training_content,
            text="Number of complete passes through the training dataset. Transfer learning (using .pt files) typically needs 30-50 epochs. Training from scratch (using .yaml files) requires 100-200 epochs. Recommended: 50 for transfer learning, 150 for training from scratch.",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8),
            justify='left',
            wraplength=500
        )
        epochs_desc.pack(anchor='w', pady=(0, 10), padx=(0, 0))

        # Batch size dropdown with enhanced options
        batch_frame = tk.Frame(training_content, bg=self.COLORS['bg_secondary'])
        batch_frame.pack(fill='x', pady=5)

        tk.Label(batch_frame, text="Batch Size:", bg=self.COLORS['bg_secondary'],
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9), width=20, anchor='w').pack(side='left')

        # Expanded batch size options from 1 to 64
        self.batch_combo = ttk.Combobox(batch_frame, textvariable=self.train_batch_size_var,
                                   values=[1, 2, 4, 8, 16, 32, 64], state='readonly', width=10)
        self.batch_combo.pack(side='left', padx=(5, 0))

        # Memory requirement label
        self.batch_memory_label = tk.Label(
            batch_frame,
            text="",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8, 'italic')
        )
        self.batch_memory_label.pack(side='left', padx=(10, 0))

        # Auto-detect button
        auto_detect_btn = tk.Button(
            batch_frame,
            text="Auto-Detect",
            bg=self.COLORS['button_bg'],
            fg=self.COLORS['button_fg'],
            font=('Segoe UI', 8),
            relief='raised',
            cursor='hand2',
            command=self._apply_recommended_batch_size
        )
        auto_detect_btn.pack(side='left', padx=(10, 0))

        # Dynamic description for batch size
        self.batch_desc_label = tk.Label(
            training_content,
            text="",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8),
            justify='left',
            wraplength=500
        )
        self.batch_desc_label.pack(anchor='w', pady=(2, 0), padx=(0, 0))

        # Static information
        batch_info = tk.Label(
            training_content,
            text="Number of images processed together. Larger batch = faster training but more memory.",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8),
            justify='left',
            wraplength=500
        )
        batch_info.pack(anchor='w', pady=(0, 10), padx=(0, 0))

        # Bind batch size change event
        self.batch_combo.bind('<<ComboboxSelected>>', self._on_batch_size_change)

        # Initialize batch size display
        self._on_batch_size_change()

        # YOLO Model Configuration Section Header
        model_config_label = tk.Label(
            training_content,
            text="YOLO Model Configuration",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['accent_primary'],
            font=('Segoe UI', 10, 'bold')
        )
        model_config_label.pack(anchor='w', pady=(10, 5))

        # YOLO Version dropdown
        version_frame = tk.Frame(training_content, bg=self.COLORS['bg_secondary'])
        version_frame.pack(fill='x', pady=5)

        tk.Label(version_frame, text="YOLO Version:", bg=self.COLORS['bg_secondary'],
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9), width=20, anchor='w').pack(side='left')

        version_combo = ttk.Combobox(version_frame, textvariable=self.yolo_version_var,
                                     values=['v8', 'v11', 'v12'],
                                     state='readonly', width=10)
        version_combo.pack(side='left', padx=(5, 0))
        version_combo.bind('<<ComboboxSelected>>', self._on_yolo_setting_changed)

        # Version description
        version_desc = tk.Label(
            training_content,
            text="• YOLOv8: Stable, widely tested, excellent documentation and community support\n• YOLOv11: Latest improvements, better accuracy with similar speed to v8 (Recommended)\n• YOLOv12: Newest version with cutting-edge features and optimizations",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8),
            justify='left',
            wraplength=500
        )
        version_desc.pack(anchor='w', pady=(0, 10), padx=(20, 0))

        # Model Size dropdown
        size_frame = tk.Frame(training_content, bg=self.COLORS['bg_secondary'])
        size_frame.pack(fill='x', pady=5)

        tk.Label(size_frame, text="Model Size:", bg=self.COLORS['bg_secondary'],
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9), width=20, anchor='w').pack(side='left')

        size_combo = ttk.Combobox(size_frame, textvariable=self.yolo_size_var,
                                  values=['n', 's', 'm', 'l', 'x'],
                                  state='readonly', width=10)
        size_combo.pack(side='left', padx=(5, 0))
        size_combo.bind('<<ComboboxSelected>>', self._on_yolo_setting_changed)

        # Size description
        size_desc = tk.Label(
            training_content,
            text="• nano (n): ~1.8M params, fastest training, lowest accuracy - ideal for testing and prototyping\n• small (s): ~11M params, good balance of speed and accuracy - recommended for most use cases\n• medium (m): ~25M params, better accuracy, moderate GPU memory usage\n• large (l): ~43M params, high accuracy, requires 6GB+ GPU memory\n• xlarge (x): ~68M params, best accuracy, needs 8GB+ GPU memory, slowest training",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8),
            justify='left',
            wraplength=500
        )
        size_desc.pack(anchor='w', pady=(0, 10), padx=(20, 0))

        # Architecture preview (shows what will be used)
        preview_frame = tk.Frame(training_content, bg=self.COLORS['bg_secondary'])
        preview_frame.pack(fill='x', pady=5)

        tk.Label(preview_frame, text="Architecture File:", bg=self.COLORS['bg_secondary'],
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9), width=20, anchor='w').pack(side='left')

        tk.Label(preview_frame, textvariable=self.architecture_display_var,
                bg=self.COLORS['bg_tertiary'],
                fg=self.COLORS['accent_primary'],
                font=('Segoe UI', 9, 'bold'),
                relief='sunken',
                padx=10, pady=3).pack(side='left', padx=(5, 0))

        # Important note about training modes
        scratch_note = tk.Label(
            training_content,
            text="TRANSFER LEARNING (.pt files): Uses pretrained weights, needs 30-50 epochs, works with 5-10 images per class.\nFROM SCRATCH (.yaml files): Random weights, needs 100+ epochs and more data, but fully custom to your dataset.",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['warning'],
            font=('Segoe UI', 8, 'italic'),
            justify='left',
            wraplength=500
        )
        scratch_note.pack(anchor='w', pady=(5, 10), padx=(0, 0))

        # Training device dropdown
        device_frame = tk.Frame(training_content, bg=self.COLORS['bg_secondary'])
        device_frame.pack(fill='x', pady=5)

        tk.Label(device_frame, text="Training Device:", bg=self.COLORS['bg_secondary'],
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9), width=20, anchor='w').pack(side='left')

        device_combo = ttk.Combobox(device_frame, textvariable=self.train_device_var,
                                    values=['auto', 'cuda', 'mps', 'cpu'], state='readonly', width=15)
        device_combo.pack(side='left', padx=(5, 0))

        # Detect current device and show as label
        device_detect_button = ttk.Button(
            device_frame,
            text="🔍 Detect",
            command=self._detect_training_device,
            width=10
        )
        device_detect_button.pack(side='left', padx=(5, 0))

        self.detected_device_label = tk.Label(
            device_frame,
            text="",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8)
        )
        self.detected_device_label.pack(side='left', padx=(10, 0))

        # Description for training device
        device_desc = tk.Label(
            training_content,
            text="Device for model training:\n• Auto: Automatically detect and use the best available device (Recommended)\n• CUDA: Use NVIDIA GPU (requires CUDA-compatible GPU)\n• MPS: Use Apple Metal (Mac only, requires M1/M2/M3 chip)\n• CPU: Use CPU only (slower but works everywhere)\n\nGPU training is typically 10-50x faster than CPU training.",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8),
            justify='left',
            wraplength=500
        )
        device_desc.pack(anchor='w', pady=(0, 10), padx=(0, 0))

        # Performance Optimization Section Header
        perf_config_label = tk.Label(
            training_content,
            text="Performance Optimization Settings",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['accent_primary'],
            font=('Segoe UI', 10, 'bold')
        )
        perf_config_label.pack(anchor='w', pady=(10, 5))

        # Workers dropdown
        workers_frame = tk.Frame(training_content, bg=self.COLORS['bg_secondary'])
        workers_frame.pack(fill='x', pady=5)

        tk.Label(workers_frame, text="Data Loading Workers:", bg=self.COLORS['bg_secondary'],
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9), width=20, anchor='w').pack(side='left')

        workers_combo = ttk.Combobox(workers_frame, textvariable=self.training_workers_var,
                                     values=[0, 1, 2, 4], state='readonly', width=10)
        workers_combo.pack(side='left', padx=(5, 0))

        # Description for workers
        workers_desc = tk.Label(
            training_content,
            text="Number of parallel threads for loading training data. More workers = better GPU utilization but more CPU/memory usage.\nRecommended: 2 for most systems (good balance), 0 only if memory errors occur, 4 for high-end systems",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8),
            justify='left',
            wraplength=500
        )
        workers_desc.pack(anchor='w', pady=(0, 10), padx=(0, 0))

        # Cache mode dropdown
        cache_frame = tk.Frame(training_content, bg=self.COLORS['bg_secondary'])
        cache_frame.pack(fill='x', pady=5)

        tk.Label(cache_frame, text="Dataset Caching:", bg=self.COLORS['bg_secondary'],
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9), width=20, anchor='w').pack(side='left')

        cache_combo = ttk.Combobox(cache_frame, textvariable=self.training_cache_var,
                                   values=['False', 'ram', 'disk', 'auto'], state='readonly', width=10)
        cache_combo.pack(side='left', padx=(5, 0))

        # Description for cache
        cache_desc = tk.Label(
            training_content,
            text="Cache training images in memory for faster epochs (after first epoch).\n• False: No caching, load from disk every epoch (slowest but lowest memory)\n• ram: Cache in RAM (fastest, recommended for small datasets <500 images)\n• disk: Cache on disk (moderate speed, low memory)\n• auto: Automatically choose based on dataset size",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8),
            justify='left',
            wraplength=500
        )
        cache_desc.pack(anchor='w', pady=(0, 10), padx=(0, 0))

        # # Region of Interest Section
        # roi_section, roi_content = self._create_section_frame(scrollable_frame, "📐 Region of Interest")
        # roi_section.pack(fill='x', pady=(0, 10))
        
        # # Enable ROI
        # self._create_checkbox(roi_content, "Enable Region of Interest (ROI)",
        #                      self.enable_roi_var).pack(anchor='w', pady=5)
        
        # # ROI coordinates
        # roi_coord_frame = tk.Frame(roi_content, bg=self.COLORS['bg_secondary'])
        # roi_coord_frame.pack(fill='x', pady=5)
        
        # # X and Y
        # tk.Label(roi_coord_frame, text="X:", bg=self.COLORS['bg_secondary'], 
        #         fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).grid(row=0, column=0, padx=(0, 5))
        
        # tk.Spinbox(roi_coord_frame, from_=0, to=4096, textvariable=self.roi_x_var,
        #           bg=self.COLORS['bg_tertiary'], fg=self.COLORS['text_primary'], width=8).grid(row=0, column=1, padx=(0, 10))
        
        # tk.Label(roi_coord_frame, text="Y:", bg=self.COLORS['bg_secondary'], 
        #         fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).grid(row=0, column=2, padx=(0, 5))
        
        # tk.Spinbox(roi_coord_frame, from_=0, to=4096, textvariable=self.roi_y_var,
        #           bg=self.COLORS['bg_tertiary'], fg=self.COLORS['text_primary'], width=8).grid(row=0, column=3)
        
        # # Width and Height
        # tk.Label(roi_coord_frame, text="Width:", bg=self.COLORS['bg_secondary'], 
        #         fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).grid(row=1, column=0, padx=(0, 5), pady=(5, 0))
        
        # tk.Spinbox(roi_coord_frame, from_=0, to=4096, textvariable=self.roi_width_var,
        #           bg=self.COLORS['bg_tertiary'], fg=self.COLORS['text_primary'], width=8).grid(row=1, column=1, padx=(0, 10), pady=(5, 0))
        
        # tk.Label(roi_coord_frame, text="Height:", bg=self.COLORS['bg_secondary'], 
        #         fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).grid(row=1, column=2, padx=(0, 5), pady=(5, 0))
        
        # tk.Spinbox(roi_coord_frame, from_=0, to=4096, textvariable=self.roi_height_var,
        #           bg=self.COLORS['bg_tertiary'], fg=self.COLORS['text_primary'], width=8).grid(row=1, column=3, pady=(5, 0))
        
        # Model and Export Section
        model_section, model_content = self._create_section_frame(scrollable_frame, "⚙️ Model & Export")
        model_section.pack(fill='x', pady=(0, 10))

        # Model info label
        model_info_frame = tk.Frame(model_content, bg=self.COLORS['bg_secondary'])
        model_info_frame.pack(fill='x', pady=(5, 5))

        model_info = tk.Label(
            model_info_frame,
            text="Model: Always uses trained model at data/models/model.pt\nTrain your custom model in the Objects tab before using AI analysis.",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 9),
            justify='left'
        )
        model_info.pack(anchor='w')

        # Export settings - Format removed, quality fixed at 100%
        export_frame = tk.Frame(model_content, bg=self.COLORS['bg_secondary'])
        export_frame.pack(fill='x', pady=5)

        tk.Label(export_frame, text="Export Quality: 100% (PNG format)", bg=self.COLORS['bg_secondary'],
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        # # Difference Detection Section
        # diff_section, diff_content = self._create_section_frame(scrollable_frame, "🔍 Difference Detection")
        # diff_section.pack(fill='x')
        
        # # Sensitivity
        # sens_frame = tk.Frame(diff_content, bg=self.COLORS['bg_secondary'])
        # sens_frame.pack(fill='x', pady=5)
        
        # tk.Label(sens_frame, text="Sensitivity:", bg=self.COLORS['bg_secondary'], 
        #         fg=self.COLORS['text_primary'], font=('Segoe UI', 9), width=15, anchor='w').pack(side='left')
        
        # sens_scale = tk.Scale(sens_frame, from_=0.0, to=1.0, resolution=0.01, orient='horizontal',
        #                     variable=self.difference_sensitivity_var, bg=self.COLORS['bg_secondary'],
        #                     fg=self.COLORS['text_primary'], highlightthickness=0, length=200)
        # sens_scale.pack(side='left', padx=(5, 0))
        
        # # Highlight differences
        # self._create_checkbox(diff_content, "Highlight differences automatically",
        #                      self.highlight_differences_var).pack(anchor='w', pady=5)
        
        return tab
    
    def _build_chatbot_tab(self) -> tk.Frame:
        """Build the Chatbot settings tab."""
        tab = tk.Frame(self.notebook, bg=self.COLORS['bg_primary'])
        
        # Scrollable frame
        canvas = tk.Canvas(tab, bg=self.COLORS['bg_primary'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.COLORS['bg_primary'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # API Configuration Section
        api_section, api_content = self._create_section_frame(scrollable_frame, "🔑 API Configuration")
        api_section.pack(fill='x', pady=(0, 10))
        
        # API Key
        api_frame = tk.Frame(api_content, bg=self.COLORS['bg_secondary'])
        api_frame.pack(fill='x', pady=5)
        
        tk.Label(api_frame, text="Gemini API Key:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(anchor='w', pady=(0, 5))
        
        key_entry_frame = tk.Frame(api_frame, bg=self.COLORS['bg_secondary'])
        key_entry_frame.pack(fill='x')
        
        self.api_key_entry = tk.Entry(key_entry_frame, textvariable=self.api_key_var,
                                     bg=self.COLORS['bg_tertiary'], fg=self.COLORS['text_primary'],
                                     font=('Segoe UI', 9), show='*', borderwidth=0,
                                     insertbackground=self.COLORS['text_primary'])
        self.api_key_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        # Show/Hide API key button
        self.show_key_button = tk.Button(key_entry_frame, text="👁", bg=self.COLORS['bg_tertiary'],
                                        fg=self.COLORS['text_primary'], borderwidth=0,
                                        command=self._toggle_api_key_visibility, font=('Segoe UI', 8), width=3)
        self.show_key_button.pack(side='right')
        
        # API help text
        help_label = tk.Label(api_content, text="Get your free API key from Google AI Studio:\nhttps://makersuite.google.com/app/apikey",
                             bg=self.COLORS['bg_secondary'], fg=self.COLORS['text_muted'],
                             font=('Segoe UI', 8), justify='left')
        help_label.pack(anchor='w', pady=(5, 0))
        
        # Model Configuration Section
        model_section, model_content = self._create_section_frame(scrollable_frame, "🤖 Model Configuration")
        model_section.pack(fill='x', pady=(0, 10))
        
        # Model selection
        model_frame = tk.Frame(model_content, bg=self.COLORS['bg_secondary'])
        model_frame.pack(fill='x', pady=5)
        
        tk.Label(model_frame, text="Model:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        gemini_model_combo = ttk.Combobox(model_frame, textvariable=self.gemini_model_var,
                                         values=SettingsValidator.VALID_GEMINI_MODELS, state='readonly', width=20)
        gemini_model_combo.pack(side='left', padx=(10, 0))
        
        # Temperature
        temp_frame = tk.Frame(model_content, bg=self.COLORS['bg_secondary'])
        temp_frame.pack(fill='x', pady=5)
        
        tk.Label(temp_frame, text="Creativity (Temperature):", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9), width=20, anchor='w').pack(side='left')
        
        temp_scale = tk.Scale(temp_frame, from_=0.0, to=1.0, resolution=0.1, orient='horizontal',
                            variable=self.temperature_var, bg=self.COLORS['bg_secondary'],
                            fg=self.COLORS['text_primary'], highlightthickness=0, length=200)
        temp_scale.pack(side='left', padx=(5, 0))
        
        # Max tokens
        tokens_frame = tk.Frame(model_content, bg=self.COLORS['bg_secondary'])
        tokens_frame.pack(fill='x', pady=5)
        
        tk.Label(tokens_frame, text="Max Tokens:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        tokens_spin = tk.Spinbox(tokens_frame, from_=100, to=8192, increment=256,
                               textvariable=self.max_tokens_var, bg=self.COLORS['bg_tertiary'],
                               fg=self.COLORS['text_primary'], width=8)
        tokens_spin.pack(side='left', padx=(10, 20))
        
        # Timeout
        tk.Label(tokens_frame, text="Timeout (s):", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        timeout_spin = tk.Spinbox(tokens_frame, from_=5, to=300, increment=5,
                                textvariable=self.timeout_var, bg=self.COLORS['bg_tertiary'],
                                fg=self.COLORS['text_primary'], width=6)
        timeout_spin.pack(side='left', padx=(10, 0))

        # Analysis Mode Section
        analysis_section, analysis_content = self._create_section_frame(scrollable_frame, "🔬 Analysis Mode")
        analysis_section.pack(fill='x', pady=(0, 10))

        # Mode selection label
        mode_label = tk.Label(
            analysis_content,
            text="Image Analysis Method:",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 9)
        )
        mode_label.pack(anchor='w', pady=(0, 5))

        # Radio button for YOLO Detection mode
        yolo_radio = tk.Radiobutton(
            analysis_content,
            text="YOLO Detection + AI Analysis (Recommended)",
            variable=self.analysis_mode_var,
            value="yolo_detection",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_primary'],
            selectcolor=self.COLORS['bg_tertiary'],
            activebackground=self.COLORS['bg_secondary'],
            activeforeground=self.COLORS['text_primary'],
            font=('Segoe UI', 9)
        )
        yolo_radio.pack(anchor='w', pady=2)

        # Description for YOLO mode
        yolo_desc = tk.Label(
            analysis_content,
            text="Runs YOLO object detection on images, sends structured detection data to AI.\nProvides detailed object analysis with counts, types, and comparisons.",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8),
            justify='left'
        )
        yolo_desc.pack(anchor='w', padx=(25, 0), pady=(0, 10))

        # Radio button for Gemini Auto mode
        gemini_radio = tk.Radiobutton(
            analysis_content,
            text="Gemini Auto-Analysis",
            variable=self.analysis_mode_var,
            value="gemini_auto",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_primary'],
            selectcolor=self.COLORS['bg_tertiary'],
            activebackground=self.COLORS['bg_secondary'],
            activeforeground=self.COLORS['text_primary'],
            font=('Segoe UI', 9)
        )
        gemini_radio.pack(anchor='w', pady=2)

        # Description for Gemini mode
        gemini_desc = tk.Label(
            analysis_content,
            text="Sends images directly to Gemini AI without YOLO detection.\nGemini analyzes images independently using its vision capabilities.",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8),
            justify='left'
        )
        gemini_desc.pack(anchor='w', padx=(25, 0), pady=(0, 5))

        # # Chat Behavior Section
        # chat_section, chat_content = self._create_section_frame(scrollable_frame, "💬 Chat Behavior")
        # chat_section.pack(fill='x', pady=(0, 10))
        
        # # Enable AI analysis
        # self._create_checkbox(chat_content, "Enable AI analysis features",
        #                      self.enable_ai_var).pack(anchor='w', pady=2)
        
        # # Response format
        # format_frame = tk.Frame(chat_content, bg=self.COLORS['bg_secondary'])
        # format_frame.pack(fill='x', pady=5)
        
        # tk.Label(format_frame, text="Response Format:", bg=self.COLORS['bg_secondary'], 
        #         fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        # response_combo = ttk.Combobox(format_frame, textvariable=self.response_format_var,
        #                              values=SettingsValidator.VALID_RESPONSE_FORMATS, state='readonly', width=15)
        # response_combo.pack(side='left', padx=(10, 0))
        
        # # Chat history limit
        # history_frame = tk.Frame(chat_content, bg=self.COLORS['bg_secondary'])
        # history_frame.pack(fill='x', pady=5)
        
        # tk.Label(history_frame, text="History Limit:", bg=self.COLORS['bg_secondary'], 
        #         fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        # history_spin = tk.Spinbox(history_frame, from_=10, to=1000, increment=10,
        #                         textvariable=self.chat_history_limit_var, bg=self.COLORS['bg_tertiary'],
        #                         fg=self.COLORS['text_primary'], width=8)
        # history_spin.pack(side='left', padx=(10, 0))
        
        # # Auto-save and memory
        # self._create_checkbox(chat_content, "Auto-save chat history",
        #                      self.chat_auto_save_var).pack(anchor='w', pady=2)
        
        # self._create_checkbox(chat_content, "Enable conversation memory",
        #                      self.conversation_memory_var).pack(anchor='w', pady=2)
        
        # # Chat export format
        # export_format_frame = tk.Frame(chat_content, bg=self.COLORS['bg_secondary'])
        # export_format_frame.pack(fill='x', pady=5)
        
        # tk.Label(export_format_frame, text="Export Format:", bg=self.COLORS['bg_secondary'], 
        #         fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        # export_format_combo = ttk.Combobox(export_format_frame, textvariable=self.chat_export_format_var,
        #                                   values=SettingsValidator.VALID_CHAT_EXPORT_FORMATS, state='readonly', width=10)
        # export_format_combo.pack(side='left', padx=(10, 0))
        
        # Custom Persona Section
        persona_section, persona_content = self._create_section_frame(scrollable_frame, "🎭 Custom Role/Persona")
        persona_section.pack(fill='x', pady=(0, 10))
        
        # Persona label
        persona_label_frame = tk.Frame(persona_content, bg=self.COLORS['bg_secondary'])
        persona_label_frame.pack(fill='x', pady=(0, 5))
        
        tk.Label(persona_label_frame, text="Custom Role/Persona:", 
                bg=self.COLORS['bg_secondary'], fg=self.COLORS['text_primary'], 
                font=('Segoe UI', 9)).pack(anchor='w')
        
        tk.Label(persona_label_frame, text="Define how the AI should behave and respond (leave empty for default):",
                bg=self.COLORS['bg_secondary'], fg=self.COLORS['text_muted'], 
                font=('Segoe UI', 8)).pack(anchor='w', pady=(2, 0))
        
        # Persona text widget with scrollbar
        persona_frame = tk.Frame(persona_content, bg=self.COLORS['bg_secondary'])
        persona_frame.pack(fill='x', pady=5)
        
        # Create text widget container
        text_container = tk.Frame(persona_frame, bg=self.COLORS['bg_tertiary'], relief='sunken', bd=1)
        text_container.pack(fill='x', padx=2, pady=2)
        
        # Scrollable text widget
        self.persona_text = tk.Text(
            text_container,
            height=6,  # 6 lines visible
            width=60,
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 9),
            wrap='word',
            borderwidth=0,
            insertbackground=self.COLORS['text_primary'],
            selectbackground=self.COLORS['accent_primary'],
            selectforeground=self.COLORS['text_primary']
        )
        
        persona_scrollbar = ttk.Scrollbar(text_container, command=self.persona_text.yview)
        self.persona_text.configure(yscrollcommand=persona_scrollbar.set)
        
        self.persona_text.pack(side='left', fill='both', expand=True)
        persona_scrollbar.pack(side='right', fill='y')
        
        # Load current persona text
        current_persona = self.config.get('chatbot_persona', '')
        # Don't set default display text - persona should only define behavior, not display content
        # Leave empty if no persona is configured
        self.persona_text.insert('1.0', current_persona)
        
        # Bind text changes to update the variable (for change tracking)
        def on_persona_change(*args):
            try:
                text_content = self.persona_text.get('1.0', tk.END).strip()
                self.chatbot_persona_var.set(text_content)
            except tk.TclError:
                pass  # Widget might be destroyed
        
        self.persona_text.bind('<KeyRelease>', on_persona_change)
        self.persona_text.bind('<FocusOut>', on_persona_change)
        
        # Character count label
        char_count_frame = tk.Frame(persona_content, bg=self.COLORS['bg_secondary'])
        char_count_frame.pack(fill='x', pady=(5, 0))
        
        self.persona_char_count = tk.Label(char_count_frame, 
                                          text=f"Characters: {len(current_persona)}/5000",
                                          bg=self.COLORS['bg_secondary'], 
                                          fg=self.COLORS['text_muted'], 
                                          font=('Segoe UI', 8))
        self.persona_char_count.pack(anchor='e')
        
        # Update character count when text changes
        def update_char_count(*args):
            try:
                text_content = self.persona_text.get('1.0', tk.END).strip()
                char_count = len(text_content)
                
                # Color coding for character count
                if char_count > 5000:
                    color = self.COLORS['error']
                elif char_count > 4500:
                    color = self.COLORS['warning']
                else:
                    color = self.COLORS['text_muted']
                
                self.persona_char_count.config(
                    text=f"Characters: {char_count}/5000",
                    fg=color
                )
            except tk.TclError:
                pass  # Widget might be destroyed
        
        self.persona_text.bind('<KeyRelease>', lambda e: (on_persona_change(), update_char_count()))
        self.persona_text.bind('<FocusOut>', lambda e: (on_persona_change(), update_char_count()))
        
        # # Rate Limiting Section
        # rate_section, rate_content = self._create_section_frame(scrollable_frame, "⏱️ Rate Limiting")
        # rate_section.pack(fill='x', pady=(0, 10))
        
        # # Enable rate limiting
        # self._create_checkbox(rate_content, "Enable rate limiting",
        #                      self.enable_rate_limiting_var).pack(anchor='w', pady=2)
        
        # # Requests per minute
        # rate_frame = tk.Frame(rate_content, bg=self.COLORS['bg_secondary'])
        # rate_frame.pack(fill='x', pady=5)
        
        # tk.Label(rate_frame, text="Requests/minute:", bg=self.COLORS['bg_secondary'], 
        #         fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        # rate_spin = tk.Spinbox(rate_frame, from_=1, to=60, textvariable=self.requests_per_minute_var,
        #                      bg=self.COLORS['bg_tertiary'], fg=self.COLORS['text_primary'], width=6)
        # rate_spin.pack(side='left', padx=(10, 20))
        
        # # Context window size
        # tk.Label(rate_frame, text="Context Window:", bg=self.COLORS['bg_secondary'], 
        #         fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        # context_spin = tk.Spinbox(rate_frame, from_=1000, to=32000, increment=1000,
        #                         textvariable=self.context_window_var, bg=self.COLORS['bg_tertiary'],
        #                         fg=self.COLORS['text_primary'], width=8)
        # context_spin.pack(side='left', padx=(10, 0))
        
        # Test Connection Section
        test_section, test_content = self._create_section_frame(scrollable_frame, "🧪 Connection Test")
        test_section.pack(fill='x')
        
        # Test button
        test_frame = tk.Frame(test_content, bg=self.COLORS['bg_secondary'])
        test_frame.pack(fill='x', pady=5)
        
        self.api_test_button = ttk.Button(test_frame, text="🧪 Test API Connection", command=self._test_api_connection)
        self.api_test_button.pack(side='left')
        
        # Test result
        self.test_result_label = tk.Label(test_content, textvariable=self.test_result_var,
                                         bg=self.COLORS['bg_secondary'], fg=self.COLORS['text_muted'],
                                         font=('Segoe UI', 9), wraplength=500, justify='left')
        self.test_result_label.pack(anchor='w', pady=(5, 0))
        
        return tab
    
    def _build_buttons(self, parent):
        """Build the dialog action buttons."""
        button_frame = tk.Frame(parent, bg=self.COLORS['bg_primary'])
        button_frame.pack(fill='x', pady=(10, 0))
        
        # Left side - Validation status
        left_frame = tk.Frame(button_frame, bg=self.COLORS['bg_primary'])
        left_frame.pack(side='left')
        
        self.validation_status_label = tk.Label(
            left_frame,
            text="Ready",
            bg=self.COLORS['bg_primary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 9)
        )
        self.validation_status_label.pack()
        
        # Right side - Action buttons
        right_frame = tk.Frame(button_frame, bg=self.COLORS['bg_primary'])
        right_frame.pack(side='right')
        
        # Cancel button
        cancel_button = ttk.Button(
            right_frame,
            text="Cancel",
            command=self._on_cancel
        )
        cancel_button.pack(side='right', padx=(10, 0))
        
        # Apply button
        self.apply_button = ttk.Button(
            right_frame,
            text="Apply",
            command=self._on_apply,
            state='disabled'
        )
        self.apply_button.pack(side='right', padx=(10, 0))
        
        # OK button
        ok_button = ttk.Button(
            right_frame,
            text="OK",
            command=self._on_ok
        )
        ok_button.pack(side='right')
        
        # Configure button styles
        style = ttk.Style()
        style.configure('Action.TButton', padding=(15, 8))

    # ========== Change Tracking ==========

    def _on_setting_changed(self, *args):
        """Called when any setting is changed."""
        # Check if current values differ from original values
        has_changes = (
            self.language_var.get() != self.original_values['language'] or
            self.data_dir_var.get() != self.original_values['data_dir'] or
            self.models_dir_var.get() != self.original_values['models_dir'] or
            self.results_dir_var.get() != self.original_values['results_dir'] or
            self.camera_index_var.get() != self.original_values['camera_index'] or
            self.camera_device_name_var.get() != self.original_values['camera_device_name'] or
            self.video_codec_var.get() != self.original_values['video_codec'] or
            self.confidence_threshold_var.get() != self.original_values['confidence_threshold'] or
            self.iou_threshold_var.get() != self.original_values['iou_threshold'] or
            self.train_epochs_var.get() != self.original_values['train_epochs'] or
            self.train_batch_size_var.get() != self.original_values['train_batch_size'] or
            self.train_device_var.get() != self.original_values['train_device'] or
            self.training_workers_var.get() != self.original_values.get('training_workers', 2) or
            self.training_cache_var.get() != str(self.original_values.get('training_cache', 'ram')) or
            self.model_architecture_var.get() != self.original_values['model_architecture'] or
            self.yolo_version_var.get() != self.original_values['yolo_version'] or
            self.yolo_size_var.get() != self.original_values['yolo_size'] or
            self.analysis_mode_var.get() != self.original_values['analysis_mode'] or
            self.api_key_var.get() != self.original_values['api_key'] or
            self.gemini_model_var.get() != self.original_values['gemini_model'] or
            self.temperature_var.get() != self.original_values['temperature'] or
            self.max_tokens_var.get() != self.original_values['max_tokens'] or
            self.timeout_var.get() != self.original_values['timeout'] or
            self.chatbot_persona_var.get() != self.original_values['chatbot_persona']
        )

        # Update state and UI
        self.has_changes = has_changes
        if has_changes:
            self.apply_button.config(state='normal')
            self.validation_status_label.config(
                text="Unsaved changes",
                fg=self.COLORS['warning']
            )
        else:
            self.apply_button.config(state='disabled')
            self.validation_status_label.config(
                text="Ready",
                fg=self.COLORS['text_muted']
            )

    # ========== Validation ==========

    def _validate_settings(self) -> tuple[bool, str]:
        """Validate all settings. Returns (is_valid, error_message)."""
        # Validate confidence thresholds
        valid, msg = SettingsValidator.validate_confidence(
            self.confidence_threshold_var.get()
        )
        if not valid:
            return False, f"Confidence Threshold: {msg}"

        valid, msg = SettingsValidator.validate_confidence(
            self.iou_threshold_var.get()
        )
        if not valid:
            return False, f"IoU Threshold: {msg}"

        # Validate directories
        for dir_name, dir_var in [
            ('Data Directory', self.data_dir_var),
            ('Models Directory', self.models_dir_var),
            ('Results Directory', self.results_dir_var)
        ]:
            valid, msg = SettingsValidator.validate_directory(dir_var.get())
            if not valid:
                return False, f"{dir_name}: {msg}"

        # Validate positive integers
        valid, msg = SettingsValidator.validate_positive_int(
            self.max_tokens_var.get(), min_val=20, max_val=8192
        )
        if not valid:
            return False, f"Max Tokens: {msg}"

        valid, msg = SettingsValidator.validate_positive_int(
            self.timeout_var.get(), min_val=5, max_val=300
        )
        if not valid:
            return False, f"Timeout: {msg}"

        valid, msg = SettingsValidator.validate_positive_int(
            self.camera_index_var.get(), min_val=0, max_val=10
        )
        if not valid:
            return False, f"Camera Index: {msg}"

        # Validate training settings
        valid, msg = SettingsValidator.validate_positive_int(
            self.train_epochs_var.get(), min_val=1, max_val=500
        )
        if not valid:
            return False, f"Training Epochs: {msg}"

        valid, msg = SettingsValidator.validate_positive_int(
            self.train_batch_size_var.get(), min_val=1, max_val=128
        )
        if not valid:
            return False, f"Training Batch Size: {msg}"

        valid, msg = SettingsValidator.validate_positive_int(
            self.training_workers_var.get(), min_val=0, max_val=8
        )
        if not valid:
            return False, f"Training Workers: {msg}"

        # Validate API key if provided
        api_key = self.api_key_var.get().strip()
        if api_key:  # Only validate if not empty
            valid, msg = SettingsValidator.validate_api_key(api_key)
            if not valid:
                return False, f"API Key: {msg}"

        # Validate persona
        persona = self.chatbot_persona_var.get()
        valid, msg = SettingsValidator.validate_persona(persona)
        if not valid:
            return False, f"Chatbot Persona: {msg}"

        # All validations passed
        return True, ""

    def _apply_settings(self) -> bool:
        """Apply settings to config and save. Returns True if successful."""
        try:
            # Update config with new values
            self.config['language'] = self.language_var.get()
            self.config['data_dir'] = self.data_dir_var.get()
            self.config['models_dir'] = self.models_dir_var.get()
            self.config['results_dir'] = self.results_dir_var.get()
            self.config['last_webcam_index'] = self.camera_index_var.get()
            self.config['camera_device_name'] = self.camera_device_name_var.get()
            self.config['video_codec'] = self.video_codec_var.get()

            # Save detection method: convert display value to config value
            detection_method_display = self.detection_method_var.get()
            detection_method_config = 'yolo' if detection_method_display == 'YOLO' else 'opencv'
            self.config['detection_method'] = detection_method_config

            self.config['detection_confidence_threshold'] = self.confidence_threshold_var.get()
            self.config['detection_iou_threshold'] = self.iou_threshold_var.get()
            self.config['train_epochs'] = self.train_epochs_var.get()

            # Validate and warn about extreme batch sizes
            batch_size = self.train_batch_size_var.get()
            if not self._validate_batch_size_selection(batch_size):
                return False  # User cancelled due to batch size warning

            self.config['batch_size'] = batch_size
            self.config['training_device'] = self.train_device_var.get()
            self.config['training_workers'] = self.training_workers_var.get()
            # Convert cache setting: 'False' string to boolean False, otherwise keep as string
            cache_value = self.training_cache_var.get()
            self.config['training_cache'] = False if cache_value == 'False' else cache_value

            # Save YOLO model configuration
            self.config['yolo_version'] = self.yolo_version_var.get()
            self.config['yolo_model_size'] = self.yolo_size_var.get()
            # Construct and save the architecture from version + size
            self.config['model_architecture'] = self._get_model_architecture()

            self.config['analysis_mode'] = self.analysis_mode_var.get()
            self.config['gemini_api_key'] = self.api_key_var.get()
            self.config['gemini_model'] = self.gemini_model_var.get()
            self.config['gemini_temperature'] = self.temperature_var.get()
            self.config['gemini_max_tokens'] = self.max_tokens_var.get()
            self.config['gemini_timeout'] = self.timeout_var.get()
            self.config['chatbot_persona'] = self.chatbot_persona_var.get()

            # Save config to file
            self._save_config()

            # Update original values to current values (changes are now saved)
            self._store_original_values()

            # Notify parent window via callback
            if self.callback:
                self.callback(self.config)

            logger.info("Settings applied and saved successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to apply settings: {e}", exc_info=True)
            messagebox.showerror(
                "Error",
                f"Failed to apply settings:\n{str(e)}",
                parent=self.dialog
            )
            return False

    def _save_config(self):
        """Save configuration to config.json file."""
        try:
            config_path = "config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}", exc_info=True)
            raise

    # ========== Button Handlers ==========

    def _on_ok(self):
        """Handle OK button - validate, apply, save, and close."""
        # Validate settings
        valid, error_msg = self._validate_settings()
        if not valid:
            messagebox.showerror(
                "Validation Error",
                f"Invalid settings:\n\n{error_msg}",
                parent=self.dialog
            )
            return

        # Apply settings
        if self._apply_settings():
            # Close dialog
            self._cleanup_and_close()

    def _on_apply(self):
        """Handle Apply button - validate, apply, save, but keep dialog open."""
        # Validate settings
        valid, error_msg = self._validate_settings()
        if not valid:
            messagebox.showerror(
                "Validation Error",
                f"Invalid settings:\n\n{error_msg}",
                parent=self.dialog
            )
            return

        # Apply settings
        if self._apply_settings():
            # Show success feedback
            self.validation_status_label.config(
                text="Settings applied successfully",
                fg=self.COLORS['success']
            )

            # Disable apply button (no pending changes)
            self.apply_button.config(state='disabled')
            self.has_changes = False

            # Reset status after 3 seconds
            self.dialog.after(3000, lambda: self.validation_status_label.config(
                text="Ready",
                fg=self.COLORS['text_muted']
            ))

    def _on_cancel(self):
        """Handle Cancel button - warn if unsaved changes, then close."""
        # Check for unsaved changes
        if self.has_changes:
            result = messagebox.askyesno(
                "Unsaved Changes",
                "You have unsaved changes. Are you sure you want to discard them?",
                parent=self.dialog,
                icon='warning'
            )
            if not result:
                return  # User chose not to discard changes

        # Close without saving
        self._cleanup_and_close()

    def _on_window_close(self):
        """Handle window close button (X) - same as cancel."""
        self._on_cancel()

    def _cleanup_and_close(self):
        """Cleanup resources and close the dialog."""
        # Stop camera test if running
        if self._testing_camera:
            self._stop_test()

        # Release dialog grab
        try:
            self.dialog.grab_release()
        except:
            pass

        # Destroy dialog
        self.dialog.destroy()

    # ========== Utility Methods ==========

    def _browse_directory(self, var: tk.StringVar, title: str):
        """Browse for a directory and update the variable."""
        current_dir = var.get()
        directory = filedialog.askdirectory(
            title=title,
            initialdir=current_dir if os.path.exists(current_dir) else os.getcwd(),
            parent=self.dialog
        )
        if directory:
            var.set(directory)

    def _toggle_api_key_visibility(self):
        """Toggle API key visibility between shown and hidden."""
        if self.api_key_entry.cget('show') == '*':
            self.api_key_entry.config(show='')
            self.show_key_button.config(text='🔒')
        else:
            self.api_key_entry.config(show='*')
            self.show_key_button.config(text='👁')

    def _detect_cameras(self):
        """Detect available cameras and populate the dropdown.

        Runs camera detection in a background thread to avoid blocking UI.
        This is important because cv2.VideoCapture is slow on Windows.
        """
        # Disable button during detection
        self.detect_cameras_button.config(state='disabled', text="Detecting...")
        self.camera_name_label.config(text="Detecting cameras...", fg=self.COLORS['text_muted'])

        def detect_in_background():
            """Background thread function for camera detection."""
            try:
                # Import WebcamService for camera enumeration
                from app.services.webcam_service import WebcamService

                # Get list of available cameras (this is the slow part)
                cameras = WebcamService.list_available_cameras(max_cameras=10)

                # Update UI on main thread
                self.dialog.after(0, lambda: self._on_cameras_detected(cameras))

            except Exception as e:
                logger.error(f"Camera detection failed: {e}", exc_info=True)
                error_msg = str(e)
                self.dialog.after(0, lambda: self._on_camera_detection_error(error_msg))

        # Run detection in background thread
        detection_thread = threading.Thread(target=detect_in_background, daemon=True)
        detection_thread.start()

    def _on_cameras_detected(self, cameras: List[Dict[str, Any]]):
        """Handle successful camera detection (runs on main thread).

        Args:
            cameras: List of detected camera information dictionaries
        """
        try:
            self._available_cameras = cameras

            if self._available_cameras:
                # Build dropdown options
                camera_options = []
                for cam in self._available_cameras:
                    option = f"[{cam['index']}] {cam['name']} ({cam['width']}x{cam['height']})"
                    camera_options.append(option)

                # Update combobox
                self.camera_combo['values'] = camera_options

                # Select current camera if it exists in the list
                current_index = self.camera_index_var.get()
                selected_idx = 0
                for i, cam in enumerate(self._available_cameras):
                    if cam['index'] == current_index:
                        selected_idx = i
                        break

                self.camera_combo.current(selected_idx)
                self._on_camera_selected(None)

                self.camera_name_label.config(
                    text=f"Found {len(self._available_cameras)} camera(s)",
                    fg=self.COLORS['success']
                )
            else:
                self.camera_combo['values'] = []
                self.camera_name_label.config(
                    text="No cameras detected",
                    fg=self.COLORS['warning']
                )

        except Exception as e:
            logger.error(f"Error updating camera list: {e}")
            self.camera_name_label.config(
                text=f"Error: {str(e)}",
                fg=self.COLORS['error']
            )
        finally:
            self.detect_cameras_button.config(state='normal', text="🔄 Detect Cameras")

    def _on_camera_detection_error(self, error_msg: str):
        """Handle camera detection error (runs on main thread).

        Args:
            error_msg: Error message to display
        """
        self.camera_name_label.config(
            text=f"Error detecting cameras: {error_msg}",
            fg=self.COLORS['error']
        )
        self.detect_cameras_button.config(state='normal', text="🔄 Detect Cameras")

    def _on_camera_selected(self, event):
        """Handle camera selection from dropdown."""
        try:
            selection_idx = self.camera_combo.current()
            if selection_idx >= 0 and selection_idx < len(self._available_cameras):
                selected_camera = self._available_cameras[selection_idx]
                self.camera_index_var.set(selected_camera['index'])
                self.camera_device_name_var.set(selected_camera['name'])

                # Update info label
                self.camera_name_label.config(
                    text=f"Selected: {selected_camera['name']} (Index {selected_camera['index']})",
                    fg=self.COLORS['text_muted']
                )
        except Exception as e:
            logger.error(f"Error handling camera selection: {e}")

    def _test_camera(self):
        """Start testing the selected camera with live preview."""
        if self._testing_camera:
            return

        camera_index = self.camera_index_var.get()
        video_codec = self.video_codec_var.get()

        try:
            self._test_cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

            if not self._test_cap.isOpened():
                messagebox.showerror(
                    "Camera Error",
                    f"Failed to open camera at index {camera_index}",
                    parent=self.dialog
                )
                return
            
            # Apply video codec setting if not "Auto"
            if video_codec and video_codec != "Auto":
                try:
                    # Convert codec string to FourCC code
                    fourcc = cv2.VideoWriter_fourcc(*video_codec.ljust(4)[:4])
                    self._test_cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                    logger.info(f"Applied video codec: {video_codec} (FourCC: {fourcc})")
                except Exception as codec_err:
                    logger.warning(f"Failed to set codec {video_codec}: {codec_err}")
                    # Continue anyway - camera may still work with default codec
                
            # Get camera properties
            width = int(self._test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self._test_cap.get(cv2.CAP_PROP_FPS))

            self.camera_info_var.set(
                f"Camera {camera_index}: {width}x{height} @ {fps}fps"
            )

            # Update button states
            self.camera_test_button.config(state='disabled')
            self.stop_test_button.config(state='normal')
            self._testing_camera = True

            # Start update loop
            self._update_camera_preview()

        except Exception as e:
            logger.error(f"Camera test failed: {e}", exc_info=True)
            messagebox.showerror(
                "Error",
                f"Failed to test camera:\n{str(e)}",
                parent=self.dialog
            )
            if self._test_cap:
                self._test_cap.release()
                self._test_cap = None

    def _update_camera_preview(self):
        """Update the camera preview canvas with the latest frame."""
        if not self._testing_camera or not self._test_cap:
            return

        try:
            ret, frame = self._test_cap.read()

            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize to fit canvas
                canvas_width = 400
                canvas_height = 240
                frame_resized = cv2.resize(frame_rgb, (canvas_width, canvas_height))

                # Convert to PIL Image and then to PhotoImage
                image = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(image)

                # Update canvas
                self._preview_canvas.delete("all")
                self._preview_canvas.create_image(0, 0, anchor='nw', image=photo)

                # Keep a reference to prevent garbage collection
                self._preview_canvas.image = photo

                # Schedule next update (30 FPS)
                self.dialog.after(33, self._update_camera_preview)
            else:
                self._stop_test()
                messagebox.showwarning(
                    "Camera Warning",
                    "Failed to read frame from camera",
                    parent=self.dialog
                )

        except Exception as e:
            logger.error(f"Camera preview update failed: {e}", exc_info=True)
            self._stop_test()

    def _stop_test(self):
        """Stop camera testing."""
        self._testing_camera = False

        if self._test_cap:
            self._test_cap.release()
            self._test_cap = None

        # Update button states
        self.camera_test_button.config(state='normal')
        self.stop_test_button.config(state='disabled')

        # Clear canvas
        self._preview_canvas.delete("all")
        self._preview_canvas.create_text(
            200, 120,
            text="Camera Preview\nSelect camera and click 'Test' to preview",
            fill='white',
            font=('Segoe UI', 12),
            justify='center'
        )

        self.camera_info_var.set("Select a camera to test")

    def _test_api_connection(self):
        """Test the Gemini API connection."""
        # Get current API key
        api_key = self.api_key_var.get().strip()

        if not api_key:
            self.test_result_var.set("Please enter an API key first")
            self.test_result_label.config(fg=self.COLORS['error'])
            return

        # Disable button during test
        self.api_test_button.config(state='disabled')
        self.test_result_var.set("Testing connection...")
        self.test_result_label.config(fg=self.COLORS['text_muted'])

        def test_in_background():
            """Test API in background thread."""
            try:
                import google.generativeai as genai

                # Configure with current settings
                genai.configure(api_key=api_key)

                # Try to create a model
                model_name = self.gemini_model_var.get()
                model = genai.GenerativeModel(model_name)

                # Send a simple test prompt
                response = model.generate_content(
                    "Hello! Please respond with 'API connection successful'",
                    generation_config={
                        'temperature': 0.1,
                        'max_output_tokens': 50
                    }
                )

                # Check response
                if response and response.text:
                    self.dialog.after(0, lambda: self._show_api_test_success())
                else:
                    self.dialog.after(0, lambda: self._show_api_test_error(
                        "No response received from API"
                    ))

            except Exception as e:
                error_msg = str(e)
                self.dialog.after(0, lambda: self._show_api_test_error(error_msg))

        # Run test in background thread
        thread = threading.Thread(target=test_in_background, daemon=True)
        thread.start()

    def _show_api_test_success(self):
        """Show successful API test result."""
        self.test_result_var.set("Connection successful! API key is valid.")
        self.test_result_label.config(fg=self.COLORS['success'])
        self.api_test_button.config(state='normal')

    def _show_api_test_error(self, error_msg: str):
        """Show failed API test result."""
        self.test_result_var.set(f"Connection failed: {error_msg}")
        self.test_result_label.config(fg=self.COLORS['error'])
        self.api_test_button.config(state='normal')

    def _detect_training_device(self):
        """Detect available training device and display it."""
        try:
            # Import device detection function from training service
            from app.services.training_service import get_best_device, get_device_memory_info

            device_str, device_name = get_best_device()

            # Get memory info if available
            memory_info = get_device_memory_info(device_str)

            if memory_info:
                info_text = f"Detected: {device_name} ({memory_info['free_mb']:.0f}MB free)"
            else:
                info_text = f"Detected: {device_name}"

            self.detected_device_label.config(
                text=info_text,
                fg=self.COLORS['success'] if device_str != 'cpu' else self.COLORS['text_muted']
            )

            logger.info(f"Detected training device: {device_name} ({device_str})")

        except Exception as e:
            logger.error(f"Error detecting training device: {e}")
            self.detected_device_label.config(
                text=f"Error: {str(e)}",
                fg=self.COLORS['error']
            )

    def _on_batch_size_change(self, event=None):
        """Update batch size description and memory label when selection changes."""
        try:
            batch_size = int(self.train_batch_size_var.get())

            # Batch size descriptions
            descriptions = {
                1: "Ultra-low memory (2GB VRAM) - Very slow but most stable",
                2: "Low memory (2-4GB VRAM) - Slow but stable",
                4: "Moderate memory (4-6GB VRAM) - Balanced (Recommended)",
                8: "Good memory (6-8GB VRAM) - Fast",
                16: "High memory (8-12GB VRAM) - Very fast",
                32: "Very high memory (12-16GB VRAM) - Extremely fast",
                64: "Enterprise GPUs (16+ GB VRAM) - Maximum performance"
            }

            # Memory requirements
            memory_reqs = {
                1: "~1-2 GB VRAM",
                2: "~2-3 GB VRAM",
                4: "~3-5 GB VRAM",
                8: "~5-8 GB VRAM",
                16: "~8-12 GB VRAM",
                32: "~12-16 GB VRAM",
                64: "~16+ GB VRAM"
            }

            # Update description label
            description = descriptions.get(batch_size, "Custom batch size")
            self.batch_desc_label.config(text=f"→ {description}")

            # Update memory label
            memory_req = memory_reqs.get(batch_size, "Unknown VRAM")
            self.batch_memory_label.config(text=f"({memory_req})")

            # Color code based on memory requirement
            if batch_size <= 2:
                color = "#90EE90"  # Light green - safe
            elif batch_size <= 8:
                color = "#FFD700"  # Gold - moderate
            elif batch_size <= 16:
                color = "#FFA500"  # Orange - high
            else:
                color = "#FF6347"  # Tomato red - very high

            self.batch_memory_label.config(fg=color)

        except (ValueError, TypeError) as e:
            logger.warning(f"Error updating batch size display: {e}")

    def _get_recommended_batch_size(self) -> int:
        """Get recommended batch size based on available GPU memory."""
        try:
            # Try to import device detection functions
            from app.services.training_service import get_best_device, get_device_memory_info

            device_str, device_name = get_best_device()

            # Get memory info if available
            memory_info = get_device_memory_info(device_str)

            if memory_info and 'total_mb' in memory_info:
                gpu_memory_mb = memory_info['total_mb']
                gpu_memory_gb = gpu_memory_mb / 1024

                # Recommend batch size based on available memory
                if gpu_memory_gb < 3:
                    return 1  # Very limited (< 3GB)
                elif gpu_memory_gb < 5:
                    return 2  # Limited (3-5GB)
                elif gpu_memory_gb < 7:
                    return 4  # Moderate (5-7GB) - default
                elif gpu_memory_gb < 10:
                    return 8  # Good (7-10GB)
                elif gpu_memory_gb < 14:
                    return 16  # High (10-14GB)
                elif gpu_memory_gb < 18:
                    return 32  # Very high (14-18GB)
                else:
                    return 64  # Enterprise (18+ GB)
            else:
                # CPU training or unknown device - use small batch
                if device_str == 'cpu':
                    return 2
                else:
                    return 4  # Safe default

        except Exception as e:
            logger.warning(f"Could not detect GPU memory for batch size recommendation: {e}")
            return 4  # Safe default

    def _apply_recommended_batch_size(self):
        """Apply recommended batch size based on detected GPU memory."""
        try:
            recommended = self._get_recommended_batch_size()

            # Get device info for display
            try:
                from app.services.training_service import get_best_device, get_device_memory_info
                device_str, device_name = get_best_device()
                memory_info = get_device_memory_info(device_str)

                if memory_info and 'total_mb' in memory_info:
                    gpu_memory_gb = memory_info['total_mb'] / 1024
                    device_info = f"{device_name} ({gpu_memory_gb:.1f} GB)"
                else:
                    device_info = device_name
            except:
                device_info = "Unknown device"

            # Show confirmation dialog
            response = messagebox.showinfo(
                "Recommended Batch Size",
                f"Detected: {device_info}\n\n"
                f"Recommended batch size: {recommended}\n\n"
                f"This value has been applied. You can change it if needed.",
                parent=self.dialog
            )

            # Apply the recommended value
            self.train_batch_size_var.set(recommended)

            # Trigger the update to show description
            self._on_batch_size_change()

            logger.info(f"Applied recommended batch size: {recommended} for {device_info}")

        except Exception as e:
            logger.error(f"Error applying recommended batch size: {e}")
            messagebox.showerror(
                "Error",
                f"Could not detect optimal batch size: {str(e)}\n\n"
                "Please select manually based on your GPU memory.",
                parent=self.dialog
            )

    def _validate_batch_size_selection(self, batch_size: int) -> bool:
        """Validate batch size selection and show warnings for extreme values.

        Args:
            batch_size: The selected batch size

        Returns:
            True if user accepts the value, False if user cancels
        """
        try:
            batch_size = int(batch_size)

            # Warn about batch size of 1
            if batch_size == 1:
                response = messagebox.askokcancel(
                    "Batch Size Warning",
                    "⚠️ Batch size of 1 is extremely slow but uses minimal memory.\n\n"
                    "This is recommended ONLY if you have very limited GPU memory (< 2GB).\n\n"
                    "Training will be significantly slower compared to larger batch sizes.\n\n"
                    "Continue with batch size 1?",
                    icon='warning',
                    parent=self.dialog
                )
                if not response:
                    logger.info("User cancelled batch size 1 selection")
                    return False

            # Warn about batch sizes 32 and above
            elif batch_size >= 32:
                memory_reqs = {
                    32: "~12-16 GB VRAM",
                    64: "~16+ GB VRAM"
                }
                memory_req = memory_reqs.get(batch_size, "16+ GB VRAM")

                response = messagebox.askokcancel(
                    "High Batch Size Warning",
                    f"⚠️ Batch size of {batch_size} requires significant GPU memory ({memory_req}).\n\n"
                    "This may cause out-of-memory (OOM) errors if your GPU doesn't have enough VRAM.\n\n"
                    "If training fails with CUDA out of memory errors, try a smaller batch size.\n\n"
                    f"Continue with batch size {batch_size}?",
                    icon='warning',
                    parent=self.dialog
                )
                if not response:
                    logger.info(f"User cancelled batch size {batch_size} selection")
                    return False

            return True  # No warning needed or user accepted

        except (ValueError, TypeError) as e:
            logger.error(f"Error validating batch size: {e}")
            return True  # Allow to proceed
