"""Modern main application window with comprehensive UI/UX design and performance optimizations."""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import numpy as np
import cv2
import os
import json
import threading
import logging
from logging.handlers import RotatingFileHandler
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime

# Import services
from app.services.webcam_service import WebcamService
from app.services.inference_service import InferenceService
from app.services.training_service import TrainingService
from app.services.gemini_service import GeminiService
from app.services.reference_manager import ReferenceManager

# Import UI components
from app.ui.components.optimized_canvas import OptimizedCanvas
from app.ui.components.object_selector import ObjectSelector

# Import dialogs
from app.ui.dialogs.comprehensive_settings_dialog import ComprehensiveSettingsDialog
from app.ui.dialogs.training_progress_dialog import TrainingProgressDialog
from app.ui.dialogs.object_naming_dialog import ObjectNamingDialog
from app.ui.dialogs.bbox_drawing_dialog import BboxDrawingDialog, ImageSourceDialog
from app.ui.dialogs.data_augmentation_dialog import DataAugmentationDialog

# Setup logging
def setup_logging():
    """Configure logging with console and optional file output based on config.json settings."""
    # Load config to get logging settings
    config = {}
    config_path = "config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config for logging setup: {e}")

    # Get logging configuration from config.json
    log_level = config.get('log_level', 'INFO').upper()
    enable_file_logging = config.get('enable_file_logging', False)
    log_file_path = config.get('log_file_path', 'logs/app.log')
    max_file_size_mb = config.get('max_file_size_mb', 10)
    backup_count = config.get('backup_count', 5)

    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level, logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Configure file handler if enabled
    if enable_file_logging:
        try:
            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            # Create rotating file handler
            file_handler = RotatingFileHandler(
                filename=log_file_path,
                maxBytes=max_file_size_mb * 1024 * 1024,  # Convert MB to bytes
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

            # Log that file logging is enabled
            root_logger.info(f"File logging enabled: {log_file_path} (max {max_file_size_mb}MB, {backup_count} backups)")
        except Exception as e:
            # If file logging fails, log error to console but continue
            root_logger.error(f"Failed to setup file logging: {e}")

    return root_logger

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


class MainWindow:
    """Modern main application window with comprehensive UI/UX design."""

    # Theme definitions
    THEMES = {
        'Dark': {
            'bg_primary': '#1e1e1e',        # Dark background
            'bg_secondary': '#2d2d2d',      # Secondary panels
            'bg_tertiary': '#3c3c3c',       # Tertiary elements
            'accent_primary': '#007acc',    # Blue accent
            'accent_secondary': '#005a9e',  # Darker blue
            'text_primary': '#ffffff',      # White text
            'text_secondary': '#cccccc',    # Light gray text
            'text_muted': '#999999',        # Muted gray text
            'success': '#4caf50',           # Green for success
            'warning': '#ff9800',           # Orange for warnings
            'error': '#f44336',             # Red for errors
            'border': '#404040',            # Border color
        },
    }

    def __init__(self, root: tk.Tk):
        """Initialize main window.

        Args:
            root: Tkinter root window
        """
        self.root = root
        self.COLORS = self.THEMES['Dark']

        # Load configuration
        self.config = self._load_config()

        # Initialize services
        self._init_services()

        # State variables
        self._training_image: Optional[np.ndarray] = None
        self._selected_bbox: Optional[tuple] = None
        self._object_selector: Optional[ObjectSelector] = None
        self._temp_selector: Optional[ObjectSelector] = None  # Temporary selector for multi-canvas selection

        # Live detection state
        self._live_detection_active = False
        self._class_colors = {}  # Color mapping for detected classes
        self._detection_in_progress = False  # Flag to prevent concurrent detections

        # Objects management multi-selection state
        self.checked_objects: set[str] = set()  # Set of checked object IDs

        # Check Gemini configuration
        self._gemini_configured = bool(self.config.get('gemini_api_key', '').strip())

        # Setup locale (simplified)
        self.locale = {}

        # Setup window
        self._setup_window()
        self._setup_styles()
        self._build_ui()

        # Auto-display reference image if it exists
        self._auto_display_reference()

        # Setup debug mode on canvases if enabled
        self._update_debug_mode()

        # Start video stream update loop
        self._update_video_stream()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file.

        Returns:
            Configuration dictionary
        """
        config_path = "config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading config: {e}")

        # Return default config
        return {
            'camera_width': 1920,
            'camera_height': 1080,
            'target_fps': 30,
            'preferred_model': 'yolo12n',
            'detection_confidence_threshold': 0.5,
            'detection_iou_threshold': 0.45,
            'gemini_api_key': '',
            'gemini_model': 'gemini-2.5-flash',
            'gemini_temperature': 0.7,
            'gemini_max_tokens': 2048,
            'chatbot_persona': 'You are a helpful AI assistant for image analysis.'
        }

    def _save_config(self):
        """Save configuration to file."""
        try:
            with open("config.json", 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def _init_services(self):
        """Initialize all application services."""
        # Webcam service
        self.webcam_service = WebcamService(
            camera_index=self.config.get('last_webcam_index', 0),
            width=self.config.get('camera_width', 1920),
            height=self.config.get('camera_height', 1080),
            fps=self.config.get('target_fps', 30),
            codec=self.config.get('video_codec', 'Auto')
        )

        # Inference service - check for trained custom model first
        trained_model_path = os.path.join("data", "models", "model.pt")
        if os.path.exists(trained_model_path):
            logger.info(f"Found trained custom model at {trained_model_path}, will use it for inference")
            model_path = trained_model_path
        else:
            logger.info("No trained custom model found, using default pretrained model")
            model_path = self.config.get('preferred_model', 'yolo12n')

        self.inference_service = InferenceService(
            model_path=model_path,
            confidence_threshold=self.config.get('detection_confidence_threshold', 0.5),
            iou_threshold=self.config.get('detection_iou_threshold', 0.45)
        )

        # Load model in background
        threading.Thread(target=self.inference_service.load_model, daemon=True).start()

        # Training service (pass config for augmentation settings)
        self.training_service = TrainingService(data_dir="data/training", config=self.config)

        # Gemini service
        api_key = self.config.get('gemini_api_key', '')
        self.gemini_service = GeminiService(
            api_key=api_key,
            model=self.config.get('gemini_model', 'gemini-2.5-pro'),
            temperature=self.config.get('gemini_temperature', 0.7),
            max_tokens=self.config.get('gemini_max_tokens', 2048),
            persona=self.config.get('chatbot_persona', '')
        )

        if api_key:
            threading.Thread(target=self.gemini_service.initialize, daemon=True).start()

        # Reference manager
        self.reference_manager = ReferenceManager(data_dir="data/reference")

        # Load reference image if configured
        ref_path = self.config.get('reference_image_path')
        if ref_path and os.path.exists(ref_path):
            self.reference_manager.load_reference_from_file(ref_path)

    def _setup_window(self):
        """Setup main window properties with modern styling."""
        self.root.title("Vision Analysis System")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)

        # Configure root window
        self.root.configure(bg=self.COLORS['bg_primary'])

        # Configure grid weights for responsive layout
        self.root.grid_rowconfigure(1, weight=1)  # Main content area
        self.root.grid_columnconfigure(0, weight=1)

        # Set up window close protocol to ensure proper cleanup
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)
    
    def _setup_styles(self):
        """Setup custom styles for ttk widgets."""
        style = ttk.Style()
        
        # Configure style theme
        style.theme_use('clam')
        
        # Custom button styles
        style.configure('Modern.TButton',
                       background=self.COLORS['accent_primary'],
                       foreground=self.COLORS['text_primary'],
                       borderwidth=0,
                       focuscolor='none',
                       padding=(12, 8))
        
        style.map('Modern.TButton',
                  background=[('active', self.COLORS['accent_secondary']),
                             ('pressed', self.COLORS['accent_secondary'])])
        
        # Secondary button style
        style.configure('Secondary.TButton',
                       background=self.COLORS['bg_tertiary'],
                       foreground=self.COLORS['text_primary'],
                       borderwidth=1,
                       focuscolor='none',
                       padding=(10, 6))
        
        # Frame styles
        style.configure('Modern.TFrame',
                       background=self.COLORS['bg_secondary'],
                       borderwidth=1,
                       relief='solid')
        
        # Notebook styles
        style.configure('Modern.TNotebook',
                       background=self.COLORS['bg_primary'],
                       borderwidth=0)
        
        style.configure('Modern.TNotebook.Tab',
                       background=self.COLORS['bg_tertiary'],
                       foreground=self.COLORS['text_primary'],
                       padding=(12, 8))
        
        style.map('Modern.TNotebook.Tab',
                  background=[('selected', self.COLORS['accent_primary']),
                             ('active', self.COLORS['bg_secondary'])])
    
    def _load_locale(self) -> dict:
        """Load localization strings."""
        pass
    
    def t(self, key: str, fallback: str) -> str:
        """Get localized string."""
        return self.locale.get(key, fallback)
    
    def _build_ui(self):
        """Build the complete modern user interface."""
        # Top toolbar
        self._build_toolbar()
        
        # Main content area with panels
        self._build_main_content()
        
        # Status bar
        self._build_status_bar()
    
    def _build_toolbar(self):
        """Build the top toolbar with primary controls."""
        toolbar_frame = tk.Frame(self.root, bg=self.COLORS['bg_secondary'], height=60)
        toolbar_frame.grid(row=0, column=0, sticky='ew', padx=0, pady=0)
        toolbar_frame.grid_propagate(False)
        
        # Left side - Primary controls
        left_frame = tk.Frame(toolbar_frame, bg=self.COLORS['bg_secondary'])
        left_frame.pack(side='left', padx=20, pady=15)
        
        self.start_button = ttk.Button(
            left_frame,
            text="🎥 Start Stream",
            style='Modern.TButton',
            command=self._on_start_stream
        )
        self.start_button.pack(side='left', padx=(0, 10))

        self.stop_button = ttk.Button(
            left_frame,
            text="⏹ Stop",
            style='Secondary.TButton',
            command=self._on_stop_stream,
            state='disabled'
        )
        self.stop_button.pack(side='left', padx=(0, 10))

        self.save_frame_button = ttk.Button(
            left_frame,
            text="💾 Save Frame",
            style='Secondary.TButton',
            command=self._on_save_frame,
            state='disabled'
        )
        self.save_frame_button.pack(side='left', padx=(0, 10))

        # Right side - Settings and help
        right_frame = tk.Frame(toolbar_frame, bg=self.COLORS['bg_secondary'])
        right_frame.pack(side='right', padx=20, pady=15)
        
        ttk.Button(
            right_frame, 
            text="⚙ Settings",
            style='Secondary.TButton',
            command=self._open_settings
        ).pack(side='right', padx=(10, 0))
    
    def _build_main_content(self):
        """Build the main content area with three panels."""
        # Main content frame
        main_frame = tk.Frame(self.root, bg=self.COLORS['bg_primary'])
        main_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=10)
        
        # Configure main frame grid
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=2)  # Video panel (larger)
        main_frame.grid_columnconfigure(1, weight=1)  # Right panels
        
        # Left side - Video panel
        self._build_video_panel(main_frame)
        
        # Right side - Tabbed panels
        self._build_right_panels(main_frame)
    
    def _build_video_panel(self, parent):
        """Build the main video display panel with video stream and reference image."""
        # Container for left panel with both video and reference
        left_container = tk.Frame(parent, bg=self.COLORS['bg_primary'])
        left_container.grid(row=0, column=0, sticky='nsew', padx=(0, 5))

        # Configure left container to split vertically (60% video, 40% reference)
        left_container.grid_rowconfigure(0, weight=3)  # Video stream (60%)
        left_container.grid_rowconfigure(1, weight=2)  # Reference image (40%)
        left_container.grid_columnconfigure(0, weight=1)

        # === Top: Video Stream Panel ===
        video_frame = tk.Frame(left_container, bg=self.COLORS['bg_secondary'], relief='solid', bd=1)
        video_frame.grid(row=0, column=0, sticky='nsew', pady=(0, 5))

        # Title bar
        title_frame = tk.Frame(video_frame, bg=self.COLORS['bg_tertiary'], height=30)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text="📹 Live Video Stream",
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 10, 'bold')
        )
        title_label.pack(side='left', padx=10, pady=5)

        # Video display area
        video_content = tk.Frame(video_frame, bg=self.COLORS['bg_primary'])
        video_content.pack(fill='both', expand=True, padx=5, pady=5)

        # Optimized canvas for video display
        self._video_canvas = OptimizedCanvas(
            video_content,
            bg='black',
            highlightthickness=0,
            target_fps=30
        )
        self._video_canvas.pack(fill='both', expand=True)

        # Set video canvas to medium quality for optimal performance
        self._video_canvas.set_render_quality('medium')

        # Create right-click context menu for video canvas
        self._create_video_context_menu()

        # Bind right-click event to video canvas
        self._video_canvas.bind("<Button-3>", self._on_video_right_click)

        # Video controls
        controls_frame = tk.Frame(video_frame, bg=self.COLORS['bg_secondary'], height=40)
        controls_frame.pack(fill='x')
        controls_frame.pack_propagate(False)

        # Camera name, resolution and FPS indicators
        self.camera_name_label = tk.Label(
            controls_frame,
            text="Camera: --",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8)
        )
        self.camera_name_label.pack(side='left', padx=10, pady=10)

        self.resolution_label = tk.Label(
            controls_frame,
            text="Resolution: --",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8)
        )
        self.resolution_label.pack(side='left', padx=(20, 10), pady=10)

        self.fps_label = tk.Label(
            controls_frame,
            text="FPS: --",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8)
        )
        self.fps_label.pack(side='left', padx=(20, 10), pady=10)

        # === Bottom: Reference Image Panel ===
        reference_frame = tk.Frame(left_container, bg=self.COLORS['bg_secondary'], relief='solid', bd=1)
        reference_frame.grid(row=1, column=0, sticky='nsew')

        # Title and controls
        ref_header_frame = tk.Frame(reference_frame, bg=self.COLORS['bg_tertiary'], height=30)
        ref_header_frame.pack(fill='x')
        ref_header_frame.pack_propagate(False)

        tk.Label(
            ref_header_frame,
            text="📋 Reference Image",
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 10, 'bold')
        ).pack(side='left', padx=10, pady=5)

        # Reference control buttons
        ref_controls_frame = tk.Frame(reference_frame, bg=self.COLORS['bg_secondary'], height=35)
        ref_controls_frame.pack(fill='x', padx=5, pady=2)
        ref_controls_frame.pack_propagate(False)

        ttk.Button(
            ref_controls_frame,
            text="📹 From Stream",
            style='Secondary.TButton',
            command=self._load_reference_from_stream,
            width=15
        ).pack(side='left', padx=(0, 2))

        ttk.Button(
            ref_controls_frame,
            text="📁 Load Image",
            style='Secondary.TButton',
            command=self._load_reference_image,
            width=15
        ).pack(side='left', padx=2)

        ttk.Button(
            ref_controls_frame,
            text="🗑 Clear",
            style='Secondary.TButton',
            command=self._clear_reference_image,
            width=8
        ).pack(side='left', padx=2)

        # Reference image display
        ref_image_frame = tk.Frame(reference_frame, bg=self.COLORS['bg_primary'])
        ref_image_frame.pack(fill='both', expand=True, padx=5, pady=(0, 2))

        self._reference_canvas = OptimizedCanvas(
            ref_image_frame,
            bg='black',
            highlightthickness=0
        )
        self._reference_canvas.pack(fill='both', expand=True)
        self._reference_canvas.set_render_quality('high')

        # Reference image info
        ref_info_frame = tk.Frame(reference_frame, bg=self.COLORS['bg_secondary'], height=25)
        ref_info_frame.pack(fill='x')
        ref_info_frame.pack_propagate(False)

        self.ref_info_label = tk.Label(
            ref_info_frame,
            text="No reference image loaded",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8)
        )
        self.ref_info_label.pack(padx=10, pady=5)
    
    def _build_right_panels(self, parent):
        """Build the right side tabbed panels."""
        right_frame = tk.Frame(parent, bg=self.COLORS['bg_primary'])
        right_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0))
        
        # Configure right frame
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(right_frame, style='Modern.TNotebook')
        notebook.grid(row=0, column=0, sticky='nsew')

        # Objects Training Tab
        objects_frame = self._build_objects_panel()
        notebook.add(objects_frame, text="🎯 Objects")

        # Chat Analysis Tab
        chat_frame = self._build_chat_panel()
        notebook.add(chat_frame, text="🤖 Analysis")
    
    
    def _build_objects_panel(self):
        """Build the comprehensive objects training panel."""
        panel = tk.Frame(bg=self.COLORS['bg_secondary'])
        
        # Title and controls
        header_frame = tk.Frame(panel, bg=self.COLORS['bg_tertiary'], height=40)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame,
            text="Object Training",
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 9, 'bold')
        ).pack(side='left', padx=10, pady=10)
        
        # Main content - split into two sections
        main_paned = tk.PanedWindow(panel, orient='vertical', bg=self.COLORS['bg_secondary'])
        main_paned.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Top section - Image capture and display
        top_frame = tk.Frame(main_paned, bg=self.COLORS['bg_secondary'])
        main_paned.add(top_frame, height=300)
        
        self._build_objects_capture_section(top_frame)
        
        # Bottom section - Objects management
        bottom_frame = tk.Frame(main_paned, bg=self.COLORS['bg_secondary'])
        main_paned.add(bottom_frame, height=200)
        
        self._build_objects_management_section(bottom_frame)
        
        return panel
    
    def _build_objects_capture_section(self, parent):
        """Build the image capture and selection section."""
        # Control buttons
        controls_frame = tk.Frame(parent, bg=self.COLORS['bg_secondary'], height=50)
        controls_frame.pack(fill='x', pady=(0, 5))
        controls_frame.pack_propagate(False)
        
        ttk.Button(
            controls_frame,
            text="📷 Capture Image",
            style='Modern.TButton',
            command=self._capture_for_training
        ).pack(side='left', padx=(10, 5), pady=10)
        
        ttk.Button(
            controls_frame,
            text="📁 Load Image",
            style='Secondary.TButton',
            command=self._load_image_for_training
        ).pack(side='left', padx=5, pady=10)
        
        ttk.Button(
            controls_frame,
            text="✂️ Select Object",
            style='Modern.TButton',
            command=self._start_object_selection
        ).pack(side='left', padx=5, pady=10)

        
        ttk.Button(
            controls_frame,
            text="🎲 Data Augmentation",
            style='Secondary.TButton',
            command=self._open_augmentation_dialog
        ).pack(side='left', padx=2, pady=5)

        
        # Image display
        image_frame = tk.Frame(parent, bg=self.COLORS['bg_primary'])
        image_frame.pack(fill='both', expand=True, padx=5, pady=(0, 5))
        
        self._objects_canvas = OptimizedCanvas(
            image_frame,
            bg='black',
            highlightthickness=0,
            relief='sunken',
            bd=2
        )
        self._objects_canvas.pack(fill='both', expand=True)
        self._objects_canvas.set_render_quality('medium')
        
        # Initialize object selector (DEPRECATED - now using BboxDrawingDialog)
        # Kept for backward compatibility, but no longer actively used
        self._object_selector = ObjectSelector(
            self._objects_canvas,
            lambda bbox: None  # Dummy callback, not used anymore
        )
        
        # Status label
        status_frame = tk.Frame(parent, bg=self.COLORS['bg_secondary'], height=30)
        status_frame.pack(fill='x')
        status_frame.pack_propagate(False)
        
        self.objects_status_label = tk.Label(
            status_frame,
            text="Capture or load an image to start training objects",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8)
        )
        self.objects_status_label.pack(padx=10, pady=5)
    
    def _build_objects_management_section(self, parent):
        """Build the objects management section."""
        # Objects list frame
        list_frame = tk.Frame(parent, bg=self.COLORS['bg_secondary'])
        list_frame.pack(fill='both', expand=True, padx=5)
        
        # List header
        list_header = tk.Frame(list_frame, bg=self.COLORS['bg_tertiary'], height=30)
        list_header.pack(fill='x')
        list_header.pack_propagate(False)
        
        tk.Label(
            list_header,
            text="Training Objects",
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 9, 'bold')
        ).pack(side='left', padx=10, pady=5)
        
        # Objects count label
        self.objects_count_label = tk.Label(
            list_header,
            text="(0 objects)",
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8)
        )
        self.objects_count_label.pack(side='right', padx=10, pady=5)
        
        # Listbox with scrollbar
        listbox_frame = tk.Frame(list_frame, bg=self.COLORS['bg_primary'])
        listbox_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self._objects_listbox = tk.Listbox(
            listbox_frame,
            bg=self.COLORS['bg_primary'],
            fg=self.COLORS['text_primary'],
            selectbackground=self.COLORS['accent_primary'],
            selectforeground=self.COLORS['text_primary'],
            font=('Segoe UI', 9),
            selectmode=tk.SINGLE,
            borderwidth=0,
            highlightthickness=0
        )
        
        scrollbar = ttk.Scrollbar(listbox_frame, orient='vertical')
        self._objects_listbox.configure(yscrollcommand=scrollbar.set)
        scrollbar.configure(command=self._objects_listbox.yview)
        
        self._objects_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Add mouse wheel support for objects listbox
        self._objects_listbox.bind('<MouseWheel>', self._on_objects_listbox_mousewheel)

        # Add click-to-view: single click on object displays it
        self._objects_listbox.bind('<<ListboxSelect>>', self._on_listbox_item_clicked)
        
        # Management buttons
        buttons_frame = tk.Frame(list_frame, bg=self.COLORS['bg_secondary'], height=40)
        buttons_frame.pack(fill='x', pady=(5, 0))
        buttons_frame.pack_propagate(False)

        # Selection control buttons
        ttk.Button(
            buttons_frame,
            text="☑ Select All",
            style='Secondary.TButton',
            command=self._select_all_objects
        ).pack(side='left', padx=(5, 2), pady=5)

        ttk.Button(
            buttons_frame,
            text="☐ Deselect All",
            style='Secondary.TButton',
            command=self._deselect_all_objects
        ).pack(side='left', padx=2, pady=5)

        ttk.Button(
            buttons_frame,
            text="✏️ Edit",
            style='Secondary.TButton',
            command=self._edit_selected_object
        ).pack(side='left', padx=2, pady=5)

        ttk.Button(
            buttons_frame,
            text="🗑️ Delete",
            style='Secondary.TButton',
            command=self._delete_selected_objects
        ).pack(side='left', padx=2, pady=5)

        # Note: View button removed - clicking on an item in the list now displays it automatically

        ttk.Button(
            buttons_frame,
            text="🚀 Train Model",
            style='Modern.TButton',
            command=self._train_model_with_all_objects
        ).pack(side='right', padx=(2, 5), pady=5)
        
        # Load initial objects
        self._refresh_objects_list()
    
    def _build_chat_panel(self):
        """Build the modern ChatBot interface with bubble-style messages."""
        panel = tk.Frame(bg=self.COLORS['bg_secondary'])

        # Enhanced header with status indicators
        header_frame = tk.Frame(panel, bg=self.COLORS['bg_tertiary'], height=50)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)

        # Title and status
        title_frame = tk.Frame(header_frame, bg=self.COLORS['bg_tertiary'])
        title_frame.pack(side='left', fill='y', padx=15, pady=8)

        tk.Label(
            title_frame,
            text="AI Analysis Chat",
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 11, 'bold')
        ).pack(anchor='w')

        # Connection status indicator
        self._chat_status_label = tk.Label(
            title_frame,
            text="● Ready" if self._gemini_configured else "● Not Configured",
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['success'] if self._gemini_configured else self.COLORS['warning'],
            font=('Segoe UI', 9)
        )
        self._chat_status_label.pack(anchor='w')

        # Chat controls
        controls_frame = tk.Frame(header_frame, bg=self.COLORS['bg_tertiary'])
        controls_frame.pack(side='right', fill='y', padx=15, pady=10)

        clear_btn = tk.Button(
            controls_frame,
            text="Clear Chat",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 9),
            borderwidth=0,
            padx=12,
            pady=6,
            command=self._clear_chat,
            cursor='hand2',
            relief='flat'
        )
        clear_btn.pack(side='right')

        # Hover effects for clear button
        def on_clear_enter(e):
            clear_btn.configure(bg=self.COLORS['bg_primary'])

        def on_clear_leave(e):
            clear_btn.configure(bg=self.COLORS['bg_secondary'])

        clear_btn.bind('<Enter>', on_clear_enter)
        clear_btn.bind('<Leave>', on_clear_leave)

        # Chat display area with modern bubble messages
        chat_display_frame = tk.Frame(panel, bg='#2B2B2B')
        chat_display_frame.pack(fill='both', expand=True, padx=0, pady=0)

        # Create canvas for scrollable chat
        self._chat_canvas = tk.Canvas(
            chat_display_frame,
            bg='#2B2B2B',
            highlightthickness=0,
            borderwidth=0
        )

        self._chat_scrollbar = ttk.Scrollbar(
            chat_display_frame,
            orient='vertical',
            command=self._chat_canvas.yview
        )
        self._chat_canvas.configure(yscrollcommand=self._chat_scrollbar.set)

        # Scrollable frame for messages
        self._messages_frame = tk.Frame(self._chat_canvas, bg='#2B2B2B')
        self._messages_frame.bind(
            '<Configure>',
            lambda e: self._chat_canvas.configure(scrollregion=self._chat_canvas.bbox("all"))
        )

        # Bind canvas resize to update messages frame width
        def _on_canvas_configure(event):
            canvas_width = event.width
            self._chat_canvas.itemconfig(self._messages_canvas_window, width=canvas_width)
            self._chat_canvas.configure(scrollregion=self._chat_canvas.bbox("all"))

        self._chat_canvas.bind('<Configure>', _on_canvas_configure)

        self._messages_canvas_window = self._chat_canvas.create_window(
            (0, 0),
            window=self._messages_frame,
            anchor="nw"
        )

        self._chat_canvas.pack(side="left", fill="both", expand=True)
        self._chat_scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel scrolling
        self._chat_canvas.bind('<MouseWheel>', self._on_chat_mousewheel)
        self._messages_frame.bind('<MouseWheel>', self._on_chat_mousewheel)

        # Modern input area
        input_container = tk.Frame(panel, bg='#2D2D2D', height=100)
        input_container.pack(fill='x', padx=0, pady=0)
        input_container.pack_propagate(False)

        # Separator line
        separator = tk.Frame(input_container, bg='#404040', height=1)
        separator.pack(fill='x')

        # Input field container
        input_field_container = tk.Frame(input_container, bg='#2D2D2D')
        input_field_container.pack(fill='both', expand=True, padx=15, pady=12)

        # Text input frame with border
        text_input_border = tk.Frame(input_field_container, bg='#404040')
        text_input_border.pack(side='left', fill='both', expand=True, padx=(0, 10))

        text_input_inner = tk.Frame(text_input_border, bg='#252525')
        text_input_inner.pack(fill='both', expand=True, padx=1, pady=1)

        # Text input with scrollbar
        text_scroll_frame = tk.Frame(text_input_inner, bg='#252525')
        text_scroll_frame.pack(fill='both', expand=True)

        self._chat_input = tk.Text(
            text_scroll_frame,
            height=3,
            bg='#252525',
            fg='#E0E0E0',
            font=('Segoe UI', 10),
            borderwidth=0,
            insertbackground='#E0E0E0',
            wrap=tk.WORD,
            padx=10,
            pady=8,
            relief='flat'
        )

        self._chat_input_scrollbar = ttk.Scrollbar(
            text_scroll_frame,
            orient='vertical',
            command=self._chat_input.yview
        )
        self._chat_input.configure(yscrollcommand=self._chat_input_scrollbar.set)

        self._chat_input.pack(side='left', fill='both', expand=True)
        self._chat_input_scrollbar.pack(side='right', fill='y')

        # Focus border effect
        def on_input_focus(e):
            text_input_border.configure(bg=self.COLORS['accent_primary'])

        def on_input_unfocus(e):
            text_input_border.configure(bg='#404040')

        self._chat_input.bind('<FocusIn>', on_input_focus)
        self._chat_input.bind('<FocusOut>', on_input_unfocus)

        # Bind events
        self._chat_input.bind('<Return>', self._on_chat_send_enhanced)
        self._chat_input.bind('<Shift-Return>', self._on_chat_newline)
        self._chat_input.bind('<KeyRelease>', self._on_chat_typing)
        self._chat_input.bind('<MouseWheel>', self._on_chat_input_mousewheel)

        # Send button
        self._send_button = tk.Button(
            input_field_container,
            text="Send",
            bg=self.COLORS['accent_primary'],
            fg='#FFFFFF',
            font=('Segoe UI', 10, 'bold'),
            borderwidth=0,
            padx=24,
            pady=10,
            command=self._on_chat_send_enhanced,
            cursor='hand2',
            relief='flat',
            state='normal'
        )
        self._send_button.pack(side='right')

        # Send button hover effects
        def on_send_enter(e):
            if self._send_button['state'] == 'normal':
                self._send_button.configure(bg=self.COLORS['accent_secondary'])

        def on_send_leave(e):
            self._send_button.configure(bg=self.COLORS['accent_primary'])

        self._send_button.bind('<Enter>', on_send_enter)
        self._send_button.bind('<Leave>', on_send_leave)

        # Initialize message state tracking
        self._message_widgets = []
        self._typing_indicator = None
        self._last_message_id = 0

        # Add welcome message
        self.root.after(100, self._add_welcome_message)

        return panel
    
    def _build_status_bar(self):
        """Build the status bar at the bottom."""
        status_frame = tk.Frame(self.root, bg=self.COLORS['bg_tertiary'], height=25)
        status_frame.grid(row=2, column=0, sticky='ew')
        status_frame.grid_propagate(False)
        
        self.status_label = tk.Label(
            status_frame,
            text="Ready",
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_secondary'],
            font=('Segoe UI', 8),
            anchor='w'
        )
        self.status_label.pack(side='left', padx=10, fill='x', expand=True)
        
        # Connection status indicator
        self.connection_label = tk.Label(
            status_frame,
            text="⚪ Disconnected",
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8)
        )
        self.connection_label.pack(side='right', padx=10)

    # ========== VIDEO STREAM HANDLERS ==========

    def _on_start_stream(self):
        """Handle start stream button click."""
        try:
            if self.webcam_service.start_stream():
                self.start_button.config(state='disabled')
                self.stop_button.config(state='normal')
                self.save_frame_button.config(state='normal')
                self.status_label.config(text="Webcam stream started")
                self.connection_label.config(text="🟢 Connected", fg=self.COLORS['success'])
                logger.info("Webcam stream started")
            else:
                messagebox.showerror("Error", "Failed to start webcam stream")

        except Exception as e:
            logger.error(f"Error starting stream: {e}")
            messagebox.showerror("Error", f"Failed to start stream: {e}")

    def _on_stop_stream(self):
        """Handle stop stream button click.

        The last captured frame remains visible on the canvas so users
        can work with it (save, capture, annotate, etc.). The frame
        stays accessible through the webcam service until the next stream starts.
        """
        try:
            self.webcam_service.stop_stream()
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            # Keep save_frame_button enabled so users can save the last frame
            self.status_label.config(text="Webcam stream stopped - last frame preserved")
            self.connection_label.config(text="⚪ Disconnected", fg=self.COLORS['text_muted'])

            # DO NOT clear canvas - keep last frame visible for user operations
            # DO NOT clear current_frame - keep it accessible for save/capture operations

            logger.info("Webcam stream stopped - last frame kept on display")

        except Exception as e:
            logger.error(f"Error stopping stream: {e}")

    def _on_save_frame(self) -> None:
        """
        Handle save frame button click to save the current video stream frame.

        Retrieves the current frame from the webcam service, prompts the user
        for a save location using a file dialog, and saves the frame to disk
        in the selected format (JPEG or PNG).

        Features:
        - Validates frame availability before prompting for save location
        - Suggests timestamped filename for organization
        - Supports multiple image formats (JPEG, PNG)
        - Provides user feedback on success or failure
        - Comprehensive error handling with logging

        Returns:
            None
        """
        try:
            # Get current frame from webcam service
            frame = self.webcam_service.get_current_frame()

            # Validate frame availability
            if frame is None:
                messagebox.showerror(
                    "No Frame Available",
                    "No video frame is currently available to save.\n\n"
                    "Please ensure the video stream is active and displaying frames."
                )
                logger.warning("Save frame attempted but no frame available")
                return

            # Generate timestamp for default filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"frame_{timestamp}.jpg"

            # Open file save dialog
            file_path = filedialog.asksaveasfilename(
                title="Save Video Frame",
                initialfile=default_filename,
                defaultextension=".jpg",
                filetypes=[
                    ("JPEG Image", "*.jpg"),
                    ("PNG Image", "*.png"),
                    ("All files", "*.*")
                ]
            )

            # Check if user cancelled the dialog
            if not file_path:
                logger.info("Save frame cancelled by user")
                return

            # Save the frame using OpenCV
            success = cv2.imwrite(file_path, frame)

            if success:
                # Show success message
                messagebox.showinfo(
                    "Frame Saved",
                    f"Video frame saved successfully!\n\n"
                    f"Location: {file_path}"
                )
                logger.info(f"Frame saved successfully to: {file_path}")
            else:
                # Save operation failed
                messagebox.showerror(
                    "Save Failed",
                    f"Failed to save frame to:\n{file_path}\n\n"
                    "Please check the file path and try again."
                )
                logger.error(f"Failed to save frame to: {file_path}")

        except PermissionError as e:
            messagebox.showerror(
                "Permission Denied",
                f"Permission denied when saving frame.\n\n"
                f"Error: {e}\n\n"
                "Please check file permissions and try again."
            )
            logger.error(f"Permission error saving frame: {e}")

        except Exception as e:
            messagebox.showerror(
                "Error Saving Frame",
                f"An unexpected error occurred while saving the frame:\n\n"
                f"{type(e).__name__}: {e}"
            )
            logger.error(f"Error saving frame: {type(e).__name__}: {e}", exc_info=True)

    def _update_video_stream(self):
        """Update video display with current frame."""
        try:
            if self.webcam_service.is_streaming():
                frame = self.webcam_service.get_current_frame()

                if frame is not None:
                    # Apply live detection if active
                    if self._live_detection_active:
                        frame = self._apply_live_detection(frame)

                    # Display frame on canvas
                    self._video_canvas.display_image(frame)

                    # Update camera info, FPS and resolution
                    camera_name = self.webcam_service.get_camera_name()
                    fps = self.webcam_service.get_fps()
                    width, height = self.webcam_service.get_resolution()

                    self.camera_name_label.config(text=f"Camera: {camera_name}")
                    self.fps_label.config(text=f"FPS: {fps:.1f}")
                    self.resolution_label.config(text=f"Resolution: {width}x{height}")

        except Exception as e:
            logger.error(f"Error updating video stream: {e}")

        # Schedule next update
        self.root.after(33, self._update_video_stream)  # ~30 FPS

    # ========== VIDEO CONTEXT MENU AND LIVE DETECTION ==========

    def _create_video_context_menu(self):
        """Create right-click context menu for video canvas."""
        self._video_context_menu = tk.Menu(self.root, tearoff=0)
        self._video_context_menu.add_command(
            label="🔍 Test Model (Live Detection)",
            command=self._toggle_live_detection
        )
        self._video_context_menu.add_separator()
        self._video_context_menu.add_command(
            label="📸 Capture Frame",
            command=self._capture_for_training
        )

    def _on_video_right_click(self, event):
        """Show context menu on right-click."""
        try:
            self._video_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._video_context_menu.grab_release()

    def _toggle_live_detection(self):
        """Toggle live object detection on video stream."""
        if self._live_detection_active:
            # Stop live detection
            self._live_detection_active = False
            self._video_context_menu.entryconfig(
                0,  # First menu item
                label="🔍 Test Model (Live Detection)"
            )
            self.status_label.config(text="Live detection stopped")
            logger.info("Live detection stopped")
        else:
            # Check if inference service is loaded
            if not self.inference_service:
                messagebox.showerror(
                    "Service Not Available",
                    "Inference service is not initialized."
                )
                return

            if not self.inference_service.is_loaded():
                response = messagebox.askyesno(
                    "Model Not Loaded",
                    "No model is currently loaded. Would you like to load the default model?\n\n"
                    "This will load the YOLO model for object detection."
                )
                if response:
                    # Try to load the model
                    success = self.inference_service.load_model()
                    if not success:
                        messagebox.showerror(
                            "Model Load Failed",
                            "Failed to load the detection model. Please check the logs."
                        )
                        return
                else:
                    return

            # Check if camera is streaming
            if not self.webcam_service.is_streaming():
                messagebox.showwarning(
                    "Camera Not Running",
                    "Please start the webcam stream first before testing live detection."
                )
                return

            # Start detection
            self._live_detection_active = True
            self._video_context_menu.entryconfig(
                0,  # First menu item
                label="✅ Test Model (Live Detection) - Active"
            )
            self.status_label.config(text="Live detection active - right-click to stop")
            logger.info("Live detection started")

    def _apply_live_detection(self, frame: np.ndarray) -> np.ndarray:
        """Apply object detection to frame and draw bounding boxes with overlays.

        Args:
            frame: Input frame from webcam

        Returns:
            Annotated frame with detections and overlays
        """
        # Skip if detection is already in progress to avoid performance issues
        if self._detection_in_progress:
            return frame

        try:
            self._detection_in_progress = True

            # Log frame dimensions
            h, w = frame.shape[:2]
            logger.info(f"[LIVE DETECTION] Processing frame: {w}x{h}")

            # Run inference
            detections = self.inference_service.detect(frame)
            logger.info(f"[LIVE DETECTION] Received {len(detections)} detections from inference service")

            # Draw detections and overlays
            annotated_frame = self._draw_live_detections(frame, detections)

            return annotated_frame

        except Exception as e:
            logger.error(f"Error in live detection: {e}", exc_info=True)
            return frame  # Return original frame on error
        finally:
            self._detection_in_progress = False

    def _draw_live_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detection bounding boxes, labels, and status overlay on frame.

        Args:
            frame: Input frame
            detections: List of detection dictionaries from inference service

        Returns:
            Annotated frame with all visual elements
        """
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]

        logger.info(f"[DRAW DETECTIONS] Drawing {len(detections)} detections on {w}x{h} frame")

        # Draw bounding boxes and labels for each detection
        for idx, detection in enumerate(detections):
            bbox = detection.get('bbox')  # [x1, y1, x2, y2]
            label = detection.get('class_name', 'Unknown')
            confidence = detection.get('confidence', 0.0)

            logger.info(f"[DRAW DETECTIONS] Detection {idx}: label='{label}', confidence={confidence:.3f}")
            logger.info(f"[DRAW DETECTIONS] Detection {idx}: Raw bbox from detection dict: {bbox}")

            if bbox is None:
                logger.warning(f"[DRAW DETECTIONS] Detection {idx}: Bbox is None, skipping")
                continue

            x1, y1, x2, y2 = map(int, bbox)

            logger.info(f"[DRAW DETECTIONS] Detection {idx}: Drawing bbox at ({x1},{y1})-({x2},{y2})")
            logger.info(f"[DRAW DETECTIONS] Detection {idx}: Bbox size: {x2-x1}x{y2-y1} pixels")
            logger.info(f"[DRAW DETECTIONS] Detection {idx}: Bbox size vs frame: "
                      f"{(x2-x1)/w*100:.1f}% width x {(y2-y1)/h*100:.1f}% height")

            # Validate bbox coordinates
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                logger.warning(f"[DRAW DETECTIONS] Detection {idx}: BBOX OUT OF BOUNDS! "
                             f"({x1},{y1})-({x2},{y2}) for {w}x{h} frame")

            if x2 <= x1 or y2 <= y1:
                logger.error(f"[DRAW DETECTIONS] Detection {idx}: INVALID BBOX DIMENSIONS! "
                           f"({x1},{y1})-({x2},{y2})")
                continue

            if (x2 - x1) > w * 0.8 or (y2 - y1) > h * 0.8:
                logger.warning(f"[DRAW DETECTIONS] Detection {idx}: VERY LARGE BBOX! "
                             f"Covers {(x2-x1)/w*100:.1f}% x {(y2-y1)/h*100:.1f}% of frame")

            # Get consistent color for this class
            color = self._get_class_color(label)

            # Draw bounding box with thicker line for visibility
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Prepare label text
            label_text = f"{label} {confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Draw label background (filled rectangle)
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1  # Filled
            )

            # Draw label text (white for contrast)
            cv2.putText(
                annotated_frame,
                label_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                1,
                cv2.LINE_AA
            )

        # Draw status overlay
        annotated_frame = self._draw_detection_overlay(annotated_frame, detections)

        logger.info(f"[DRAW DETECTIONS] Completed drawing all detections")

        return annotated_frame

    def _draw_detection_overlay(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw status overlay showing live detection info.

        Args:
            frame: Input frame
            detections: List of detections

        Returns:
            Frame with overlay
        """
        # Draw semi-transparent background for overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (250, 75), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Draw "LIVE DETECTION" indicator
        cv2.putText(
            frame,
            "LIVE DETECTION",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),  # Green
            2,
            cv2.LINE_AA
        )

        # Draw detection count
        detection_count = len(detections)
        cv2.putText(
            frame,
            f"Objects: {detection_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),  # Green
            1,
            cv2.LINE_AA
        )

        return frame

    def _get_class_color(self, class_name: str) -> tuple:
        """Get consistent color for each class.

        Args:
            class_name: Name of the detected class

        Returns:
            BGR color tuple
        """
        if class_name not in self._class_colors:
            # Generate random but consistent color based on class name hash
            import random
            random.seed(hash(class_name))
            color = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )
            self._class_colors[class_name] = color

        return self._class_colors[class_name]

    # ========== REFERENCE TAB HANDLERS ==========

    def _auto_display_reference(self):
        """Automatically display reference image on startup if one exists."""
        try:
            # Check if reference manager has a reference image
            if self.reference_manager.has_reference():
                logger.info("Auto-displaying reference image on startup")
                self._display_reference_image()
                self.status_label.config(text="Reference image loaded from previous session")
            else:
                logger.info("No reference image found for auto-display")
        except Exception as e:
            logger.error(f"Error auto-displaying reference image: {e}")
            # Don't show error to user as this is a non-critical startup operation

    def _load_reference_image(self):
        """Load reference image from file and automatically set as default reference."""
        try:
            filepath = filedialog.askopenfilename(
                title="Select Reference Image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
            )

            if filepath:
                if self.reference_manager.load_reference_from_file(filepath):
                    self._display_reference_image()

                    # Auto-set as default reference
                    self.config['reference_image_path'] = filepath
                    self._save_config()
                    logger.info(f"Reference image auto-set as default: {filepath}")

                    # Update status with success indicator
                    import os
                    filename = os.path.basename(filepath)
                    self.status_label.config(text=f"✓ Reference image loaded and set as default: {filename}")
                else:
                    messagebox.showerror("Error", "Failed to load reference image")

        except Exception as e:
            logger.error(f"Error loading reference: {e}")
            messagebox.showerror("Error", f"Failed to load reference: {e}")

    def _load_reference_from_stream(self):
        """Load reference image from current webcam stream frame.

        This method captures the current frame from the video stream, saves it to disk,
        and automatically sets it as the default reference image. The image is both
        saved to the reference directory and displayed in the reference canvas.
        """
        try:
            # Check if camera is streaming
            if not self.webcam_service.is_streaming():
                messagebox.showwarning(
                    "Camera Not Running",
                    "Please start the webcam stream first before loading a frame as reference."
                )
                return

            # Get current frame from webcam
            frame = self.webcam_service.get_current_frame()

            if frame is None:
                messagebox.showwarning(
                    "No Frame Available",
                    "No frame is currently available from the video stream. Please try again."
                )
                return

            # Set frame as reference image and save to disk
            if self.reference_manager.set_reference_from_array(frame.copy(), save=True):
                # Display the reference image immediately
                self._display_reference_image()

                # Get the saved path and store in config (auto-set as default reference)
                ref_info = self.reference_manager.get_reference_info()
                if ref_info and ref_info.get('path'):
                    self.config['reference_image_path'] = ref_info['path']
                    self._save_config()
                    logger.info(f"Reference image auto-set as default: {ref_info['path']}")

                # Update status
                self.status_label.config(text="✓ Reference image captured and set as default")
                logger.info("Reference image loaded from current video stream frame and saved")
            else:
                messagebox.showerror(
                    "Failed to Load",
                    "Failed to set the current frame as reference image."
                )

        except Exception as e:
            logger.error(f"Error loading reference from stream: {e}")
            messagebox.showerror(
                "Error",
                f"Failed to load reference from stream: {e}"
            )

    def _clear_reference_image(self):
        """Clear current reference image."""
        try:
            self.reference_manager.clear_reference()
            self._reference_canvas.clear()
            self.ref_info_label.config(text="No reference image loaded")
            self.status_label.config(text="Reference image cleared")

        except Exception as e:
            logger.error(f"Error clearing reference: {e}")

    def _display_reference_image(self):
        """Display reference image on canvas."""
        try:
            ref_image = self.reference_manager.get_reference()
            if ref_image is not None:
                self._reference_canvas.display_image(ref_image)

                # Update info label
                info = self.reference_manager.get_reference_info()
                if info:
                    self.ref_info_label.config(
                        text=f"{info['width']}x{info['height']} - {info.get('timestamp', 'Unknown')}"
                    )

        except Exception as e:
            logger.error(f"Error displaying reference: {e}")

    # ========== OBJECTS TAB HANDLERS ==========

    def _capture_for_training(self):
        """Capture frame with image source selection and annotate multiple objects.

        NEW MULTI-OBJECT WORKFLOW:
        1. User selects image source (video stream or reference image)
        2. User draws multiple bboxes on full frame
        3. User selects class for each object
        4. Store all objects from single annotation session
        """
        try:
            # Check available image sources
            has_video = self.webcam_service.is_streaming()
            has_reference = self.reference_manager.has_reference()

            if not has_video and not has_reference:
                messagebox.showwarning(
                    "No Image Source",
                    "Please start the webcam stream or load a reference image first."
                )
                return

            # Show image source selection dialog
            source_dialog = ImageSourceDialog(
                self.root,
                has_video_stream=has_video,
                has_reference=has_reference
            )
            source = source_dialog.show()

            if not source:
                return  # User cancelled

            # Get image based on selected source
            if source == 'video':
                frame = self.webcam_service.capture_frame()
                if frame is None:
                    messagebox.showwarning("Warning", "No video frame available")
                    return
                source_name = "video stream"
            elif source == 'reference':
                frame = self.reference_manager.get_reference()
                if frame is None:
                    messagebox.showwarning("Warning", "No reference image loaded")
                    return
                source_name = "reference image"
            else:
                return

            # Get existing classes for quick selection
            existing_classes = list(set(obj.label for obj in self.training_service.get_all_objects()))

            # Open multi-object bbox drawing dialog
            dialog = BboxDrawingDialog(self.root, frame, existing_classes=existing_classes)
            result = dialog.show()

            if result and len(result) > 0:
                # Result is now a list of objects with optional background region and segmentation
                # Generate a single image_id for all objects from this capture
                image_id = str(uuid.uuid4())

                # Add all objects to training service with the same image_id
                for obj_data in result:
                    self.training_service.add_object(
                        image=frame,  # FULL FRAME!
                        label=obj_data['class'],
                        bbox=obj_data['bbox'],  # Actual bbox coordinates
                        background_region=obj_data.get('background_region'),  # Optional background for augmentation
                        segmentation=obj_data.get('segmentation', []),  # YOLO segmentation points
                        threshold=obj_data.get('threshold'),  # Threshold value used
                        image_id=image_id  # NEW: Same ID for all objects from this image
                    )

                # Refresh UI
                self._refresh_objects_list()
                self.objects_status_label.config(
                    text=f"✓ {len(result)} object(s) added from {source_name}"
                )

                # Display annotated frame with all bboxes
                annotated_frame = frame.copy()
                for obj_data in result:
                    x1, y1, x2, y2 = obj_data['bbox']
                    class_name = obj_data['class']

                    # Draw bbox
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw label with background
                    cv2.putText(
                        annotated_frame,
                        class_name,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                self._objects_canvas.display_image(annotated_frame)

                logger.info(f"Added {len(result)} training objects from {source_name}")

        except Exception as e:
            logger.error(f"Error capturing for training: {e}")
            messagebox.showerror("Error", f"Failed to capture image: {e}")

    def _load_image_for_training(self):
        """Load and display an image file to the canvas.

        This method ONLY loads and displays the image - no automatic
        detection, processing, dialog opening, or object addition.

        Use the "Select Object" button after loading to annotate objects.
        """
        try:
            # Open file dialog to select image
            filepath = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
            )

            if not filepath:
                return

            # Load image from file
            image = cv2.imread(filepath)
            if image is None:
                messagebox.showerror("Error", "Failed to load image")
                return

            # Display the raw image on canvas (no annotations, no processing)
            self._objects_canvas.display_image(image)

            # Update status to indicate image is loaded and ready
            self.objects_status_label.config(
                text=f"✓ Image loaded: {filepath.split('/')[-1]} - Use 'Select Object' to annotate"
            )

            logger.info(f"Image loaded for display: {filepath}")

        except Exception as e:
            logger.error(f"Error loading image: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def _start_object_selection(self):
        """Start object selection with image source selection and multi-object support.

        NEW MULTI-OBJECT WORKFLOW:
        1. User selects image source (video stream or reference image)
        2. User draws multiple bboxes on full frame
        3. User selects class for each object
        4. Store all objects from single annotation session
        """
        try:
            # Check available image sources
            has_video = self.webcam_service.get_current_frame() is not None
            has_reference = self.reference_manager.has_reference()

            if not has_video and not has_reference:
                messagebox.showwarning(
                    "No Image Source",
                    "Please start the webcam stream or load a reference image first."
                )
                return

            # Show image source selection dialog
            source_dialog = ImageSourceDialog(
                self.root,
                has_video_stream=has_video,
                has_reference=has_reference
            )
            source = source_dialog.show()

            if not source:
                return  # User cancelled

            # Get image based on selected source
            if source == 'video':
                selected_image = self.webcam_service.get_current_frame()
                if selected_image is None:
                    messagebox.showwarning("Warning", "No video frame available")
                    return
                source_name = "video stream"
            elif source == 'reference':
                selected_image = self.reference_manager.get_reference()
                if selected_image is None:
                    messagebox.showwarning("Warning", "No reference image loaded")
                    return
                source_name = "reference image"
            else:
                return

            logger.info(f"Opening bbox drawing dialog for {source_name}")

            # Get existing classes for quick selection
            existing_classes = list(set(obj.label for obj in self.training_service.get_all_objects()))

            # Open multi-object bbox drawing dialog
            dialog = BboxDrawingDialog(self.root, selected_image, existing_classes=existing_classes)
            result = dialog.show()

            if result and len(result) > 0:
                # Result is now a list of objects with optional background region and segmentation
                # Generate a single image_id for all objects from this capture
                image_id = str(uuid.uuid4())

                # Add all objects to training service with the same image_id
                for obj_data in result:
                    self.training_service.add_object(
                        image=selected_image,  # FULL FRAME!
                        label=obj_data['class'],
                        bbox=obj_data['bbox'],  # Actual bbox coordinates
                        background_region=obj_data.get('background_region'),  # Optional background for augmentation
                        segmentation=obj_data.get('segmentation', []),  # YOLO segmentation points
                        threshold=obj_data.get('threshold'),  # Threshold value used
                        image_id=image_id  # NEW: Same ID for all objects from this image
                    )

                # Refresh UI
                self._refresh_objects_list()
                self.objects_status_label.config(
                    text=f"✓ {len(result)} object(s) added from {source_name}"
                )

                # Display annotated frame with all bboxes
                annotated_frame = selected_image.copy()
                for obj_data in result:
                    x1, y1, x2, y2 = obj_data['bbox']
                    class_name = obj_data['class']

                    # Draw bbox
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw label
                    cv2.putText(
                        annotated_frame,
                        class_name,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                self._objects_canvas.display_image(annotated_frame)

                logger.info(f"Added {len(result)} training objects from {source_name}")

        except Exception as e:
            logger.error(f"Error in object selection: {e}")
            messagebox.showerror("Error", f"Failed to select object: {e}")

    def _cleanup_multi_canvas_selection(self):
        """Clean up all active canvas selectors and remove visual feedback."""
        if hasattr(self, '_active_selectors'):
            for selector in self._active_selectors:
                selector.deactivate()
            self._active_selectors = []

        # Remove visual feedback (green borders)
        for canvas in [self._video_canvas, self._reference_canvas, self._objects_canvas]:
            canvas.config(highlightthickness=0)

        self._selection_active = False

    # DEPRECATED: Old object selection method (replaced by BboxDrawingDialog)
    # Kept for reference but no longer used
    # The new workflow uses _capture_for_training, _load_image_for_training,
    # and _start_object_selection methods instead

    def _refresh_objects_list(self):
        """Refresh the objects listbox with checkbox indicators.

        Maintains checkbox state after refresh by checking the checked_objects set.
        Displays checkboxes as: ☐ (unchecked) or ☑ (checked)
        """
        try:
            self._objects_listbox.delete(0, tk.END)

            objects = self.training_service.get_all_objects()
            for obj in objects:
                # Determine checkbox state based on checked_objects set
                checkbox = "☑" if obj.object_id in self.checked_objects else "☐"
                display_text = f"{checkbox} {obj.label} ({obj.object_id})"
                self._objects_listbox.insert(tk.END, display_text)

            # Update count
            counts = self.training_service.get_object_count()
            self.objects_count_label.config(
                text=f"({counts['total']} objects)"
            )

        except Exception as e:
            logger.error(f"Error refreshing objects list: {e}")

    def _select_all_objects(self):
        """Select all objects and check all checkboxes.

        Marks all items in the listbox as checked and updates the visual display.
        Adds all object IDs to the checked_objects set.
        """
        try:
            objects = self.training_service.get_all_objects()

            if not objects:
                messagebox.showinfo("Info", "No objects to select")
                return

            # Add all object IDs to checked set
            for obj in objects:
                self.checked_objects.add(obj.object_id)

            # Refresh display to show all checkboxes as checked
            self._refresh_objects_list()

            # Update status
            count = len(objects)
            self.status_label.config(text=f"Selected all {count} object(s)")
            logger.info(f"Selected all {count} objects")

        except Exception as e:
            logger.error(f"Error selecting all objects: {e}")
            messagebox.showerror("Error", f"Failed to select all objects: {e}")

    def _deselect_all_objects(self):
        """Deselect all objects and uncheck all checkboxes.

        Marks all items in the listbox as unchecked and updates the visual display.
        Clears the checked_objects set.
        """
        try:
            # Clear the checked objects set
            self.checked_objects.clear()

            # Refresh display to show all checkboxes as unchecked
            self._refresh_objects_list()

            # Update status
            self.status_label.config(text="Deselected all objects")
            logger.info("Deselected all objects")

        except Exception as e:
            logger.error(f"Error deselecting all objects: {e}")
            messagebox.showerror("Error", f"Failed to deselect all objects: {e}")

    def _edit_selected_object(self):
        """Edit selected object label."""
        try:
            selection = self._objects_listbox.curselection()
            if not selection:
                messagebox.showwarning("Warning", "Please select an object first")
                return

            idx = selection[0]
            objects = self.training_service.get_all_objects()

            if idx < len(objects):
                obj = objects[idx]

                # Simple input dialog
                new_label = simpledialog.askstring(
                    "Edit Label",
                    f"Enter new label for object:",
                    initialvalue=obj.label,
                    parent=self.root
                )

                if new_label and new_label.strip():
                    self.training_service.update_object(obj.object_id, label=new_label.strip())
                    self._refresh_objects_list()
                    self.status_label.config(text=f"Object label updated: {new_label}")

        except Exception as e:
            logger.error(f"Error editing object: {e}")

    def _delete_selected_objects(self):
        """Delete all checked objects in batch.

        Handles multi-delete functionality by deleting all objects that have been
        checked via the checkbox system. Shows confirmation dialog with count.

        Edge cases handled:
        - No objects checked: Shows warning
        - Invalid object indices: Validates before deletion
        - All objects selected: Confirms and deletes all
        """
        try:
            # Check if any objects are checked
            if not self.checked_objects:
                messagebox.showwarning("Warning", "Please check at least one object to delete")
                return

            # Get all objects and filter to checked ones
            all_objects = self.training_service.get_all_objects()
            objects_to_delete = [obj for obj in all_objects if obj.object_id in self.checked_objects]

            if not objects_to_delete:
                messagebox.showwarning("Warning", "No valid objects selected for deletion")
                self.checked_objects.clear()  # Clear stale IDs
                self._refresh_objects_list()
                return

            # Confirm deletion with count
            count = len(objects_to_delete)
            object_labels = ", ".join([obj.label for obj in objects_to_delete[:3]])  # Show first 3
            if count > 3:
                object_labels += f", and {count - 3} more"

            confirmation_msg = f"Delete {count} selected object(s)?\n\nObjects: {object_labels}"
            if not messagebox.askyesno("Confirm Delete", confirmation_msg):
                return

            # Delete all checked objects
            deleted_count = 0
            failed_count = 0
            for obj in objects_to_delete:
                try:
                    self.training_service.delete_object(obj.object_id)
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete object {obj.object_id}: {e}")
                    failed_count += 1

            # Clear checked objects set
            self.checked_objects.clear()

            # Refresh the list
            self._refresh_objects_list()

            # Update status with results
            if failed_count == 0:
                self.status_label.config(text=f"Successfully deleted {deleted_count} object(s)")
                logger.info(f"Deleted {deleted_count} objects")
            else:
                self.status_label.config(
                    text=f"Deleted {deleted_count} object(s), {failed_count} failed"
                )
                logger.warning(f"Deleted {deleted_count} objects, {failed_count} failed")
                messagebox.showwarning(
                    "Partial Deletion",
                    f"Successfully deleted {deleted_count} object(s)\nFailed to delete {failed_count} object(s)"
                )

        except Exception as e:
            logger.error(f"Error deleting objects: {e}")
            messagebox.showerror("Error", f"Failed to delete objects: {e}")

    def _view_selected_object(self):
        """View selected object image in the Objects tab canvas.

        Displays the full frame with contour/bbox annotation, not just the cropped object.
        Prioritizes showing segmentation contours when available.
        """
        try:
            selection = self._objects_listbox.curselection()
            if not selection:
                messagebox.showwarning("Warning", "Please select an object first")
                return

            idx = selection[0]
            objects = self.training_service.get_all_objects()

            if idx < len(objects):
                obj = objects[idx]

                # Display full frame with contour/bbox annotation
                annotated_frame = self._draw_object_contour(obj.image, obj, color=(0, 255, 0), thickness=2)

                # Add label text
                x1, y1, x2, y2 = obj.bbox
                cv2.putText(
                    annotated_frame,
                    obj.label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                self._objects_canvas.display_image(annotated_frame)

                # Update status to reflect what's being displayed
                display_type = "contour" if obj.segmentation and len(obj.segmentation) > 0 else "bbox"
                self.objects_status_label.config(text=f"Viewing: {obj.label} (full frame with {display_type})")
                logger.info(f"Displaying object '{obj.label}' with {display_type} annotation")

        except Exception as e:
            logger.error(f"Error viewing object: {e}")

    def _train_model_with_all_objects(self):
        """Start model training with all objects."""
        try:
            counts = self.training_service.get_object_count()

            if counts['total'] == 0:
                messagebox.showwarning(
                    "No Objects",
                    "Please add at least one object before training."
                )
                return

            if not messagebox.askyesno(
                "Confirm Training",
                f"Train model with {counts['total']} objects?\nThis may take several minutes."
            ):
                return

            # Show progress dialog
            progress = TrainingProgressDialog(self.root)

            def train_thread():
                try:
                    progress.update_status("Preparing dataset...")

                    # Get training parameters from config
                    # Note: training from scratch requires more epochs (default 100)
                    epochs = self.config.get('train_epochs', 100)
                    batch_size = self.config.get('batch_size', 8)

                    # Model architecture for training from scratch
                    # Available: yolo11n.yaml (nano), yolo11s.yaml (small), yolo11m.yaml (medium)
                    model_architecture = self.config.get('model_architecture', 'yolo11n.yaml')

                    logger.info(f"Starting training from scratch with {epochs} epochs")
                    logger.info(f"Model architecture: {model_architecture}")

                    # Define progress callback that updates the dialog
                    def on_progress(metrics):
                        """Called by training service with updated metrics."""
                        try:
                            progress.update_metrics(metrics)
                        except Exception as e:
                            logger.error(f"Error in progress callback: {e}")

                    # Define cancellation check callback
                    def check_cancelled():
                        """Called by training service to check if user cancelled."""
                        return progress.is_cancelled()

                    # Start training with callbacks
                    device = self.config.get('training_device', 'auto')
                    success = self.training_service.train_model(
                        epochs=epochs,
                        batch_size=batch_size,
                        progress_callback=on_progress,
                        cancellation_check=check_cancelled,
                        device=device,
                        model_architecture=model_architecture
                    )

                    # Check if cancelled
                    if progress.is_cancelled():
                        logger.info("Training cancelled by user")
                        progress.set_complete(False, "Training cancelled by user.")
                        return

                    if success:
                        # Reload the inference service with the newly trained model
                        logger.info("Training successful, reloading inference model...")
                        trained_model_path = os.path.join("data", "models", "model.pt")

                        if os.path.exists(trained_model_path):
                            # Reload the model in the inference service
                            self.inference_service.load_model(custom_model_path=trained_model_path)
                            logger.info("Inference model reloaded with newly trained model")
                            progress.set_complete(True, "Training completed successfully!\nModel ready for use in chat.")
                        else:
                            logger.error(f"Trained model not found at {trained_model_path}")
                            progress.set_complete(True, "Training completed but model file not found.")
                    else:
                        progress.set_complete(False, "Training failed.")

                except Exception as e:
                    logger.error(f"Training error: {e}")
                    progress.set_complete(False, f"Error: {e}")

            # Start training in background
            thread = threading.Thread(target=train_thread, daemon=True)
            thread.start()

            progress.show()

        except Exception as e:
            logger.error(f"Error starting training: {e}")
            messagebox.showerror("Error", f"Failed to start training: {e}")

    def _open_augmentation_dialog(self):
        """Open the data augmentation dialog."""
        try:
            # Check if we have training objects
            if not self.training_service.objects:
                messagebox.showwarning(
                    "No Objects",
                    "Please add training objects first before using data augmentation."
                )
                return

            # Open augmentation dialog
            dialog = DataAugmentationDialog(
                self.root,
                self.training_service,
                self.webcam_service
            )
            dialog.show()

            # Refresh objects list to show newly augmented objects
            self._refresh_objects_list()

            # Update objects status label
            objects_count = len(self.training_service.get_all_objects())
            self.objects_status_label.config(
                text=f"Total objects: {objects_count}"
            )

            logger.info("Data augmentation dialog closed, objects list refreshed")

        except Exception as e:
            logger.error(f"Error opening augmentation dialog: {e}")
            messagebox.showerror("Error", f"Failed to open augmentation dialog: {e}")

    def _on_objects_listbox_mousewheel(self, event):
        """Handle mouse wheel scrolling on objects listbox."""
        self._objects_listbox.yview_scroll(-1 * int(event.delta / 120), "units")

    def _draw_object_contour(self, image: np.ndarray, obj, color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """Draw object contour or bbox on image.

        Prioritizes drawing segmentation contours over bounding boxes for more accurate
        visualization of the training data.

        Args:
            image: Image to draw on (will be copied)
            obj: TrainingObject with segmentation or bbox
            color: BGR color tuple for the contour/bbox
            thickness: Line thickness

        Returns:
            Image with contour or bbox drawn
        """
        result = image.copy()
        h, w = image.shape[:2]

        if obj.segmentation and len(obj.segmentation) > 0:
            # Convert normalized YOLO segmentation to pixel coordinates
            points = []
            for i in range(0, len(obj.segmentation), 2):
                x = int(obj.segmentation[i] * w)
                y = int(obj.segmentation[i + 1] * h)
                points.append([x, y])

            contour = np.array(points, dtype=np.int32)

            # Draw contour outline
            cv2.polylines(result, [contour], True, color, thickness)

            logger.debug(f"Drew contour with {len(points)} points for object '{obj.label}'")
        else:
            # Fallback to bounding box if no segmentation
            x1, y1, x2, y2 = obj.bbox
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

            logger.debug(f"Drew bbox (no segmentation) for object '{obj.label}'")

        return result

    def _on_listbox_item_clicked(self, event):
        """Handle listbox item selection - toggle checkbox and display the selected object.

        This method provides dual functionality:
        1. Toggles the checkbox state for the clicked item
        2. Displays the object on the canvas (preserves existing behavior)

        Args:
            event: Listbox selection event
        """
        try:
            selection = self._objects_listbox.curselection()
            if not selection:
                return

            # Get the clicked index (the last item in selection for multiple select mode)
            idx = selection[-1]  # Use last selected item
            objects = self.training_service.get_all_objects()

            if idx >= len(objects):
                logger.warning(f"Index {idx} out of range for objects list (length: {len(objects)})")
                return

            obj = objects[idx]

            # Log the click for debugging
            logger.debug(f"Listbox item clicked - Index: {idx}, Object ID: {obj.object_id}, Label: {obj.label}")

            # Toggle checkbox state for the clicked item
            if obj.object_id in self.checked_objects:
                self.checked_objects.remove(obj.object_id)
                checkbox_state = "unchecked"
            else:
                self.checked_objects.add(obj.object_id)
                checkbox_state = "checked"

            # Refresh the entire list to reflect the new checkbox state
            # This is more robust than delete/insert which can cause event re-entrancy issues
            # where queued events operate on stale indices, toggling the wrong item
            logger.debug(f"Refreshing objects list after toggling '{obj.label}' to {checkbox_state}")
            self._refresh_objects_list()

            # Restore selection to maintain visual feedback
            # Must clear all selections first to ensure clean state in MULTIPLE selectmode
            self._objects_listbox.selection_clear(0, tk.END)
            self._objects_listbox.selection_set(idx)
            logger.debug(f"Restored selection to index {idx}")

            # Clear any existing contours/debug boxes from canvas before displaying new object
            self._objects_canvas.clear_debug_boxes()

            # Display full frame with contour/bbox annotation (preserve existing functionality)
            annotated_frame = self._draw_object_contour(obj.image, obj, color=(0, 255, 0), thickness=2)
            self._objects_canvas.display_image(annotated_frame)

            # Update status to reflect what's being displayed
            display_type = "contour" if obj.segmentation and len(obj.segmentation) > 0 else "bbox"
            self.objects_status_label.config(
                text=f"Viewing: {obj.label} (full frame with {display_type}) - {checkbox_state}"
            )
            logger.info(f"Toggled checkbox for '{obj.label}' (ID: {obj.object_id}, index: {idx}) to {checkbox_state}, displaying with {display_type} annotation")

        except Exception as e:
            logger.error(f"Error handling listbox item click: {e}", exc_info=True)

    # ========== CHAT TAB HANDLERS ==========

    def _on_chat_send_enhanced(self, event=None):
        """Handle chat send button click or Enter key."""
        try:
            message = self._chat_input.get("1.0", tk.END).strip()

            if not message:
                return "break"  # Prevent default behavior

            if event and event.keysym == "Return" and not event.state & 0x1:
                # Enter without Shift - send message
                self._send_chat_message(message)
                return "break"

            if event is None:
                # Button click
                self._send_chat_message(message)

        except Exception as e:
            logger.error(f"Error sending chat: {e}")

        return "break"

    def _on_chat_newline(self, event):
        """Handle Shift+Enter for new line."""
        return None  # Allow default behavior

    def _on_chat_typing(self, event):
        """Handle typing in chat input."""
        pass  # Placeholder for future typing indicators

    def _send_chat_message(self, message: str):
        """Send chat message and get AI response.

        Args:
            message: User message text
        """
        try:
            # Log user message
            logger.info(f"User message: {message}")

            # Add user message to chat
            self._add_chat_message("User", message)

            # Clear input
            self._chat_input.delete("1.0", tk.END)

            # Check if Gemini is configured
            if not self._gemini_configured:
                self._add_chat_message(
                    "System",
                    "Gemini AI is not configured. Please set your API key in Settings."
                )
                return

            # Show typing indicator
            self._show_typing_indicator()

            # Disable send button
            self._send_button.config(state='disabled')
            self.status_label.config(text="AI is thinking...")

            def process_chat():
                try:
                    # Get current frame (may be None if camera not running)
                    frame = self.webcam_service.get_current_frame()

                    # Get reference image (may be None if not set)
                    reference = self.reference_manager.get_reference()

                    # Get analysis mode from config
                    analysis_mode = self.config.get('analysis_mode', 'gemini_auto')

                    response = None

                    if analysis_mode == 'yolo_detection':
                        # YOLO Detection Mode: Run YOLO on available images, send text results to Gemini
                        logger.info("Using YOLO Detection mode for analysis")

                        # Get class names from the model
                        class_names = self.inference_service.get_class_names()

                        # Run YOLO on current frame if available
                        frame_detections = None
                        frame_width, frame_height = None, None
                        if frame is not None:
                            frame_detections = self.inference_service.detect(frame)
                            frame_height, frame_width = frame.shape[:2]

                        # Run YOLO on reference image if available
                        ref_detections = None
                        ref_width, ref_height = None, None
                        if reference is not None:
                            ref_detections = self.inference_service.detect(reference)
                            ref_height, ref_width = reference.shape[:2]

                        # ALWAYS use compare_with_text_only to show BOTH image sections
                        # This ensures the prompt always displays both reference and video stream sections,
                        # with "Status: Not provided" for any missing images
                        response = self.gemini_service.compare_with_text_only(
                            prompt=message,
                            ref_detections=ref_detections,
                            curr_detections=frame_detections,
                            class_names=class_names,
                            ref_width=ref_width,
                            ref_height=ref_height,
                            curr_width=frame_width,
                            curr_height=frame_height,
                            curr_is_video_frame=True  # Current frame is from webcam
                        )

                    elif analysis_mode == 'gemini_auto':
                        # Gemini Auto-Analysis Mode: Send images directly to Gemini for vision analysis
                        logger.info("Using Gemini Auto-Analysis mode for analysis")
                        response = self.gemini_service.chat_with_images(message, frame, reference)

                    else:
                        # Default to gemini_auto if mode is unrecognized
                        logger.warning(f"Unknown analysis mode '{analysis_mode}', defaulting to gemini_auto")
                        response = self.gemini_service.chat_with_images(message, frame, reference)

                    # Remove typing indicator
                    self.root.after(0, self._remove_typing_indicator)

                    if response:
                        # Log AI response
                        logger.info(f"AI response: {response}")
                        self.root.after(0, lambda: self._add_chat_message("AI", response))
                        self.root.after(0, lambda: self.status_label.config(text="Response received"))
                    else:
                        # Log empty response
                        logger.warning("Empty response from Gemini API")
                        self.root.after(0, lambda: self._add_chat_message(
                            "System",
                            "Failed to get AI response. Please check your API key and connection."
                        ))

                except Exception as e:
                    logger.error(f"Error processing chat: {e}")
                    # Remove typing indicator on error
                    error_msg = str(e)  # Capture error immediately to avoid scoping issues
                    self.root.after(0, self._remove_typing_indicator)
                    self.root.after(0, lambda msg=error_msg: self._add_chat_message(
                        "System",
                        f"Error: {msg}"
                    ))

                finally:
                    self.root.after(0, lambda: self._send_button.config(state='normal'))

            # Process in background
            threading.Thread(target=process_chat, daemon=True).start()

        except Exception as e:
            logger.error(f"Error in send chat: {e}")
            self._remove_typing_indicator()
            self._send_button.config(state='normal')

    def _format_detections_as_text(self, detections: List[Dict[str, Any]], image_label: str) -> str:
        """Format YOLO detections as human-readable text.

        Args:
            detections: List of detection dictionaries from YOLO
            image_label: Label for the image (e.g., "Current Frame" or "Reference Image")

        Returns:
            Formatted string describing detected objects
        """
        if not detections:
            return f"{image_label}: No objects detected"

        # Count objects by class
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # Build summary
        items = [f"{class_name} ({count})" for class_name, count in sorted(class_counts.items())]
        return f"{image_label}: {', '.join(items)}"

    def _parse_markdown(self, text: str) -> list:
        """Parse markdown formatting in text and return list of (text, styles) tuples.

        Supports:
        - **bold** or __bold__
        - *italic* or _italic_
        - ~~strikethrough~~
        - Combined styles (e.g., ***bold italic***)

        Args:
            text: Text with markdown formatting

        Returns:
            List of (text_segment, styles_set) tuples where styles_set contains
            'bold', 'italic', 'strikethrough' as applicable
        """
        import re

        # This will store our result segments
        segments = []

        # Pattern to match markdown syntax (order matters for nested styles)
        # Matches: ***text***, **text**, __text__, *text*, _text_, ~~text~~
        pattern = r'(\*\*\*[^*]+\*\*\*|\*\*[^*]+\*\*|__[^_]+__|~~[^~]+~~|\*[^*]+\*|_[^_]+_)'

        # Split text by markdown patterns while keeping the delimiters
        parts = re.split(pattern, text)

        for part in parts:
            if not part:  # Skip empty strings
                continue

            styles = set()
            plain_text = part

            # Check for bold italic (***text***)
            if part.startswith('***') and part.endswith('***') and len(part) > 6:
                styles.add('bold')
                styles.add('italic')
                plain_text = part[3:-3]
            # Check for bold (**text** or __text__)
            elif (part.startswith('**') and part.endswith('**') and len(part) > 4) or \
                 (part.startswith('__') and part.endswith('__') and len(part) > 4):
                styles.add('bold')
                plain_text = part[2:-2]
            # Check for italic (*text* or _text_)
            elif (part.startswith('*') and part.endswith('*') and len(part) > 2) or \
                 (part.startswith('_') and part.endswith('_') and len(part) > 2):
                styles.add('italic')
                plain_text = part[1:-1]
            # Check for strikethrough (~~text~~)
            elif part.startswith('~~') and part.endswith('~~') and len(part) > 4:
                styles.add('strikethrough')
                plain_text = part[2:-2]

            segments.append((plain_text, styles))

        return segments if segments else [(text, set())]

    def _add_chat_message(self, sender: str, message: str):
        """Add modern bubble-style message to chat display.

        Args:
            sender: Message sender (User, AI, System)
            message: Message text
        """
        try:
            from datetime import datetime

            # Message container for proper alignment
            msg_container = tk.Frame(self._messages_frame, bg='#2B2B2B')
            msg_container.pack(fill='x', padx=15, pady=8)

            # Determine message styling based on sender
            if sender == "User":
                # User messages - right aligned, dark blue background
                bubble_bg = '#0E639C'
                text_color = '#FFFFFF'
                anchor_side = 'e'
                max_width = 350
            elif sender == "AI":
                # AI messages - left aligned, dark gray background
                bubble_bg = '#3C3C3C'
                text_color = '#E0E0E0'
                anchor_side = 'w'
                max_width = 400
            else:
                # System messages - centered, dark orange/amber background
                bubble_bg = '#8B6914'
                text_color = '#FFFFFF'
                anchor_side = 'center'
                max_width = 400

            # Create message bubble frame
            if anchor_side == 'e':
                bubble_frame = tk.Frame(msg_container, bg='#2B2B2B')
                bubble_frame.pack(side='right')
            elif anchor_side == 'center':
                bubble_frame = tk.Frame(msg_container, bg='#2B2B2B')
                bubble_frame.pack(anchor='center')
            else:
                bubble_frame = tk.Frame(msg_container, bg='#2B2B2B')
                bubble_frame.pack(side='left')

            # Sender name label (small, subtle)
            if sender != "System":
                sender_label = tk.Label(
                    bubble_frame,
                    text=sender,
                    bg='#2B2B2B',
                    fg='#999999',
                    font=('Segoe UI', 8),
                    anchor=anchor_side
                )
                sender_label.pack(anchor=anchor_side, padx=12, pady=(0, 2))

            # Message bubble with shadow effect
            bubble_shadow = tk.Frame(bubble_frame, bg='#1A1A1A')
            bubble_shadow.pack()

            bubble = tk.Frame(bubble_shadow, bg=bubble_bg)
            bubble.pack(padx=1, pady=1)

            # Message text - use Text widget for AI messages (supports markdown),
            # Label for User/System messages (plain text)
            if sender == "AI":
                # Use Text widget for AI messages to support markdown formatting
                message_text = tk.Text(
                    bubble,
                    bg=bubble_bg,
                    fg=text_color,
                    font=('Segoe UI', 10),
                    wrap='word',
                    width=45,  # Approximate width in characters
                    borderwidth=0,
                    highlightthickness=0,
                    padx=15,
                    pady=12,
                    cursor='arrow',
                    state='normal'
                )

                # Configure text tags for markdown styles
                message_text.tag_config('bold', font=('Segoe UI', 10, 'bold'))
                message_text.tag_config('italic', font=('Segoe UI', 10, 'italic'))
                message_text.tag_config('bold_italic', font=('Segoe UI', 10, 'bold italic'))
                message_text.tag_config('strikethrough', overstrike=True)

                # Parse markdown and insert formatted text
                segments = self._parse_markdown(message)
                for text_segment, styles in segments:
                    if not text_segment:
                        continue

                    # Determine which tag(s) to apply
                    if 'bold' in styles and 'italic' in styles:
                        message_text.insert('end', text_segment, 'bold_italic')
                    elif 'bold' in styles:
                        message_text.insert('end', text_segment, 'bold')
                    elif 'italic' in styles:
                        message_text.insert('end', text_segment, 'italic')
                    elif 'strikethrough' in styles:
                        message_text.insert('end', text_segment, 'strikethrough')
                    else:
                        message_text.insert('end', text_segment)

                # Make text read-only
                message_text.config(state='disabled')

                # Pack the widget first so it can calculate wrapped dimensions
                message_text.pack()

                # Force update to calculate actual dimensions after packing
                message_text.update_idletasks()

                # Calculate required height based on DISPLAYED lines (accounts for wrapping)
                # Count visible/wrapped lines by checking display line info
                try:
                    # Get the last visible line index
                    last_index = message_text.index('end-1c')
                    # Count display lines (wrapped lines) not just text lines
                    display_line_count = 0
                    current_index = '1.0'
                    while message_text.compare(current_index, '<=', last_index):
                        dline = message_text.dlineinfo(current_index)
                        if dline is None:
                            break
                        display_line_count += 1
                        # Move to next display line
                        current_index = message_text.index(f'{current_index}+1 display lines')

                    # Set height to actual display line count
                    if display_line_count > 0:
                        message_text.config(height=display_line_count)
                except Exception as e:
                    # Fallback: use text line count if display line calculation fails
                    logger.warning(f"Could not calculate display lines, using text lines: {e}")
                    line_count = int(message_text.index('end-1c').split('.')[0])
                    message_text.config(height=line_count)
            else:
                # Use Label for User/System messages (no markdown parsing)
                message_label = tk.Label(
                    bubble,
                    text=message,
                    bg=bubble_bg,
                    fg=text_color,
                    font=('Segoe UI', 10),
                    wraplength=max_width,
                    justify='left',
                    padx=15,
                    pady=12,
                    anchor='w'
                )
                message_label.pack()

            # Timestamp (subtle)
            timestamp = datetime.now().strftime("%H:%M")
            time_label = tk.Label(
                bubble_frame,
                text=timestamp,
                bg='#2B2B2B',
                fg='#777777',
                font=('Segoe UI', 7),
                anchor=anchor_side
            )
            time_label.pack(anchor=anchor_side, padx=12, pady=(2, 0))

            # Auto-scroll to bottom with smooth animation
            self._chat_canvas.update_idletasks()
            self._chat_canvas.yview_moveto(1.0)

            # Store message widget for future reference
            self._message_widgets.append(msg_container)

        except Exception as e:
            logger.error(f"Error adding chat message: {e}")

    def _insert_markdown(self, text_widget: tk.Text, markdown_text: str):
        """Parse and insert markdown-formatted text into Text widget.

        Args:
            text_widget: The Text widget to insert into
            markdown_text: Markdown-formatted text
        """
        import re

        # Split by code blocks first
        parts = re.split(r'```(.*?)```', markdown_text, flags=re.DOTALL)

        for i, part in enumerate(parts):
            if i % 2 == 1:  # Code block
                text_widget.insert('end', part, 'code')
                continue

            # Process inline markdown
            pos = 0
            while pos < len(part):
                # Check for **bold**
                bold_match = re.match(r'\*\*(.*?)\*\*', part[pos:])
                if bold_match:
                    text_widget.insert('end', bold_match.group(1), 'bold')
                    pos += len(bold_match.group(0))
                    continue

                # Check for *italic*
                italic_match = re.match(r'\*(.*?)\*', part[pos:])
                if italic_match:
                    text_widget.insert('end', italic_match.group(1), 'italic')
                    pos += len(italic_match.group(0))
                    continue

                # Check for `code`
                code_match = re.match(r'`(.*?)`', part[pos:])
                if code_match:
                    text_widget.insert('end', code_match.group(1), 'code')
                    pos += len(code_match.group(0))
                    continue

                # Check for headers (# at start of line)
                if pos == 0 or part[pos-1] == '\n':
                    header_match = re.match(r'#+\s+(.*?)(\n|$)', part[pos:])
                    if header_match:
                        text_widget.insert('end', header_match.group(1) + '\n', 'header')
                        pos += len(header_match.group(0))
                        continue

                # Regular character
                text_widget.insert('end', part[pos])
                pos += 1

    def _add_welcome_message(self):
        """Add a welcome message to the chat on startup."""
        try:
            welcome_text = "Hello! I'm your AI assistant for image analysis. I can help you analyze live video frames, compare images, and detect objects. Start the webcam or load a reference image, then ask me questions!"
            self._add_chat_message("AI", welcome_text)
        except Exception as e:
            logger.error(f"Error adding welcome message: {e}")

    def _show_typing_indicator(self):
        """Show typing indicator while AI is processing."""
        try:
            # Message container
            msg_container = tk.Frame(self._messages_frame, bg='#2B2B2B')
            msg_container.pack(fill='x', padx=15, pady=8)

            # Bubble frame (left aligned for AI)
            bubble_frame = tk.Frame(msg_container, bg='#2B2B2B')
            bubble_frame.pack(side='left')

            # Sender label
            sender_label = tk.Label(
                bubble_frame,
                text="AI",
                bg='#2B2B2B',
                fg='#999999',
                font=('Segoe UI', 8)
            )
            sender_label.pack(anchor='w', padx=12, pady=(0, 2))

            # Typing bubble
            bubble_shadow = tk.Frame(bubble_frame, bg='#1A1A1A')
            bubble_shadow.pack()

            bubble = tk.Frame(bubble_shadow, bg='#3C3C3C')
            bubble.pack(padx=1, pady=1)

            # Typing animation (three dots)
            typing_label = tk.Label(
                bubble,
                text="●●●",
                bg='#3C3C3C',
                fg='#888888',
                font=('Segoe UI', 10),
                padx=20,
                pady=12
            )
            typing_label.pack()

            # Store reference to remove later
            self._typing_indicator = msg_container

            # Auto-scroll to bottom
            self._chat_canvas.update_idletasks()
            self._chat_canvas.yview_moveto(1.0)

            # Animate typing indicator
            self._animate_typing_indicator(typing_label, 0)

        except Exception as e:
            logger.error(f"Error showing typing indicator: {e}")

    def _animate_typing_indicator(self, label, state):
        """Animate the typing indicator dots.

        Args:
            label: The label widget to animate
            state: Current animation state (0-2)
        """
        try:
            if self._typing_indicator is None:
                return  # Indicator was removed

            animations = ["●○○", "●●○", "●●●"]
            label.configure(text=animations[state % 3])

            # Schedule next animation frame
            self.root.after(400, lambda: self._animate_typing_indicator(label, state + 1))

        except Exception as e:
            logger.error(f"Error animating typing indicator: {e}")

    def _remove_typing_indicator(self):
        """Remove typing indicator from chat."""
        try:
            if self._typing_indicator:
                self._typing_indicator.destroy()
                self._typing_indicator = None
        except Exception as e:
            logger.error(f"Error removing typing indicator: {e}")

    def _clear_chat(self):
        """Clear all chat messages."""
        try:
            for widget in self._messages_frame.winfo_children():
                widget.destroy()

            self._message_widgets.clear()
            self.status_label.config(text="Chat cleared")

            # Add welcome message again after clearing
            self.root.after(100, self._add_welcome_message)

        except Exception as e:
            logger.error(f"Error clearing chat: {e}")

    def _on_chat_mousewheel(self, event):
        """Handle mouse wheel scrolling on chat canvas."""
        self._chat_canvas.yview_scroll(-1 * int(event.delta / 120), "units")

    def _on_chat_input_mousewheel(self, event):
        """Handle mouse wheel scrolling on chat input."""
        return None  # Allow default scrolling

    # ========== SETTINGS DIALOG ==========

    def _open_settings(self):
        """Open settings dialog."""
        try:
            # Prepare services dict for dialog
            services = {
                'webcam_service': self.webcam_service,
                'gemini_service': self.gemini_service,
                'inference_service': self.inference_service,
                'training_service': self.training_service,
                'reference_manager': self.reference_manager
            }

            dialog = ComprehensiveSettingsDialog(
                self.root,
                self.config,
                services,
                callback=self._on_settings_apply
            )

            # Modal dialog
            self.root.wait_window(dialog.dialog)

        except Exception as e:
            logger.error(f"Error opening settings: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to open settings:\n{str(e)}")

    def _on_settings_apply(self, new_config: Dict[str, Any]):
        """Handle settings apply.

        Args:
            new_config: Updated configuration dictionary
        """
        try:
            # Update config
            self.config.update(new_config)
            self._save_config()

            # Update services with new config
            self.inference_service.update_thresholds(
                confidence=new_config.get('detection_confidence_threshold'),
                iou=new_config.get('detection_iou_threshold')
            )

            self.gemini_service.update_config(
                model=new_config.get('gemini_model'),
                temperature=new_config.get('gemini_temperature'),
                max_tokens=new_config.get('gemini_max_tokens'),
                persona=new_config.get('chatbot_persona', '')
            )

            # Update training service config (for augmentation settings)
            self.training_service.config = new_config
            logger.info("Training service config updated with new augmentation settings")

            # Update webcam settings if changed
            new_camera_index = new_config.get('last_webcam_index', 0)
            new_codec = new_config.get('video_codec', 'Auto')
            
            if hasattr(self.webcam_service, 'camera_index') and self.webcam_service.camera_index != new_camera_index:
                self.webcam_service.set_camera_index(new_camera_index)
                logger.info(f"Webcam camera index updated to: {new_camera_index}")
            
            if hasattr(self.webcam_service, 'codec') and self.webcam_service.codec != new_codec:
                self.webcam_service.set_codec(new_codec)
                logger.info(f"Webcam codec updated to: {new_codec}")

            # Update Gemini configuration status
            self._gemini_configured = bool(new_config.get('gemini_api_key', '').strip())

            # Update debug mode on canvases
            self._update_debug_mode()

            self.status_label.config(text="Settings applied successfully")
            logger.info("Settings updated")

        except Exception as e:
            logger.error(f"Error applying settings: {e}")
            messagebox.showerror("Error", f"Failed to apply settings: {e}")

    # ========== DEBUG MODE ==========

    def _update_debug_mode(self):
        """Update debug mode on all canvases based on config."""
        try:
            debug_enabled = self.config.get('debug_mode', False)

            # Model test callback for running YOLO inference
            def model_test_callback(image):
                """Run YOLO inference on image and return detections."""
                try:
                    detections = self.inference_service.detect(image)
                    logger.info(f"Debug mode: Found {len(detections) if detections else 0} detections")
                    return detections if detections else []
                except Exception as e:
                    logger.error(f"Error in debug model test: {e}")
                    return []

            # Enable debug mode on all canvases
            # self._video_canvas.enable_debug_mode(debug_enabled, model_test_callback)
            self._reference_canvas.enable_debug_mode(debug_enabled, model_test_callback)
            self._objects_canvas.enable_debug_mode(debug_enabled, model_test_callback)

            logger.info(f"Debug mode {'enabled' if debug_enabled else 'disabled'} on all canvases")

        except Exception as e:
            logger.error(f"Error updating debug mode: {e}")

    # ========== WINDOW MANAGEMENT ==========

    def _on_window_close(self):
        """Handle window close event."""
        try:
            # Stop webcam stream
            if self.webcam_service.is_streaming():
                self.webcam_service.stop_stream()

            # Destroy window
            self.root.destroy()

        except Exception as e:
            logger.error(f"Error closing window: {e}")
            self.root.destroy()


# ========== MAIN ENTRY POINT ==========

def main():
    """Main application entry point."""
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()