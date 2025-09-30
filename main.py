"""Modern main application window with comprehensive UI/UX design and performance optimizations."""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import json
import threading
import logging
from typing import Optional, Dict, Any, Callable, List
from datetime import datetime

# Import services
from app.services.webcam_service import WebcamService
from app.services.inference_service import InferenceService
from app.services.training_service import TrainingService
from app.services.gemini_service import GeminiService
from app.services.reference_manager import ReferenceManager

# Import UI components
from app.ui.components.optimized_canvas import OptimizedCanvas, ChatCanvas
from app.ui.components.object_selector import ObjectSelector

# Import dialogs
from app.ui.dialogs.comprehensive_settings_dialog import ComprehensiveSettingsDialog
from app.ui.dialogs.training_progress_dialog import TrainingProgressDialog
from app.ui.dialogs.object_naming_dialog import ObjectNamingDialog

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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

        # Check Gemini configuration
        self._gemini_configured = bool(self.config.get('gemini_api_key', '').strip())

        # Setup locale (simplified)
        self.locale = {}

        # Setup window
        self._setup_window()
        self._setup_styles()
        self._build_ui()

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
            fps=self.config.get('target_fps', 30)
        )

        # Inference service
        model_path = self.config.get('preferred_model', 'yolo12n')
        self.inference_service = InferenceService(
            model_path=model_path,
            confidence_threshold=self.config.get('detection_confidence_threshold', 0.5),
            iou_threshold=self.config.get('detection_iou_threshold', 0.45)
        )

        # Load model in background
        threading.Thread(target=self.inference_service.load_model, daemon=True).start()

        # Training service
        self.training_service = TrainingService(data_dir="data/training")

        # Gemini service
        api_key = self.config.get('gemini_api_key', '')
        self.gemini_service = GeminiService(
            api_key=api_key,
            model=self.config.get('gemini_model', 'gemini-2.5-pro'),
            temperature=self.config.get('gemini_temperature', 0.7),
            max_tokens=self.config.get('gemini_max_tokens', 2048)
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
        self.root.title("Vision Analysis System - Modern Interface")
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
        
        self.capture_button = ttk.Button(
            left_frame, 
            text="📷 Capture",
            style='Modern.TButton',
            command=self._on_capture_image,
            state='disabled'
        )
        self.capture_button.pack(side='left', padx=(0, 10))
        
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
        """Build the main video display panel."""
        video_frame = tk.Frame(parent, bg=self.COLORS['bg_secondary'], relief='solid', bd=1)
        video_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        
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

        # Video controls
        controls_frame = tk.Frame(video_frame, bg=self.COLORS['bg_secondary'], height=40)
        controls_frame.pack(fill='x')
        controls_frame.pack_propagate(False)
        
        # Resolution and FPS indicators
        self.resolution_label = tk.Label(
            controls_frame,
            text="Resolution: --",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8)
        )
        self.resolution_label.pack(side='left', padx=10, pady=10)
        
        self.fps_label = tk.Label(
            controls_frame,
            text="FPS: --",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8)
        )
        self.fps_label.pack(side='left', padx=(20, 10), pady=10)
    
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
        
        # Reference Image Tab
        ref_frame = self._build_reference_panel()
        notebook.add(ref_frame, text="📋 Reference")
        
        # Objects Training Tab (replaces Current Image Tab)
        objects_frame = self._build_objects_panel()
        notebook.add(objects_frame, text="🎯 Objects")
        
        # Chat Analysis Tab
        chat_frame = self._build_chat_panel()
        notebook.add(chat_frame, text="🤖 Analysis")
    
    def _build_reference_panel(self):
        """Build the reference image management panel."""
        panel = tk.Frame(bg=self.COLORS['bg_secondary'])
        
        # Title and controls
        header_frame = tk.Frame(panel, bg=self.COLORS['bg_tertiary'], height=40)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame,
            text="Reference Image",
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 9, 'bold')
        ).pack(side='left', padx=10, pady=10)
        
        # Control buttons
        controls_frame = tk.Frame(panel, bg=self.COLORS['bg_secondary'], height=50)
        controls_frame.pack(fill='x', padx=5, pady=5)
        controls_frame.pack_propagate(False)
        
        ttk.Button(
            controls_frame,
            text="📁 Load Image",
            style='Secondary.TButton',
            command=self._load_reference_image
        ).pack(side='left', padx=(0, 5))
        
        ttk.Button(
            controls_frame,
            text="📷 From Stream",
            style='Secondary.TButton',
            command=self._set_reference_from_stream
        ).pack(side='left', padx=5)
        
        ttk.Button(
            controls_frame,
            text="🗑 Clear",
            style='Secondary.TButton',
            command=self._clear_reference_image
        ).pack(side='left', padx=5)
        
        # Image display
        image_frame = tk.Frame(panel, bg=self.COLORS['bg_primary'])
        image_frame.pack(fill='both', expand=True, padx=5, pady=(0, 5))

        self._reference_canvas = OptimizedCanvas(
            image_frame,
            bg='black',
            highlightthickness=0
        )
        self._reference_canvas.pack(fill='both', expand=True)
        self._reference_canvas.set_render_quality('high')
        
        # Image info
        info_frame = tk.Frame(panel, bg=self.COLORS['bg_secondary'], height=30)
        info_frame.pack(fill='x')
        info_frame.pack_propagate(False)
        
        self.ref_info_label = tk.Label(
            info_frame,
            text="No reference image loaded",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8)
        )
        self.ref_info_label.pack(padx=10, pady=5)
        
        return panel
    
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
        
        # Initialize object selector
        self._object_selector = ObjectSelector(
            self._objects_canvas,
            self._on_object_selection_complete
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
        
        # Management buttons
        buttons_frame = tk.Frame(list_frame, bg=self.COLORS['bg_secondary'], height=40)
        buttons_frame.pack(fill='x', pady=(5, 0))
        buttons_frame.pack_propagate(False)
        
        ttk.Button(
            buttons_frame,
            text="✅ Confirm",
            style='Secondary.TButton',
            command=self._confirm_selected_object
        ).pack(side='left', padx=(5, 2), pady=5)
        
        ttk.Button(
            buttons_frame,
            text="❌ Unconfirm",
            style='Secondary.TButton',
            command=self._unconfirm_selected_object
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
            command=self._delete_selected_object
        ).pack(side='left', padx=2, pady=5)
        
        ttk.Button(
            buttons_frame,
            text="👁️ View",
            style='Secondary.TButton',
            command=self._view_selected_object
        ).pack(side='left', padx=2, pady=5)
        
        ttk.Button(
            buttons_frame,
            text="🚀 Train Model",
            style='Modern.TButton',
            command=self._train_model_with_confirmed_objects
        ).pack(side='right', padx=(2, 5), pady=5)
        
        ttk.Button(
            buttons_frame,
            text="📤 Export Dataset",
            style='Modern.TButton',
            command=self._export_confirmed_dataset
        ).pack(side='right', padx=2, pady=5)
        
        # Load initial objects
        self._refresh_objects_list()
    
    def _build_chat_panel(self):
        """Build the enhanced ChatBot conversation interface with modern UI."""
        panel = tk.Frame(bg=self.COLORS['bg_secondary'])
        
        # Enhanced header with status indicators
        header_frame = tk.Frame(panel, bg=self.COLORS['bg_tertiary'], height=45)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        # Title and status
        title_frame = tk.Frame(header_frame, bg=self.COLORS['bg_tertiary'])
        title_frame.pack(side='left', fill='y', padx=10, pady=5)
        
        tk.Label(
            title_frame,
            text="🤖 AI Analysis Chat",
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 10, 'bold')
        ).pack(anchor='w')
        
        # Connection status indicator
        self._chat_status_label = tk.Label(
            title_frame,
            text="● Ready" if self._gemini_configured else "● Not Configured",
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['success'] if self._gemini_configured else self.COLORS['warning'],
            font=('Segoe UI', 8)
        )
        self._chat_status_label.pack(anchor='w')
        
        # Chat controls
        controls_frame = tk.Frame(header_frame, bg=self.COLORS['bg_tertiary'])
        controls_frame.pack(side='right', fill='y', padx=10, pady=8)
        
        ttk.Button(
            controls_frame,
            text="🧹",
            style='Secondary.TButton',
            command=self._clear_chat,
            width=3
        ).pack(side='right', padx=(5, 0))
        
        # Enhanced chat display area with custom scrollable frame
        chat_frame = tk.Frame(panel, bg=self.COLORS['bg_primary'])
        chat_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create optimized chat canvas for smooth scrolling
        self._chat_canvas = ChatCanvas(
            chat_frame,
            bg=self.COLORS['bg_primary'],
            highlightthickness=0,
            borderwidth=0
        )
        # Enable virtual scrolling for performance with many messages
        self._chat_canvas.enable_virtual_scrolling(True)
        
        self._chat_scrollbar = ttk.Scrollbar(
            chat_frame, 
            orient='vertical', 
            command=self._chat_canvas.yview
        )
        self._chat_canvas.configure(yscrollcommand=self._chat_scrollbar.set)
        
        # Scrollable frame for messages
        self._messages_frame = tk.Frame(self._chat_canvas, bg=self.COLORS['bg_primary'])
        self._messages_frame.bind(
            '<Configure>',
            lambda e: self._chat_canvas.configure(scrollregion=self._chat_canvas.bbox("all"))
        )
        
        # Bind canvas resize to update messages frame width for responsive design
        def _on_canvas_configure(event):
            # Update the messages frame width to match canvas width
            canvas_width = event.width
            self._chat_canvas.itemconfig(self._messages_canvas_window, width=canvas_width)
            # Update scroll region
            self._chat_canvas.configure(scrollregion=self._chat_canvas.bbox("all"))
        
        self._chat_canvas.bind('<Configure>', _on_canvas_configure)
        
        self._messages_canvas_window = self._chat_canvas.create_window((0, 0), window=self._messages_frame, anchor="nw")
        
        self._chat_canvas.pack(side="left", fill="both", expand=True)
        self._chat_scrollbar.pack(side="right", fill="y")
        
        # Bind mouse wheel scrolling
        self._chat_canvas.bind('<MouseWheel>', self._on_chat_mousewheel)
        self._messages_frame.bind('<MouseWheel>', self._on_chat_mousewheel)
        
        # Enhanced input area with better styling
        input_frame = tk.Frame(panel, bg=self.COLORS['bg_secondary'], height=120)
        input_frame.pack(fill='x', padx=5, pady=(0, 5))
        input_frame.pack_propagate(False)
        
        # Enhanced text input with send indicator
        input_container = tk.Frame(input_frame, bg=self.COLORS['bg_secondary'])
        input_container.pack(fill='x', pady=(8, 5))
        
        # Input field with modern styling
        input_border = tk.Frame(input_container, bg=self.COLORS['accent_primary'], height=2)
        input_border.pack(fill='x', pady=(0, 1))
        
        input_field_frame = tk.Frame(input_container, bg=self.COLORS['bg_tertiary'])
        input_field_frame.pack(fill='x')
        
        # Frame for text input and scrollbar
        text_input_frame = tk.Frame(input_field_frame, bg=self.COLORS['bg_tertiary'])
        text_input_frame.pack(side='left', fill='both', expand=True)

        self._chat_input = tk.Text(
            text_input_frame,
            height=5,
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 10),
            borderwidth=0,
            insertbackground=self.COLORS['text_primary'],
            wrap=tk.WORD,
            padx=10,
            pady=8
        )

        # Add scrollbar for chat input
        self._chat_input_scrollbar = ttk.Scrollbar(
            text_input_frame,
            orient='vertical',
            command=self._chat_input.yview
        )
        self._chat_input.configure(yscrollcommand=self._chat_input_scrollbar.set)

        # Pack text input and scrollbar
        self._chat_input.pack(side='left', fill='both', expand=True)
        self._chat_input_scrollbar.pack(side='right', fill='y')

        # Bind events for chat input
        self._chat_input.bind('<Return>', self._on_chat_send_enhanced)
        self._chat_input.bind('<Shift-Return>', self._on_chat_newline)
        self._chat_input.bind('<KeyRelease>', self._on_chat_typing)

        # Add mouse wheel support for chat input
        self._chat_input.bind('<MouseWheel>', self._on_chat_input_mousewheel)
        
        # Enhanced send button
        send_frame = tk.Frame(input_field_frame, bg=self.COLORS['bg_tertiary'])
        send_frame.pack(side='right', padx=(5, 10), pady=8)
        
        self._send_button = tk.Button(
            send_frame,
            text="➤",
            bg=self.COLORS['accent_primary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 12, 'bold'),
            borderwidth=0,
            padx=12,
            pady=6,
            command=self._on_chat_send_enhanced,
            cursor='hand2',
            state='normal'
        )
        self._send_button.pack()
        
        # Send button hover effects
        def on_send_enter(e):
            self._send_button.configure(bg=self.COLORS['accent_secondary'])
        
        def on_send_leave(e):
            self._send_button.configure(bg=self.COLORS['accent_primary'])
        
        self._send_button.bind('<Enter>', on_send_enter)
        self._send_button.bind('<Leave>', on_send_leave)
        
        # Initialize message state tracking
        self._message_widgets = []
        self._typing_indicator = None
        self._last_message_id = 0
        
        # Setup accessibility features
        self.root.after(100, self._setup_chat_accessibility)
        
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
                self.capture_button.config(state='normal')
                self.status_label.config(text="Webcam stream started")
                self.connection_label.config(text="🟢 Connected", fg=self.COLORS['success'])
                logger.info("Webcam stream started")
            else:
                messagebox.showerror("Error", "Failed to start webcam stream")

        except Exception as e:
            logger.error(f"Error starting stream: {e}")
            messagebox.showerror("Error", f"Failed to start stream: {e}")

    def _on_stop_stream(self):
        """Handle stop stream button click."""
        try:
            self.webcam_service.stop_stream()
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.capture_button.config(state='disabled')
            self.status_label.config(text="Webcam stream stopped")
            self.connection_label.config(text="⚪ Disconnected", fg=self.COLORS['text_muted'])

            # Clear video canvas
            self._video_canvas.clear()

            # Reset FPS and resolution labels
            self.fps_label.config(text="FPS: --")
            self.resolution_label.config(text="Resolution: --")

            logger.info("Webcam stream stopped")

        except Exception as e:
            logger.error(f"Error stopping stream: {e}")

    def _on_capture_image(self):
        """Handle capture button click."""
        try:
            frame = self.webcam_service.capture_frame()
            if frame is not None:
                # Save to disk
                os.makedirs("data/captures", exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filepath = f"data/captures/capture_{timestamp}.png"
                cv2.imwrite(filepath, frame)
                self.status_label.config(text=f"Image saved: {filepath}")
                logger.info(f"Image captured: {filepath}")
            else:
                messagebox.showwarning("Warning", "No frame available to capture")

        except Exception as e:
            logger.error(f"Error capturing image: {e}")
            messagebox.showerror("Error", f"Failed to capture image: {e}")

    def _update_video_stream(self):
        """Update video display with current frame."""
        try:
            if self.webcam_service.is_streaming():
                frame = self.webcam_service.get_current_frame()

                if frame is not None:
                    # Display frame on canvas
                    self._video_canvas.display_image(frame)

                    # Update FPS and resolution
                    fps = self.webcam_service.get_fps()
                    width, height = self.webcam_service.get_resolution()

                    self.fps_label.config(text=f"FPS: {fps:.1f}")
                    self.resolution_label.config(text=f"Resolution: {width}x{height}")

        except Exception as e:
            logger.error(f"Error updating video stream: {e}")

        # Schedule next update
        self.root.after(33, self._update_video_stream)  # ~30 FPS

    # ========== REFERENCE TAB HANDLERS ==========

    def _load_reference_image(self):
        """Load reference image from file."""
        try:
            filepath = filedialog.askopenfilename(
                title="Select Reference Image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
            )

            if filepath:
                if self.reference_manager.load_reference_from_file(filepath):
                    self._display_reference_image()
                    self.config['reference_image_path'] = filepath
                    self._save_config()
                    self.status_label.config(text="Reference image loaded")
                else:
                    messagebox.showerror("Error", "Failed to load reference image")

        except Exception as e:
            logger.error(f"Error loading reference: {e}")
            messagebox.showerror("Error", f"Failed to load reference: {e}")

    def _set_reference_from_stream(self):
        """Set reference image from current webcam frame."""
        try:
            if not self.webcam_service.is_streaming():
                messagebox.showwarning("Warning", "Please start the webcam stream first")
                return

            frame = self.webcam_service.capture_frame()
            if frame is not None:
                if self.reference_manager.set_reference_from_array(frame, save=True):
                    self._display_reference_image()
                    saved_path = self.reference_manager.save_current_reference()
                    if saved_path:
                        self.config['reference_image_path'] = saved_path
                        self._save_config()
                    self.status_label.config(text="Reference image set from stream")
                else:
                    messagebox.showerror("Error", "Failed to set reference image")
            else:
                messagebox.showwarning("Warning", "No frame available")

        except Exception as e:
            logger.error(f"Error setting reference: {e}")
            messagebox.showerror("Error", f"Failed to set reference: {e}")

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
        """Capture current frame for training."""
        try:
            if not self.webcam_service.is_streaming():
                messagebox.showwarning("Warning", "Please start the webcam stream first")
                return

            frame = self.webcam_service.capture_frame()
            if frame is not None:
                self._training_image = frame
                self._objects_canvas.display_image(frame)
                self.objects_status_label.config(text="Image captured. Click 'Select Object' to mark an object.")
                self._object_selector.set_image(frame)
            else:
                messagebox.showwarning("Warning", "No frame available")

        except Exception as e:
            logger.error(f"Error capturing for training: {e}")
            messagebox.showerror("Error", f"Failed to capture image: {e}")

    def _load_image_for_training(self):
        """Load image from file for training."""
        try:
            filepath = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
            )

            if filepath:
                image = cv2.imread(filepath)
                if image is not None:
                    self._training_image = image
                    self._objects_canvas.display_image(image)
                    self.objects_status_label.config(text="Image loaded. Click 'Select Object' to mark an object.")
                    self._object_selector.set_image(image)
                else:
                    messagebox.showerror("Error", "Failed to load image")

        except Exception as e:
            logger.error(f"Error loading image: {e}")
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def _start_object_selection(self):
        """Start interactive object selection."""
        try:
            if self._training_image is None:
                messagebox.showwarning("Warning", "Please capture or load an image first")
                return

            self._object_selector.activate()
            self.objects_status_label.config(text="Draw a rectangle around the object...")

        except Exception as e:
            logger.error(f"Error starting selection: {e}")

    def _on_object_selection_complete(self, bbox: tuple):
        """Handle object selection completion.

        Args:
            bbox: Bounding box (x1, y1, x2, y2)
        """
        try:
            self._object_selector.deactivate()

            if self._training_image is None:
                return

            # Extract object from image
            x1, y1, x2, y2 = map(int, bbox)
            h, w = self._training_image.shape[:2]

            # Ensure bbox is within image bounds
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))

            object_image = self._training_image[y1:y2, x1:x2]

            if object_image.size == 0:
                messagebox.showerror("Error", "Invalid selection")
                return

            # Show naming dialog
            dialog = ObjectNamingDialog(self.root, object_image)
            confirmed, label = dialog.show()

            if confirmed and label:
                # Add to training service
                self.training_service.add_object(object_image, label, bbox=(x1, y1, x2, y2))
                self._refresh_objects_list()
                self.objects_status_label.config(text=f"Object '{label}' added to training dataset")
                logger.info(f"Training object added: {label}")

        except Exception as e:
            logger.error(f"Error completing selection: {e}")
            messagebox.showerror("Error", f"Failed to add object: {e}")

    def _refresh_objects_list(self):
        """Refresh the objects listbox."""
        try:
            self._objects_listbox.delete(0, tk.END)

            objects = self.training_service.get_all_objects()
            for obj in objects:
                status = "✓" if obj.confirmed else "○"
                self._objects_listbox.insert(tk.END, f"{status} {obj.label} ({obj.object_id})")

            # Update count
            counts = self.training_service.get_object_count()
            self.objects_count_label.config(
                text=f"({counts['confirmed']}/{counts['total']} confirmed)"
            )

        except Exception as e:
            logger.error(f"Error refreshing objects list: {e}")

    def _confirm_selected_object(self):
        """Confirm selected object for training."""
        try:
            selection = self._objects_listbox.curselection()
            if not selection:
                messagebox.showwarning("Warning", "Please select an object first")
                return

            idx = selection[0]
            objects = self.training_service.get_all_objects()

            if idx < len(objects):
                obj = objects[idx]
                self.training_service.update_object(obj.object_id, confirmed=True)
                self._refresh_objects_list()
                self.status_label.config(text=f"Object confirmed: {obj.label}")

        except Exception as e:
            logger.error(f"Error confirming object: {e}")

    def _unconfirm_selected_object(self):
        """Unconfirm selected object."""
        try:
            selection = self._objects_listbox.curselection()
            if not selection:
                messagebox.showwarning("Warning", "Please select an object first")
                return

            idx = selection[0]
            objects = self.training_service.get_all_objects()

            if idx < len(objects):
                obj = objects[idx]
                self.training_service.update_object(obj.object_id, confirmed=False)
                self._refresh_objects_list()
                self.status_label.config(text=f"Object unconfirmed: {obj.label}")

        except Exception as e:
            logger.error(f"Error unconfirming object: {e}")

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

    def _delete_selected_object(self):
        """Delete selected object."""
        try:
            selection = self._objects_listbox.curselection()
            if not selection:
                messagebox.showwarning("Warning", "Please select an object first")
                return

            idx = selection[0]
            objects = self.training_service.get_all_objects()

            if idx < len(objects):
                obj = objects[idx]

                if messagebox.askyesno("Confirm Delete", f"Delete object '{obj.label}'?"):
                    self.training_service.delete_object(obj.object_id)
                    self._refresh_objects_list()
                    self.status_label.config(text=f"Object deleted: {obj.label}")

        except Exception as e:
            logger.error(f"Error deleting object: {e}")

    def _view_selected_object(self):
        """View selected object image."""
        try:
            selection = self._objects_listbox.curselection()
            if not selection:
                messagebox.showwarning("Warning", "Please select an object first")
                return

            idx = selection[0]
            objects = self.training_service.get_all_objects()

            if idx < len(objects):
                obj = objects[idx]
                self._objects_canvas.display_image(obj.image)
                self.objects_status_label.config(text=f"Viewing: {obj.label}")

        except Exception as e:
            logger.error(f"Error viewing object: {e}")

    def _train_model_with_confirmed_objects(self):
        """Start model training with confirmed objects."""
        try:
            counts = self.training_service.get_object_count()

            if counts['confirmed'] == 0:
                messagebox.showwarning(
                    "No Objects",
                    "Please confirm at least one object before training."
                )
                return

            if not messagebox.askyesno(
                "Confirm Training",
                f"Train model with {counts['confirmed']} confirmed objects?\nThis may take several minutes."
            ):
                return

            # Show progress dialog
            progress = TrainingProgressDialog(self.root)

            def train_thread():
                try:
                    progress.update_status("Starting training...")

                    base_model = self.config.get('preferred_model', 'yolo12n')
                    epochs = self.config.get('train_epochs', 10)
                    batch_size = self.config.get('batch_size', 8)

                    success = self.training_service.train_model(
                        base_model=base_model,
                        epochs=epochs,
                        batch_size=batch_size
                    )

                    if success:
                        progress.set_complete(True, "Training completed successfully!")
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

    def _export_confirmed_dataset(self):
        """Export confirmed objects as dataset."""
        try:
            counts = self.training_service.get_object_count()

            if counts['confirmed'] == 0:
                messagebox.showwarning(
                    "No Objects",
                    "Please confirm at least one object before exporting."
                )
                return

            # Ask for export directory
            export_dir = filedialog.askdirectory(title="Select Export Directory")

            if export_dir:
                if self.training_service.export_dataset(export_dir, format='yolo'):
                    messagebox.showinfo(
                        "Export Complete",
                        f"Dataset exported to:\n{export_dir}"
                    )
                    self.status_label.config(text="Dataset exported successfully")
                else:
                    messagebox.showerror("Error", "Failed to export dataset")

        except Exception as e:
            logger.error(f"Error exporting dataset: {e}")
            messagebox.showerror("Error", f"Failed to export dataset: {e}")

    def _on_objects_listbox_mousewheel(self, event):
        """Handle mouse wheel scrolling on objects listbox."""
        self._objects_listbox.yview_scroll(-1 * int(event.delta / 120), "units")

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

            # Disable send button
            self._send_button.config(state='disabled')
            self.status_label.config(text="Processing...")

            def process_chat():
                try:
                    # Get current frame
                    frame = self.webcam_service.get_current_frame()

                    if frame is None:
                        self.root.after(0, lambda: self._add_chat_message(
                            "System",
                            "No webcam frame available. Please start the stream."
                        ))
                        return

                    # Run YOLO detection
                    detections = self.inference_service.detect(frame)

                    # Get reference image if available
                    ref_image = self.reference_manager.get_reference()

                    # Get AI response
                    if ref_image is not None:
                        # Compare mode
                        ref_detections = self.inference_service.detect(ref_image)
                        response = self.gemini_service.compare_images(
                            ref_image, frame, message, ref_detections, detections
                        )
                    else:
                        # Single image analysis
                        response = self.gemini_service.analyze_image(frame, message, detections)

                    if response:
                        self.root.after(0, lambda: self._add_chat_message("AI", response))
                        self.root.after(0, lambda: self.status_label.config(text="Response received"))
                    else:
                        self.root.after(0, lambda: self._add_chat_message(
                            "System",
                            "Failed to get AI response. Please check your API key and connection."
                        ))

                except Exception as e:
                    logger.error(f"Error processing chat: {e}")
                    self.root.after(0, lambda: self._add_chat_message(
                        "System",
                        f"Error: {e}"
                    ))

                finally:
                    self.root.after(0, lambda: self._send_button.config(state='normal'))

            # Process in background
            threading.Thread(target=process_chat, daemon=True).start()

        except Exception as e:
            logger.error(f"Error in send chat: {e}")
            self._send_button.config(state='normal')

    def _add_chat_message(self, sender: str, message: str):
        """Add message to chat display.

        Args:
            sender: Message sender (User, AI, System)
            message: Message text
        """
        try:
            # Create message frame
            msg_frame = tk.Frame(self._messages_frame, bg=self.COLORS['bg_primary'])
            msg_frame.pack(fill='x', padx=10, pady=5)

            # Sender label
            sender_color = {
                'User': self.COLORS['accent_primary'],
                'AI': self.COLORS['success'],
                'System': self.COLORS['warning']
            }.get(sender, self.COLORS['text_secondary'])

            sender_label = tk.Label(
                msg_frame,
                text=f"{sender}:",
                bg=self.COLORS['bg_primary'],
                fg=sender_color,
                font=('Segoe UI', 9, 'bold')
            )
            sender_label.pack(anchor='w')

            # Message label
            message_label = tk.Label(
                msg_frame,
                text=message,
                bg=self.COLORS['bg_primary'],
                fg=self.COLORS['text_primary'],
                font=('Segoe UI', 9),
                wraplength=400,
                justify='left'
            )
            message_label.pack(anchor='w', pady=(2, 0))

            # Scroll to bottom
            self._chat_canvas.update_idletasks()
            self._chat_canvas.yview_moveto(1.0)

        except Exception as e:
            logger.error(f"Error adding chat message: {e}")

    def _clear_chat(self):
        """Clear all chat messages."""
        try:
            for widget in self._messages_frame.winfo_children():
                widget.destroy()

            self.status_label.config(text="Chat cleared")

        except Exception as e:
            logger.error(f"Error clearing chat: {e}")

    def _on_chat_mousewheel(self, event):
        """Handle mouse wheel scrolling on chat canvas."""
        self._chat_canvas.yview_scroll(-1 * int(event.delta / 120), "units")

    def _on_chat_input_mousewheel(self, event):
        """Handle mouse wheel scrolling on chat input."""
        return None  # Allow default scrolling

    def _setup_chat_accessibility(self):
        """Setup accessibility features for chat."""
        pass  # Placeholder for future accessibility features

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
            logger.error(f"Error opening settings: {e}")
            messagebox.showerror("Error", f"Failed to open settings: {e}")

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
                max_tokens=new_config.get('gemini_max_tokens')
            )

            # Update Gemini configuration status
            self._gemini_configured = bool(new_config.get('gemini_api_key', '').strip())

            self.status_label.config(text="Settings applied successfully")
            logger.info("Settings updated")

        except Exception as e:
            logger.error(f"Error applying settings: {e}")
            messagebox.showerror("Error", f"Failed to apply settings: {e}")

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