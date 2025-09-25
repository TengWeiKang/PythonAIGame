"""Modern main application window with comprehensive UI/UX design and performance optimizations."""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import json
import threading
import logging
from typing import Optional, Dict, Any, Callable
from datetime import datetime

from ..config.settings import Config, save_config
from ..services.webcam_service import WebcamService
from ..services.gemini_service import AsyncGeminiService
from ..services.inference_service import InferenceService
from ..services.detection_service import DetectionService
from ..services.annotation_service import AnnotationService
from ..services.training_service import TrainingService
from ..services.object_training_service import ObjectTrainingService
from ..services.image_analysis_service import ImageAnalysisService
from ..services.integrated_analysis_service import IntegratedAnalysisService
from ..services.reference_manager import ReferenceImageManager
from ..services.yolo_workflow_orchestrator import YoloWorkflowOrchestrator, WorkflowConfig
from ..core.entities import PipelineState
from ..utils.file_utils import (
    save_reference_image,
    load_reference_image,
    get_latest_reference_image,
    cleanup_old_reference_images,
    get_image_info
)
from ..utils.detection_formatter import format_detection_data, format_detection_summary_compact

# Fallback to ASCII formatter for Windows compatibility
try:
    from ..utils.detection_formatter_ascii import (
        format_detection_data_ascii,
        format_detection_summary_compact_ascii
    )
    HAS_ASCII_FORMATTER = True
except ImportError:
    HAS_ASCII_FORMATTER = False
from .components.status_bar import StatusBar
from .components.object_selector import ObjectSelector
from .dialogs.object_naming_dialog import ObjectNamingDialog
from .dialogs.object_edit_dialog import ObjectEditDialog
from .dialogs.training_progress_dialog import TrainingProgressDialog

# Import performance optimizations
from ..core.performance import PerformanceMonitor, performance_timer
from ..core.cache_manager import CacheManager, generate_image_hash
from ..core.memory_manager import get_memory_manager
from ..core.threading_manager import get_threading_manager, TaskPriority, run_in_thread
from ..ui.optimized_canvas import OptimizedCanvas, VideoCanvas, ChatCanvas


class ModernMainWindow:
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
        'Light': {
            'bg_primary': '#ffffff',        # Light background
            'bg_secondary': '#f5f5f5',      # Secondary panels
            'bg_tertiary': '#e0e0e0',       # Tertiary elements
            'accent_primary': '#0078d4',    # Blue accent
            'accent_secondary': '#106ebe',  # Darker blue
            'text_primary': '#000000',      # Black text
            'text_secondary': '#333333',    # Dark gray text
            'text_muted': '#666666',        # Muted gray text
            'success': '#4caf50',           # Green for success
            'warning': '#ff9800',           # Orange for warnings
            'error': '#f44336',             # Red for errors
            'border': '#cccccc',            # Border color
        }
    }
    
    def __init__(self, root: tk.Tk, config: Config):
        self.root = root
        self.config = config
        self.current_theme = config.app_theme
        self.COLORS = self.THEMES.get(self.current_theme, self.THEMES['Dark'])
        
        # Initialize performance systems first with error handling
        try:
            self.performance_monitor = PerformanceMonitor.instance()
        except Exception as e:
            logging.error(f"Failed to initialize performance monitor: {e}")
            self.performance_monitor = None
            
        try:
            # Ensure data and cache directories exist
            os.makedirs(config.data_dir, exist_ok=True)
            cache_dir = os.path.join(config.data_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_manager = CacheManager(cache_dir)
        except Exception as e:
            logging.error(f"Failed to initialize cache manager: {e}")
            self.cache_manager = None
            
        try:
            self.memory_manager = get_memory_manager()
        except Exception as e:
            logging.error(f"Failed to initialize memory manager: {e}")
            self.memory_manager = None
            
        try:
            self.threading_manager = get_threading_manager()
        except Exception as e:
            logging.error(f"Failed to initialize threading manager: {e}")
            self.threading_manager = None
        
        # Load localization
        self.locale = self._load_locale()
        
        # Initialize services with comprehensive error handling
        try:
            self.webcam_service = WebcamService()
        except Exception as e:
            logging.error(f"Failed to initialize webcam service: {e}")
            self.webcam_service = None
        
        # Initialize Gemini service with enhanced configuration detection
        try:
            # Check if we have environment configuration with API key
            env_api_key = None
            if hasattr(config, '_environment_config') and config._environment_config:
                env_api_key = config._environment_config.gemini_api_key
                if env_api_key:
                    logging.info("Using Gemini API key from environment configuration")

            # Use environment API key if available, otherwise fall back to config
            effective_api_key = env_api_key or getattr(config, 'gemini_api_key', '')

            self.gemini_service = AsyncGeminiService(
                api_key=effective_api_key,
                model=getattr(config, 'gemini_model', 'gemini-1.5-flash'),
                timeout=getattr(config, 'gemini_timeout', 30),
                temperature=getattr(config, 'gemini_temperature', 0.7),
                max_tokens=getattr(config, 'gemini_max_tokens', 2048),
                persona=getattr(config, 'chatbot_persona', '')
            )

            # Log configuration status for debugging
            if hasattr(self.gemini_service, 'get_configuration_status'):
                status = self.gemini_service.get_configuration_status()
                logging.info(f"Gemini service configuration status: {status}")

        except Exception as e:
            logging.error(f"Failed to initialize Gemini service: {e}")
            self.gemini_service = None

        # Register memory pressure callback if available
        if self.memory_manager:
            try:
                self.memory_manager.register_memory_pressure_callback(self._on_memory_pressure)
            except Exception as e:
                logging.error(f"Failed to register memory pressure callback: {e}")

        # Enhanced chat session initialization with better detection
        self._gemini_configured = False
        if self.gemini_service:
            try:
                # Use the service's own configuration check rather than config attribute
                if self.gemini_service.is_configured():
                    self.gemini_service.start_chat_session(getattr(config, 'chatbot_persona', ''))
                    self._gemini_configured = True
                    logging.info("Gemini chat session started successfully")
                else:
                    logging.warning("Gemini service not properly configured - chat session not started")
                    # Get detailed status for debugging
                    if hasattr(self.gemini_service, 'get_configuration_status'):
                        status = self.gemini_service.get_configuration_status()
                        logging.debug(f"Gemini configuration details: {status}")
            except Exception as e:
                logging.error(f"Failed to start Gemini chat session: {e}")
                self._gemini_configured = False
        
        try:
            self.inference_service = InferenceService(config)
        except Exception as e:
            logging.error(f"Failed to initialize inference service: {e}")
            self.inference_service = None
            
        try:
            self.annotation_service = AnnotationService(config)
        except Exception as e:
            logging.error(f"Failed to initialize annotation service: {e}")
            self.annotation_service = None
        try:
            self.training_service = TrainingService(config)
        except Exception as e:
            logging.error(f"Failed to initialize training service: {e}")
            self.training_service = None
            
        try:
            self.object_training_service = ObjectTrainingService(config)
        except Exception as e:
            logging.error(f"Failed to initialize object training service: {e}")
            self.object_training_service = None

        try:
            # ImageAnalysisService requires InferenceService, so initialize it after inference_service is ready
            if self.inference_service:
                self.image_analysis_service = ImageAnalysisService(self.inference_service, config)
            else:
                logging.warning("InferenceService not available, ImageAnalysisService will not be initialized")
                self.image_analysis_service = None
        except Exception as e:
            logging.error(f"Failed to initialize image analysis service: {e}")
            self.image_analysis_service = None

        # Initialize IntegratedAnalysisService for comprehensive YOLO+Chatbot integration
        self.reference_manager = None
        self.workflow_orchestrator = None

        try:
            if self.gemini_service and hasattr(config, 'model_size'):
                # Initialize YOLO backend
                from ..backends.yolo_backend import YoloBackend

                yolo_config = {
                    'model_size': getattr(config, 'model_size', 'yolo12n'),
                    'confidence_threshold': getattr(config, 'confidence_threshold', 0.5),
                    'device': getattr(config, 'device', 'auto')
                }

                self.yolo_backend = YoloBackend(yolo_config)

                # Load the YOLO model
                model_name = yolo_config['model_size']
                if self.yolo_backend.load_model(model_name):
                    # Initialize Reference Image Manager
                    try:
                        reference_data_dir = os.path.join(self.config.data_dir, 'references')
                        os.makedirs(reference_data_dir, exist_ok=True)

                        self.reference_manager = ReferenceImageManager(
                            yolo_backend=self.yolo_backend,
                            data_dir=reference_data_dir,
                            max_references=getattr(config, 'max_references', 100),
                            max_memory_mb=getattr(config, 'reference_max_memory_mb', 50),
                            enable_compression=getattr(config, 'reference_compression', True)
                        )
                        logging.info("ReferenceImageManager initialized successfully")
                    except Exception as e:
                        logging.error(f"Failed to initialize ReferenceImageManager: {e}")
                        self.reference_manager = None

                    # Initialize IntegratedAnalysisService
                    integration_config = {
                        'enable_image_comparison': getattr(config, 'enable_image_comparison', True),
                        'enable_scene_analysis': getattr(config, 'enable_scene_analysis', True),
                        'chatbot_persona': getattr(config, 'chatbot_persona', ''),
                        'response_format': getattr(config, 'response_format', 'Detailed')
                    }

                    self.integrated_analysis_service = IntegratedAnalysisService(
                        yolo_backend=self.yolo_backend,
                        gemini_service=self.gemini_service,
                        config=integration_config
                    )
                    logging.info("IntegratedAnalysisService initialized successfully")

                    # Initialize Workflow Orchestrator
                    workflow_config = WorkflowConfig(
                        auto_yolo_analysis=getattr(config, 'auto_yolo_analysis', True),
                        reference_comparison_enabled=getattr(config, 'reference_comparison_enabled', True),
                        min_detection_confidence=getattr(config, 'min_detection_confidence', 0.5),
                        max_objects_to_analyze=getattr(config, 'max_objects_to_analyze', 10),
                        async_timeout_seconds=getattr(config, 'workflow_async_timeout', 5.0),
                        enable_performance_monitoring=getattr(config, 'workflow_performance_monitoring', True),
                        cache_analysis_results=getattr(config, 'cache_analysis_results', True)
                    )

                    self.workflow_orchestrator = YoloWorkflowOrchestrator(
                        yolo_backend=self.yolo_backend,
                        reference_manager=self.reference_manager,
                        gemini_service=self.gemini_service,
                        integrated_service=self.integrated_analysis_service,
                        config=workflow_config
                    )
                    logging.info("YoloWorkflowOrchestrator initialized successfully")
                else:
                    logging.warning("Failed to load YOLO model, IntegratedAnalysisService not available")
                    self.integrated_analysis_service = None
                    self.yolo_backend = None
            else:
                logging.warning("Gemini service or model configuration not available, IntegratedAnalysisService not initialized")
                self.integrated_analysis_service = None
                self.yolo_backend = None
        except Exception as e:
            logging.error(f"Failed to initialize integrated analysis service: {e}")
            self.integrated_analysis_service = None
            self.yolo_backend = None
            self.workflow_orchestrator = None
        
        # Initialize state variables
        self._current_frame = None
        self._reference_image = None
        self._captured_image = None
        self._is_streaming = False
        self._stream_thread = None
        self._detection_service: Optional[DetectionService] = None
        
        # UI components
        self._video_canvas = None
        self._reference_canvas = None
        self._current_canvas = None
        self._chat_canvas = None
        self._messages_frame = None
        self._chat_input = None
        self._status_bar = None
        
        # Objects tab components
        self._objects_canvas = None
        self._objects_listbox = None
        self._object_selector = None
        self._training_objects = []
        
        # Dialog references
        self._settings_dialog = None
        
        # Chat history
        self._chat_history = []
        
        # Setup the window and UI
        self._setup_window()
        self._setup_styles()
        self._build_ui()
        
        # Initialize services with current config values
        self._update_all_services_with_config()
        
        # Apply current performance mode
        self.set_performance_mode(self.config.performance_mode)
        
        # Auto-load reference image if available
        self.root.after(200, self._auto_load_reference_image)
        
        # Show welcome message in chat if Gemini is configured
        self.root.after(100, self._show_welcome_message)
    
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
        locale_path = os.path.join(
            self.config.locales_dir, 
            self.config.default_locale + ".json"
        )
        try:
            with open(locale_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[ModernMainWindow] Locale fallback empty ({e})")
            return {}
    
    def t(self, key: str, fallback: str) -> str:
        """Get localized string."""
        return self.locale.get(key, fallback)
    
    def apply_theme(self, theme_name: str) -> bool:
        """Apply a theme to the entire application.
        
        Args:
            theme_name: Name of the theme ('Dark' or 'Light')
            
        Returns:
            bool: True if theme was applied successfully
        """
        if theme_name not in self.THEMES:
            logging.warning(f"Unknown theme: {theme_name}")
            return False
        
        try:
            self.current_theme = theme_name
            self.COLORS = self.THEMES[theme_name]
            
            # Update config
            self.config.app_theme = theme_name
            
            # Reapply styles and colors
            self._setup_styles()
            self._apply_theme_to_widgets()
            
            logging.info(f"Theme applied successfully: {theme_name}")
            return True
        except Exception as e:
            logging.error(f"Failed to apply theme {theme_name}: {e}")
            return False
    
    def set_theme(self, theme_name: str) -> bool:
        """Alias for apply_theme for service integration compatibility."""
        return self.apply_theme(theme_name)
    
    def _apply_theme_to_widgets(self):
        """Apply current theme colors to all widgets recursively."""
        try:
            self._apply_theme_to_widget(self.root)
        except Exception as e:
            logging.error(f"Failed to apply theme to widgets: {e}")
    
    def _apply_theme_to_widget(self, widget):
        """Recursively apply theme to a widget and its children."""
        try:
            # Apply theme based on widget type
            widget_class = widget.__class__.__name__
            
            if widget_class == 'Tk':
                widget.configure(bg=self.COLORS['bg_primary'])
            elif widget_class in ['Frame', 'LabelFrame']:
                current_bg = widget.cget('bg')
                # Only update if it matches a theme color pattern
                if current_bg in ['#1e1e1e', '#2d2d2d', '#3c3c3c', '#ffffff', '#f5f5f5', '#e0e0e0']:
                    if current_bg in ['#1e1e1e', '#ffffff']:
                        widget.configure(bg=self.COLORS['bg_primary'])
                    elif current_bg in ['#2d2d2d', '#f5f5f5']:
                        widget.configure(bg=self.COLORS['bg_secondary'])
                    elif current_bg in ['#3c3c3c', '#e0e0e0']:
                        widget.configure(bg=self.COLORS['bg_tertiary'])
            elif widget_class == 'Label':
                current_bg = widget.cget('bg')
                current_fg = widget.cget('fg')
                # Update background and foreground for theme colors
                if current_bg in ['#1e1e1e', '#2d2d2d', '#3c3c3c', '#ffffff', '#f5f5f5', '#e0e0e0']:
                    if current_bg in ['#1e1e1e', '#ffffff']:
                        widget.configure(bg=self.COLORS['bg_primary'])
                    elif current_bg in ['#2d2d2d', '#f5f5f5']:
                        widget.configure(bg=self.COLORS['bg_secondary'])
                    elif current_bg in ['#3c3c3c', '#e0e0e0']:
                        widget.configure(bg=self.COLORS['bg_tertiary'])
                
                if current_fg in ['#ffffff', '#cccccc', '#999999', '#000000', '#333333', '#666666']:
                    if current_fg in ['#ffffff', '#000000']:
                        widget.configure(fg=self.COLORS['text_primary'])
                    elif current_fg in ['#cccccc', '#333333']:
                        widget.configure(fg=self.COLORS['text_secondary'])
                    elif current_fg in ['#999999', '#666666']:
                        widget.configure(fg=self.COLORS['text_muted'])
            elif widget_class == 'Text':
                current_bg = widget.cget('bg')
                current_fg = widget.cget('fg')
                if current_bg in ['#1e1e1e', '#2d2d2d', '#3c3c3c', '#ffffff', '#f5f5f5', '#e0e0e0']:
                    if current_bg in ['#3c3c3c', '#e0e0e0']:
                        widget.configure(
                            bg=self.COLORS['bg_tertiary'],
                            fg=self.COLORS['text_primary'],
                            insertbackground=self.COLORS['text_primary']
                        )
            
            # Recursively apply to children
            for child in widget.winfo_children():
                self._apply_theme_to_widget(child)
        except Exception as e:
            logging.debug(f"Error applying theme to widget {widget}: {e}")
    
    def set_performance_mode(self, mode: str) -> bool:
        """Set performance mode and apply optimizations.
        
        Args:
            mode: Performance mode ('Performance', 'Balanced', 'Power_Saving')
            
        Returns:
            bool: True if performance mode was applied successfully
        """
        valid_modes = ['Performance', 'Balanced', 'Power_Saving']
        if mode not in valid_modes:
            logging.warning(f"Invalid performance mode: {mode}")
            return False
        
        try:
            # Update config
            self.config.performance_mode = mode
            
            # Apply performance optimizations
            if mode == 'Performance':
                self._apply_performance_optimizations()
            elif mode == 'Balanced':
                self._apply_balanced_optimizations()
            else:  # Power_Saving
                self._apply_power_saving_optimizations()
            
            logging.info(f"Performance mode set to: {mode}")
            return True
        except Exception as e:
            logging.error(f"Failed to set performance mode {mode}: {e}")
            return False
    
    def _apply_performance_optimizations(self):
        """Apply optimizations for maximum performance."""
        try:
            # Increase FPS targets for real-time processing
            if self.config.target_fps < 60:
                self.config.target_fps = 60
            
            # Increase camera FPS if supported
            if self.config.camera_fps < 30:
                self.config.camera_fps = 30
            
            # Optimize memory usage for performance
            if self.memory_manager:
                self.memory_manager.set_memory_pressure_threshold(0.85)  # Higher threshold
            
            # Optimize threading for performance
            if self.threading_manager:
                self.threading_manager.set_max_workers('detection', 4)  # More workers
                self.threading_manager.set_max_workers('inference', 2)
            
            # Enable GPU usage if available
            self.config.use_gpu = True
            
            # Optimize cache settings
            if self.cache_manager:
                self.cache_manager.set_cache_size_mb(512)  # Larger cache
                
            logging.debug("Performance optimizations applied")
        except Exception as e:
            logging.error(f"Failed to apply performance optimizations: {e}")
    
    def _apply_balanced_optimizations(self):
        """Apply balanced optimizations for good performance with reasonable resource usage."""
        try:
            # Standard FPS targets
            self.config.target_fps = 30
            self.config.camera_fps = 30
            
            # Balanced memory usage
            if self.memory_manager:
                self.memory_manager.set_memory_pressure_threshold(0.75)  # Standard threshold
            
            # Balanced threading
            if self.threading_manager:
                self.threading_manager.set_max_workers('detection', 2)  # Moderate workers
                self.threading_manager.set_max_workers('inference', 1)
            
            # Keep GPU setting as configured
            # self.config.use_gpu remains unchanged
            
            # Standard cache settings
            if self.cache_manager:
                self.cache_manager.set_cache_size_mb(256)  # Standard cache
                
            logging.debug("Balanced optimizations applied")
        except Exception as e:
            logging.error(f"Failed to apply balanced optimizations: {e}")
    
    def _apply_power_saving_optimizations(self):
        """Apply optimizations to minimize resource usage."""
        try:
            # Lower FPS targets to save power
            self.config.target_fps = 15
            self.config.camera_fps = 15
            
            # Conservative memory usage
            if self.memory_manager:
                self.memory_manager.set_memory_pressure_threshold(0.60)  # Lower threshold
            
            # Minimal threading
            if self.threading_manager:
                self.threading_manager.set_max_workers('detection', 1)  # Single worker
                self.threading_manager.set_max_workers('inference', 1)
            
            # Prefer CPU over GPU for lower power consumption
            self.config.use_gpu = False
            
            # Minimal cache to save memory
            if self.cache_manager:
                self.cache_manager.set_cache_size_mb(128)  # Smaller cache
                
            logging.debug("Power saving optimizations applied")
        except Exception as e:
            logging.error(f"Failed to apply power saving optimizations: {e}")
    
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
        
        ttk.Button(
            right_frame, 
            text="❓ Help",
            style='Secondary.TButton',
            command=self._show_help
        ).pack(side='right')
    
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
        self._video_canvas = VideoCanvas(
            video_content,
            bg='black',
            highlightthickness=0,
            target_fps=30
        )
        self._video_canvas.pack(fill='both', expand=True)
        
        # Set video canvas to high quality for better performance
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
        
        ttk.Button(
            controls_frame,
            text="❌ Cancel Selection",
            style='Secondary.TButton',
            command=self._cancel_object_selection
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
    
    # Event handlers and functionality methods
    
    def _on_start_stream(self):
        """Start the webcam stream."""
        try:
            if not self.webcam_service or not self.webcam_service.is_opened():
                success = self.webcam_service and self.webcam_service.open(
                    self.config.last_webcam_index,
                    self.config.camera_width,
                    self.config.camera_height,
                    self.config.camera_fps
                )
                if not success:
                    messagebox.showerror("Error", "Failed to open webcam")
                    return
            
            self._is_streaming = True
            self.start_button.configure(state='disabled')
            self.stop_button.configure(state='normal')
            self.capture_button.configure(state='normal')
            
            # Start streaming thread
            self._stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
            self._stream_thread.start()
            
            self._update_status("Stream started")
            self.connection_label.configure(text="🟢 Connected", fg=self.COLORS['success'])
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start stream: {e}")
    
    def _on_stop_stream(self):
        """Stop the webcam stream."""
        try:
            self._is_streaming = False
            
            if self._stream_thread and self._stream_thread.is_alive():
                self._stream_thread.join(timeout=1.0)
            
            if self.webcam_service:
                self.webcam_service.close()

            self.start_button.configure(state='normal')
            self.stop_button.configure(state='disabled')
            self.capture_button.configure(state='disabled')
            
            self._update_status("Stream stopped")
            self.connection_label.configure(text="⚪ Disconnected", fg=self.COLORS['text_muted'])
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop stream: {e}")
    
    def _on_capture_image(self):
        """Capture current frame as image."""
        if self._current_frame is not None:
            self._captured_image = self._current_frame.copy()
            self._display_image_on_canvas(self._current_canvas, self._captured_image)
            self.current_info_label.configure(
                text=f"Captured: {datetime.now().strftime('%H:%M:%S')}"
            )
            self._update_status("Image captured")
        else:
            messagebox.showwarning("Warning", "No frame available to capture")
    
    def _stream_worker(self):
        """Background thread worker for video streaming."""
        fps_counter = 0
        import time
        last_fps_time = time.time()
        
        while self._is_streaming:
            try:
                if self.webcam_service:
                    ret, frame = self.webcam_service.read()
                else:
                    ret, frame = False, None
                if ret and frame is not None:
                    self._current_frame = frame.copy()
                    
                    # Update video display
                    self.root.after(0, self._update_video_display, frame)
                    
                    # Calculate FPS
                    fps_counter += 1
                    current_time = time.time()
                    if current_time - last_fps_time >= 1.0:
                        fps = fps_counter / (current_time - last_fps_time)
                        self.root.after(0, self._update_fps_display, fps)
                        fps_counter = 0
                        last_fps_time = current_time
                    
                    # Update resolution info
                    h, w = frame.shape[:2]
                    self.root.after(0, self._update_resolution_display, w, h)
                
                time.sleep(1.0 / 30.0)  # Limit to ~30 FPS
                
            except Exception as e:
                print(f"Stream error: {e}")
                break
    
    @performance_timer("ui_update_video_display")
    def _update_video_display(self, frame):
        """Update the video display canvas with optimizations."""
        try:
            # Use optimized canvas display method
            success = self._video_canvas.display_frame(frame)
            if not success:
                print("Failed to display video frame")
        except Exception as e:
            print(f"Error updating video display: {e}")
    
    def _update_fps_display(self, fps):
        """Update the FPS display."""
        self.fps_label.configure(text=f"FPS: {fps:.1f}")
    
    def _update_resolution_display(self, width, height):
        """Update the resolution display."""
        self.resolution_label.configure(text=f"Resolution: {width}x{height}")
    
    @performance_timer("ui_display_image_on_canvas")
    def _display_image_on_canvas(self, canvas, image):
        """Display an image on the specified canvas with optimizations."""
        if image is None:
            return False
        
        try:
            # Use optimized canvas if available
            if hasattr(canvas, 'display_image_optimized'):
                return canvas.display_image_optimized(image)
            
            # Fallback to original implementation for non-optimized canvases
            return self._display_image_fallback(canvas, image)
        except Exception as e:
            print(f"Error displaying image: {e}")
            return False
    
    def _display_image_fallback(self, canvas, image):
        """Fallback image display for non-optimized canvases."""
        if image is None:
            return False
        
        # Get canvas dimensions
        canvas.update_idletasks()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return False
        
        # Check cache first
        image_hash = generate_image_hash(image)
        cache_key = f"display:{image_hash}:{canvas_width}x{canvas_height}"
        
        cached_photo = self.cache_manager.image_cache.get(cache_key)
        if cached_photo is not None:
            canvas.delete("all")
            canvas.create_image(
                canvas_width // 2, 
                canvas_height // 2, 
                anchor="center", 
                image=cached_photo
            )
            canvas.image = cached_photo  # Keep reference
            return True
        
        # Process image
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Calculate scaling
        img_height, img_width = image_rgb.shape[:2]
        scale = min(canvas_width / img_width, canvas_height / img_height)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        resized_image = cv2.resize(image_rgb, (new_width, new_height))
        
        # Convert to PhotoImage
        pil_image = Image.fromarray(resized_image)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Cache the PhotoImage
        self.cache_manager.image_cache.put(cache_key, photo)
        
        # Update canvas
        canvas.delete("all")
        canvas.create_image(
            canvas_width // 2, 
            canvas_height // 2, 
            anchor="center", 
            image=photo
        )
        canvas.image = photo  # Keep reference
        
        return True
    
    def _update_status(self, message: str):
        """Update the status bar message."""
        self.status_label.configure(text=message)
    
    # Additional methods for image management, chat, settings, etc. will be continued in the next part...
    
    def _load_reference_image(self):
        """Load reference image from file."""
        file_path = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                image = cv2.imread(file_path)
                if image is not None:
                    self._reference_image = image
                    self._display_image_on_canvas(self._reference_canvas, image)

                    # Set reference image in IntegratedAnalysisService for YOLO comparison
                    self._set_reference_in_integrated_service(self._reference_image)

                    # Auto-save reference image for persistence
                    saved_path = self._save_reference_image_persistent(image, "file")
                    if saved_path:
                        # Update config with saved path
                        self.config.reference_image_path = saved_path
                        self._save_config_async()

                    self.ref_info_label.configure(
                        text=f"Loaded: {os.path.basename(file_path)}"
                    )
                    self._update_status("Reference image loaded and saved")

                    # Show user feedback about auto-save
                    self.root.after(2000, lambda: self._show_reference_auto_save_feedback("loaded"))
                else:
                    messagebox.showerror("Error", "Could not load image file")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def _set_reference_from_stream(self):
        """Set reference image from current stream frame."""
        if self._current_frame is not None:
            self._reference_image = self._current_frame.copy()
            self._display_image_on_canvas(self._reference_canvas, self._reference_image)

            # Set reference image in IntegratedAnalysisService for YOLO comparison
            self._set_reference_in_integrated_service(self._reference_image)

            # Auto-save reference image for persistence
            saved_path = self._save_reference_image_persistent(self._reference_image, "camera")
            if saved_path:
                # Update config with saved path
                self.config.reference_image_path = saved_path
                self._save_config_async()

            self.ref_info_label.configure(
                text=f"From stream: {datetime.now().strftime('%H:%M:%S')}"
            )
            self._update_status("Reference set from stream and saved")

            # Show user feedback about auto-save
            self.root.after(2000, lambda: self._show_reference_auto_save_feedback("captured"))
        else:
            messagebox.showwarning("Warning", "No stream frame available")
    
    def _clear_reference_image(self):
        """Clear the reference image."""
        self._reference_image = None
        self._reference_canvas.delete("all")
        self.ref_info_label.configure(text="No reference image loaded")
        
        # Clear from config as well
        self.config.reference_image_path = ""
        self._save_config_async()
        
        self._update_status("Reference image cleared")
    
    # Objects Tab Methods
    
    def _capture_for_training(self):
        """Capture current frame for object training."""
        if self._current_frame is not None:
            image = self._current_frame.copy()
            self._object_selector.set_image(image)
            self.objects_status_label.configure(
                text=f"Image captured: {datetime.now().strftime('%H:%M:%S')} - Select an object to train"
            )
            self._update_status("Image captured for training")
        else:
            messagebox.showwarning("Warning", "No stream frame available. Start the video stream first.")
    
    def _load_image_for_training(self):
        """Load an image from file for object training."""
        file_path = filedialog.askopenfilename(
            title="Select Image for Training",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                image = cv2.imread(file_path)
                if image is not None:
                    self._object_selector.set_image(image)
                    filename = os.path.basename(file_path)
                    self.objects_status_label.configure(
                        text=f"Loaded: {filename} - Select an object to train"
                    )
                    self._update_status("Image loaded for training")
                else:
                    messagebox.showerror("Error", "Could not load image file")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def _start_object_selection(self):
        """Start object selection mode."""
        if not self._object_selector.has_image():
            messagebox.showwarning("Warning", "Please capture or load an image first")
            return
        
        self._object_selector.start_selection()
        self.objects_status_label.configure(
            text="Selection mode active - Draw rectangle around object to train"
        )
        self._update_status("Object selection started - draw rectangle around object")
    
    def _cancel_object_selection(self):
        """Cancel object selection mode."""
        self._object_selector.stop_selection()
        self.objects_status_label.configure(
            text="Selection cancelled - Click 'Select Object' to try again"
        )
        self._update_status("Object selection cancelled")
    
    def _on_object_selection_complete(self, cropped_image, coordinates):
        """Handle completed object selection."""
        try:
            # Get source image
            source_image = self._object_selector.current_image
            
            # Open naming dialog
            dialog = ObjectNamingDialog(
                self.root,
                coordinates,
                source_image,
                cropped_image
            )
            
            # Wait for dialog to complete
            self.root.wait_window(dialog.window)
            result = dialog.get_result()
            
            if result:
                # Save object to training service
                object_id = self.object_training_service.save_object(
                    name=result['name'],
                    image=result['cropped_image'],
                    coordinates=coordinates,
                    source_image=result['source_image'],
                    metadata={
                        'description': result.get('description', ''),
                        'confidence': result.get('confidence', 0.8),
                        'auto_generate': result.get('auto_generate', False)
                    }
                )
                
                # Refresh objects list
                self._refresh_objects_list()
                
                # Update status
                self.objects_status_label.configure(
                    text=f"Object '{result['name']}' saved successfully"
                )
                self._update_status(f"Training object '{result['name']}' saved")
                
                messagebox.showinfo("Success", f"Object '{result['name']}' saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save training object: {e}")
            self._update_status("Failed to save training object")
    
    def _refresh_objects_list(self):
        """Refresh the training objects list."""
        try:
            # Clear listbox
            self._objects_listbox.delete(0, tk.END)
            
            # Load objects from service
            self._training_objects = self.object_training_service.load_objects()
            
            # Update count label with confirmed/total
            count = len(self._training_objects)
            confirmed_count = self.object_training_service.get_confirmed_count()
            self.objects_count_label.configure(
                text=f"({confirmed_count}/{count} confirmed)"
            )
            
            # Populate listbox with confirmation status
            for obj in self._training_objects:
                timestamp = obj.get('timestamp', 'Unknown')
                confirmed = obj.get('confirmed', False)
                status_icon = "✅" if confirmed else "❌"
                display_text = f"{status_icon} {obj['name']} ({timestamp})"
                self._objects_listbox.insert(tk.END, display_text)
                
        except Exception as e:
            print(f"Error refreshing objects list: {e}")
            self.objects_count_label.configure(text="(Error loading)")
    
    def _edit_selected_object(self):
        """Edit the selected training object."""
        selection = self._objects_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an object to edit")
            return
        
        try:
            # Get selected object
            index = selection[0]
            if index >= len(self._training_objects):
                return
            
            object_data = self._training_objects[index]
            
            # Open edit dialog
            dialog = ObjectEditDialog(self.root, object_data)
            
            # Wait for dialog to complete
            self.root.wait_window(dialog.window)
            result = dialog.get_result()
            
            if result:
                if result.get('delete'):
                    # Handle deletion
                    self._delete_object(object_data['id'])
                else:
                    # Handle update
                    success = self.object_training_service.update_object(
                        object_data['id'],
                        result
                    )
                    
                    if success:
                        self._refresh_objects_list()
                        self._update_status("Object updated successfully")
                        messagebox.showinfo("Success", "Object updated successfully!")
                    else:
                        messagebox.showerror("Error", "Failed to update object")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to edit object: {e}")
    
    def _delete_selected_object(self):
        """Delete the selected training object."""
        selection = self._objects_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an object to delete")
            return
        
        try:
            # Get selected object
            index = selection[0]
            if index >= len(self._training_objects):
                return
            
            object_data = self._training_objects[index]
            object_name = object_data['name']
            
            # Confirm deletion
            if messagebox.askyesno(
                "Confirm Deletion",
                f"Are you sure you want to delete '{object_name}'?\n\nThis action cannot be undone."
            ):
                self._delete_object(object_data['id'])
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete object: {e}")
    
    def _delete_object(self, object_id):
        """Delete object by ID."""
        success = self.object_training_service.delete_object(object_id)
        
        if success:
            self._refresh_objects_list()
            self._update_status("Object deleted successfully")
            messagebox.showinfo("Success", "Object deleted successfully!")
        else:
            messagebox.showerror("Error", "Failed to delete object")
    
    def _view_selected_object(self):
        """View the selected training object."""
        selection = self._objects_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an object to view")
            return
        
        try:
            # Get selected object
            index = selection[0]
            if index >= len(self._training_objects):
                return
            
            object_data = self._training_objects[index]
            
            # Create preview window
            self._create_object_preview_window(object_data)
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to view object: {e}")
    
    def _confirm_selected_object(self):
        """Confirm the selected training object."""
        selection = self._objects_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an object to confirm")
            return
        
        try:
            # Get selected object
            index = selection[0]
            if index >= len(self._training_objects):
                return
            
            object_data = self._training_objects[index]
            object_id = object_data['id']
            object_name = object_data['name']
            
            # Confirm the object
            success = self.object_training_service.confirm_object(object_id)
            
            if success:
                self._refresh_objects_list()
                self._update_status(f"Object '{object_name}' confirmed for training")
                messagebox.showinfo("Success", f"Object '{object_name}' confirmed for training!")
            else:
                messagebox.showerror("Error", "Failed to confirm object")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to confirm object: {e}")
    
    def _unconfirm_selected_object(self):
        """Unconfirm the selected training object."""
        selection = self._objects_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an object to unconfirm")
            return
        
        try:
            # Get selected object
            index = selection[0]
            if index >= len(self._training_objects):
                return
            
            object_data = self._training_objects[index]
            object_id = object_data['id']
            object_name = object_data['name']
            
            # Unconfirm the object
            success = self.object_training_service.unconfirm_object(object_id)
            
            if success:
                self._refresh_objects_list()
                self._update_status(f"Object '{object_name}' unconfirmed")
                messagebox.showinfo("Success", f"Object '{object_name}' unconfirmed!")
            else:
                messagebox.showerror("Error", "Failed to unconfirm object")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to unconfirm object: {e}")
    
    def _train_model_with_confirmed_objects(self):
        """Start training a YOLO model with confirmed objects."""
        try:
            # Check if we have confirmed objects
            confirmed_count = self.object_training_service.get_confirmed_count()
            if confirmed_count == 0:
                messagebox.showwarning("No Confirmed Objects", 
                                     "No confirmed objects found. Please confirm some objects first.")
                return
            
            # Show confirmation dialog
            if not messagebox.askyesno("Train Model", 
                                     f"Start training YOLO model with {confirmed_count} confirmed objects?\n\n"
                                     "This process may take several minutes depending on your hardware."):
                return
            
            # Open training progress dialog
            training_dialog = TrainingProgressDialog(
                self.root, 
                self.config, 
                self.object_training_service
            )
            
        except Exception as e:
            messagebox.showerror("Training Error", f"Failed to start training: {e}")
            self._update_status(f"Training failed: {e}")
    
    def _export_confirmed_dataset(self):
        """Export only confirmed objects as a dataset."""
        try:
            # Check if we have confirmed objects
            confirmed_count = self.object_training_service.get_confirmed_count()
            if confirmed_count == 0:
                messagebox.showwarning("No Confirmed Objects", 
                                     "No confirmed objects found. Please confirm some objects first.")
                return
            
            # Ask user for export location
            export_path = filedialog.askdirectory(
                title="Select Export Directory",
                parent=self.root
            )
            
            if not export_path:
                return
            
            # Export confirmed dataset
            self._update_status("Exporting confirmed dataset...")
            result_path = self.object_training_service.export_confirmed_dataset("yolo", export_path)
            
            self._update_status("Dataset exported successfully")
            messagebox.showinfo("Export Complete", 
                              f"Confirmed dataset exported successfully!\n\n"
                              f"Location: {result_path}\n"
                              f"Exported {confirmed_count} confirmed objects")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export dataset: {e}")
            self._update_status(f"Export failed: {e}")
    
    def _create_object_preview_window(self, object_data):
        """Create a preview window for the selected object."""
        preview_window = tk.Toplevel(self.root)
        preview_window.title(f"Preview: {object_data['name']}")
        preview_window.geometry("500x400")
        preview_window.resizable(True, True)
        
        # Center on main window
        x = self.root.winfo_x() + 50
        y = self.root.winfo_y() + 50
        preview_window.geometry(f"+{x}+{y}")
        
        # Main frame
        main_frame = ttk.Frame(preview_window, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        ttk.Label(
            main_frame,
            text=f"Object: {object_data['name']}",
            font=('Arial', 12, 'bold')
        ).pack(pady=(0, 10))
        
        # Image preview
        if object_data.get('image') is not None:
            canvas = tk.Canvas(main_frame, bg='black', height=250)
            canvas.pack(fill='both', expand=True, pady=(0, 10))
            
            # Display image
            self._display_image_on_canvas(canvas, object_data['image'])
        
        # Object information
        info_frame = ttk.LabelFrame(main_frame, text="Information", padding="10")
        info_frame.pack(fill='x')
        
        # Create info text
        info_lines = []
        info_lines.append(f"Name: {object_data['name']}")
        info_lines.append(f"Created: {object_data.get('created_at', 'Unknown')}")
        
        if object_data.get('description'):
            info_lines.append(f"Description: {object_data['description']}")
        
        if 'image_shape' in object_data:
            shape = object_data['image_shape']
            if len(shape) >= 2:
                height, width = shape[:2]
                info_lines.append(f"Dimensions: {width} x {height}")
        
        info_text = '\n'.join(info_lines)
        
        ttk.Label(info_frame, text=info_text, justify='left').pack(anchor='w')
        
        # Close button
        ttk.Button(
            main_frame,
            text="Close",
            command=preview_window.destroy
        ).pack(pady=(10, 0))
    
    def _export_training_dataset(self):
        """Export the training dataset."""
        try:
            # Check if we have objects to export
            if not self._training_objects:
                messagebox.showwarning("No Data", "No training objects to export")
                return
            
            # Ask user for export format
            format_dialog = tk.Toplevel(self.root)
            format_dialog.title("Export Format")
            format_dialog.geometry("300x150")
            format_dialog.resizable(False, False)
            format_dialog.transient(self.root)
            format_dialog.grab_set()
            
            # Center dialog
            x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 150
            y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 75
            format_dialog.geometry(f"+{x}+{y}")
            
            # Dialog content
            main_frame = ttk.Frame(format_dialog, padding="20")
            main_frame.pack(fill='both', expand=True)
            
            ttk.Label(main_frame, text="Select export format:", font=('Arial', 11, 'bold')).pack(pady=(0, 10))
            
            format_var = tk.StringVar(value="yolo")
            
            ttk.Radiobutton(main_frame, text="YOLO Format", variable=format_var, value="yolo").pack(anchor='w', pady=2)
            ttk.Radiobutton(main_frame, text="COCO Format", variable=format_var, value="coco").pack(anchor='w', pady=2)
            ttk.Radiobutton(main_frame, text="Pascal VOC Format", variable=format_var, value="pascal_voc").pack(anchor='w', pady=2)
            
            # Buttons
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill='x', pady=(15, 0))
            
            result = {'format': None}
            
            def on_export():
                result['format'] = format_var.get()
                format_dialog.destroy()
            
            def on_cancel():
                result['format'] = None
                format_dialog.destroy()
            
            ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side='right', padx=(5, 0))
            ttk.Button(button_frame, text="Export", command=on_export).pack(side='right')
            
            # Wait for dialog
            self.root.wait_window(format_dialog)
            
            if result['format']:
                # Choose output directory
                output_dir = filedialog.askdirectory(title="Select Export Directory")
                if output_dir:
                    # Export dataset
                    export_path = self.object_training_service.export_dataset(
                        result['format'],
                        output_dir
                    )
                    
                    self._update_status(f"Dataset exported to {export_path}")
                    messagebox.showinfo("Success", f"Dataset exported successfully to:\n{export_path}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export dataset: {e}")
            self._update_status("Dataset export failed")
    
    def _analyze_current_image(self):
        """Analyze current captured image using Gemini."""
        if not self.gemini_service or not self.gemini_service.is_configured():
            messagebox.showwarning("Warning", "Gemini API key not configured. Please check settings.")
            return
        
        if self._captured_image is None:
            messagebox.showwarning("Warning", "No image captured to analyze")
            return
        
        self._add_chat_message("User", "Analyzing current image...")
        self._update_status("Analyzing image...")
        
        def on_result(result, error):
            if error:
                self.root.after(0, lambda: self._add_chat_message("System", f"Error: {error}"))
                self.root.after(0, lambda: self._update_status("Analysis failed"))
            else:
                self.root.after(0, lambda: self._add_chat_message("AI", result))
                self.root.after(0, lambda: self._update_status("Analysis complete"))
        
        if self.gemini_service:
            self.gemini_service.analyze_single_image_async(self._captured_image, on_result)
        else:
            messagebox.showerror("Error", "Gemini service not available")
    
    def _compare_images(self):
        """Compare reference and current images using Gemini."""
        if not self.gemini_service or not self.gemini_service.is_configured():
            messagebox.showwarning("Warning", "Gemini API key not configured. Please check settings.")
            return
        
        if self._reference_image is None:
            messagebox.showwarning("Warning", "No reference image loaded")
            return
        
        if self._captured_image is None:
            messagebox.showwarning("Warning", "No current image captured")
            return
        
        self._add_chat_message("User", "Comparing reference and current images...")
        self._update_status("Comparing images...")
        
        def on_result(result, error):
            if error:
                self.root.after(0, lambda: self._add_chat_message("System", f"Error: {error}"))
                self.root.after(0, lambda: self._update_status("Comparison failed"))
            else:
                self.root.after(0, lambda: self._add_chat_message("AI", result))
                self.root.after(0, lambda: self._update_status("Comparison complete"))
        
        if self.gemini_service:
            self.gemini_service.compare_images_async(
                self._reference_image,
                self._captured_image,
                on_result
            )
        else:
            messagebox.showerror("Error", "Gemini service not available")
    
    
    def _on_chat_send(self, event=None):
        """Legacy chat send method - redirects to enhanced version."""
        return self._on_chat_send_enhanced(event)
    
    
    def _add_chat_message(self, sender: str, message: str, message_status: str = 'sent'):
        """Add a message to the chat display with modern bubble design and proper alignment."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        message_id = self._last_message_id + 1
        self._last_message_id = message_id
        
        # Create message container with proper alignment setup
        message_container = tk.Frame(self._messages_frame, bg=self.COLORS['bg_primary'])
        message_container.pack(fill='x', padx=10, pady=5)
        
        # Determine message alignment and styling
        if sender == "User":
            alignment = 'e'  # Right alignment for user messages
            bubble_color = self.COLORS['accent_primary']
            text_color = self.COLORS['text_primary']
            sender_icon = "👤"
            # User messages: right-aligned with space on left
            container_padx = (40, 0)  # More space on left, none on right
        elif sender == "AI":
            alignment = 'w'  # Left alignment for AI messages
            bubble_color = self.COLORS['bg_tertiary']
            text_color = self.COLORS['text_primary']
            sender_icon = "🤖"
            # AI messages: left-aligned with space on right
            container_padx = (0, 40)  # No space on left, more on right
        else:  # System
            alignment = 'w'  # Left alignment for system messages
            bubble_color = self.COLORS['warning']
            text_color = self.COLORS['text_primary']
            sender_icon = "⚙️"
            # System messages: left-aligned with space on right
            container_padx = (0, 40)
        
        # Create message bubble frame with proper alignment
        bubble_frame = tk.Frame(message_container, bg=self.COLORS['bg_primary'])
        bubble_frame.pack(anchor=alignment, fill='x', padx=container_padx, pady=2)
        
        # Sender info (for non-user messages)
        if sender != "User":
            sender_frame = tk.Frame(bubble_frame, bg=self.COLORS['bg_primary'])
            sender_frame.pack(anchor='w', padx=5, pady=(0, 2))
            
            tk.Label(
                sender_frame,
                text=f"{sender_icon} {sender}",
                bg=self.COLORS['bg_primary'],
                fg=self.COLORS['text_muted'],
                font=('Segoe UI', 8, 'bold')
            ).pack(side='left')
            
            # Timestamp for AI/System messages
            tk.Label(
                sender_frame,
                text=f"• {timestamp}",
                bg=self.COLORS['bg_primary'],
                fg=self.COLORS['text_muted'],
                font=('Segoe UI', 7)
            ).pack(side='left', padx=(5, 0))
        
        # Message bubble with improved modern styling
        bubble = tk.Frame(
            bubble_frame,
            bg=bubble_color,
            relief='flat',  # Flat relief for cleaner look
            bd=0,           # No border for modern appearance
            padx=2,         # Add internal padding for better bubble appearance
            pady=1
        )
        # Pack bubble with side alignment for user messages
        if sender == "User":
            bubble.pack(side='right', pady=2)  # Pack to right side
        else:
            bubble.pack(side='left', pady=2)   # Pack to left side
        
        # Message text with responsive word wrapping
        max_width = 45 if len(message) > 80 else 35
        wrapped_message = self._wrap_message_text(message, max_width)
        
        message_label = tk.Label(
            bubble,
            text=wrapped_message,
            bg=bubble_color,
            fg=text_color,
            font=('Segoe UI', 9),
            justify='right' if sender == "User" else 'left',
            padx=12,
            pady=8,
            wraplength=350  # Slightly smaller wrap length for better mobile-like appearance
        )
        message_label.pack()
        
        # User message timestamp and status
        if sender == "User":
            status_frame = tk.Frame(bubble_frame, bg=self.COLORS['bg_primary'])
            status_frame.pack(side='right', padx=(0, 5), pady=(2, 0))
            
            # Status indicator
            status_icon = "✓" if message_status == 'sent' else "⏳" if message_status == 'sending' else "❌"
            status_color = self.COLORS['success'] if message_status == 'sent' else self.COLORS['warning'] if message_status == 'sending' else self.COLORS['error']
            
            tk.Label(
                status_frame,
                text=f"{timestamp} {status_icon}",
                bg=self.COLORS['bg_primary'],
                fg=status_color,
                font=('Segoe UI', 7)
            ).pack(side='right')
        
        # Add context menu for copy functionality
        def show_context_menu(event):
            context_menu = tk.Menu(self.root, tearoff=0)
            context_menu.add_command(
                label="📋 Copy Message",
                command=lambda: self._copy_message_to_clipboard(message)
            )
            context_menu.add_separator()
            if sender == "AI":
                context_menu.add_command(
                    label="👍 Good Response",
                    command=lambda: self._rate_message(message_id, 'positive')
                )
                context_menu.add_command(
                    label="👎 Poor Response", 
                    command=lambda: self._rate_message(message_id, 'negative')
                )
            
            try:
                context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                context_menu.grab_release()
        
        # Bind right-click context menu
        message_label.bind('<Button-3>', show_context_menu)
        bubble.bind('<Button-3>', show_context_menu)
        
        # Store message data
        message_data = {
            'id': message_id,
            'timestamp': timestamp,
            'sender': sender,
            'message': message,
            'status': message_status,
            'widget': message_container
        }
        
        self._message_widgets.append(message_data)
        self._chat_history.append(message_data)
        
        # Auto-scroll to bottom with animation
        self.root.after(10, self._scroll_to_bottom)
        
        # Animate message appearance
        self._animate_message_appearance(message_container)
        
        return message_id
    
    def _remove_last_ai_message(self):
        """Remove the last AI message from chat display (e.g., typing indicator)."""
        if not self._message_widgets:
            return
        
        # Check if last message was from AI
        last_message = self._message_widgets[-1]
        if last_message['sender'] != 'AI':
            return
        
        # Remove the widget
        last_message['widget'].destroy()
        
        # Remove from tracking lists
        self._message_widgets.pop()
        if self._chat_history and self._chat_history[-1]['id'] == last_message['id']:
            self._chat_history.pop()
        
        # Update canvas scroll region
        self._chat_canvas.update_idletasks()
        self._chat_canvas.configure(scrollregion=self._chat_canvas.bbox("all"))
    
    def _setup_chat_accessibility(self):
        """Setup accessibility features for the chat interface."""
        # Keyboard navigation bindings
        self._chat_canvas.bind('<Tab>', self._focus_next_widget)
        self._chat_canvas.bind('<Shift-Tab>', self._focus_previous_widget)
        
        # Enable focus for keyboard navigation
        self._chat_canvas.configure(takefocus=True)
        self._chat_input.configure(takefocus=True)
        self._send_button.configure(takefocus=True)
        
        # Screen reader accessibility
        self._chat_canvas.configure(relief='sunken', bd=1)
        
        # Add accessibility descriptions
        self._add_accessibility_labels()
    
    def _add_accessibility_labels(self):
        """Add accessibility labels and descriptions."""
        # Set proper widget names for accessibility (using widget class names)
        try:
            # Use the widget's string representation for identification instead
            self._chat_canvas.widget_name = "chat_messages_area"
            self._chat_input.widget_name = "message_input_field"
            if hasattr(self, '_send_button'):
                self._send_button.widget_name = "send_message_button"
        except (AttributeError, tk.TclError):
            pass  # Gracefully handle if attributes don't exist
    
    def _focus_next_widget(self, event):
        """Navigate to next focusable widget."""
        event.widget.tk_focusNext().focus()
        return 'break'
    
    def _focus_previous_widget(self, event):
        """Navigate to previous focusable widget."""
        event.widget.tk_focusPrev().focus()
        return 'break'
    
    def _clear_chat(self):
        """Clear the chat display."""
        # Clear all message widgets
        for widget_data in self._message_widgets:
            widget_data['widget'].destroy()
        
        self._message_widgets.clear()
        self._chat_history.clear()
        self._last_message_id = 0
        
        # Reset typing indicator
        if self._typing_indicator:
            self._typing_indicator.destroy()
            self._typing_indicator = None
        
        self._update_status("Chat cleared")
    
    # Enhanced chat helper methods
    
    def _wrap_message_text(self, text: str, max_width: int) -> str:
        """Wrap message text for better display with improved handling."""
        import textwrap
        
        # Handle empty or very short messages
        if not text or len(text.strip()) <= 3:
            return text
        
        # Split by existing newlines and wrap each paragraph
        paragraphs = text.split('\n')
        wrapped_paragraphs = []
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                wrapped_paragraphs.append('')  # Preserve empty lines
            else:
                # Use textwrap with break_long_words=False for better word boundaries
                wrapped = textwrap.fill(
                    paragraph, 
                    width=max_width, 
                    break_long_words=False, 
                    break_on_hyphens=False,
                    expand_tabs=True
                )
                wrapped_paragraphs.append(wrapped)
        
        return '\n'.join(wrapped_paragraphs)
    
    def _scroll_to_bottom(self):
        """Smoothly scroll chat to bottom."""
        self._chat_canvas.update_idletasks()
        self._chat_canvas.yview_moveto(1.0)
    
    def _on_chat_mousewheel(self, event):
        """Handle mouse wheel scrolling in chat."""
        self._chat_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_chat_input_mousewheel(self, event):
        """Handle mouse wheel scrolling in chat input text area."""
        self._chat_input.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_objects_listbox_mousewheel(self, event):
        """Handle mouse wheel scrolling in objects listbox."""
        self._objects_listbox.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _animate_message_appearance(self, message_widget):
        """Animate message appearance with fade-in effect."""
        # Simple fade-in animation by adjusting alpha
        message_widget.configure(bg=self.COLORS['bg_primary'])
        # Note: Tkinter doesn't support true alpha blending, so we simulate with color transitions
        
    def _copy_message_to_clipboard(self, message: str):
        """Copy message text to clipboard."""
        self.root.clipboard_clear()
        self.root.clipboard_append(message)
        self._update_status("Message copied to clipboard")
    
    def _rate_message(self, message_id: int, rating: str):
        """Rate an AI message (for future learning/improvement)."""
        # Find the message and update rating
        for msg_data in self._message_widgets:
            if msg_data['id'] == message_id:
                msg_data['rating'] = rating
                break
        
        rating_text = "positive" if rating == 'positive' else "negative"
        self._update_status(f"Message rated as {rating_text}")
    
    def _on_chat_send_enhanced(self, event=None):
        """Enhanced chat send handler with improved UX."""
        # Get message text
        message = self._chat_input.get(1.0, tk.END).strip()
        if not message:
            return 'break' if event else None
        
        # Clear input
        self._chat_input.delete(1.0, tk.END)
        
        # Add user message with sending status
        message_id = self._add_chat_message("User", message, "sending")
        
        # Update message status to sent
        self.root.after(100, lambda: self._update_message_status(message_id, "sent"))
        
        # Handle message processing
        self._process_chat_message(message)
        
        return 'break' if event else None
    
    def _on_chat_newline(self, event):
        """Handle Shift+Return for new line."""
        return None  # Allow default behavior (new line)
    
    def _on_chat_typing(self, event):
        """Handle typing indicators and input validation."""
        message = self._chat_input.get(1.0, tk.END).strip()
        
        # Update send button state
        if message:
            self._send_button.configure(bg=self.COLORS['accent_primary'], state='normal')
        else:
            self._send_button.configure(bg=self.COLORS['bg_tertiary'], state='normal')
    
    def _update_message_status(self, message_id: int, new_status: str):
        """Update message status indicator."""
        for msg_data in self._message_widgets:
            if msg_data['id'] == message_id:
                msg_data['status'] = new_status
                # Update visual indicator if needed
                break
    
    def _validate_stream_for_analysis(self) -> tuple[bool, Optional[str]]:
        """
        Validate that the video stream is active and ready for automatic YOLO analysis.

        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            # Check if streaming is active
            if not hasattr(self, '_is_streaming') or not self._is_streaming:
                return False, "Video stream is not active. Please start the webcam to enable automatic analysis."

            # Check if webcam service is available and properly initialized
            if not hasattr(self, 'webcam_service') or not self.webcam_service:
                return False, "Webcam service is not available or not initialized."

            # Check if webcam is opened and functioning
            try:
                if not self.webcam_service.is_opened():
                    return False, "Webcam is not connected or opened. Please check camera connection."
            except Exception as e:
                return False, f"Webcam service error: {str(e)}"

            # Check if current frame is available and valid
            if not hasattr(self, '_current_frame') or self._current_frame is None:
                return False, "No current frame available. Please ensure the webcam is working properly."

            # Check frame validity and dimensions
            try:
                if self._current_frame.size == 0:
                    return False, "Current frame is empty or corrupted."

                # Check minimum frame dimensions (reasonable for analysis)
                height, width = self._current_frame.shape[:2]
                if height < 100 or width < 100:
                    return False, f"Frame too small for analysis ({width}x{height}). Minimum 100x100 required."

            except Exception as e:
                return False, f"Frame validation error: {str(e)}"

            # Check if integrated analysis service is available and properly configured
            if not hasattr(self, 'integrated_analysis_service') or self.integrated_analysis_service is None:
                return False, "YOLO analysis service is not available or not initialized."

            # Validate YOLO backend is loaded and ready
            if not hasattr(self, 'yolo_backend') or self.yolo_backend is None:
                return False, "YOLO backend is not available."

            try:
                if not hasattr(self.yolo_backend, 'is_loaded') or not self.yolo_backend.is_loaded:
                    return False, "YOLO model is not loaded. Please ensure model is properly initialized."
            except Exception as e:
                return False, f"YOLO backend validation error: {str(e)}"

            # Check Gemini service is configured for the integrated analysis
            if not hasattr(self, 'gemini_service') or not self.gemini_service:
                return False, "Gemini service is not available for AI analysis."

            try:
                if not self.gemini_service.is_configured():
                    return False, "Gemini service is not properly configured. Please check API key in settings."
            except Exception as e:
                return False, f"Gemini service validation error: {str(e)}"

            # All validations passed
            return True, None

        except Exception as e:
            # Catch any unexpected errors in validation
            import traceback
            logging.error(f"Stream validation failed with unexpected error: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return False, f"Validation error: {str(e)}"

    def _set_reference_in_integrated_service(self, reference_image: np.ndarray) -> bool:
        """
        Helper method to set reference image in IntegratedAnalysisService and ReferenceManager.

        Args:
            reference_image: Reference image to set

        Returns:
            bool: True if successful, False otherwise
        """
        success = False

        # Set in reference manager first if available
        if hasattr(self, 'reference_manager') and self.reference_manager:
            try:
                # Capture reference with current configuration
                reference_id = self.reference_manager.capture_reference(
                    reference_image,
                    confidence_threshold=self.config.min_detection_confidence
                )
                if reference_id:
                    logging.info(f"Reference image captured successfully in ReferenceManager: {reference_id}")
                    success = True
                else:
                    logging.warning("Failed to capture reference image in ReferenceManager")
            except Exception as e:
                logging.error(f"Error capturing reference in ReferenceManager: {e}")

        # Also set in integrated service for backward compatibility
        if hasattr(self, 'integrated_analysis_service') and self.integrated_analysis_service:
            try:
                service_success = self.integrated_analysis_service.set_reference_image(reference_image)
                if service_success:
                    logging.info("Reference image set successfully in IntegratedAnalysisService")
                    success = True
                else:
                    logging.warning("Failed to set reference image in IntegratedAnalysisService")
            except Exception as e:
                logging.error(f"Error setting reference image in IntegratedAnalysisService: {e}")

        return success

    def _fallback_to_basic_analysis(self, message: str, current_frame: Optional[np.ndarray], chat_callback):
        """
        Fallback method for basic analysis when automatic YOLO inference is not available.

        Args:
            message: User's original message
            current_frame: Current video frame (if available)
            chat_callback: Callback function to handle the response
        """
        try:
            enhanced_message = message

            # Try basic image analysis if available
            if (current_frame is not None and
                hasattr(self, 'image_analysis_service') and
                self.image_analysis_service is not None):
                try:
                    logging.info("Attempting basic image analysis fallback")
                    analysis_result = self.image_analysis_service.analyze_frame_comprehensive(
                        current_frame, message
                    )

                    if analysis_result:
                        enhanced_message = self.image_analysis_service.format_for_chatbot(
                            analysis_result, message
                        )
                        logging.info(f"Basic image analysis successful: {len(enhanced_message)} characters")
                    else:
                        logging.warning("Basic image analysis returned no results")

                except Exception as e:
                    logging.error(f"Basic image analysis failed: {e}")
                    # Continue with original message

            # Send to Gemini (with or without basic analysis enhancement)
            if hasattr(self, 'gemini_service') and self.gemini_service:
                # Include image if available
                if current_frame is not None:
                    try:
                        import cv2
                        _, image_bytes = cv2.imencode('.jpg', current_frame)
                        image_data = image_bytes.tobytes()
                        result = self.gemini_service.send_message(enhanced_message, image_data)
                    except Exception as e:
                        logging.warning(f"Failed to encode image for Gemini: {e}")
                        result = self.gemini_service.send_message(enhanced_message)
                else:
                    result = self.gemini_service.send_message(enhanced_message)

                chat_callback(result, None)
            else:
                chat_callback(None, "Gemini service not available")

        except Exception as e:
            logging.error(f"Fallback analysis failed: {e}")
            chat_callback(None, f"Analysis failed: {str(e)}")

    def _process_chat_message(self, message: str):
        """Process chat message with enhanced handling and diagnostics."""
        # Enhanced service availability check with detailed diagnostics
        if not self.gemini_service:
            self._add_chat_message("System",
                "Chatbot service not initialized. Please restart the application.")
            return

        if not self.gemini_service.is_configured():
            # Get detailed configuration status for better error messages
            if hasattr(self.gemini_service, 'get_configuration_status'):
                try:
                    status = self.gemini_service.get_configuration_status()
                    if status.get('errors'):
                        error_msg = f"Chatbot configuration issue: {'; '.join(status['errors'])}"
                    elif not status.get('has_api_key'):
                        error_msg = "No API key found. Please set GEMINI_API_KEY environment variable or configure in Settings."
                    elif not status.get('sdk_available'):
                        error_msg = "Google Generative AI SDK not available. Please install: pip install google-generativeai"
                    elif not status.get('api_key_valid_format'):
                        error_msg = "API key format is invalid. Please check your Gemini API key."
                    elif not status.get('has_model'):
                        error_msg = "AI model not initialized. Check your API key and internet connection."
                    else:
                        error_msg = "Chatbot not configured. Please check Settings > Chatbot."

                    logging.debug(f"Chatbot configuration status: {status}")
                except Exception as e:
                    logging.error(f"Failed to get configuration status: {e}")
                    error_msg = "Chatbot configuration check failed. Please check Settings > Chatbot."
            else:
                error_msg = "Chatbot not configured. Please check Settings > Chatbot to set up your API key."

            self._add_chat_message("System", error_msg)
            return
        
        # Handle special commands (help command removed - now acts as normal chat)
        if message.lower().strip() == "analyze current":
            self._analyze_current_image()
            return
        elif message.lower().strip() == "compare":
            self._compare_images()
            return
        
        # Show enhanced typing indicator
        self._show_typing_indicator()
        
        # Send message to Gemini API
        def chat_callback(result, error):
            # Remove typing indicator
            self.root.after(0, self._hide_typing_indicator)
            if error:
                self.root.after(0, lambda: self._add_chat_message("System", f"Error: {error}", "error"))
            else:
                self.root.after(0, lambda: self._add_chat_message("AI", result))
        
        # Send message in background thread with automatic YOLO inference
        try:
            def send_message():
                try:
                    if not self.gemini_service:
                        chat_callback(None, "Gemini service not available")
                        return

                    # AUTOMATIC YOLO INFERENCE - Use new workflow orchestrator
                    stream_valid, validation_error = self._validate_stream_for_analysis()

                    # Try workflow orchestrator first if available
                    if stream_valid and hasattr(self, 'workflow_orchestrator') and self.workflow_orchestrator:
                        try:
                            logging.info(f"Starting orchestrated YOLO workflow for message: '{message[:50]}...'")

                            # Capture current frame for analysis
                            current_frame = self._current_frame.copy()

                            # Run orchestrated workflow asynchronously in this thread
                            import asyncio

                            # Create event loop for async operation
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)

                            try:
                                # Run the orchestrated workflow
                                workflow_result = loop.run_until_complete(
                                    self.workflow_orchestrator.orchestrate_analysis(
                                        current_frame, message, self._reference_image
                                    )
                                )

                                if workflow_result and workflow_result.success:
                                    # Success! Use formatted data and AI response
                                    logging.info(f"Orchestrated workflow completed in {workflow_result.workflow_time_ms:.1f}ms")

                                    # Combine detection data with AI response
                                    if workflow_result.formatted_data and workflow_result.ai_response:
                                        enhanced_response = f"{workflow_result.formatted_data}\n\n{workflow_result.ai_response}"
                                    elif workflow_result.ai_response:
                                        enhanced_response = workflow_result.ai_response
                                    else:
                                        enhanced_response = workflow_result.formatted_data or "Analysis completed"

                                    chat_callback(enhanced_response, None)
                                    return
                                else:
                                    # Log the error but continue to fallback
                                    error_msg = workflow_result.error_message if workflow_result else "Unknown workflow error"
                                    logging.warning(f"Orchestrated workflow failed: {error_msg}")

                            finally:
                                loop.close()

                        except Exception as e:
                            logging.error(f"Orchestrated workflow failed: {e}")
                            # Continue to fallback

                    # Fallback to integrated analysis service if orchestrator not available
                    elif stream_valid and hasattr(self, 'integrated_analysis_service') and self.integrated_analysis_service:
                        # FALLBACK: INTEGRATED ANALYSIS SERVICE
                        try:
                            logging.info(f"Falling back to integrated analysis for message: '{message[:50]}...'")

                            # Capture current frame for analysis
                            current_frame = self._current_frame.copy()

                            # Run integrated YOLO+Gemini analysis synchronously in this thread
                            import asyncio

                            # Create event loop for async operation
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)

                            try:
                                # Run the integrated analysis (YOLO data will be hidden in the prompt)
                                analysis_result = loop.run_until_complete(
                                    self.integrated_analysis_service.analyze_with_chatbot(
                                        current_frame, message
                                    )
                                )

                                if analysis_result and analysis_result.success:
                                    # Success! Create enhanced response with visible YOLO detection data
                                    logging.info(f"Integrated analysis completed successfully in {analysis_result.analysis_duration_ms:.1f}ms")

                                    # Extract detection data for formatting
                                    enhanced_response = self._create_enhanced_chat_response(
                                        analysis_result, current_frame.shape[:2][::-1]  # (width, height)
                                    )

                                    chat_callback(enhanced_response, None)
                                    return
                                else:
                                    # Log the error but continue to fallback
                                    error_msg = analysis_result.error_message if analysis_result else "Unknown analysis error"
                                    logging.warning(f"Automatic YOLO analysis failed: {error_msg}")

                            finally:
                                loop.close()

                        except Exception as e:
                            logging.error(f"Automatic YOLO inference failed: {e}")
                            # Continue to fallback
                    else:
                        # Log why automatic YOLO is not available
                        if validation_error:
                            logging.info(f"Automatic YOLO analysis not available: {validation_error}")
                        else:
                            logging.info("Integrated analysis service not available")

                    # FALLBACK: Basic analysis or simple chat (when YOLO fails or unavailable)
                    self._fallback_to_basic_analysis(message, self._current_frame, chat_callback)

                except Exception as e:
                    logging.error(f"Chat message processing failed: {e}")
                    chat_callback(None, str(e))

            threading.Thread(target=send_message, daemon=True).start()
        except Exception as e:
            self._add_chat_message("System", f"Error sending message: {e}", "error")
    
    def _show_typing_indicator(self):
        """Show enhanced typing indicator."""
        if self._typing_indicator:
            return  # Already showing
        
        self._typing_indicator = tk.Frame(self._messages_frame, bg=self.COLORS['bg_primary'])
        self._typing_indicator.pack(fill='x', padx=10, pady=5)
        
        # Typing indicator content
        indicator_frame = tk.Frame(self._typing_indicator, bg=self.COLORS['bg_primary'])
        indicator_frame.pack(anchor='w', pady=2)
        
        # AI sender info
        sender_frame = tk.Frame(indicator_frame, bg=self.COLORS['bg_primary'])
        sender_frame.pack(anchor='w', padx=5)
        
        tk.Label(
            sender_frame,
            text="🤖 AI",
            bg=self.COLORS['bg_primary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8, 'bold')
        ).pack(side='left')
        
        # Typing indicator bubble
        typing_bubble = tk.Frame(
            indicator_frame,
            bg=self.COLORS['bg_tertiary'],
            relief='solid',
            bd=1
        )
        typing_bubble.pack(anchor='w', padx=5, pady=2)
        
        # Animated typing dots
        self._typing_dots_label = tk.Label(
            typing_bubble,
            text="●●●",
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 9),
            padx=12,
            pady=8
        )
        self._typing_dots_label.pack()
        
        # Start typing animation
        self._animate_typing_dots()
        
        # Scroll to bottom
        self.root.after(10, self._scroll_to_bottom)
    
    def _hide_typing_indicator(self):
        """Hide typing indicator."""
        if self._typing_indicator:
            self._typing_indicator.destroy()
            self._typing_indicator = None
    
    def _animate_typing_dots(self):
        """Animate the typing indicator dots."""
        if not self._typing_indicator or not hasattr(self, '_typing_dots_label'):
            return
        
        current_text = self._typing_dots_label.cget('text')
        if current_text == "●●●":
            new_text = "●●○"
        elif current_text == "●●○":
            new_text = "●○○"
        elif current_text == "●○○":
            new_text = "○○○"
        else:
            new_text = "●●●"
        
        try:
            self._typing_dots_label.configure(text=new_text)
            self.root.after(500, self._animate_typing_dots)
        except tk.TclError:
            # Widget destroyed, stop animation
            pass
    
    def _get_help_text(self) -> str:
        """Get help text for chat commands."""
        return """Available commands and features:

🔍 **analyze current** - Analyze the current captured image
📊 **compare** - Compare reference and current images

**Interactive Features:**
• Right-click on any message to copy or rate it
• Use Shift+Enter for new lines in your message
• Scroll through chat history with mouse wheel
• Type commands directly into the chat

**Tips:**
• Ensure good lighting for better image analysis
• Use clear, specific questions for better AI responses
• The AI can analyze images, compare them, and answer questions
• Internet connection required for AI features"""

    def _create_enhanced_chat_response(self, analysis_result, frame_dimensions: tuple) -> str:
        """
        Create enhanced chat response with visible YOLO detection data.

        Args:
            analysis_result: IntegratedAnalysisResult from the analysis service
            frame_dimensions: Tuple of (width, height) for the frame

        Returns:
            Enhanced response string with detection data and AI response
        """
        try:
            response_parts = []

            # Extract detection data from analysis result
            detections = []
            yolo_comparison = analysis_result.yolo_comparison
            image_analysis = analysis_result.image_analysis

            # Try multiple sources for detection data (prioritized)
            # 1. From YOLO comparison current objects
            if yolo_comparison and yolo_comparison.object_comparisons:
                for comp in yolo_comparison.object_comparisons:
                    if comp.current_object:
                        detections.append(comp.current_object)

            # 2. From image analysis objects
            if not detections and image_analysis and image_analysis.objects:
                detections = image_analysis.objects

            # 3. Fallback: Run direct YOLO detection on current frame if no detections found
            if not detections and hasattr(self, 'integrated_analysis_service') and self.integrated_analysis_service:
                try:
                    # Get current frame and run YOLO directly
                    if hasattr(self, '_current_frame') and self._current_frame is not None:
                        yolo_backend = self.integrated_analysis_service.yolo_backend
                        if yolo_backend and yolo_backend.is_loaded:
                            detections = yolo_backend.predict(
                                self._current_frame,
                                conf=0.5,
                                iou=0.45,
                                verbose=False
                            )
                            logging.debug(f"Direct YOLO detection fallback found {len(detections)} objects")
                except Exception as e:
                    logging.warning(f"Direct YOLO detection fallback failed: {e}")

            # Enhance detections with class names if not already present
            if detections and hasattr(self, 'integrated_analysis_service') and self.integrated_analysis_service:
                try:
                    yolo_backend = self.integrated_analysis_service.yolo_backend
                    if yolo_backend and yolo_backend.is_loaded and hasattr(yolo_backend.model, 'names'):
                        class_names = yolo_backend.model.names
                        for detection in detections:
                            if not detection.class_name and detection.class_id in class_names:
                                detection.class_name = class_names[detection.class_id]
                except Exception as e:
                    logging.warning(f"Failed to add class names to detections: {e}")

            # Format detection data if available
            if detections:
                try:
                    # Try the full unicode formatter first
                    detection_data = format_detection_data(
                        detections=detections,
                        frame_dimensions=frame_dimensions,
                        yolo_comparison=yolo_comparison,
                        image_analysis=image_analysis,
                        include_coordinates=True,
                        include_angles=True,
                        include_confidence=True,
                        include_size_info=True
                    )
                except (UnicodeEncodeError, UnicodeError) as e:
                    # Fallback to ASCII formatter for Windows compatibility
                    logging.warning(f"Unicode formatter failed, using ASCII fallback: {e}")
                    if HAS_ASCII_FORMATTER:
                        detection_data = format_detection_data_ascii(
                            detections=detections,
                            frame_dimensions=frame_dimensions,
                            yolo_comparison=yolo_comparison,
                            image_analysis=image_analysis,
                            include_coordinates=True,
                            include_angles=True,
                            include_confidence=True,
                            include_size_info=True
                        )
                    else:
                        detection_data = f"Detection data: {len(detections)} objects found"
                except Exception as e:
                    # General fallback
                    logging.error(f"Detection formatting failed: {e}")
                    detection_data = f"Detection data formatting error: {len(detections)} objects detected"

                response_parts.append(detection_data)
                response_parts.append("")  # Add spacing

            # Add the AI response
            if analysis_result.chatbot_response:
                response_parts.append("AI ANALYSIS:")
                response_parts.append(analysis_result.chatbot_response)

            # Add performance info if verbose mode is enabled
            if hasattr(self, 'config') and self.config and self.config.get('ui_verbose_mode', False):
                response_parts.append("")
                response_parts.append(f"Analysis Time: {analysis_result.analysis_duration_ms:.1f}ms")

            return "\n".join(response_parts)

        except Exception as e:
            logging.error(f"Failed to create enhanced chat response: {e}")
            # Fallback to original response
            return analysis_result.chatbot_response if analysis_result.chatbot_response else "Analysis completed but response formatting failed."

    def _open_settings(self):
        """Open the comprehensive settings dialog."""
        if self._settings_dialog is None or not self._settings_dialog.winfo_exists():
            from .dialogs.comprehensive_settings_dialog import ComprehensiveSettingsDialog
            
            # Prepare services dictionary for the settings dialog
            services = {
                'webcam_service': self.webcam_service,
                'gemini_service': self.gemini_service,
                'inference_service': self.inference_service,
                'annotation_service': self.annotation_service,
                'training_service': self.training_service,
                'object_training_service': self.object_training_service,
                'image_analysis_service': self.image_analysis_service,
                'integrated_analysis_service': getattr(self, 'integrated_analysis_service', None),
                'yolo_backend': getattr(self, 'yolo_backend', None)
            }
            
            self._settings_dialog = ComprehensiveSettingsDialog(
                self.root, 
                self.config, 
                services,
                self._on_settings_changed
            )
        else:
            self._settings_dialog.lift()
    
    def _show_help(self):
        """Show help dialog."""
        help_text = """Vision Analysis System - Help
        
Features:
• Live webcam streaming with controls
• Reference image management (load from file or capture from stream)
• Current image capture and saving
• AI-powered image analysis using Google's Gemini API
• Automatic difference detection between images
• Interactive chat interface for analysis

Getting Started:
1. Configure your Gemini API key in Settings
2. Start the video stream
3. Load or capture a reference image
4. Capture current images for comparison
5. Use the AI analysis features to detect differences

Tips:
• Ensure good lighting for better analysis results
• Use high-quality reference images
• The AI analysis requires an active internet connection
"""
        messagebox.showinfo("Help", help_text)
    
    def _on_settings_changed(self):
        """Handle settings changes and update all services."""
        self._update_all_services_with_config()
        self._update_status("Settings updated")
    
    def _update_all_services_with_config(self):
        """Update all services with current configuration settings."""
        try:
            # Update Gemini service configuration
            if hasattr(self, 'gemini_service'):
                gemini_config = {
                    'api_key': getattr(self.config, 'gemini_api_key', ''),
                    'model': getattr(self.config, 'gemini_model', 'gemini-1.5-flash'),
                    'timeout': getattr(self.config, 'gemini_timeout', 30),
                    'temperature': getattr(self.config, 'gemini_temperature', 0.7),
                    'max_tokens': getattr(self.config, 'gemini_max_tokens', 2048),
                    'persona': getattr(self.config, 'chatbot_persona', ''),
                    'enable_rate_limiting': getattr(self.config, 'enable_rate_limiting', True),
                    'requests_per_minute': getattr(self.config, 'requests_per_minute', 15)
                }
                if self.gemini_service:
                    self.gemini_service.update_configuration(**gemini_config)

                    # Restart chat session with new configuration if API key is available
                    if getattr(self.config, 'gemini_api_key', ''):
                        try:
                            self.gemini_service.start_chat_session(getattr(self.config, 'chatbot_persona', ''))
                        except Exception as e:
                            print(f"Failed to restart chat session: {e}")
            
            # Update webcam service if streaming (restart stream with new settings)
            if hasattr(self, 'webcam_service') and self._is_streaming:
                # Store current streaming state
                was_streaming = self._is_streaming
                
                # Stop current stream
                self._on_stop_stream()
                
                # Apply new webcam settings and restart if it was streaming
                if was_streaming:
                    # Small delay to ensure clean stop
                    self.root.after(500, self._restart_stream_with_new_config)
            
            # Update other services as needed
            # Note: Other services can be updated here when they support configuration changes
            
        except Exception as e:
            print(f"Error updating services with new config: {e}")
    
    def _restart_stream_with_new_config(self):
        """Restart webcam stream with new configuration."""
        try:
            self._on_start_stream()
        except Exception as e:
            print(f"Error restarting stream with new config: {e}")
            messagebox.showerror("Stream Error", f"Failed to restart stream with new settings: {e}")
    
    
    # Reference Image Persistence Methods
    
    def _auto_load_reference_image(self):
        """Auto-load reference image on startup if available."""
        try:
            # Check if we have a reference image path in config
            if hasattr(self.config, 'reference_image_path') and self.config.reference_image_path:
                saved_image = load_reference_image(self.config.reference_image_path)
                if saved_image is not None:
                    self._reference_image = saved_image
                    self._display_image_on_canvas(self._reference_canvas, saved_image)

                    # Set reference image in IntegratedAnalysisService for YOLO comparison
                    self._set_reference_in_integrated_service(self._reference_image)

                    filename = os.path.basename(self.config.reference_image_path)
                    self.ref_info_label.configure(
                        text=f"Auto-loaded: {filename}"
                    )
                    self._update_status("Previous reference image restored")

                    # Show user feedback
                    self.root.after(1000, lambda: self._show_reference_auto_save_feedback("restored"))
                    return
            
            # If no path in config, try to find the latest reference image
            latest_path = get_latest_reference_image(self.config.data_dir)
            if latest_path:
                saved_image = load_reference_image(latest_path)
                if saved_image is not None:
                    self._reference_image = saved_image
                    self._display_image_on_canvas(self._reference_canvas, saved_image)

                    # Set reference image in IntegratedAnalysisService for YOLO comparison
                    self._set_reference_in_integrated_service(self._reference_image)

                    # Update config with found path
                    self.config.reference_image_path = latest_path
                    self._save_config_async()

                    filename = os.path.basename(latest_path)
                    self.ref_info_label.configure(
                        text=f"Auto-loaded: {filename}"
                    )
                    self._update_status("Latest reference image restored")

                    # Show user feedback
                    self.root.after(1000, lambda: self._show_reference_auto_save_feedback("restored"))

                    # Clean up old reference images
                    self.root.after(3000, lambda: cleanup_old_reference_images(self.config.data_dir))
                    
        except Exception as e:
            print(f"Error auto-loading reference image: {e}")
            # Don't show error to user, just log it
    
    def _save_reference_image_persistent(self, image: np.ndarray, source: str) -> Optional[str]:
        """Save reference image for persistence."""
        try:
            return save_reference_image(image, self.config.data_dir, source)
        except Exception as e:
            print(f"Error saving reference image: {e}")
            return None
    
    def _save_config_async(self):
        """Save configuration asynchronously to avoid blocking UI."""
        def save_config_worker():
            try:
                save_config(self.config)
            except Exception as e:
                print(f"Error saving config: {e}")
        
        threading.Thread(target=save_config_worker, daemon=True).start()
    
    def _show_reference_auto_save_feedback(self, action: str):
        """Show temporary feedback about reference image persistence."""
        try:
            if not hasattr(self, 'ref_info_label'):
                return
                
            original_text = self.ref_info_label.cget('text')
            
            if action == "loaded":
                feedback_text = f"{original_text} ✓ Saved for next session"
            elif action == "captured":
                feedback_text = f"{original_text} ✓ Saved for next session"
            elif action == "restored":
                feedback_text = f"{original_text} ✓ From previous session"
            else:
                return
            
            # Show feedback briefly
            self.ref_info_label.configure(text=feedback_text)
            
            # Restore original text after 3 seconds
            self.root.after(3000, lambda: self.ref_info_label.configure(text=original_text))
            
        except Exception as e:
            print(f"Error showing reference feedback: {e}")

    def _on_memory_pressure(self, level: str, stats):
        """Handle memory pressure events."""
        if level == 'critical':
            # Critical memory pressure - aggressive cleanup
            self._emergency_memory_cleanup()
            self._update_status(f"Critical memory usage: {stats.memory_percent:.1f}% - Performing cleanup")
        elif level == 'warning':
            # Warning level - gentle cleanup
            self._optimize_memory_usage()
            self._update_status(f"High memory usage: {stats.memory_percent:.1f}% - Optimizing")
    
    def _emergency_memory_cleanup(self):
        """Emergency memory cleanup for critical memory pressure."""
        try:
            # Clear all image caches
            self.cache_manager.clear_all_caches()
            
            # Clear canvas caches
            if hasattr(self._video_canvas, 'clear_caches'):
                self._video_canvas.clear_caches()
            if hasattr(self._reference_canvas, 'clear_caches'):
                self._reference_canvas.clear_caches()
            if hasattr(self._objects_canvas, 'clear_caches'):
                self._objects_canvas.clear_caches()
            
            # Force memory manager cleanup
            self.memory_manager.emergency_cleanup()
            
            print("Emergency memory cleanup completed")
            
        except Exception as e:
            print(f"Error during emergency memory cleanup: {e}")
    
    def _optimize_memory_usage(self):
        """Optimize memory usage during warnings."""
        try:
            # Gentle cache cleanup
            if hasattr(self._video_canvas, 'clear_caches'):
                self._video_canvas.clear_caches()
            
            # Optimize memory manager
            self.memory_manager.optimize_memory_usage()
            
        except Exception as e:
            print(f"Error during memory optimization: {e}")
    
    @run_in_thread(pool_type='background')
    def _background_performance_monitoring(self):
        """Background performance monitoring and optimization."""
        while hasattr(self, '_monitoring_active') and self._monitoring_active:
            try:
                # Get current performance metrics
                metrics = self.performance_monitor.get_current_metrics()
                
                if metrics:
                    # Check for performance issues
                    if metrics.cpu_percent > 90:
                        # High CPU usage - reduce video quality
                        if hasattr(self._video_canvas, 'set_render_quality'):
                            self._video_canvas.set_render_quality('low')
                    elif metrics.cpu_percent < 50:
                        # Low CPU usage - can increase quality
                        if hasattr(self._video_canvas, 'set_render_quality'):
                            self._video_canvas.set_render_quality('medium')
                    
                    # Update FPS from webcam service
                    if hasattr(self.webcam_service, 'get_current_fps'):
                        actual_fps = self.webcam_service.get_current_fps() if self.webcam_service else 0
                        self.root.after(0, lambda: self._update_fps_display(actual_fps))
                
                # Sleep for monitoring interval
                import time
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                print(f"Performance monitoring error: {e}")
                import time
                time.sleep(1.0)
    
    def start_performance_monitoring(self):
        """Start background performance monitoring."""
        self._monitoring_active = True
        self._background_performance_monitoring()
    
    def stop_performance_monitoring(self):
        """Stop background performance monitoring."""
        self._monitoring_active = False
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'memory': self.memory_manager.get_memory_report(),
            'threading': self.threading_manager.get_comprehensive_stats(),
            'cache': self.cache_manager.get_cache_stats(),
            'performance': {
                'current_metrics': self.performance_monitor.get_current_metrics(),
                'operation_stats': {
                    op: self.performance_monitor.get_operation_stats(op)
                    for op in ['webcam_read', 'ui_display_image_on_canvas', 'gemini_send_message']
                }
            }
        }
    
    def __del__(self):
        """Cleanup resources."""
        try:
            # Stop performance monitoring
            self.stop_performance_monitoring()
            
            # Stop streaming
            self._is_streaming = False
            
            # Close services
            if hasattr(self, 'webcam_service') and self.webcam_service:
                self.webcam_service.close()
            if hasattr(self, 'gemini_service') and self.gemini_service:
                self.gemini_service.cleanup_threads()
            
            # Stop optimized canvases
            if hasattr(self._video_canvas, 'stop_async_rendering'):
                self._video_canvas.stop_async_rendering()
            
            # Shutdown threading manager
            if hasattr(self, 'threading_manager'):
                self.threading_manager.shutdown_all(timeout=5.0)
            
            # Stop memory manager monitoring
            if hasattr(self, 'memory_manager'):
                self.memory_manager.stop_monitoring()
            
            # Stop performance monitoring
            if hasattr(self, 'performance_monitor'):
                self.performance_monitor.stop_monitoring()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def _show_performance_dialog(self):
        """Show performance monitoring dialog."""
        try:
            from .dialogs.performance_dialog import PerformanceDialog
            
            dialog = PerformanceDialog(
                self.root,
                self.get_performance_report(),
                self.cache_manager,
                self.memory_manager,
                self.performance_monitor
            )
        except ImportError:
            # Fallback to simple messagebox if dialog doesn't exist
            report = self.get_performance_report()
            memory_info = report.get('memory', {}).get('current', {})
            
            message = f"""Performance Report:
            
Memory Usage: {memory_info.get('memory_percent', 0):.1f}%
Used Memory: {memory_info.get('used_memory_mb', 0):.1f} MB
Active Threads: {report.get('threading', {}).get('system', {}).get('total_threads', 0)}

Cache Hit Rates:
{self._format_cache_stats(report.get('cache', {}))}
"""
            
            messagebox.showinfo("Performance Report", message)
    
    def _format_cache_stats(self, cache_stats: Dict) -> str:
        """Format cache statistics for display."""
        lines = []
        for cache_name, stats in cache_stats.items():
            if isinstance(stats, dict) and 'hit_rate' in stats:
                lines.append(f"  {cache_name}: {stats['hit_rate']:.1%}")
        return '\n'.join(lines) if lines else "  No cache statistics available"
    
    def _show_welcome_message(self):
        """Show welcome message in chat."""
        if self._gemini_configured and hasattr(self, '_messages_frame'):
            # Use a user-friendly welcome message instead of the internal persona configuration
            welcome_msg = "Hello! I'm your AI assistant powered by Gemini. I can help with image analysis and answer your questions. Just type your question and I'll analyze what I see!"
            self._add_chat_message("AI", welcome_msg)
        elif hasattr(self, '_messages_frame'):
            self._add_chat_message("System",
                "Welcome! To enable AI chat, please configure your Gemini API key in Settings > Chatbot.")

    def cleanup_resources(self):
        """Clean up resources when closing the application."""
        logging.info("Starting application cleanup...")

        # Stop webcam service
        if hasattr(self, 'webcam_service') and self.webcam_service:
            try:
                self.webcam_service.stop()
                logging.info("Webcam service stopped")
            except Exception as e:
                logging.error(f"Failed to stop webcam service: {e}")

        # Shutdown workflow orchestrator
        if hasattr(self, 'workflow_orchestrator') and self.workflow_orchestrator:
            try:
                self.workflow_orchestrator.shutdown()
                logging.info("Workflow orchestrator shutdown successfully")
            except Exception as e:
                logging.error(f"Failed to shutdown workflow orchestrator: {e}")

        # Cleanup reference manager
        if hasattr(self, 'reference_manager') and self.reference_manager:
            try:
                # Save any pending references
                if hasattr(self.reference_manager, 'cleanup'):
                    self.reference_manager.cleanup()
                logging.info("Reference manager cleanup completed")
            except Exception as e:
                logging.error(f"Failed to cleanup reference manager: {e}")

        # Clean up other services
        for service_name in ['detection_service', 'inference_service', 'training_service',
                            'annotation_service', 'object_training_service', 'image_analysis_service']:
            if hasattr(self, service_name):
                service = getattr(self, service_name)
                if service and hasattr(service, 'cleanup'):
                    try:
                        service.cleanup()
                        logging.info(f"{service_name} cleanup completed")
                    except Exception as e:
                        logging.error(f"Failed to cleanup {service_name}: {e}")

        # Save configuration
        if hasattr(self, 'config_manager'):
            try:
                self.config_manager.save()
                logging.info("Configuration saved")
            except Exception as e:
                logging.error(f"Failed to save config: {e}")

        logging.info("Application cleanup completed")