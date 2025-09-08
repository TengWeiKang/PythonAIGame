"""General settings dialog."""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from ...config.settings import Config, save_config

class SettingsDialog:
    """Dialog for general application settings."""
    
    def __init__(self, parent, config: Config):
        self.parent = parent
        self.config = config
        
        # Create dialog window
        self.window = tk.Toplevel(parent)
        self.window.title("General Settings")
        self.window.geometry("600x500")
        self.window.resizable(True, True)
        
        # Make it modal
        self.window.transient(parent)
        self.window.grab_set()
        
        # Center on parent
        self._center_window()
        
        # Build UI
        self._build_ui()
        
        # Load current settings
        self._load_settings()

    def _center_window(self):
        """Center the dialog on the parent window."""
        self.window.update_idletasks()
        x = self.parent.winfo_x() + (self.parent.winfo_width() // 2) - (self.window.winfo_width() // 2)
        y = self.parent.winfo_y() + (self.parent.winfo_height() // 2) - (self.window.winfo_height() // 2)
        self.window.geometry(f"+{x}+{y}")

    def _build_ui(self):
        """Build the dialog UI."""
        # Create notebook for tabbed interface
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # General tab
        self._build_general_tab(notebook)
        
        # Model tab
        self._build_model_tab(notebook)
        
        # Directories tab
        self._build_directories_tab(notebook)
        
        # Detection tab
        self._build_detection_tab(notebook)
        
        # Dialog buttons
        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(button_frame, text="OK", command=self._on_ok).pack(side='right', padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=self._on_cancel).pack(side='right')
        ttk.Button(button_frame, text="Apply", command=self._on_apply).pack(side='right', padx=(0, 5))
        ttk.Button(button_frame, text="Reset to Defaults", command=self._reset_defaults).pack(side='left')

    def _build_general_tab(self, notebook):
        """Build the general settings tab."""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="General")
        
        # Language settings
        lang_frame = ttk.LabelFrame(frame, text="Language", padding="5")
        lang_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(lang_frame, text="Default Locale:").pack(anchor='w')
        self.locale_var = tk.StringVar()
        locale_combo = ttk.Combobox(lang_frame, textvariable=self.locale_var, 
                                   values=['en', 'es', 'fr', 'de'], state='readonly')
        locale_combo.pack(fill='x', pady=(5, 0))
        
        # Performance settings
        perf_frame = ttk.LabelFrame(frame, text="Performance", padding="5")
        perf_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(perf_frame, text="Target FPS:").pack(anchor='w')
        self.target_fps_var = tk.StringVar()
        fps_spin = tk.Spinbox(perf_frame, from_=1, to=60, textvariable=self.target_fps_var, width=10)
        fps_spin.pack(anchor='w', pady=(5, 5))
        
        self.use_gpu_var = tk.BooleanVar()
        ttk.Checkbutton(perf_frame, text="Use GPU acceleration (if available)", 
                       variable=self.use_gpu_var).pack(anchor='w')
        
        # Debug settings
        debug_frame = ttk.LabelFrame(frame, text="Debug", padding="5")
        debug_frame.pack(fill='x')
        
        self.debug_var = tk.BooleanVar()
        ttk.Checkbutton(debug_frame, text="Enable debug mode", 
                       variable=self.debug_var).pack(anchor='w')

    def _build_model_tab(self, notebook):
        """Build the model settings tab."""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="Model")
        
        # Model selection
        model_frame = ttk.LabelFrame(frame, text="Model Selection", padding="5")
        model_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(model_frame, text="Model Size:").pack(anchor='w')
        self.model_size_var = tk.StringVar()
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_size_var,
                                  values=['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x',
                                         'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                                  state='readonly')
        model_combo.pack(fill='x', pady=(5, 5))
        
        ttk.Label(model_frame, text="Image Size:").pack(anchor='w')
        self.img_size_var = tk.StringVar()
        size_combo = ttk.Combobox(model_frame, textvariable=self.img_size_var,
                                 values=['320', '640', '1280'], state='readonly')
        size_combo.pack(fill='x', pady=(5, 0))
        
        # Training settings
        train_frame = ttk.LabelFrame(frame, text="Training", padding="5")
        train_frame.pack(fill='x')
        
        ttk.Label(train_frame, text="Training Epochs:").pack(anchor='w')
        self.train_epochs_var = tk.StringVar()
        epochs_spin = tk.Spinbox(train_frame, from_=1, to=1000, textvariable=self.train_epochs_var, width=10)
        epochs_spin.pack(anchor='w', pady=(5, 5))
        
        ttk.Label(train_frame, text="Batch Size:").pack(anchor='w')
        self.batch_size_var = tk.StringVar()
        batch_spin = tk.Spinbox(train_frame, from_=1, to=64, textvariable=self.batch_size_var, width=10)
        batch_spin.pack(anchor='w', pady=(5, 0))

    def _build_directories_tab(self, notebook):
        """Build the directories settings tab."""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="Directories")
        
        # Directory settings
        dirs = [
            ("Data Directory:", "data_dir"),
            ("Models Directory:", "models_dir"),
            ("Master Directory:", "master_dir"),
            ("Results Export Directory:", "results_export_dir"),
            ("Locales Directory:", "locales_dir")
        ]
        
        self.dir_vars = {}
        
        for label_text, attr_name in dirs:
            dir_frame = ttk.Frame(frame)
            dir_frame.pack(fill='x', pady=5)
            
            ttk.Label(dir_frame, text=label_text, width=20).pack(side='left')
            
            var = tk.StringVar()
            self.dir_vars[attr_name] = var
            
            entry = ttk.Entry(dir_frame, textvariable=var)
            entry.pack(side='left', fill='x', expand=True, padx=(5, 5))
            
            ttk.Button(dir_frame, text="Browse...", 
                      command=lambda v=var: self._browse_directory(v)).pack(side='right')

    def _build_detection_tab(self, notebook):
        """Build the detection settings tab."""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="Detection")
        
        # Matching tolerances
        match_frame = ttk.LabelFrame(frame, text="Matching Tolerances", padding="5")
        match_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(match_frame, text="IoU Match Threshold:").pack(anchor='w')
        self.iou_threshold_var = tk.StringVar()
        iou_spin = tk.Spinbox(match_frame, from_=0.1, to=1.0, increment=0.1,
                             textvariable=self.iou_threshold_var, width=10)
        iou_spin.pack(anchor='w', pady=(5, 5))
        
        ttk.Label(match_frame, text="Master Tolerance (pixels):").pack(anchor='w')
        self.master_tolerance_var = tk.StringVar()
        tolerance_spin = tk.Spinbox(match_frame, from_=1, to=100,
                                   textvariable=self.master_tolerance_var, width=10)
        tolerance_spin.pack(anchor='w', pady=(5, 5))
        
        ttk.Label(match_frame, text="Angle Tolerance (degrees):").pack(anchor='w')
        self.angle_tolerance_var = tk.StringVar()
        angle_spin = tk.Spinbox(match_frame, from_=1, to=180,
                               textvariable=self.angle_tolerance_var, width=10)
        angle_spin.pack(anchor='w', pady=(5, 0))
        
        # Display settings
        display_frame = ttk.LabelFrame(frame, text="Display", padding="5")
        display_frame.pack(fill='x')
        
        ttk.Label(display_frame, text="Preview Max Width:").pack(anchor='w')
        self.preview_width_var = tk.StringVar()
        width_spin = tk.Spinbox(display_frame, from_=320, to=1920, increment=160,
                               textvariable=self.preview_width_var, width=10)
        width_spin.pack(anchor='w', pady=(5, 5))
        
        ttk.Label(display_frame, text="Preview Max Height:").pack(anchor='w')
        self.preview_height_var = tk.StringVar()
        height_spin = tk.Spinbox(display_frame, from_=240, to=1080, increment=120,
                                textvariable=self.preview_height_var, width=10)
        height_spin.pack(anchor='w', pady=(5, 0))

    def _load_settings(self):
        """Load current settings into the dialog."""
        # General tab
        self.locale_var.set(self.config.default_locale)
        self.target_fps_var.set(str(self.config.target_fps))
        self.use_gpu_var.set(self.config.use_gpu)
        self.debug_var.set(self.config.debug)
        
        # Model tab
        self.model_size_var.set(self.config.model_size)
        self.img_size_var.set(str(self.config.img_size))
        self.train_epochs_var.set(str(self.config.train_epochs))
        self.batch_size_var.set(str(self.config.batch_size))
        
        # Directories tab
        self.dir_vars['data_dir'].set(self.config.data_dir)
        self.dir_vars['models_dir'].set(self.config.models_dir)
        self.dir_vars['master_dir'].set(self.config.master_dir)
        self.dir_vars['results_export_dir'].set(self.config.results_export_dir)
        self.dir_vars['locales_dir'].set(self.config.locales_dir)
        
        # Detection tab
        self.iou_threshold_var.set(str(self.config.iou_match_threshold))
        self.master_tolerance_var.set(str(self.config.master_tolerance_px))
        self.angle_tolerance_var.set(str(self.config.angle_tolerance_deg))
        self.preview_width_var.set(str(self.config.preview_max_width))
        self.preview_height_var.set(str(self.config.preview_max_height))

    def _browse_directory(self, var):
        """Browse for directory."""
        directory = filedialog.askdirectory(initialdir=var.get())
        if directory:
            var.set(directory)

    def _save_settings(self):
        """Save settings to config."""
        try:
            # General tab
            self.config.default_locale = self.locale_var.get()
            self.config.target_fps = int(self.target_fps_var.get())
            self.config.use_gpu = self.use_gpu_var.get()
            self.config.debug = self.debug_var.get()
            
            # Model tab
            self.config.model_size = self.model_size_var.get()
            self.config.img_size = int(self.img_size_var.get())
            self.config.train_epochs = int(self.train_epochs_var.get())
            self.config.batch_size = int(self.batch_size_var.get())
            
            # Directories tab
            self.config.data_dir = self.dir_vars['data_dir'].get()
            self.config.models_dir = self.dir_vars['models_dir'].get()
            self.config.master_dir = self.dir_vars['master_dir'].get()
            self.config.results_export_dir = self.dir_vars['results_export_dir'].get()
            self.config.locales_dir = self.dir_vars['locales_dir'].get()
            
            # Detection tab
            self.config.iou_match_threshold = float(self.iou_threshold_var.get())
            self.config.master_tolerance_px = int(self.master_tolerance_var.get())
            self.config.angle_tolerance_deg = int(self.angle_tolerance_var.get())
            self.config.preview_max_width = int(self.preview_width_var.get())
            self.config.preview_max_height = int(self.preview_height_var.get())
            
            # Save to file
            save_config(self.config)
            
        except ValueError as e:
            raise ValueError(f"Invalid value in settings: {e}")

    def _reset_defaults(self):
        """Reset all settings to defaults."""
        result = messagebox.askyesno(
            "Reset Settings", 
            "This will reset all settings to their default values. Continue?"
        )
        
        if result:
            from ...config.defaults import DEFAULT_CONFIG
            from ...config.settings import Config
            
            # Create new config with defaults
            self.config = Config()
            
            # Reload the UI with defaults
            self._load_settings()

    def _on_apply(self):
        """Apply settings without closing."""
        try:
            self._save_settings()
            messagebox.showinfo("Settings", "Settings applied successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    def _on_ok(self):
        """Save settings and close."""
        try:
            self._save_settings()
            self.window.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    def _on_cancel(self):
        """Close without saving."""
        self.window.destroy()

    def winfo_exists(self):
        """Check if dialog exists."""
        try:
            return self.window.winfo_exists()
        except tk.TclError:
            return False

    def lift(self):
        """Bring dialog to front."""
        self.window.lift()