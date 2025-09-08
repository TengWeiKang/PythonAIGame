"""Main application window."""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
import json
from typing import Optional

from ..config.settings import Config
from ..services.webcam_service import WebcamService
from ..services.inference_service import InferenceService
from ..services.detection_service import DetectionService
from ..services.annotation_service import AnnotationService
from ..services.training_service import TrainingService
from ..core.entities import PipelineState
from .components.status_bar import StatusBar
# Dialogs will be imported when needed to avoid circular imports

class MainWindow:
    """Main application window."""
    
    def __init__(self, root: tk.Tk, config: Config):
        self.root = root
        self.config = config
        
        # Load localization first
        self.locale = self._load_locale()
        
        # Setup window (now that locale is available)
        self._setup_window()
        
        # Initialize services
        self.webcam_service = WebcamService()
        self.inference_service = InferenceService(config)
        self.annotation_service = AnnotationService(config)
        self.training_service = TrainingService(config)
        
        # Pipeline will be created when needed
        self.detection_service: Optional[DetectionService] = None
        
        # State variables
        self._frame = None  # last processed frame
        self.master = None  # master image data
        self.object_entries = []  # collected ROIs
        
        # Dialog references
        self._webcam_dialog = None
        self._settings_dialog = None
        
        # Build UI
        self._build_ui()

    def _setup_window(self):
        """Setup main window properties."""
        self.root.title(self.t('app_title', 'Webcam Master Checker'))
        self.root.geometry("1200x800")
        
        # Configure grid weights
        self.root.grid_rowconfigure(4, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

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
            print(f"[MainWindow] Locale fallback empty ({e})")
            return {}

    def t(self, key: str, fallback: str) -> str:
        """Get localized string."""
        return self.locale.get(key, fallback)

    def _build_ui(self):
        """Build the main user interface."""
        self._build_menubar()
        self._build_video_panel()
        self._build_control_buttons()
        self._build_status_area()

    def _build_menubar(self):
        """Build the menu bar."""
        menubar = tk.Menu(self.root)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(
            label='General Settings', 
            command=self.open_general_settings
        )
        settings_menu.add_command(
            label='Webcam Settings', 
            command=self.open_webcam_settings
        )
        settings_menu.add_separator()
        settings_menu.add_command(
            label='Object Classification Settings', 
            command=self.open_object_classification_settings
        )
        
        menubar.add_cascade(label='Settings', menu=settings_menu)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label='Annotate Image', command=self.open_annotator)
        tools_menu.add_command(label='Train Model', command=self.train_model)
        tools_menu.add_command(label='Test Model', command=self.test_model)
        
        menubar.add_cascade(label='Tools', menu=tools_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label='About', command=self.show_about)
        
        menubar.add_cascade(label='Help', menu=help_menu)
        
        self.root.config(menu=menubar)

    def _build_video_panel(self):
        """Build the main video display panel."""
        # Main container for video
        video_frame = ttk.Frame(self.root)
        video_frame.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)
        
        # Fixed-size main detection preview
        self._main_prev_w = self.config.preview_max_width
        self._main_prev_h = self.config.preview_max_height
        
        # Create blank image
        blank_array = np.zeros((self._main_prev_h, self._main_prev_w, 3), dtype=np.uint8)
        blank_main = ImageTk.PhotoImage(Image.fromarray(blank_array))
        
        self.video_panel = tk.Label(
            video_frame, 
            image=blank_main,
            width=self._main_prev_w,
            height=self._main_prev_h,
            relief='sunken',
            bg='black'
        )
        self.video_panel.image = blank_main
        self.video_panel.pack(padx=4, pady=4)

    def _build_control_buttons(self):
        """Build the control button panels."""
        # Primary controls
        controls_frame1 = ttk.Frame(self.root)
        controls_frame1.grid(row=1, column=0, sticky='ew', padx=4, pady=2)
        
        ttk.Button(
            controls_frame1, 
            text=self.t('btn_start', 'Start Stream'), 
            command=self.start_stream
        ).pack(side='left', padx=2)
        
        ttk.Button(
            controls_frame1, 
            text=self.t('btn_stop', 'Stop'), 
            command=self.stop_stream
        ).pack(side='left', padx=2)
        
        ttk.Button(
            controls_frame1, 
            text=self.t('btn_capture', 'Capture Image'), 
            command=self.capture_image
        ).pack(side='left', padx=2)
        
        # Secondary controls
        controls_frame2 = ttk.Frame(self.root)
        controls_frame2.grid(row=2, column=0, sticky='ew', padx=4, pady=2)
        
        ttk.Button(
            controls_frame2, 
            text=self.t('btn_load_master', 'Load Master Image'), 
            command=self.load_master
        ).pack(side='left', padx=2)
        
        ttk.Button(
            controls_frame2, 
            text=self.t('btn_export', 'Export Results'), 
            command=self.export_results
        ).pack(side='left', padx=2)

    def _build_status_area(self):
        """Build the status and feedback area."""
        # Status bar
        self.status_bar = StatusBar(self.root)
        self.status_bar.grid(row=3, column=0, sticky='ew', padx=4, pady=2)
        
        # Feedback text area
        self.feedback = tk.Text(
            self.root, 
            width=80, 
            height=8, 
            state='disabled', 
            wrap='word'
        )
        self.feedback.grid(row=4, column=0, padx=4, pady=4, sticky='nsew')

    # Event handlers
    def start_stream(self):
        """Start the webcam stream."""
        try:
            if not self.webcam_service.is_opened():
                success = self.webcam_service.open(
                    self.config.last_webcam_index,
                    self.config.camera_width,
                    self.config.camera_height,
                    self.config.camera_fps
                )
                if not success:
                    messagebox.showerror("Error", "Failed to open webcam")
                    return
            
            # Create detection service if not exists
            if not self.detection_service:
                self.detection_service = DetectionService(
                    frame_source=self.webcam_service,
                    inference_service=self.inference_service,
                    matcher=None,  # TODO: Implement matcher
                    feedback_builder=self._generate_feedback,
                    config=self.config
                )
                self.detection_service.add_listener(self._on_pipeline_update)
            
            # Start the detection pipeline
            self.detection_service.start(self.root)
            self.status_bar.set_status("Stream started")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start stream: {e}")

    def stop_stream(self):
        """Stop the webcam stream."""
        try:
            if self.detection_service:
                self.detection_service.stop()
            
            self.webcam_service.close()
            self.status_bar.set_status("Stream stopped")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop stream: {e}")

    def capture_image(self):
        """Capture current frame as image."""
        if self._frame is not None:
            # TODO: Implement image capture
            self.status_bar.set_status("Image captured")
        else:
            messagebox.showwarning("Warning", "No frame available to capture")

    def load_master(self):
        """Load master image for comparison."""
        # TODO: Implement master image loading
        self.status_bar.set_status("Master image loading not implemented")

    def export_results(self):
        """Export detection results."""
        # TODO: Implement results export
        self.status_bar.set_status("Results export not implemented")

    def open_webcam_settings(self):
        """Open webcam settings dialog."""
        if self._webcam_dialog is None or not self._webcam_dialog.winfo_exists():
            from .dialogs.webcam_dialog import WebcamDialog
            self._webcam_dialog = WebcamDialog(self.root, self.config, self.webcam_service)
        else:
            self._webcam_dialog.lift()

    def open_general_settings(self):
        """Open general settings dialog."""
        if self._settings_dialog is None or not self._settings_dialog.winfo_exists():
            from .dialogs.settings_dialog import SettingsDialog
            self._settings_dialog = SettingsDialog(self.root, self.config)
        else:
            self._settings_dialog.lift()

    def open_object_classification_settings(self):
        """Open object classification settings dialog."""
        from .dialogs.object_classification_dialog import ObjectClassificationDialog
        dialog = ObjectClassificationDialog(self.root, self.config, self.webcam_service)

    def open_annotator(self):
        """Open annotation tool."""
        # TODO: Implement annotation tool
        messagebox.showinfo("Info", "Annotation tool not implemented")

    def train_model(self):
        """Start model training."""
        # TODO: Implement model training dialog
        messagebox.showinfo("Info", "Model training not implemented")

    def test_model(self):
        """Test current model."""
        # TODO: Implement model testing
        messagebox.showinfo("Info", "Model testing not implemented")

    def show_about(self):
        """Show about dialog."""
        about_text = f"""Python Game Detection System
Version 2.0.0

A computer vision application for object detection
and master image comparison.

Configuration loaded from: config.json
Data directory: {self.config.data_dir}
Models directory: {self.config.models_dir}
"""
        messagebox.showinfo("About", about_text)

    def _on_pipeline_update(self, state: PipelineState):
        """Handle pipeline state updates."""
        # Update video panel
        if state.frame is not None:
            self._frame = state.frame
            self._update_video_panel(state.frame, state.detections)
        
        # Update status
        self.status_bar.set_fps(state.fps)
        self.status_bar.set_detections(len(state.detections))
        
        # Update feedback
        self._update_feedback(state.feedback)

    def _update_video_panel(self, frame, detections):
        """Update the main video panel with annotated frame."""
        try:
            # Annotate frame with detections
            annotated_frame = self.annotation_service.annotate_image(
                frame, detections, draw_labels=True, draw_confidence=True
            )
            
            # Resize to fit panel
            h, w = annotated_frame.shape[:2]
            scale = min(self._main_prev_w / w, self._main_prev_h / h)
            
            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                import cv2
                annotated_frame = cv2.resize(annotated_frame, (new_w, new_h))
            
            # Convert BGR to RGB for PIL
            rgb_frame = annotated_frame[:, :, ::-1]
            
            # Create PIL image and PhotoImage
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update panel
            self.video_panel.configure(image=photo)
            self.video_panel.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Error updating video panel: {e}")

    def _update_feedback(self, feedback_lines):
        """Update the feedback text area."""
        try:
            self.feedback.configure(state='normal')
            self.feedback.delete('1.0', tk.END)
            
            if feedback_lines:
                feedback_text = '\n'.join(feedback_lines)
                self.feedback.insert('1.0', feedback_text)
            
            self.feedback.configure(state='disabled')
            
        except Exception as e:
            print(f"Error updating feedback: {e}")

    def _generate_feedback(self, matches) -> list:
        """Generate feedback lines from match results."""
        if not matches:
            return ["No matches to display"]
        
        feedback = []
        for match in matches:
            feedback.append(f"Match: {match.verdict} (IOU: {match.iou:.2f})")
        
        return feedback

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'detection_service') and self.detection_service:
            self.detection_service.stop()
        if hasattr(self, 'webcam_service') and self.webcam_service:
            self.webcam_service.close()