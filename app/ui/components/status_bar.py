"""Status bar component."""

import tkinter as tk
from tkinter import ttk

class StatusBar(ttk.Frame):
    """Status bar showing application state."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        """Build the status bar UI."""
        # Main status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self, textvariable=self.status_var)
        self.status_label.pack(side='left', padx=(0, 10))
        
        # FPS display
        self.fps_var = tk.StringVar(value="FPS: --")
        self.fps_label = ttk.Label(self, textvariable=self.fps_var)
        self.fps_label.pack(side='left', padx=(0, 10))
        
        # Detections count
        self.detections_var = tk.StringVar(value="Detections: 0")
        self.detections_label = ttk.Label(self, textvariable=self.detections_var)
        self.detections_label.pack(side='left', padx=(0, 10))
        
        # Separator
        separator = ttk.Separator(self, orient='vertical')
        separator.pack(side='left', fill='y', padx=5)
        
        # Model info
        self.model_var = tk.StringVar(value="Model: None")
        self.model_label = ttk.Label(self, textvariable=self.model_var)
        self.model_label.pack(side='left')

    def set_status(self, status: str):
        """Update the main status message."""
        self.status_var.set(status)

    def set_fps(self, fps: float):
        """Update the FPS display."""
        self.fps_var.set(f"FPS: {fps:.1f}")

    def set_detections(self, count: int):
        """Update the detections count."""
        self.detections_var.set(f"Detections: {count}")

    def set_model_info(self, model_name: str):
        """Update the model information."""
        self.model_var.set(f"Model: {model_name}")

    def grid(self, **kwargs):
        """Override grid to configure the frame properly."""
        super().grid(**kwargs)
        # Make sure the status bar expands to fill width
        if 'sticky' not in kwargs:
            self.configure(padding=5)