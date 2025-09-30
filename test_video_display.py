"""Test script for video canvas display system."""

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import time
from app.services.webcam_service import WebcamService
from app.ui.components.optimized_canvas import OptimizedCanvas


class VideoDisplayTest:
    """Test application for video display."""

    def __init__(self, root: tk.Tk):
        """Initialize test window."""
        self.root = root
        self.root.title("Video Display Test")
        self.root.geometry("1000x700")

        # Initialize services
        self.webcam_service = WebcamService(
            camera_index=0,
            width=1920,
            height=1080,
            fps=30
        )

        self._build_ui()

        # Start update loop
        self._update_video_stream()

    def _build_ui(self):
        """Build test UI."""
        # Control frame
        control_frame = tk.Frame(self.root, bg='#2d2d2d', height=60)
        control_frame.pack(fill='x')
        control_frame.pack_propagate(False)

        # Buttons
        self.start_button = ttk.Button(
            control_frame,
            text="Start Stream",
            command=self._start_stream
        )
        self.start_button.pack(side='left', padx=10, pady=15)

        self.stop_button = ttk.Button(
            control_frame,
            text="Stop Stream",
            command=self._stop_stream,
            state='disabled'
        )
        self.stop_button.pack(side='left', padx=10, pady=15)

        # Toggle overlay button
        self.overlay_enabled = tk.BooleanVar(value=False)
        self.overlay_button = ttk.Checkbutton(
            control_frame,
            text="Show Overlays",
            variable=self.overlay_enabled
        )
        self.overlay_button.pack(side='left', padx=10, pady=15)

        # Video canvas
        video_frame = tk.Frame(self.root, bg='black')
        video_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.video_canvas = OptimizedCanvas(
            video_frame,
            bg='black',
            highlightthickness=0,
            target_fps=30
        )
        self.video_canvas.pack(fill='both', expand=True)
        self.video_canvas.set_render_quality('medium')

        # Status labels
        status_frame = tk.Frame(self.root, bg='#2d2d2d', height=40)
        status_frame.pack(fill='x')
        status_frame.pack_propagate(False)

        self.fps_label = tk.Label(
            status_frame,
            text="FPS: --",
            bg='#2d2d2d',
            fg='white',
            font=('Arial', 10)
        )
        self.fps_label.pack(side='left', padx=10)

        self.resolution_label = tk.Label(
            status_frame,
            text="Resolution: --",
            bg='#2d2d2d',
            fg='white',
            font=('Arial', 10)
        )
        self.resolution_label.pack(side='left', padx=10)

        self.status_label = tk.Label(
            status_frame,
            text="Ready",
            bg='#2d2d2d',
            fg='white',
            font=('Arial', 10)
        )
        self.status_label.pack(side='right', padx=10)

    def _start_stream(self):
        """Start webcam stream."""
        if self.webcam_service.start_stream():
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.status_label.config(text="Streaming", fg='green')
            print("Stream started successfully")
        else:
            self.status_label.config(text="Failed to start", fg='red')
            print("Failed to start stream")

    def _stop_stream(self):
        """Stop webcam stream."""
        self.webcam_service.stop_stream()
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.video_canvas.clear()
        self.fps_label.config(text="FPS: --")
        self.resolution_label.config(text="Resolution: --")
        self.status_label.config(text="Stopped", fg='orange')
        print("Stream stopped")

    def _update_video_stream(self):
        """Update video display."""
        try:
            if self.webcam_service.is_streaming():
                frame = self.webcam_service.get_current_frame()

                if frame is not None:
                    # Get FPS and resolution
                    fps = self.webcam_service.get_fps()
                    width, height = self.webcam_service.get_resolution()

                    # Update labels
                    self.fps_label.config(text=f"FPS: {fps:.1f}")
                    self.resolution_label.config(text=f"Resolution: {width}x{height}")

                    # Display frame with optional overlays
                    if self.overlay_enabled.get():
                        overlays = {
                            'fps': fps,
                            'resolution': f"{width}x{height}"
                        }
                        self.video_canvas.display_image(frame, overlays=overlays)
                    else:
                        self.video_canvas.display_image(frame)

        except Exception as e:
            print(f"Error updating video: {e}")
            self.status_label.config(text=f"Error: {e}", fg='red')

        # Schedule next update (~30 FPS)
        self.root.after(33, self._update_video_stream)

    def cleanup(self):
        """Cleanup resources."""
        if self.webcam_service.is_streaming():
            self.webcam_service.stop_stream()


def main():
    """Main entry point."""
    root = tk.Tk()
    app = VideoDisplayTest(root)

    # Handle window close
    root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))

    root.mainloop()


if __name__ == "__main__":
    main()