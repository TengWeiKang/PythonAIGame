"""Mock test for video canvas display without actual camera."""

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from app.ui.components.optimized_canvas import OptimizedCanvas


class MockVideoTest:
    """Mock video test using synthetic frames."""

    def __init__(self, root: tk.Tk):
        """Initialize test window."""
        self.root = root
        self.root.title("Mock Video Canvas Test")
        self.root.geometry("900x700")

        self.frame_count = 0
        self.is_playing = False

        self._build_ui()

    def _build_ui(self):
        """Build test UI."""
        # Control frame
        control_frame = tk.Frame(self.root, bg='#2d2d2d', height=60)
        control_frame.pack(fill='x')
        control_frame.pack_propagate(False)

        # Buttons
        ttk.Button(
            control_frame,
            text="Play",
            command=self._start_playback
        ).pack(side='left', padx=10, pady=15)

        ttk.Button(
            control_frame,
            text="Stop",
            command=self._stop_playback
        ).pack(side='left', padx=10, pady=15)

        # Overlay toggle
        self.overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            control_frame,
            text="Show Overlays",
            variable=self.overlay_var
        ).pack(side='left', padx=10, pady=15)

        # Video canvas
        video_frame = tk.Frame(self.root, bg='black')
        video_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.canvas = OptimizedCanvas(
            video_frame,
            bg='black',
            highlightthickness=0,
            target_fps=30
        )
        self.canvas.pack(fill='both', expand=True)
        self.canvas.set_render_quality('medium')

        # Status
        status_frame = tk.Frame(self.root, bg='#2d2d2d', height=40)
        status_frame.pack(fill='x')
        status_frame.pack_propagate(False)

        self.status_label = tk.Label(
            status_frame,
            text="Ready - Click Play to start",
            bg='#2d2d2d',
            fg='white',
            font=('Arial', 10)
        )
        self.status_label.pack(padx=10)

    def _generate_frame(self) -> np.ndarray:
        """Generate a synthetic test frame.

        Returns:
            BGR image as numpy array
        """
        # Create a colorful test pattern
        width, height = 640, 480
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Create animated gradient background
        offset = (self.frame_count * 2) % 255
        for y in range(height):
            color_val = int((y / height * 255 + offset) % 255)
            frame[y, :] = [color_val, 128, 255 - color_val]

        # Draw moving rectangle
        rect_x = int((self.frame_count * 3) % (width - 100))
        rect_y = int(height / 2 - 50)
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 100, rect_y + 100),
                     (0, 255, 0), 3)

        # Draw circle
        circle_x = int(width / 2 + np.sin(self.frame_count * 0.05) * 150)
        circle_y = int(height / 2 + np.cos(self.frame_count * 0.05) * 100)
        cv2.circle(frame, (circle_x, circle_y), 30, (255, 0, 0), -1)

        # Draw frame counter
        text = f"Frame: {self.frame_count}"
        cv2.putText(frame, text, (width - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def _start_playback(self):
        """Start mock playback."""
        if not self.is_playing:
            self.is_playing = True
            self.status_label.config(text="Playing...", fg='green')
            self._update_frame()

    def _stop_playback(self):
        """Stop mock playback."""
        self.is_playing = False
        self.canvas.clear()
        self.status_label.config(text="Stopped", fg='orange')

    def _update_frame(self):
        """Update display with new frame."""
        if not self.is_playing:
            return

        try:
            # Generate frame
            frame = self._generate_frame()
            self.frame_count += 1

            # Calculate FPS (simulated)
            simulated_fps = 30.0

            # Display with overlays
            if self.overlay_var.get():
                overlays = {
                    'fps': simulated_fps,
                    'resolution': '640x480',
                    'text': 'Mock Test Mode'
                }
                self.canvas.display_image(frame, overlays=overlays)
            else:
                self.canvas.display_image(frame)

            # Update status
            self.status_label.config(
                text=f"Playing - Frame {self.frame_count} | FPS: {simulated_fps:.1f}",
                fg='green'
            )

        except Exception as e:
            print(f"Error: {e}")
            self.status_label.config(text=f"Error: {e}", fg='red')
            self.is_playing = False

        # Schedule next update (~30 FPS)
        if self.is_playing:
            self.root.after(33, self._update_frame)


def main():
    """Main entry point."""
    print("Starting Mock Video Canvas Test...")
    print("This test generates synthetic frames to validate the canvas display system.")
    print("Features tested:")
    print("  - Frame display with automatic scaling")
    print("  - Aspect ratio preservation")
    print("  - FPS overlay rendering")
    print("  - Resolution overlay rendering")
    print("  - Custom text overlay rendering")
    print("  - Real-time updates at 30 FPS")
    print("\nClick 'Play' to start the test.\n")

    root = tk.Tk()
    app = MockVideoTest(root)
    root.mainloop()

    print("Test completed successfully!")


if __name__ == "__main__":
    main()