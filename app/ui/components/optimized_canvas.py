"""Optimized canvas widget for high-performance image display."""

import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import cv2
from typing import Optional


class OptimizedCanvas(tk.Canvas):
    """High-performance canvas for displaying images with automatic scaling."""

    def __init__(self, master, **kwargs):
        """Initialize optimized canvas.

        Args:
            master: Parent widget
            **kwargs: Additional canvas configuration
        """
        # Extract custom parameters
        self.target_fps = kwargs.pop('target_fps', 30)

        super().__init__(master, **kwargs)

        self._current_image: Optional[ImageTk.PhotoImage] = None
        self._image_id: Optional[int] = None
        self._render_quality = 'medium'

        # Bind resize event
        self.bind('<Configure>', self._on_resize)

        # Store last displayed image for resize
        self._last_image_array: Optional[np.ndarray] = None

    def set_render_quality(self, quality: str):
        """Set rendering quality.

        Args:
            quality: One of 'low', 'medium', 'high'
        """
        valid_qualities = ['low', 'medium', 'high']
        if quality in valid_qualities:
            self._render_quality = quality

    def display_image(self, image: np.ndarray, overlays: Optional[dict] = None):
        """Display image on canvas with automatic scaling and optional overlays.

        Args:
            image: Image as numpy array (BGR format from OpenCV)
            overlays: Optional dictionary with overlay information:
                - 'fps': FPS value to display
                - 'resolution': Resolution string to display
                - 'text': Custom text to display
        """
        try:
            if image is None or image.size == 0:
                return

            # Store for potential resize
            self._last_image_array = image

            # Get canvas dimensions
            canvas_width = self.winfo_width()
            canvas_height = self.winfo_height()

            if canvas_width <= 1 or canvas_height <= 1:
                # Canvas not yet sized, defer display
                self.after(50, lambda: self.display_image(image, overlays))
                return

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Calculate scaling to fit canvas while maintaining aspect ratio
            h, w = image_rgb.shape[:2]
            scale = min(canvas_width / w, canvas_height / h)

            new_w = int(w * scale)
            new_h = int(h * scale)

            # Select interpolation based on quality
            if self._render_quality == 'high':
                interpolation = cv2.INTER_LANCZOS4
            elif self._render_quality == 'medium':
                interpolation = cv2.INTER_LINEAR
            else:
                interpolation = cv2.INTER_NEAREST

            # Resize image
            if new_w != w or new_h != h:
                image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=interpolation)

            # Draw overlays on the image if provided
            if overlays:
                image_rgb = self._draw_overlays(image_rgb, overlays)

            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image=pil_image)

            # Calculate position to center image
            x = (canvas_width - new_w) // 2
            y = (canvas_height - new_h) // 2

            # Display on canvas
            if self._image_id is None:
                self._image_id = self.create_image(x, y, anchor=tk.NW, image=photo)
            else:
                self.coords(self._image_id, x, y)
                self.itemconfig(self._image_id, image=photo)

            # Keep reference to prevent garbage collection
            self._current_image = photo

        except Exception as e:
            print(f"Error displaying image: {e}")

    def _draw_overlays(self, image: np.ndarray, overlays: dict) -> np.ndarray:
        """Draw overlay information on image.

        Args:
            image: RGB image as numpy array
            overlays: Dictionary with overlay data

        Returns:
            Image with overlays drawn
        """
        try:
            # Create a copy to avoid modifying original
            img_with_overlays = image.copy()
            h, w = img_with_overlays.shape[:2]

            # Define text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            padding = 10
            line_height = 20
            y_offset = padding + 15

            # Background for text (semi-transparent)
            overlay_bg = img_with_overlays.copy()

            # Draw FPS if provided
            if 'fps' in overlays:
                fps = overlays['fps']
                text = f"FPS: {fps:.1f}"
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

                # Draw background rectangle
                cv2.rectangle(overlay_bg, (padding, padding),
                             (padding + text_w + 10, padding + text_h + 10),
                             (0, 0, 0), -1)

                # Blend with original
                cv2.addWeighted(overlay_bg, 0.5, img_with_overlays, 0.5, 0, img_with_overlays)

                # Draw text
                cv2.putText(img_with_overlays, text, (padding + 5, y_offset),
                           font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                y_offset += line_height

            # Draw resolution if provided
            if 'resolution' in overlays:
                resolution = overlays['resolution']
                text = f"Res: {resolution}"
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

                # Draw background rectangle
                overlay_bg = img_with_overlays.copy()
                cv2.rectangle(overlay_bg, (padding, y_offset - 15),
                             (padding + text_w + 10, y_offset + 5),
                             (0, 0, 0), -1)
                cv2.addWeighted(overlay_bg, 0.5, img_with_overlays, 0.5, 0, img_with_overlays)

                # Draw text
                cv2.putText(img_with_overlays, text, (padding + 5, y_offset),
                           font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                y_offset += line_height

            # Draw custom text if provided
            if 'text' in overlays:
                text = overlays['text']
                lines = text.split('\n')
                for line in lines:
                    if line.strip():
                        (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, thickness)

                        # Draw background
                        overlay_bg = img_with_overlays.copy()
                        cv2.rectangle(overlay_bg, (padding, y_offset - 15),
                                     (padding + text_w + 10, y_offset + 5),
                                     (0, 0, 0), -1)
                        cv2.addWeighted(overlay_bg, 0.5, img_with_overlays, 0.5, 0, img_with_overlays)

                        # Draw text
                        cv2.putText(img_with_overlays, line, (padding + 5, y_offset),
                                   font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                        y_offset += line_height

            return img_with_overlays

        except Exception as e:
            print(f"Error drawing overlays: {e}")
            return image

    def clear(self):
        """Clear canvas."""
        if self._image_id is not None:
            self.delete(self._image_id)
            self._image_id = None

        self._current_image = None
        self._last_image_array = None

    def _on_resize(self, event):
        """Handle canvas resize event.

        Args:
            event: Resize event
        """
        # Redisplay last image if available
        if self._last_image_array is not None:
            self.display_image(self._last_image_array)


class ChatCanvas(tk.Canvas):
    """Specialized canvas for chat message display with virtual scrolling."""

    def __init__(self, master, **kwargs):
        """Initialize chat canvas.

        Args:
            master: Parent widget
            **kwargs: Additional canvas configuration
        """
        super().__init__(master, **kwargs)

        self._virtual_scrolling = False

    def enable_virtual_scrolling(self, enabled: bool):
        """Enable or disable virtual scrolling optimization.

        Args:
            enabled: True to enable, False to disable
        """
        self._virtual_scrolling = enabled