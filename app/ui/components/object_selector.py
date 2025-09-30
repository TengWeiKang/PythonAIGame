"""Interactive object selector for bounding box selection."""

import tkinter as tk
from typing import Optional, Callable, Tuple
import numpy as np


class ObjectSelector:
    """Interactive tool for selecting objects with bounding boxes on canvas."""

    def __init__(self, canvas: tk.Canvas, callback: Optional[Callable[[Tuple[int, int, int, int]], None]] = None):
        """Initialize object selector.

        Args:
            canvas: Canvas widget to draw selection on
            callback: Function called when selection is complete (receives bbox as x1, y1, x2, y2)
        """
        self.canvas = canvas
        self.callback = callback

        self._active = False
        self._start_x: Optional[int] = None
        self._start_y: Optional[int] = None
        self._rect_id: Optional[int] = None
        self._current_image: Optional[np.ndarray] = None

        # Bind mouse events
        self.canvas.bind('<Button-1>', self._on_mouse_down)
        self.canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_up)

    def set_image(self, image: np.ndarray):
        """Set the image for object selection.

        Args:
            image: Image as numpy array
        """
        self._current_image = image

    def activate(self):
        """Activate selection mode."""
        self._active = True
        self.canvas.config(cursor='crosshair')

    def deactivate(self):
        """Deactivate selection mode."""
        self._active = False
        self.canvas.config(cursor='')
        self._clear_rectangle()

    def is_active(self) -> bool:
        """Check if selector is active.

        Returns:
            True if active, False otherwise
        """
        return self._active

    def _on_mouse_down(self, event):
        """Handle mouse button press.

        Args:
            event: Mouse event
        """
        if not self._active:
            return

        self._start_x = event.x
        self._start_y = event.y

        # Clear any existing rectangle
        self._clear_rectangle()

    def _on_mouse_drag(self, event):
        """Handle mouse drag.

        Args:
            event: Mouse event
        """
        if not self._active or self._start_x is None or self._start_y is None:
            return

        # Update rectangle
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)

        self._rect_id = self.canvas.create_rectangle(
            self._start_x, self._start_y, event.x, event.y,
            outline='green', width=2, dash=(5, 5)
        )

    def _on_mouse_up(self, event):
        """Handle mouse button release.

        Args:
            event: Mouse event
        """
        if not self._active or self._start_x is None or self._start_y is None:
            return

        end_x = event.x
        end_y = event.y

        # Calculate bounding box (normalized)
        x1 = min(self._start_x, end_x)
        y1 = min(self._start_y, end_y)
        x2 = max(self._start_x, end_x)
        y2 = max(self._start_y, end_y)

        # Ensure minimum size
        if x2 - x1 < 10 or y2 - y1 < 10:
            self._clear_rectangle()
            self._start_x = None
            self._start_y = None
            return

        # Call callback with bounding box
        if self.callback:
            try:
                self.callback((x1, y1, x2, y2))
            except Exception as e:
                print(f"Error in selection callback: {e}")

        # Reset state
        self._clear_rectangle()
        self._start_x = None
        self._start_y = None

    def _clear_rectangle(self):
        """Clear selection rectangle from canvas."""
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)
            self._rect_id = None