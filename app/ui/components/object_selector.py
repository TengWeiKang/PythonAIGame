"""Object Selector component for image selection and cropping."""

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Optional, Callable, Tuple, Dict, Any
import logging


class ObjectSelector:
    """Component for selecting and cropping objects from images."""
    
    def __init__(self, canvas: tk.Canvas, on_selection_complete: Optional[Callable] = None):
        self.canvas = canvas
        self.on_selection_complete = on_selection_complete
        
        # State variables
        self.current_image = None
        self.image_scale = 1.0
        self.image_offset_x = 0
        self.image_offset_y = 0
        self.canvas_image_id = None
        
        # Selection state
        self.selection_mode = False
        self.start_x = None
        self.start_y = None
        self.current_x = None
        self.current_y = None
        self.selection_rect_id = None
        
        # Mouse event bindings (initially disabled)
        self.mouse_bindings = {}
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def set_image(self, image: np.ndarray) -> None:
        """
        Set the image for selection.
        
        Args:
            image: Input image as numpy array
        """
        try:
            if image is None:
                self.clear_image()
                return
            
            self.current_image = image.copy()
            self._display_image()
            
        except Exception as e:
            self.logger.error(f"Failed to set image: {e}")
    
    def _display_image(self) -> None:
        """Display the current image on the canvas."""
        if self.current_image is None:
            return
        
        try:
            # Clear previous image
            if self.canvas_image_id:
                self.canvas.delete(self.canvas_image_id)
            
            # Get canvas dimensions
            self.canvas.update_idletasks()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                return
            
            # Convert BGR to RGB
            if len(self.current_image.shape) == 3 and self.current_image.shape[2] == 3:
                image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = self.current_image
            
            # Calculate scaling to fit canvas
            img_height, img_width = image_rgb.shape[:2]
            scale = min(canvas_width / img_width, canvas_height / img_height)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Store image transformation parameters
            self.image_scale = scale
            self.image_offset_x = (canvas_width - new_width) // 2
            self.image_offset_y = (canvas_height - new_height) // 2
            
            # Resize image
            if new_width > 0 and new_height > 0:
                resized_image = cv2.resize(image_rgb, (new_width, new_height))
                
                # Convert to PhotoImage
                pil_image = Image.fromarray(resized_image)
                self.photo = ImageTk.PhotoImage(pil_image)
                
                # Display on canvas
                self.canvas_image_id = self.canvas.create_image(
                    canvas_width // 2,
                    canvas_height // 2,
                    anchor="center",
                    image=self.photo
                )
                
        except Exception as e:
            self.logger.error(f"Failed to display image: {e}")
    
    def clear_image(self) -> None:
        """Clear the current image."""
        self.current_image = None
        self.canvas.delete("all")
        self.canvas_image_id = None
        self.stop_selection()
    
    def start_selection(self) -> None:
        """Enable selection mode for object cropping."""
        if self.current_image is None:
            return
        
        self.selection_mode = True
        
        # Clear any existing selection
        self._clear_selection_rectangle()
        
        # Bind mouse events
        self.mouse_bindings = {
            '<Button-1>': self.canvas.bind('<Button-1>', self._on_mouse_press),
            '<B1-Motion>': self.canvas.bind('<B1-Motion>', self._on_mouse_drag),
            '<ButtonRelease-1>': self.canvas.bind('<ButtonRelease-1>', self._on_mouse_release)
        }
        
        # Change cursor to crosshair
        self.canvas.configure(cursor='crosshair')
        
        self.logger.info("Selection mode started")
    
    def stop_selection(self) -> None:
        """Disable selection mode."""
        self.selection_mode = False
        
        # Unbind mouse events
        for event, binding_id in self.mouse_bindings.items():
            self.canvas.unbind(event, binding_id)
        self.mouse_bindings.clear()
        
        # Reset cursor
        self.canvas.configure(cursor='')
        
        # Clear selection rectangle
        self._clear_selection_rectangle()
        
        # Reset selection state
        self.start_x = None
        self.start_y = None
        self.current_x = None
        self.current_y = None
        
        self.logger.info("Selection mode stopped")
    
    def _on_mouse_press(self, event) -> None:
        """Handle mouse press event."""
        if not self.selection_mode:
            return
        
        # Store starting position
        self.start_x = event.x
        self.start_y = event.y
        self.current_x = event.x
        self.current_y = event.y
        
        # Clear any existing selection rectangle
        self._clear_selection_rectangle()
    
    def _on_mouse_drag(self, event) -> None:
        """Handle mouse drag event."""
        if not self.selection_mode or self.start_x is None:
            return
        
        self.current_x = event.x
        self.current_y = event.y
        
        # Update selection rectangle
        self._update_selection_rectangle()
    
    def _on_mouse_release(self, event) -> None:
        """Handle mouse release event."""
        if not self.selection_mode or self.start_x is None:
            return
        
        self.current_x = event.x
        self.current_y = event.y
        
        # Check if selection is large enough
        width = abs(self.current_x - self.start_x)
        height = abs(self.current_y - self.start_y)
        
        if width < 10 or height < 10:
            self._clear_selection_rectangle()
            return
        
        # Convert canvas coordinates to image coordinates
        image_coords = self._canvas_to_image_coordinates(
            self.start_x, self.start_y, self.current_x, self.current_y
        )
        
        if image_coords:
            # Crop the selected area
            cropped_image = self._crop_image(image_coords)
            
            if cropped_image is not None and self.on_selection_complete:
                # Call completion callback
                self.on_selection_complete(cropped_image, image_coords)
            
            # Stop selection mode
            self.stop_selection()
    
    def _update_selection_rectangle(self) -> None:
        """Update the selection rectangle display."""
        if self.start_x is None or self.current_x is None:
            return
        
        # Remove existing rectangle
        self._clear_selection_rectangle()
        
        # Calculate rectangle coordinates
        x1 = min(self.start_x, self.current_x)
        y1 = min(self.start_y, self.current_y)
        x2 = max(self.start_x, self.current_x)
        y2 = max(self.start_y, self.current_y)
        
        # Draw selection rectangle
        self.selection_rect_id = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline='red',
            width=2,
            dash=(5, 5)
        )
    
    def _clear_selection_rectangle(self) -> None:
        """Clear the selection rectangle."""
        if self.selection_rect_id:
            self.canvas.delete(self.selection_rect_id)
            self.selection_rect_id = None
    
    def _canvas_to_image_coordinates(self, x1: int, y1: int, x2: int, y2: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Convert canvas coordinates to image coordinates.
        
        Args:
            x1, y1, x2, y2: Canvas coordinates
        
        Returns:
            Tuple of image coordinates (x1, y1, x2, y2) or None if invalid
        """
        if self.current_image is None:
            return None
        
        try:
            # Convert to image space
            img_x1 = int((x1 - self.image_offset_x) / self.image_scale)
            img_y1 = int((y1 - self.image_offset_y) / self.image_scale)
            img_x2 = int((x2 - self.image_offset_x) / self.image_scale)
            img_y2 = int((y2 - self.image_offset_y) / self.image_scale)
            
            # Ensure correct order
            img_x1, img_x2 = min(img_x1, img_x2), max(img_x1, img_x2)
            img_y1, img_y2 = min(img_y1, img_y2), max(img_y1, img_y2)
            
            # Clamp to image bounds
            img_height, img_width = self.current_image.shape[:2]
            img_x1 = max(0, min(img_x1, img_width - 1))
            img_y1 = max(0, min(img_y1, img_height - 1))
            img_x2 = max(0, min(img_x2, img_width - 1))
            img_y2 = max(0, min(img_y2, img_height - 1))
            
            # Ensure valid rectangle
            if img_x2 <= img_x1 or img_y2 <= img_y1:
                return None
            
            return (img_x1, img_y1, img_x2, img_y2)
            
        except Exception as e:
            self.logger.error(f"Failed to convert coordinates: {e}")
            return None
    
    def _crop_image(self, coordinates: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Crop image using the specified coordinates.
        
        Args:
            coordinates: Image coordinates (x1, y1, x2, y2)
        
        Returns:
            Cropped image or None if failed
        """
        if self.current_image is None:
            return None
        
        try:
            x1, y1, x2, y2 = coordinates
            cropped = self.current_image[y1:y2, x1:x2]
            
            # Ensure we have a valid crop
            if cropped.size == 0:
                return None
            
            return cropped
            
        except Exception as e:
            self.logger.error(f"Failed to crop image: {e}")
            return None
    
    def get_selection_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current selection.
        
        Returns:
            Dictionary with selection information or None if no selection
        """
        if (self.start_x is None or self.current_x is None or 
            self.start_y is None or self.current_y is None):
            return None
        
        # Canvas coordinates
        canvas_coords = {
            'x1': min(self.start_x, self.current_x),
            'y1': min(self.start_y, self.current_y),
            'x2': max(self.start_x, self.current_x),
            'y2': max(self.start_y, self.current_y)
        }
        
        # Image coordinates
        image_coords = self._canvas_to_image_coordinates(
            self.start_x, self.start_y, self.current_x, self.current_y
        )
        
        if image_coords:
            return {
                'canvas_coordinates': canvas_coords,
                'image_coordinates': {
                    'x1': image_coords[0],
                    'y1': image_coords[1],
                    'x2': image_coords[2],
                    'y2': image_coords[3]
                },
                'width': image_coords[2] - image_coords[0],
                'height': image_coords[3] - image_coords[1]
            }
        
        return None
    
    def is_selecting(self) -> bool:
        """Check if currently in selection mode."""
        return self.selection_mode
    
    def has_image(self) -> bool:
        """Check if an image is currently loaded."""
        return self.current_image is not None
    
    def get_image_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current image.
        
        Returns:
            Dictionary with image information or None if no image
        """
        if self.current_image is None:
            return None
        
        height, width = self.current_image.shape[:2]
        channels = self.current_image.shape[2] if len(self.current_image.shape) == 3 else 1
        
        return {
            'width': width,
            'height': height,
            'channels': channels,
            'scale': self.image_scale,
            'offset_x': self.image_offset_x,
            'offset_y': self.image_offset_y
        }


class AdvancedObjectSelector(ObjectSelector):
    """Advanced object selector with additional features."""
    
    def __init__(self, canvas: tk.Canvas, on_selection_complete: Optional[Callable] = None):
        super().__init__(canvas, on_selection_complete)
        
        # Additional selection modes
        self.selection_modes = {
            'rectangle': self._rectangle_selection,
            'polygon': self._polygon_selection,
            'grabcut': self._grabcut_selection
        }
        self.current_mode = 'rectangle'
        
        # Polygon selection state
        self.polygon_points = []
        self.polygon_lines = []
        
        # GrabCut state
        self.grabcut_mask = None
        self.grabcut_foreground_points = []
        self.grabcut_background_points = []
    
    def set_selection_mode(self, mode: str) -> None:
        """
        Set the selection mode.
        
        Args:
            mode: Selection mode ('rectangle', 'polygon', 'grabcut')
        """
        if mode not in self.selection_modes:
            raise ValueError(f"Invalid selection mode: {mode}")
        
        # Clear any current selection
        self.stop_selection()
        self.current_mode = mode
        
        self.logger.info(f"Selection mode set to: {mode}")
    
    def _polygon_selection(self) -> None:
        """Handle polygon selection mode."""
        # Implementation for polygon selection
        pass
    
    def _grabcut_selection(self) -> None:
        """Handle GrabCut selection mode."""
        # Implementation for GrabCut selection
        pass
    
    def _rectangle_selection(self) -> None:
        """Handle rectangle selection mode (default behavior)."""
        # Use the base class rectangle selection
        pass