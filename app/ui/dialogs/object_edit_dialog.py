"""Object Edit Dialog for editing existing training objects."""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from typing import Optional, Dict, Any, List
import re
from datetime import datetime


class ObjectEditDialog:
    """Dialog for editing existing training objects."""
    
    def __init__(self, parent: tk.Widget, object_data: Dict[str, Any]):
        self.parent = parent
        self.object_data = object_data.copy()
        self.result = None
        
        # Create dialog window
        self.window = tk.Toplevel(parent)
        self.window.title("Edit Object")
        self.window.geometry("700x600")
        self.window.resizable(True, True)
        
        # Make it modal
        self.window.transient(parent)
        self.window.grab_set()
        
        # Center on parent
        self._center_window()
        
        # Variables
        self.object_name_var = tk.StringVar(value=object_data.get('name', ''))
        self.description_var = tk.StringVar(value=object_data.get('description', ''))
        
        # Custom metadata variables
        self.custom_metadata = object_data.get('custom', {}).copy()
        
        # Build UI
        self._build_ui()
        
        # Focus on name entry
        self.name_entry.focus_set()
        self.name_entry.select_range(0, tk.END)
        
        # Bind Enter key to OK
        self.window.bind('<Return>', lambda e: self._on_ok())
        self.window.bind('<Escape>', lambda e: self._on_cancel())
        
        # Handle window close
        self.window.protocol('WM_DELETE_WINDOW', self._on_cancel)
        
        # Display preview
        self._display_preview()
        
        # Load metadata
        self._load_custom_metadata()
    
    def _center_window(self) -> None:
        """Center the dialog on the parent window."""
        self.window.update_idletasks()
        
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        x = parent_x + (parent_width // 2) - (self.window.winfo_width() // 2)
        y = parent_y + (parent_height // 2) - (self.window.winfo_height() // 2)
        
        self.window.geometry(f"+{x}+{y}")
    
    def _build_ui(self) -> None:
        """Build the dialog UI."""
        # Main container
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Edit Training Object",
            font=('Arial', 14, 'bold')
        )
        title_label.pack(pady=(0, 15))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill='both', expand=True, pady=(0, 15))
        
        # Basic Information Tab
        basic_frame = ttk.Frame(notebook, padding="10")
        notebook.add(basic_frame, text="Basic Information")
        self._build_basic_info_tab(basic_frame)
        
        # Preview Tab
        preview_frame = ttk.Frame(notebook, padding="10")
        notebook.add(preview_frame, text="Preview")
        self._build_preview_tab(preview_frame)
        
        # Metadata Tab
        metadata_frame = ttk.Frame(notebook, padding="10")
        notebook.add(metadata_frame, text="Metadata")
        self._build_metadata_tab(metadata_frame)
        
        # Object Info Tab
        info_frame = ttk.Frame(notebook, padding="10")
        notebook.add(info_frame, text="Object Info")
        self._build_info_tab(info_frame)
        
        # Bottom buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x')
        
        ttk.Button(
            button_frame,
            text="Cancel",
            command=self._on_cancel
        ).pack(side='right', padx=(5, 0))
        
        ttk.Button(
            button_frame,
            text="Save Changes",
            command=self._on_ok,
            style='Accent.TButton'
        ).pack(side='right')
        
        ttk.Button(
            button_frame,
            text="🗑 Delete Object",
            command=self._on_delete,
            style='Danger.TButton'
        ).pack(side='left')
    
    def _build_basic_info_tab(self, parent: ttk.Frame) -> None:
        """Build the basic information tab."""
        # Object name
        name_frame = ttk.LabelFrame(parent, text="Object Name", padding="10")
        name_frame.pack(fill='x', pady=(0, 10))
        
        self.name_entry = ttk.Entry(
            name_frame,
            textvariable=self.object_name_var,
            font=('Arial', 12)
        )
        self.name_entry.pack(fill='x')
        
        # Name validation label
        self.name_validation_label = ttk.Label(
            name_frame,
            text="",
            foreground='red',
            font=('Arial', 9)
        )
        self.name_validation_label.pack(anchor='w', pady=(5, 0))
        
        # Bind validation
        self.object_name_var.trace('w', self._validate_name)
        
        # Description
        desc_frame = ttk.LabelFrame(parent, text="Description", padding="10")
        desc_frame.pack(fill='both', expand=True)
        
        # Use Text widget for multi-line description
        self.description_text = tk.Text(
            desc_frame,
            height=6,
            wrap='word',
            font=('Arial', 10)
        )
        
        # Scrollbar for description
        desc_scrollbar = ttk.Scrollbar(desc_frame, orient='vertical', command=self.description_text.yview)
        self.description_text.configure(yscrollcommand=desc_scrollbar.set)
        
        self.description_text.pack(side='left', fill='both', expand=True)
        desc_scrollbar.pack(side='right', fill='y')
        
        # Load current description
        current_desc = self.object_data.get('description', '')
        if current_desc:
            self.description_text.insert('1.0', current_desc)
    
    def _build_preview_tab(self, parent: ttk.Frame) -> None:
        """Build the preview tab."""
        # Object preview
        preview_label_frame = ttk.LabelFrame(parent, text="Object Image", padding="10")
        preview_label_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        self.object_preview_canvas = tk.Canvas(
            preview_label_frame,
            bg='black',
            relief='sunken',
            bd=1
        )
        self.object_preview_canvas.pack(fill='both', expand=True)
        
        # Source image preview (if available)
        if self.object_data.get('source_image') is not None:
            source_label_frame = ttk.LabelFrame(parent, text="Source Image", padding="10")
            source_label_frame.pack(fill='both', expand=True)
            
            self.source_preview_canvas = tk.Canvas(
                source_label_frame,
                bg='black',
                relief='sunken',
                bd=1
            )
            self.source_preview_canvas.pack(fill='both', expand=True)
    
    def _build_metadata_tab(self, parent: ttk.Frame) -> None:
        """Build the metadata tab."""
        # Custom metadata section
        metadata_label_frame = ttk.LabelFrame(parent, text="Custom Metadata", padding="10")
        metadata_label_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Metadata listbox with scrollbar
        metadata_list_frame = ttk.Frame(metadata_label_frame)
        metadata_list_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        self.metadata_listbox = tk.Listbox(
            metadata_list_frame,
            font=('Arial', 10),
            selectmode=tk.SINGLE
        )
        
        metadata_scrollbar = ttk.Scrollbar(
            metadata_list_frame,
            orient='vertical',
            command=self.metadata_listbox.yview
        )
        self.metadata_listbox.configure(yscrollcommand=metadata_scrollbar.set)
        
        self.metadata_listbox.pack(side='left', fill='both', expand=True)
        metadata_scrollbar.pack(side='right', fill='y')
        
        # Metadata buttons
        metadata_buttons_frame = ttk.Frame(metadata_label_frame)
        metadata_buttons_frame.pack(fill='x')
        
        ttk.Button(
            metadata_buttons_frame,
            text="Add Metadata",
            command=self._add_metadata
        ).pack(side='left', padx=(0, 5))
        
        ttk.Button(
            metadata_buttons_frame,
            text="Edit Selected",
            command=self._edit_metadata
        ).pack(side='left', padx=(0, 5))
        
        ttk.Button(
            metadata_buttons_frame,
            text="Remove Selected",
            command=self._remove_metadata
        ).pack(side='left')
    
    def _build_info_tab(self, parent: ttk.Frame) -> None:
        """Build the object information tab."""
        # System information (read-only)
        info_text = tk.Text(
            parent,
            wrap='word',
            font=('Consolas', 10),
            state='disabled'
        )
        
        info_scrollbar = ttk.Scrollbar(parent, orient='vertical', command=info_text.yview)
        info_text.configure(yscrollcommand=info_scrollbar.set)
        
        info_text.pack(side='left', fill='both', expand=True)
        info_scrollbar.pack(side='right', fill='y')
        
        self.info_text = info_text
        self._update_info_display()
    
    def _validate_name(self, *args) -> None:
        """Validate object name input."""
        name = self.object_name_var.get()
        
        # Clear previous validation message
        self.name_validation_label.config(text="")
        
        if not name:
            return
        
        # Check for valid characters
        if not re.match(r'^[a-zA-Z0-9\s\-_]+$', name):
            self.name_validation_label.config(text="Only letters, numbers, spaces, hyphens, and underscores allowed")
            return
        
        # Check length
        if len(name) > 50:
            self.name_validation_label.config(text="Name must be 50 characters or less")
            return
        
        # Check if starts/ends with space
        if name.startswith(' ') or name.endswith(' '):
            self.name_validation_label.config(text="Name cannot start or end with spaces")
            return
    
    def _display_preview(self) -> None:
        """Display object preview images."""
        # Display object image
        if self.object_data.get('image') is not None:
            self._display_image_on_canvas(
                self.object_preview_canvas,
                self.object_data['image']
            )
        
        # Display source image if available
        if (hasattr(self, 'source_preview_canvas') and 
            self.object_data.get('source_image') is not None):
            self._display_image_on_canvas(
                self.source_preview_canvas,
                self.object_data['source_image']
            )
    
    def _display_image_on_canvas(self, canvas: tk.Canvas, image: np.ndarray) -> None:
        """Display an image on the specified canvas."""
        try:
            if image is None or image.size == 0:
                print("Error: Invalid or empty image provided")
                return

            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume BGR format from OpenCV
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # BGRA format
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                # Grayscale or already RGB
                rgb_image = image

            # Force canvas update and get dimensions
            canvas.update()
            self.window.update_idletasks()

            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()

            # Use reasonable defaults if canvas dimensions are not available
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width, canvas_height = 300, 200
                # Schedule a retry after canvas is properly sized
                self.window.after(100, lambda: self._display_image_on_canvas(canvas, image))
                return

            # Calculate scaling to fit canvas while maintaining aspect ratio
            img_height, img_width = rgb_image.shape[:2]

            if img_width <= 0 or img_height <= 0:
                print("Error: Invalid image dimensions")
                return

            # Add some padding to prevent image from touching canvas edges
            padding = 10
            available_width = canvas_width - (2 * padding)
            available_height = canvas_height - (2 * padding)

            scale = min(available_width / img_width, available_height / img_height)
            scale = min(scale, 1.0)  # Don't upscale images

            new_width = max(1, int(img_width * scale))
            new_height = max(1, int(img_height * scale))

            # Resize image
            resized_image = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.fromarray(resized_image)
            photo = ImageTk.PhotoImage(pil_image)

            # Clear canvas and display image
            canvas.delete("all")

            # Center the image on canvas
            x_center = canvas_width // 2
            y_center = canvas_height // 2

            canvas.create_image(
                x_center,
                y_center,
                anchor="center",
                image=photo
            )

            # CRITICAL: Store reference to prevent garbage collection
            canvas.image = photo

            # Add image info text
            info_text = f"Size: {img_width}x{img_height} px"
            canvas.create_text(
                padding,
                padding,
                anchor="nw",
                text=info_text,
                fill="white",
                font=("Arial", 9)
            )

        except Exception as e:
            print(f"Error displaying image: {e}")
            # Show error message on canvas
            canvas.delete("all")
            canvas.create_text(
                canvas.winfo_width() // 2 if canvas.winfo_width() > 1 else 150,
                canvas.winfo_height() // 2 if canvas.winfo_height() > 1 else 100,
                anchor="center",
                text=f"Error loading image:\n{str(e)}",
                fill="red",
                font=("Arial", 10),
                justify="center"
            )
    
    def _load_custom_metadata(self) -> None:
        """Load custom metadata into the listbox."""
        self.metadata_listbox.delete(0, tk.END)
        
        for key, value in self.custom_metadata.items():
            display_text = f"{key}: {str(value)[:50]}{'...' if len(str(value)) > 50 else ''}"
            self.metadata_listbox.insert(tk.END, display_text)
    
    def _add_metadata(self) -> None:
        """Add new custom metadata."""
        dialog = MetadataEditDialog(self.window, "Add Metadata")
        if dialog.result:
            key, value = dialog.result
            self.custom_metadata[key] = value
            self._load_custom_metadata()
    
    def _edit_metadata(self) -> None:
        """Edit selected metadata."""
        selection = self.metadata_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a metadata item to edit", parent=self.window)
            return
        
        # Get selected key
        index = selection[0]
        keys = list(self.custom_metadata.keys())
        if index >= len(keys):
            return
        
        selected_key = keys[index]
        current_value = self.custom_metadata[selected_key]
        
        dialog = MetadataEditDialog(self.window, "Edit Metadata", selected_key, current_value)
        if dialog.result:
            new_key, new_value = dialog.result
            
            # Remove old key if changed
            if new_key != selected_key:
                del self.custom_metadata[selected_key]
            
            self.custom_metadata[new_key] = new_value
            self._load_custom_metadata()
    
    def _remove_metadata(self) -> None:
        """Remove selected metadata."""
        selection = self.metadata_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a metadata item to remove", parent=self.window)
            return
        
        index = selection[0]
        keys = list(self.custom_metadata.keys())
        if index >= len(keys):
            return
        
        selected_key = keys[index]
        
        if messagebox.askyesno("Confirm Removal", f"Remove metadata '{selected_key}'?", parent=self.window):
            del self.custom_metadata[selected_key]
            self._load_custom_metadata()
    
    def _update_info_display(self) -> None:
        """Update object information display."""
        info_lines = []
        
        # Basic information
        info_lines.append("=== Object Information ===")
        info_lines.append(f"ID: {self.object_data.get('id', 'Unknown')}")
        info_lines.append(f"Original Name: {self.object_data.get('name', 'Unknown')}")
        info_lines.append(f"Created: {self.object_data.get('created_at', 'Unknown')}")
        info_lines.append(f"Modified: {self.object_data.get('modified_at', 'Never')}")
        info_lines.append("")
        
        # Image information
        info_lines.append("=== Image Information ===")
        if 'image_shape' in self.object_data:
            shape = self.object_data['image_shape']
            if len(shape) >= 2:
                height, width = shape[:2]
                channels = shape[2] if len(shape) > 2 else 1
                info_lines.append(f"Dimensions: {width} x {height}")
                info_lines.append(f"Channels: {channels}")
                info_lines.append(f"Total Pixels: {width * height}")
        
        if 'source_image_shape' in self.object_data and self.object_data['source_image_shape']:
            shape = self.object_data['source_image_shape']
            if len(shape) >= 2:
                height, width = shape[:2]
                info_lines.append(f"Source Dimensions: {width} x {height}")
        
        info_lines.append("")
        
        # Coordinate information
        info_lines.append("=== Bounding Box ===")
        if 'coordinates' in self.object_data:
            coords = self.object_data['coordinates']
            if isinstance(coords, dict):
                x1, y1 = coords.get('x1', 0), coords.get('y1', 0)
                x2, y2 = coords.get('x2', 0), coords.get('y2', 0)
                width = x2 - x1
                height = y2 - y1
                info_lines.append(f"Top-Left: ({x1:.1f}, {y1:.1f})")
                info_lines.append(f"Bottom-Right: ({x2:.1f}, {y2:.1f})")
                info_lines.append(f"Size: {width:.1f} x {height:.1f}")
                info_lines.append(f"Area: {width * height:.1f}")
        
        info_lines.append("")
        
        # File information
        info_lines.append("=== File Paths ===")
        info_lines.append(f"Object Image: {self.object_data.get('image_path', 'Unknown')}")
        info_lines.append(f"Source Image: {self.object_data.get('source_image_path', 'None')}")
        
        # Update text widget
        self.info_text.config(state='normal')
        self.info_text.delete('1.0', tk.END)
        self.info_text.insert('1.0', '\n'.join(info_lines))
        self.info_text.config(state='disabled')
    
    def _on_ok(self) -> None:
        """Handle Save Changes button click."""
        name = self.object_name_var.get().strip()
        
        if not name:
            messagebox.showerror("Error", "Please enter an object name.", parent=self.window)
            return
        
        # Validate name
        if not re.match(r'^[a-zA-Z0-9\s\-_]+$', name):
            messagebox.showerror(
                "Error", 
                "Object name can only contain letters, numbers, spaces, hyphens, and underscores.",
                parent=self.window
            )
            return
        
        if len(name) > 50:
            messagebox.showerror("Error", "Object name must be 50 characters or less.", parent=self.window)
            return
        
        # Get description
        description = self.description_text.get('1.0', tk.END).strip()
        
        # Prepare result
        self.result = {
            'name': name,
            'description': description if description else None,
            'custom': self.custom_metadata.copy()
        }
        
        # Close dialog
        self.window.grab_release()
        self.window.destroy()
    
    def _on_cancel(self) -> None:
        """Handle Cancel button click."""
        self.result = None
        self.window.grab_release()
        self.window.destroy()
    
    def _on_delete(self) -> None:
        """Handle Delete Object button click."""
        object_name = self.object_data.get('name', 'this object')
        
        if messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete '{object_name}'?\n\nThis action cannot be undone.",
            parent=self.window
        ):
            self.result = {'delete': True}
            self.window.grab_release()
            self.window.destroy()
    
    def get_result(self) -> Optional[Dict[str, Any]]:
        """Get the dialog result after it's closed."""
        return self.result


class MetadataEditDialog:
    """Dialog for editing individual metadata key-value pairs."""
    
    def __init__(self, parent: tk.Widget, title: str, key: str = "", value: Any = ""):
        self.parent = parent
        self.result = None
        
        # Create dialog window
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry("400x200")
        self.window.resizable(True, True)
        
        # Make it modal
        self.window.transient(parent)
        self.window.grab_set()
        
        # Center on parent
        self._center_window()
        
        # Variables
        self.key_var = tk.StringVar(value=key)
        self.value_var = tk.StringVar(value=str(value))
        
        # Build UI
        self._build_ui()
        
        # Focus on key entry
        if not key:
            self.key_entry.focus_set()
        else:
            self.value_entry.focus_set()
            self.value_entry.select_range(0, tk.END)
        
        # Bind keys
        self.window.bind('<Return>', lambda e: self._on_ok())
        self.window.bind('<Escape>', lambda e: self._on_cancel())
        
        # Handle window close
        self.window.protocol('WM_DELETE_WINDOW', self._on_cancel)
    
    def _center_window(self) -> None:
        """Center the dialog on the parent window."""
        self.window.update_idletasks()
        
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        x = parent_x + (parent_width // 2) - (self.window.winfo_width() // 2)
        y = parent_y + (parent_height // 2) - (self.window.winfo_height() // 2)
        
        self.window.geometry(f"+{x}+{y}")
    
    def _build_ui(self) -> None:
        """Build the dialog UI."""
        main_frame = ttk.Frame(self.window, padding="15")
        main_frame.pack(fill='both', expand=True)
        
        # Key input
        ttk.Label(main_frame, text="Key:", font=('Arial', 10, 'bold')).pack(anchor='w')
        
        self.key_entry = ttk.Entry(
            main_frame,
            textvariable=self.key_var,
            font=('Arial', 11)
        )
        self.key_entry.pack(fill='x', pady=(5, 15))
        
        # Value input
        ttk.Label(main_frame, text="Value:", font=('Arial', 10, 'bold')).pack(anchor='w')
        
        self.value_entry = ttk.Entry(
            main_frame,
            textvariable=self.value_var,
            font=('Arial', 11)
        )
        self.value_entry.pack(fill='x', pady=(5, 20))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x')
        
        ttk.Button(
            button_frame,
            text="Cancel",
            command=self._on_cancel
        ).pack(side='right', padx=(5, 0))
        
        ttk.Button(
            button_frame,
            text="OK",
            command=self._on_ok,
            style='Accent.TButton'
        ).pack(side='right')
    
    def _on_ok(self) -> None:
        """Handle OK button click."""
        key = self.key_var.get().strip()
        value = self.value_var.get().strip()
        
        if not key:
            messagebox.showerror("Error", "Please enter a key.", parent=self.window)
            return
        
        # Try to convert value to appropriate type
        converted_value = self._convert_value(value)
        
        self.result = (key, converted_value)
        self.window.grab_release()
        self.window.destroy()
    
    def _on_cancel(self) -> None:
        """Handle Cancel button click."""
        self.result = None
        self.window.grab_release()
        self.window.destroy()
    
    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        # Try to detect and convert common types
        value = value.strip()
        
        if not value:
            return ""
        
        # Try boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def get_result(self) -> Optional[tuple]:
        """Get the dialog result after it's closed."""
        return self.result