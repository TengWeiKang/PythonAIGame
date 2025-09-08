"""Object Naming Dialog for naming selected objects."""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple
import re


class ObjectNamingDialog:
    """Dialog for naming and configuring selected objects."""
    
    def __init__(self, parent: tk.Widget, coordinates: Tuple[float, float, float, float], 
                 source_image: np.ndarray, cropped_image: Optional[np.ndarray] = None):
        self.parent = parent
        self.coordinates = coordinates
        self.source_image = source_image
        self.cropped_image = cropped_image
        self.result = None
        
        # Create dialog window
        self.window = tk.Toplevel(parent)
        self.window.title("Name Object")
        self.window.geometry("600x500")
        self.window.resizable(True, True)
        
        # Make it modal
        self.window.transient(parent)
        self.window.grab_set()
        
        # Center on parent
        self._center_window()
        
        # Variables
        self.object_name_var = tk.StringVar()
        self.description_var = tk.StringVar()
        self.confidence_var = tk.DoubleVar(value=0.8)
        self.auto_generate_var = tk.BooleanVar(value=False)
        
        # Build UI
        self._build_ui()
        
        # Focus on name entry
        self.name_entry.focus_set()
        
        # Bind Enter key to OK
        self.window.bind('<Return>', lambda e: self._on_ok())
        self.window.bind('<Escape>', lambda e: self._on_cancel())
        
        # Handle window close
        self.window.protocol('WM_DELETE_WINDOW', self._on_cancel)
        
        # If cropped image not provided, crop it now
        if self.cropped_image is None and self.source_image is not None:
            self._crop_object()
        
        # Display preview
        self._display_preview()
    
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
    
    def _crop_object(self) -> None:
        """Crop object from source image using coordinates."""
        try:
            x1, y1, x2, y2 = self.coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure coordinates are within bounds
            h, w = self.source_image.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            
            self.cropped_image = self.source_image[y1:y2, x1:x2]
            
        except Exception as e:
            print(f"Error cropping object: {e}")
            self.cropped_image = None
    
    def _build_ui(self) -> None:
        """Build the dialog UI."""
        # Main container
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Name and Configure Object",
            font=('Arial', 14, 'bold')
        )
        title_label.pack(pady=(0, 15))
        
        # Create horizontal layout
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill='both', expand=True)
        
        # Left side - Preview
        preview_frame = ttk.LabelFrame(content_frame, text="Object Preview", padding="10")
        preview_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Canvas for object preview
        self.preview_canvas = tk.Canvas(
            preview_frame,
            width=200,
            height=200,
            bg='black',
            relief='sunken',
            bd=1
        )
        self.preview_canvas.pack(expand=True, fill='both')
        
        # Right side - Configuration
        config_frame = ttk.LabelFrame(content_frame, text="Object Configuration", padding="10")
        config_frame.pack(side='right', fill='both', expand=True)
        
        self._build_configuration_panel(config_frame)
        
        # Bottom buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(15, 0))
        
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
        
        # Suggest names button
        ttk.Button(
            button_frame,
            text="🔍 Suggest Names",
            command=self._suggest_names
        ).pack(side='left')
    
    def _build_configuration_panel(self, parent: ttk.Frame) -> None:
        """Build the configuration panel."""
        # Object name
        name_frame = ttk.Frame(parent)
        name_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(name_frame, text="Object Name:").pack(anchor='w')
        self.name_entry = ttk.Entry(
            name_frame,
            textvariable=self.object_name_var,
            font=('Arial', 11)
        )
        self.name_entry.pack(fill='x', pady=(2, 0))
        
        # Name validation label
        self.name_validation_label = ttk.Label(
            name_frame,
            text="",
            foreground='red',
            font=('Arial', 9)
        )
        self.name_validation_label.pack(anchor='w')
        
        # Bind validation
        self.object_name_var.trace('w', self._validate_name)
        
        # Description
        desc_frame = ttk.Frame(parent)
        desc_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(desc_frame, text="Description (Optional):").pack(anchor='w')
        self.description_entry = ttk.Entry(
            desc_frame,
            textvariable=self.description_var,
            font=('Arial', 10)
        )
        self.description_entry.pack(fill='x', pady=(2, 0))
        
        # Confidence level
        conf_frame = ttk.Frame(parent)
        conf_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(conf_frame, text="Training Confidence:").pack(anchor='w')
        
        conf_scale_frame = ttk.Frame(conf_frame)
        conf_scale_frame.pack(fill='x', pady=(2, 0))
        
        self.confidence_scale = ttk.Scale(
            conf_scale_frame,
            from_=0.1,
            to=1.0,
            orient='horizontal',
            variable=self.confidence_var,
            command=self._update_confidence_label
        )
        self.confidence_scale.pack(side='left', fill='x', expand=True)
        
        self.confidence_label = ttk.Label(
            conf_scale_frame,
            text="80%",
            width=5
        )
        self.confidence_label.pack(side='right', padx=(5, 0))
        
        # Object dimensions info
        info_frame = ttk.LabelFrame(parent, text="Object Information", padding="5")
        info_frame.pack(fill='x', pady=(10, 0))
        
        self.info_text = tk.Text(
            info_frame,
            height=4,
            width=30,
            font=('Arial', 9),
            state='disabled',
            wrap='word'
        )
        self.info_text.pack(fill='x')
        
        # Update info
        self._update_object_info()
        
        # Options
        options_frame = ttk.LabelFrame(parent, text="Options", padding="5")
        options_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Checkbutton(
            options_frame,
            text="Auto-generate variations",
            variable=self.auto_generate_var
        ).pack(anchor='w')
    
    def _validate_name(self, *args) -> None:
        """Validate object name input."""
        name = self.object_name_var.get()
        
        # Clear previous validation message
        self.name_validation_label.config(text="")
        
        if not name:
            return
        
        # Check for valid characters (alphanumeric, spaces, hyphens, underscores)
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
    
    def _update_confidence_label(self, value) -> None:
        """Update confidence percentage label."""
        percentage = int(float(value) * 100)
        self.confidence_label.config(text=f"{percentage}%")
    
    def _update_object_info(self) -> None:
        """Update object information display."""
        if self.cropped_image is not None:
            height, width = self.cropped_image.shape[:2]
            channels = self.cropped_image.shape[2] if len(self.cropped_image.shape) == 3 else 1
            
            # Calculate bounding box dimensions
            x1, y1, x2, y2 = self.coordinates
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            info_text = f"""Dimensions: {width} x {height}
Channels: {channels}
Bounding Box: {bbox_width:.0f} x {bbox_height:.0f}
Area: {width * height} pixels"""
            
        else:
            info_text = "No object preview available"
        
        self.info_text.config(state='normal')
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
        self.info_text.config(state='disabled')
    
    def _display_preview(self) -> None:
        """Display object preview on canvas."""
        if self.cropped_image is None:
            return
        
        try:
            # Convert BGR to RGB
            if len(self.cropped_image.shape) == 3:
                rgb_image = cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = self.cropped_image
            
            # Get canvas dimensions
            self.preview_canvas.update_idletasks()
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width, canvas_height = 200, 200
            
            # Calculate scaling
            img_height, img_width = rgb_image.shape[:2]
            scale = min(canvas_width / img_width, canvas_height / img_height)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            if new_width > 0 and new_height > 0:
                # Resize image
                resized_image = cv2.resize(rgb_image, (new_width, new_height))
                
                # Convert to PhotoImage
                pil_image = Image.fromarray(resized_image)
                self.photo = ImageTk.PhotoImage(pil_image)
                
                # Display on canvas
                self.preview_canvas.delete("all")
                self.preview_canvas.create_image(
                    canvas_width // 2,
                    canvas_height // 2,
                    anchor="center",
                    image=self.photo
                )
        
        except Exception as e:
            print(f"Error displaying preview: {e}")
    
    def _suggest_names(self) -> None:
        """Suggest object names based on image analysis."""
        # Simple name suggestions based on common objects
        suggestions = [
            "person", "car", "bicycle", "dog", "cat", "bird", "book", "phone", "laptop", 
            "chair", "table", "bottle", "cup", "plate", "apple", "orange", "banana",
            "traffic_light", "stop_sign", "parking_meter", "bench", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "remote", "keyboard", "mouse", "microwave",
            "toaster", "sink", "refrigerator", "plant", "clock", "vase", "scissors"
        ]
        
        # Create suggestion dialog
        suggestion_window = tk.Toplevel(self.window)
        suggestion_window.title("Suggested Names")
        suggestion_window.geometry("300x400")
        suggestion_window.transient(self.window)
        suggestion_window.grab_set()
        
        # Center on this dialog
        x = self.window.winfo_x() + 50
        y = self.window.winfo_y() + 50
        suggestion_window.geometry(f"+{x}+{y}")
        
        # Create UI
        frame = ttk.Frame(suggestion_window, padding="10")
        frame.pack(fill='both', expand=True)
        
        ttk.Label(frame, text="Select a suggested name:", font=('Arial', 11, 'bold')).pack(pady=(0, 10))
        
        # Listbox with suggestions
        listbox_frame = ttk.Frame(frame)
        listbox_frame.pack(fill='both', expand=True)
        
        suggestion_listbox = tk.Listbox(listbox_frame, font=('Arial', 10))
        scrollbar = ttk.Scrollbar(listbox_frame, orient='vertical', command=suggestion_listbox.yview)
        suggestion_listbox.configure(yscrollcommand=scrollbar.set)
        
        suggestion_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Populate suggestions
        for suggestion in suggestions:
            suggestion_listbox.insert(tk.END, suggestion)
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill='x', pady=(10, 0))
        
        def on_select():
            selection = suggestion_listbox.curselection()
            if selection:
                selected_name = suggestion_listbox.get(selection[0])
                self.object_name_var.set(selected_name)
            suggestion_window.destroy()
        
        def on_double_click(event):
            on_select()
        
        suggestion_listbox.bind('<Double-Button-1>', on_double_click)
        
        ttk.Button(button_frame, text="Cancel", command=suggestion_window.destroy).pack(side='right')
        ttk.Button(button_frame, text="Select", command=on_select).pack(side='right', padx=(0, 5))
    
    def _on_ok(self) -> None:
        """Handle OK button click."""
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
        
        # Prepare result
        self.result = {
            'name': name,
            'description': self.description_var.get().strip(),
            'coordinates': self.coordinates,
            'confidence': self.confidence_var.get(),
            'auto_generate': self.auto_generate_var.get(),
            'cropped_image': self.cropped_image,
            'source_image': self.source_image
        }
        
        # Close dialog
        self.window.grab_release()
        self.window.destroy()
    
    def _on_cancel(self) -> None:
        """Handle Cancel button click."""
        self.result = None
        self.window.grab_release()
        self.window.destroy()
    
    def get_result(self) -> Optional[Dict[str, Any]]:
        """Get the dialog result after it's closed."""
        return self.result


class QuickObjectNamingDialog:
    """Simplified quick naming dialog for rapid object labeling."""
    
    def __init__(self, parent: tk.Widget, default_name: str = ""):
        self.parent = parent
        self.result = None
        
        # Create dialog window
        self.window = tk.Toplevel(parent)
        self.window.title("Quick Name")
        self.window.geometry("300x120")
        self.window.resizable(False, False)
        
        # Make it modal
        self.window.transient(parent)
        self.window.grab_set()
        
        # Center on parent
        self._center_window()
        
        # Variables
        self.name_var = tk.StringVar(value=default_name)
        
        # Build UI
        self._build_ui()
        
        # Focus and select all text
        self.name_entry.focus_set()
        self.name_entry.select_range(0, tk.END)
        
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
        
        # Name input
        ttk.Label(main_frame, text="Object Name:", font=('Arial', 10, 'bold')).pack(anchor='w')
        
        self.name_entry = ttk.Entry(
            main_frame,
            textvariable=self.name_var,
            font=('Arial', 11),
            width=30
        )
        self.name_entry.pack(fill='x', pady=(5, 15))
        
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
        name = self.name_var.get().strip()
        
        if not name:
            messagebox.showerror("Error", "Please enter an object name.", parent=self.window)
            return
        
        self.result = name
        self.window.grab_release()
        self.window.destroy()
    
    def _on_cancel(self) -> None:
        """Handle Cancel button click."""
        self.result = None
        self.window.grab_release()
        self.window.destroy()
    
    def get_result(self) -> Optional[str]:
        """Get the dialog result after it's closed."""
        return self.result