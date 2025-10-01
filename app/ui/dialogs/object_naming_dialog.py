"""Dialog for naming and labeling objects."""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional
import numpy as np
import cv2
from PIL import Image, ImageTk


class ObjectNamingDialog:
    """Dialog for entering object labels and confirming selections."""

    def __init__(self, parent, object_image: np.ndarray, suggested_label: str = ""):
        """Initialize object naming dialog.

        Args:
            parent: Parent window
            object_image: Cropped object image to display
            suggested_label: Suggested label (optional)
        """
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Name Object")
        self.dialog.geometry("550x650")  # Increased from 400x500 to 550x650
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Center on parent
        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 550) // 2  # Updated width
        y = parent.winfo_y() + (parent.winfo_height() - 650) // 2  # Updated height
        self.dialog.geometry(f"+{x}+{y}")

        self.object_image = object_image
        self.label_value: Optional[str] = None
        self.confirmed = False

        self._build_ui(suggested_label)

    def _build_ui(self, suggested_label: str):
        """Build dialog UI.

        Args:
            suggested_label: Suggested label text
        """
        # Main frame
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill='both', expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Name the Selected Object",
            font=('Segoe UI', 12, 'bold')
        )
        title_label.pack(pady=(0, 15))

        # Image display
        image_frame = ttk.LabelFrame(main_frame, text="Object Preview", padding=10)
        image_frame.pack(fill='both', expand=True, pady=(0, 15))

        self.image_label = ttk.Label(image_frame)
        self.image_label.pack()

        # Display object image
        self._display_object()

        # Label entry
        label_frame = ttk.LabelFrame(main_frame, text="Object Label", padding=10)
        label_frame.pack(fill='x', pady=(0, 15))

        ttk.Label(
            label_frame,
            text="Enter a descriptive name for this object:",
            font=('Segoe UI', 9)
        ).pack(anchor='w', pady=(0, 5))

        self.label_entry = ttk.Entry(label_frame, font=('Segoe UI', 11))
        self.label_entry.pack(fill='x')
        self.label_entry.insert(0, suggested_label)
        self.label_entry.focus()

        # Bind Enter key
        self.label_entry.bind('<Return>', lambda e: self._on_confirm())

        # Info label
        info_label = ttk.Label(
            main_frame,
            text="This object will be added to the training dataset.",
            font=('Segoe UI', 8),
            foreground='gray'
        )
        info_label.pack(pady=(0, 15))

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x')

        ttk.Button(
            button_frame,
            text="Cancel",
            command=self._on_cancel
        ).pack(side='right', padx=(10, 0))

        ttk.Button(
            button_frame,
            text="Confirm",
            command=self._on_confirm
        ).pack(side='right')

    def _display_object(self):
        """Display cropped object image."""
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(self.object_image, cv2.COLOR_BGR2RGB)

            # Resize if too large
            h, w = image_rgb.shape[:2]
            max_size = 450  # Increased from 300 to 450 to utilize larger dialog

            if h > max_size or w > max_size:
                scale = min(max_size / w, max_size / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Convert to PIL and display
            pil_image = Image.fromarray(image_rgb)
            photo = ImageTk.PhotoImage(image=pil_image)

            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep reference

        except Exception as e:
            print(f"Error displaying object: {e}")

    def _on_confirm(self):
        """Handle confirm button click."""
        label = self.label_entry.get().strip()

        if not label:
            messagebox.showwarning(
                "Invalid Label",
                "Please enter a label for the object.",
                parent=self.dialog
            )
            return

        self.label_value = label
        self.confirmed = True
        self.dialog.destroy()

    def _on_cancel(self):
        """Handle cancel button click."""
        self.confirmed = False
        self.dialog.destroy()

    def show(self) -> tuple[bool, Optional[str]]:
        """Show dialog modally and return result.

        Returns:
            Tuple of (confirmed, label_value)
        """
        self.dialog.wait_window()
        return (self.confirmed, self.label_value)