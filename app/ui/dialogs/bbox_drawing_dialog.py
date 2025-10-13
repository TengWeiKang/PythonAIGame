"""Interactive bounding box drawing dialog for full-frame object annotation with multi-object support."""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, List, Dict
import numpy as np
import cv2
from PIL import Image, ImageTk
import logging

logger = logging.getLogger(__name__)


class ImageSourceDialog(tk.Toplevel):
    """Dialog to select image source for object annotation."""

    def __init__(self, parent, has_video_stream=True, has_reference=True):
        """Initialize image source selection dialog.

        Args:
            parent: Parent window
            has_video_stream: Whether video stream is available
            has_reference: Whether reference image is loaded
        """
        super().__init__(parent)
        self.title("Select Image Source")
        self.result = None

        # Make dialog modal
        self.transient(parent)
        self.grab_set()

        # Build UI
        self._build_ui(has_video_stream, has_reference)

        # Center dialog
        self.geometry("450x280")
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 450) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 280) // 2
        self.geometry(f"+{x}+{y}")

        # Bind escape key
        self.bind('<Escape>', lambda e: self.destroy())

    def _build_ui(self, has_video_stream, has_reference):
        """Build the UI."""
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill='both', expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Select Image Source for Annotation",
            font=('Segoe UI', 12, 'bold')
        )
        title_label.pack(pady=(0, 10))

        # Instructions
        ttk.Label(
            main_frame,
            text="Choose which image to use for drawing bounding boxes:",
            font=('Segoe UI', 9),
            foreground='gray'
        ).pack(pady=(0, 20))

        # Button container
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=10)

        # Video stream button
        if has_video_stream:
            video_btn = ttk.Button(
                button_frame,
                text="📹 Current Video Frame",
                command=lambda: self._select('video'),
                width=35
            )
            video_btn.pack(pady=5)

            ttk.Label(
                button_frame,
                text="Capture and annotate current webcam frame",
                font=('Segoe UI', 8),
                foreground='gray'
            ).pack(pady=(0, 10))

        # Reference image button
        if has_reference:
            ref_btn = ttk.Button(
                button_frame,
                text="🖼️ Reference Image",
                command=lambda: self._select('reference'),
                width=35
            )
            ref_btn.pack(pady=5)

            ttk.Label(
                button_frame,
                text="Use loaded reference image for annotation",
                font=('Segoe UI', 8),
                foreground='gray'
            ).pack(pady=(0, 10))

        # Separator
        ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=10)

        # Cancel button
        ttk.Button(
            main_frame,
            text="Cancel",
            command=self.destroy,
            width=35
        ).pack(pady=5)

    def _select(self, source):
        """Handle source selection."""
        self.result = source
        self.destroy()

    def show(self):
        """Show dialog and return selected source."""
        self.wait_window()
        return self.result


class BboxDrawingDialog:
    """Interactive dialog for drawing multiple bounding boxes on full frames."""

    def __init__(self, parent, image: np.ndarray, existing_classes: Optional[List[str]] = None):
        """Initialize bbox drawing dialog with multi-object support.

        Args:
            parent: Parent window
            image: Full frame image (NOT cropped!)
            existing_classes: List of existing class names for quick selection
        """
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Draw Bounding Boxes and Select Classes (Multi-Object)")
        self.dialog.geometry("1600x900")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Center on parent
        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 1600) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 900) // 2
        self.dialog.geometry(f"+{x}+{y}")

        self.original_image = image.copy()
        self.display_image = image.copy()
        self.existing_classes = existing_classes or []
        self.quick_classes = self.existing_classes[:9]  # First 9 for quick buttons

        # Multi-object tracking
        self.objects: List[Dict] = []  # List of {'bbox': (x1,y1,x2,y2), 'class': str, 'background_region': Optional[tuple], 'segmentation': List[float], 'threshold': int, 'threshold_channel': str}
        self.current_bbox: Optional[tuple] = None
        self.drawing = False
        self.start_point: Optional[tuple] = None
        self.current_rect_id: Optional[int] = None
        self.saved_rect_ids: List[int] = []  # Canvas IDs for saved bboxes

        # Threshold detection settings
        self.threshold_value = 127  # Current threshold value (0-255)
        self.threshold_channel = tk.StringVar(value='Gray')  # Channel selection: 'Gray', 'R', 'G', 'B'
        self.threshold_invert = tk.BooleanVar(value=False)  # Invert binary threshold
        self.current_binary_preview = None  # Current binary preview image

        # Result
        self.result: Optional[List[Dict]] = None

        # Canvas scaling and offset tracking
        self.scale_factor = 1.0  # Scale factor: display_size / original_size
        self.display_width = 0  # Scaled image width on canvas
        self.display_height = 0  # Scaled image height on canvas
        self.canvas_offset_x = 0  # X offset if image is centered
        self.canvas_offset_y = 0  # Y offset if image is centered

        self._build_ui()
        self._bind_keyboard_shortcuts()

        # Bind window close
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _build_ui(self):
        """Build the dialog UI with horizontal layout."""
        # Main container with horizontal split
        main_container = ttk.Frame(self.dialog)
        main_container.pack(fill='both', expand=True)

        # LEFT PANEL: Image canvas (70% width)
        left_frame = ttk.Frame(main_container)
        left_frame.pack(side='left', fill='both', expand=True, padx=(10, 5), pady=10)

        # Canvas title
        ttk.Label(
            left_frame,
            text="📸 Image (Click and drag to draw bounding boxes)",
            font=('Segoe UI', 10, 'bold')
        ).pack(anchor='w', pady=(0, 5))

        # Canvas for image (no scrollbars - image will be scaled to fit)
        canvas_container = ttk.Frame(left_frame, relief='sunken', borderwidth=2)
        canvas_container.pack(fill='both', expand=True)

        self.canvas = tk.Canvas(
            canvas_container,
            bg='black',
            cursor='crosshair',
            highlightthickness=0
        )
        self.canvas.pack(fill='both', expand=True)

        # Display image after canvas is packed
        # Use after_idle to ensure canvas has been rendered and has accurate dimensions
        self.dialog.after(10, self._display_image)

        # Bind mouse events
        self.canvas.bind('<Button-1>', self._on_mouse_down)
        self.canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_up)

        # Canvas controls
        canvas_controls = ttk.Frame(left_frame)
        canvas_controls.pack(fill='x', pady=(5, 0))

        # Top row: Clear button
        top_row = ttk.Frame(canvas_controls)
        top_row.pack(fill='x', pady=(0, 5))

        ttk.Button(
            top_row,
            text="🗑️ Clear Current Box",
            command=self._clear_current_bbox,
            width=20
        ).pack(side='left', padx=(0, 5))

        # Canvas status
        status_row = ttk.Frame(canvas_controls)
        status_row.pack(fill='x')

        self.canvas_status = ttk.Label(
            status_row,
            text="Draw a box around an object",
            font=('Segoe UI', 8),
            foreground='gray'
        )
        self.canvas_status.pack(side='left', padx=10)

        # RIGHT PANEL: Controls (fixed 750px width for two-column layout)
        right_frame = ttk.Frame(main_container, width=750)
        right_frame.pack(side='right', fill='y', padx=(5, 10), pady=10)
        right_frame.pack_propagate(False)  # Fixed width

        # Create two-column container inside right panel
        columns_container = ttk.Frame(right_frame)
        columns_container.pack(fill='both', expand=True, padx=5, pady=5)

        # LEFT COLUMN (instructions, objects, threshold, preview)
        left_column = ttk.Frame(columns_container)
        left_column.pack(side='left', fill='both', expand=True, padx=(0, 5))

        # RIGHT COLUMN (class selection, action buttons)
        right_column = ttk.Frame(columns_container)
        right_column.pack(side='left', fill='both', expand=True, padx=(5, 0))

        # Build left column sections
        self._build_instructions_section(left_column)
        self._build_objects_list_section(left_column)
        self._build_threshold_section(left_column)

        # Build right column sections
        self._build_class_selection_section(right_column)
        self._build_action_buttons_section(right_column)

    def _build_instructions_section(self, parent):
        """Build instructions section."""
        instructions_frame = ttk.LabelFrame(parent, text="Instructions", padding=8)
        instructions_frame.pack(fill='x', pady=(0, 10))

        instructions = """1. Draw bbox around object
2. Adjust threshold (object = WHITE)
3. Use Invert if needed
4. Select class (right panel)
5. Click "Add Object" """

        ttk.Label(
            instructions_frame,
            text=instructions,
            justify='left',
            font=('Segoe UI', 8),
            foreground='blue'
        ).pack(anchor='w')

    def _build_objects_list_section(self, parent):
        """Build objects list section."""
        list_frame = ttk.LabelFrame(parent, text="Added Objects", padding=10)
        list_frame.pack(fill='x', pady=(0, 10))

        # Listbox with scrollbar
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill='x')

        self.objects_listbox = tk.Listbox(
            list_container,
            font=('Consolas', 9),
            selectmode='single',
            height=5
        )
        self.objects_listbox.pack(side='left', fill='x', expand=True)

        scrollbar = ttk.Scrollbar(list_container, command=self.objects_listbox.yview)
        scrollbar.pack(side='right', fill='y')
        self.objects_listbox.config(yscrollcommand=scrollbar.set)

        # Delete button
        ttk.Button(
            list_frame,
            text="🗑️ Delete Selected",
            command=self._delete_selected_object,
            width=25
        ).pack(pady=(5, 0))

    def _build_threshold_section(self, parent):
        """Build threshold slider and binary preview section."""
        threshold_frame = ttk.LabelFrame(parent, text="Threshold Detection", padding=10)
        threshold_frame.pack(fill='x', pady=(0, 10))

        # Threshold slider
        slider_frame = ttk.Frame(threshold_frame)
        slider_frame.pack(fill='x', pady=(0, 10))

        ttk.Label(
            slider_frame,
            text="Threshold Value:",
            font=('Segoe UI', 9, 'bold')
        ).pack(anchor='w', pady=(0, 5))

        # Value display and slider in same row
        value_row = ttk.Frame(slider_frame)
        value_row.pack(fill='x')

        self.threshold_value_label = ttk.Label(
            value_row,
            text=f"{self.threshold_value}",
            font=('Segoe UI', 10, 'bold'),
            foreground='#0066FF',
            width=4
        )
        self.threshold_value_label.pack(side='left', padx=(0, 10))

        self.threshold_slider = tk.Scale(
            value_row,
            from_=0,
            to=255,
            orient='horizontal',
            command=self._on_threshold_changed,
            showvalue=False,
            length=250
        )
        self.threshold_slider.set(self.threshold_value)
        self.threshold_slider.pack(side='left', fill='x', expand=True)

        # Range labels
        range_frame = ttk.Frame(slider_frame)
        range_frame.pack(fill='x')

        ttk.Label(
            range_frame,
            text="0",
            font=('Segoe UI', 7),
            foreground='gray'
        ).pack(side='left')

        ttk.Label(
            range_frame,
            text="255",
            font=('Segoe UI', 7),
            foreground='gray'
        ).pack(side='right')

        # Channel selection
        ttk.Label(
            slider_frame,
            text="Channel:",
            font=('Segoe UI', 9, 'bold')
        ).pack(anchor='w', pady=(10, 5))

        # Radio button frame for channel selection
        channel_buttons_frame = ttk.Frame(slider_frame)
        channel_buttons_frame.pack(fill='x', pady=(0, 5))

        # Radio buttons for each channel
        channels = [
            ('Gray', 'Gray'),
            ('Red', 'R'),
            ('Green', 'G'),
            ('Blue', 'B')
        ]

        for label, value in channels:
            rb = tk.Radiobutton(
                channel_buttons_frame,
                text=label,
                variable=self.threshold_channel,
                value=value,
                font=('Segoe UI', 9),
                command=self._on_channel_changed
            )
            rb.pack(side='left', padx=5)

        # Invert checkbox
        invert_checkbox = tk.Checkbutton(
            slider_frame,
            text="Invert",
            variable=self.threshold_invert,
            font=('Segoe UI', 9),
            command=self._on_threshold_changed  # Update preview when toggled
        )
        invert_checkbox.pack(anchor='w', pady=(5, 5))

        # Binary preview canvas
        ttk.Label(
            threshold_frame,
            text="Binary Preview:",
            font=('Segoe UI', 9, 'bold')
        ).pack(anchor='w', pady=(5, 5))

        preview_container = ttk.Frame(threshold_frame, relief='sunken', borderwidth=2)
        preview_container.pack(fill='both', expand=True)

        self.preview_canvas = tk.Canvas(
            preview_container,
            width=250,
            height=250,
            bg='black',
            highlightthickness=0
        )
        self.preview_canvas.pack()

        # Preview status label
        self.preview_status_label = ttk.Label(
            threshold_frame,
            text="Draw a bounding box to see preview",
            font=('Segoe UI', 8),
            foreground='gray'
        )
        self.preview_status_label.pack(pady=(5, 0))

    def _build_class_selection_section(self, parent):
        """Build class selection section."""
        class_frame = ttk.LabelFrame(parent, text="Select Class", padding=10)
        class_frame.pack(fill='x', pady=(0, 10))

        # Quick class buttons in horizontal grid layout
        if self.quick_classes:
            ttk.Label(
                class_frame,
                text="Quick Select (Press 1-9):",
                font=('Segoe UI', 9, 'bold')
            ).pack(anchor='w', pady=(0, 5))

            button_container = ttk.Frame(class_frame)
            button_container.pack(fill='x', pady=(0, 10))

            # Arrange buttons in grid: 3 columns for better space usage
            # This reduces vertical height by ~70%
            num_cols = 3
            for i, cls in enumerate(self.quick_classes, 1):
                row = (i - 1) // num_cols
                col = (i - 1) % num_cols

                btn = ttk.Button(
                    button_container,
                    text=f"{i}. {cls}",
                    command=lambda c=cls: self._select_class(c),
                    width=11  # Reduced width for 3-column layout
                )
                btn.grid(row=row, column=col, padx=2, pady=2, sticky='ew')

            # Configure columns to expand equally
            for col in range(num_cols):
                button_container.columnconfigure(col, weight=1)

        # Dropdown for all classes
        if self.existing_classes:
            ttk.Label(
                class_frame,
                text="All Classes:",
                font=('Segoe UI', 9, 'bold')
            ).pack(anchor='w', pady=(0, 5))

            self.class_var = tk.StringVar()
            self.class_combo = ttk.Combobox(
                class_frame,
                textvariable=self.class_var,
                values=sorted(self.existing_classes),
                font=('Segoe UI', 9),
                state='readonly'
            )
            self.class_combo.pack(fill='x', pady=(0, 10))
            self.class_combo.bind('<<ComboboxSelected>>', lambda e: self._select_class(self.class_var.get()))

        # New class entry
        ttk.Label(
            class_frame,
            text="Or Enter New Class:",
            font=('Segoe UI', 9, 'bold')
        ).pack(anchor='w', pady=(0, 5))

        entry_frame = ttk.Frame(class_frame)
        entry_frame.pack(fill='x')

        self.new_class_entry = ttk.Entry(entry_frame, font=('Segoe UI', 9))
        self.new_class_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))

        ttk.Button(
            entry_frame,
            text="✓",
            command=lambda: self._select_class(self.new_class_entry.get().strip()),
            width=3
        ).pack(side='right')

        # Selected class display
        self.selected_class_label = ttk.Label(
            class_frame,
            text="No class selected",
            font=('Segoe UI', 9),
            foreground='gray'
        )
        self.selected_class_label.pack(pady=(10, 0))

        # Store current selected class
        self.current_class = None

    def _build_action_buttons_section(self, parent):
        """Build action buttons section."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill='x', side='bottom', pady=(10, 0))

        # Add object button
        self.add_button = ttk.Button(
            button_frame,
            text="➕ Add Object",
            command=self._add_current_object,
            state='disabled',
            width=25
        )
        self.add_button.pack(fill='x', pady=2)

        ttk.Separator(button_frame, orient='horizontal').pack(fill='x', pady=10)

        # Done button
        ttk.Button(
            button_frame,
            text="✓ Done (Save All)",
            command=self._confirm_all,
            width=25
        ).pack(fill='x', pady=2)

        # Cancel button
        ttk.Button(
            button_frame,
            text="✗ Cancel",
            command=self._on_cancel,
            width=25
        ).pack(fill='x', pady=2)

        # Status label
        self.status_label = ttk.Label(
            button_frame,
            text="",
            font=('Segoe UI', 8),
            foreground='gray'
        )
        self.status_label.pack(pady=(10, 0))

    def _display_image(self):
        """Display the full frame image on canvas, scaled to fit without scrollbars.

        The image is scaled down (never up) to fit within the canvas dimensions while
        maintaining aspect ratio. All coordinates are properly converted between
        display space and original image space.
        """
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = image_rgb.shape[:2]

            # Get accurate canvas dimensions
            self.dialog.update_idletasks()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # Fallback dimensions if canvas not yet rendered properly
            # Account for the 70/30 split: ~840px for canvas in 1200px dialog
            if canvas_width < 10:
                canvas_width = 840
            if canvas_height < 10:
                canvas_height = 700

            # Add small padding to prevent scrollbars from appearing
            padding = 4
            available_width = canvas_width - padding
            available_height = canvas_height - padding

            # Calculate scale factor to fit image within canvas
            # Only scale DOWN, never scale UP (scale_factor <= 1.0)
            scale_w = available_width / orig_w
            scale_h = available_height / orig_h
            self.scale_factor = min(scale_w, scale_h, 1.0)

            # Calculate display dimensions
            self.display_width = int(orig_w * self.scale_factor)
            self.display_height = int(orig_h * self.scale_factor)

            # Resize image for display
            display_img = cv2.resize(
                image_rgb,
                (self.display_width, self.display_height),
                interpolation=cv2.INTER_AREA  # Best for downscaling
            )

            # Convert to PIL and create PhotoImage
            pil_image = Image.fromarray(display_img)
            self.photo = ImageTk.PhotoImage(image=pil_image)

            # Clear canvas and display image centered
            self.canvas.delete('all')

            # Center the image on canvas if it's smaller than canvas
            x_offset = max(0, (canvas_width - self.display_width) // 2)
            y_offset = max(0, (canvas_height - self.display_height) // 2)

            self.canvas.create_image(
                x_offset, y_offset,
                anchor='nw',
                image=self.photo,
                tags='image'
            )

            # Store offsets for coordinate conversion
            self.canvas_offset_x = x_offset
            self.canvas_offset_y = y_offset

            logger.info(
                f"Image displayed: {orig_w}x{orig_h} → {self.display_width}x{self.display_height} "
                f"(scale={self.scale_factor:.3f}, offset=({x_offset},{y_offset}))"
            )

        except Exception as e:
            logger.error(f"Error displaying image: {e}")
            messagebox.showerror("Error", f"Failed to display image: {e}", parent=self.dialog)

    def _canvas_to_image_coords(self, canvas_x: int, canvas_y: int) -> tuple:
        """Convert canvas coordinates to original image coordinates.

        Takes into account both the scale factor and any canvas offset.
        Returns coordinates in the original image space.

        Args:
            canvas_x: X coordinate on the canvas
            canvas_y: Y coordinate on the canvas

        Returns:
            Tuple of (image_x, image_y) in original image pixel coordinates
        """
        # Subtract canvas offset first
        display_x = canvas_x - self.canvas_offset_x
        display_y = canvas_y - self.canvas_offset_y

        # Then scale to original image coordinates
        image_x = int(display_x / self.scale_factor)
        image_y = int(display_y / self.scale_factor)

        return (image_x, image_y)

    def _image_to_canvas_coords(self, image_x: int, image_y: int) -> tuple:
        """Convert original image coordinates to canvas coordinates.

        Takes into account both the scale factor and any canvas offset.
        Returns coordinates in the canvas/display space.

        Args:
            image_x: X coordinate in original image
            image_y: Y coordinate in original image

        Returns:
            Tuple of (canvas_x, canvas_y) for display on canvas
        """
        # Scale from image to display coordinates
        display_x = int(image_x * self.scale_factor)
        display_y = int(image_y * self.scale_factor)

        # Add canvas offset
        canvas_x = display_x + self.canvas_offset_x
        canvas_y = display_y + self.canvas_offset_y

        return (canvas_x, canvas_y)

    def _image_to_canvas_coords_bbox(self, img_x1: int, img_y1: int, img_x2: int, img_y2: int) -> tuple:
        """Convert bounding box from image coordinates to canvas coordinates.

        Args:
            img_x1, img_y1, img_x2, img_y2: Bbox coordinates in original image space

        Returns:
            Tuple of (canvas_x1, canvas_y1, canvas_x2, canvas_y2)
        """
        canvas_x1, canvas_y1 = self._image_to_canvas_coords(img_x1, img_y1)
        canvas_x2, canvas_y2 = self._image_to_canvas_coords(img_x2, img_y2)
        return (canvas_x1, canvas_y1, canvas_x2, canvas_y2)

    def _apply_threshold_to_bbox(self, bbox: tuple, threshold_value: int) -> np.ndarray:
        """Apply threshold to bbox region using selected channel.

        Args:
            bbox: Bounding box (x1, y1, x2, y2) in image coordinates
            threshold_value: Threshold value (0-255)

        Returns:
            Binary thresholded image of bbox region
        """
        x1, y1, x2, y2 = bbox

        # Extract bbox region
        roi = self.original_image[y1:y2, x1:x2]

        # Get selected channel
        channel = self.threshold_channel.get()

        # Extract the appropriate channel
        if channel == 'Gray':
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        elif channel == 'R':
            # Extract Red channel (index 2 in BGR)
            gray = roi[:, :, 2]
        elif channel == 'G':
            # Extract Green channel (index 1 in BGR)
            gray = roi[:, :, 1]
        elif channel == 'B':
            # Extract Blue channel (index 0 in BGR)
            gray = roi[:, :, 0]
        else:
            # Default to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply threshold
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        # Apply inversion if checkbox is checked
        if self.threshold_invert.get():
            binary = cv2.bitwise_not(binary)

        return binary

    def _extract_segmentation_from_bbox(self, bbox: tuple, threshold_value: int) -> tuple:
        """Extract YOLO segmentation from bbox using threshold detection.

        Args:
            bbox: Bounding box (x1, y1, x2, y2) in image coordinates
            threshold_value: Threshold value (0-255)

        Returns:
            Tuple of (segmentation_points, binary_filled) where:
            - segmentation_points: List of normalized coordinates [x1, y1, x2, y2, ...]
            - binary_filled: Binary image with holes filled (for debugging)
        """
        # Apply threshold to bbox region
        binary = self._apply_threshold_to_bbox(bbox, threshold_value)

        # Fill holes using morphological closing
        kernel = np.ones((5, 5), np.uint8)
        binary_filled = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(binary_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # No contours found - return empty segmentation
            logger.warning("No contours found in thresholded bbox region")
            return [], binary_filled

        # Get largest contour (the object)
        largest_contour = max(contours, key=cv2.contourArea)

        # Convert contour to YOLO segmentation format
        segmentation = self._contour_to_yolo_segmentation(largest_contour, bbox)

        return segmentation, binary_filled

    def _contour_to_yolo_segmentation(self, contour: np.ndarray, bbox: tuple) -> List[float]:
        """Convert OpenCV contour to YOLO segmentation format.

        Args:
            contour: OpenCV contour points (N, 1, 2) relative to bbox
            bbox: Bounding box (x1, y1, x2, y2) in image pixel coordinates

        Returns:
            List of normalized coordinates: [x1, y1, x2, y2, ...]
        """
        x1, y1, x2, y2 = bbox
        h, w = self.original_image.shape[:2]

        points = []

        # Use all original contour points from OpenCV - no simplification
        for point in contour:
            px, py = point[0]

            # Convert from bbox-relative to image-relative coordinates
            abs_x = x1 + px
            abs_y = y1 + py

            # Normalize to [0, 1]
            norm_x = abs_x / w
            norm_y = abs_y / h

            # Clamp to valid range
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))

            points.extend([norm_x, norm_y])

        return points

    def _on_mouse_down(self, event):
        """Handle mouse button down - start drawing bbox."""
        self.drawing = True
        self.start_point = (event.x, event.y)

        # Clear previous current rectangle (but not saved bboxes)
        if self.current_rect_id:
            self.canvas.delete(self.current_rect_id)
            self.current_rect_id = None

        logger.debug(f"Started drawing bbox at canvas coords: {self.start_point}")

    def _on_mouse_drag(self, event):
        """Handle mouse drag - update bbox preview."""
        if not self.drawing or not self.start_point:
            return

        # Clear previous rectangle
        if self.current_rect_id:
            self.canvas.delete(self.current_rect_id)

        # Draw current rectangle
        x1, y1 = self.start_point
        x2, y2 = event.x, event.y

        # GREEN for object bbox
        outline_color = '#00FF00'
        tag = 'current_bbox'

        self.current_rect_id = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline=outline_color,
            width=3,
            tags=tag
        )

    def _on_mouse_up(self, event):
        """Handle mouse button up - finalize bbox."""
        if not self.drawing or not self.start_point:
            return

        self.drawing = False

        # Get canvas coordinates
        x1, y1 = self.start_point
        x2, y2 = event.x, event.y

        # Normalize (ensure x1 < x2, y1 < y2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Validate size (minimum 10x10 pixels on canvas)
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            messagebox.showwarning(
                "Invalid Selection",
                "Selection is too small. Please draw a larger area.",
                parent=self.dialog
            )
            self._clear_current_bbox()
            return

        # Convert to image coordinates
        img_x1, img_y1 = self._canvas_to_image_coords(x1, y1)
        img_x2, img_y2 = self._canvas_to_image_coords(x2, y2)

        # Clamp to image bounds
        h, w = self.original_image.shape[:2]
        img_x1 = max(0, min(img_x1, w - 1))
        img_x2 = max(0, min(img_x2, w))
        img_y1 = max(0, min(img_y1, h - 1))
        img_y2 = max(0, min(img_y2, h))

        # Store bbox in image coordinates
        self.current_bbox = (img_x1, img_y1, img_x2, img_y2)

        # Enable add button
        self.add_button.config(state='normal')

        # Update status
        bbox_width = img_x2 - img_x1
        bbox_height = img_y2 - img_y1
        self.canvas_status.config(
            text=f"✓ Box: {bbox_width}x{bbox_height}px at ({img_x1},{img_y1})",
            foreground='green'
        )
        self.status_label.config(
            text="Select a class and click Add Object"
        )

        # Update binary preview for the new bbox
        self._update_binary_preview()

        logger.info(f"Bbox drawn: {self.current_bbox}")

    def _on_threshold_changed(self, value=None):
        """Handle threshold slider change or invert checkbox toggle.

        Args:
            value: New threshold value (0-255) from slider, or None if called from checkbox
        """
        if value is not None:
            self.threshold_value = int(float(value))
            self.threshold_value_label.config(text=f"{self.threshold_value}")
            logger.debug(f"Threshold changed to {self.threshold_value}")

        # Update binary preview if bbox is drawn
        self._update_binary_preview()

    def _on_channel_changed(self):
        """Handle channel selection change - update binary preview."""
        channel = self.threshold_channel.get()
        logger.debug(f"Threshold channel changed to {channel}")

        # Update binary preview with new channel selection
        self._update_binary_preview()

    def _update_binary_preview(self):
        """Update the binary preview canvas with thresholded bbox region."""
        if not self.current_bbox:
            # No bbox drawn yet
            self.preview_canvas.delete('all')
            self.preview_status_label.config(
                text="Draw a bounding box to see preview",
                foreground='gray'
            )
            return

        try:
            # Apply threshold to bbox region
            binary = self._apply_threshold_to_bbox(self.current_bbox, self.threshold_value)

            # Fill holes for preview
            kernel = np.ones((5, 5), np.uint8)
            binary_filled = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

            # Store for later use
            self.current_binary_preview = binary_filled

            # Resize to fit preview canvas (250x250)
            h, w = binary_filled.shape
            scale = min(250 / w, 250 / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            if new_w > 0 and new_h > 0:
                preview_img = cv2.resize(binary_filled, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                # Convert to PIL Image
                pil_img = Image.fromarray(preview_img)
                photo = ImageTk.PhotoImage(image=pil_img)

                # Display on canvas (centered)
                self.preview_canvas.delete('all')
                x_offset = (250 - new_w) // 2
                y_offset = (250 - new_h) // 2
                self.preview_canvas.create_image(
                    x_offset, y_offset,
                    anchor='nw',
                    image=photo
                )

                # Keep reference to prevent garbage collection
                self.preview_canvas.photo = photo

                # Update status
                self.preview_status_label.config(
                    text=f"Preview: {w}x{h}px bbox region",
                    foreground='green'
                )
            else:
                self.preview_status_label.config(
                    text="Invalid bbox size",
                    foreground='red'
                )

        except Exception as e:
            logger.error(f"Error updating binary preview: {e}")
            self.preview_status_label.config(
                text="Preview error",
                foreground='red'
            )

    def _clear_current_bbox(self):
        """Clear the current bounding box being drawn."""
        if self.current_rect_id:
            self.canvas.delete(self.current_rect_id)
            self.current_rect_id = None

        self.current_bbox = None
        self.current_binary_preview = None
        self.drawing = False
        self.start_point = None
        self.add_button.config(state='disabled')

        # Clear binary preview
        self.preview_canvas.delete('all')
        self.preview_status_label.config(
            text="Draw a bounding box to see preview",
            foreground='gray'
        )

        self.canvas_status.config(
            text="Draw a box around an object",
            foreground='gray'
        )
        self.status_label.config(text="")

    def _select_class(self, class_name: str):
        """Handle class selection."""
        if not class_name:
            return

        self.current_class = class_name
        self.selected_class_label.config(
            text=f"✓ Selected: {class_name}",
            foreground='green'
        )

        # Clear new class entry if a quick button or dropdown was used
        if class_name in self.existing_classes:
            self.new_class_entry.delete(0, tk.END)

        logger.info(f"Class selected: {class_name}")

    def _add_current_object(self):
        """Add current bbox and class to objects list."""
        # Validate bbox
        if not self.current_bbox:
            messagebox.showwarning(
                "No Bounding Box",
                "Please draw a bounding box first.",
                parent=self.dialog
            )
            return

        # Validate class
        if not self.current_class:
            messagebox.showwarning(
                "No Class",
                "Please select or enter a class name.",
                parent=self.dialog
            )
            return

        # Extract segmentation from bbox using threshold detection
        try:
            segmentation, binary_filled = self._extract_segmentation_from_bbox(
                self.current_bbox,
                self.threshold_value
            )

            if not segmentation:
                logger.warning("No segmentation points extracted, using empty list")

        except Exception as e:
            logger.error(f"Error extracting segmentation: {e}")
            segmentation = []

        # Add to objects list with segmentation
        obj = {
            'bbox': self.current_bbox,
            'class': self.current_class,
            'background_region': None,  # No background region (feature removed)
            'segmentation': segmentation,  # YOLO segmentation format
            'threshold': self.threshold_value,  # Store threshold used
            'threshold_channel': self.threshold_channel.get(),  # Store channel used
            'threshold_invert': self.threshold_invert.get()  # Store invert setting
        }
        self.objects.append(obj)

        # Update listbox
        x1, y1, x2, y2 = self.current_bbox
        w = x2 - x1
        h = y2 - y1
        self.objects_listbox.insert(
            tk.END,
            f"{len(self.objects)}. {self.current_class} [{w}x{h}px]"
        )

        # Change current bbox color to BLUE (saved)
        if self.current_rect_id:
            self.canvas.itemconfig(self.current_rect_id, outline='#0066FF', width=2)
            # Change tag to saved
            self.canvas.itemconfig(self.current_rect_id, tags='saved_bbox')
            self.saved_rect_ids.append(self.current_rect_id)

        # Reset for next object
        self.current_bbox = None
        self.current_rect_id = None
        self.current_class = None
        self.add_button.config(state='disabled')
        self.new_class_entry.delete(0, tk.END)
        self.selected_class_label.config(
            text="No class selected",
            foreground='gray'
        )

        # Update status
        self.canvas_status.config(
            text="Draw next object or click Done",
            foreground='gray'
        )
        self.status_label.config(
            text=f"✓ {len(self.objects)} object(s) added",
            foreground='green'
        )

        logger.info(f"Object added: {obj}")

    def _delete_selected_object(self):
        """Delete selected object from list."""
        selection = self.objects_listbox.curselection()
        if not selection:
            messagebox.showinfo(
                "No Selection",
                "Please select an object to delete.",
                parent=self.dialog
            )
            return

        idx = selection[0]

        # Confirm deletion
        obj = self.objects[idx]
        response = messagebox.askyesno(
            "Confirm Delete",
            f"Delete object: {obj['class']}?",
            parent=self.dialog
        )
        if not response:
            return

        # Remove from objects list
        self.objects.pop(idx)

        # Refresh listbox and canvas
        self._refresh_objects_list()
        self._redraw_all_bboxes()

        # Update status
        self.status_label.config(
            text=f"{len(self.objects)} object(s) remaining",
            foreground='orange'
        )

        logger.info(f"Object deleted at index {idx}")

    def _refresh_objects_list(self):
        """Refresh the objects listbox."""
        self.objects_listbox.delete(0, tk.END)

        for i, obj in enumerate(self.objects, 1):
            x1, y1, x2, y2 = obj['bbox']
            w = x2 - x1
            h = y2 - y1
            self.objects_listbox.insert(
                tk.END,
                f"{i}. {obj['class']} [{w}x{h}px]"
            )

    def _redraw_all_bboxes(self):
        """Redraw all saved bboxes on canvas."""
        # Clear all bbox rectangles
        self.canvas.delete('saved_bbox')
        self.saved_rect_ids.clear()

        # Redraw image
        self._display_image()

        # Redraw all saved bboxes in BLUE
        for obj in self.objects:
            x1, y1, x2, y2 = obj['bbox']

            # Convert to canvas coordinates
            cx1, cy1 = self._image_to_canvas_coords(x1, y1)
            cx2, cy2 = self._image_to_canvas_coords(x2, y2)

            rect_id = self.canvas.create_rectangle(
                cx1, cy1, cx2, cy2,
                outline='#0066FF',  # Blue
                width=2,
                tags='saved_bbox'
            )
            self.saved_rect_ids.append(rect_id)

    def _confirm_all(self):
        """Confirm and return all objects."""
        if not self.objects:
            response = messagebox.askyesno(
                "No Objects",
                "No objects have been added. Exit anyway?",
                icon='warning',
                parent=self.dialog
            )
            if not response:
                return

        # Return list of objects
        self.result = self.objects
        logger.info(f"Confirmed {len(self.objects)} objects")
        self.dialog.destroy()

    def _on_cancel(self):
        """Cancel and return None."""
        if self.objects:
            response = messagebox.askyesno(
                "Discard Changes",
                f"{len(self.objects)} object(s) will be lost. Cancel anyway?",
                icon='warning',
                parent=self.dialog
            )
            if not response:
                return

        self.result = None
        logger.info("Dialog cancelled")
        self.dialog.destroy()

    def _bind_keyboard_shortcuts(self):
        """Bind keyboard shortcuts."""
        # Number keys 1-9 for quick class selection
        for i in range(1, 10):
            if i <= len(self.quick_classes):
                self.dialog.bind(
                    str(i),
                    lambda e, idx=i-1: self._select_class(self.quick_classes[idx])
                )

        # Enter to add object
        self.dialog.bind('<Return>', lambda e: self._add_current_object())

        # Escape to cancel
        self.dialog.bind('<Escape>', lambda e: self._on_cancel())

        # Delete key to remove selected object
        self.dialog.bind('<Delete>', lambda e: self._delete_selected_object())

        logger.info("Keyboard shortcuts bound")

    def show(self) -> Optional[List[Dict]]:
        """Show dialog modally and return result.

        Returns:
            List of objects if confirmed: [{'bbox': (x1,y1,x2,y2), 'class': 'name'}, ...]
            None if cancelled
        """
        self.dialog.wait_window()
        return self.result
