"""Data augmentation dialog for randomly placing training objects onto background images."""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Optional, List, Dict, Tuple
import numpy as np
import cv2
from PIL import Image, ImageTk
import logging
import random
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class DataAugmentationDialog:
    """Dialog for generating augmented training data by randomly placing objects on backgrounds."""

    def __init__(self, parent, training_service, webcam_service):
        """Initialize data augmentation dialog.

        Args:
            parent: Parent window
            training_service: TrainingService instance
            webcam_service: WebcamService instance
        """
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Data Augmentation - Random Object Placement")
        self.dialog.geometry("1600x900")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Center on parent
        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 1600) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 900) // 2
        self.dialog.geometry(f"+{x}+{y}")

        # Services
        self.training_service = training_service
        self.webcam_service = webcam_service

        # State
        self.background_image: Optional[np.ndarray] = None
        self.augmented_data: List[Dict] = []  # List of {'image': ndarray, 'labels': List[str], 'filename': str}
        self.selected_objects_ids: List[str] = []  # List of selected object IDs
        self.current_preview_index: int = -1
        self.preview_bg_button: Optional[ttk.Button] = None  # Preview button reference
        self.batch_timestamp: str = ""  # Timestamp for current batch to ensure unique filenames

        # Get all training objects with segmentation
        all_objects = self.training_service.get_all_objects()
        self.available_objects = [obj for obj in all_objects if obj.segmentation and len(obj.segmentation) > 0]

        if not self.available_objects:
            messagebox.showerror(
                "No Segmented Objects",
                "No training objects with segmentation data found.\n\n"
                "Please add objects using threshold detection first.",
                parent=self.dialog
            )
            self.dialog.destroy()
            return

        logger.info(f"Found {len(self.available_objects)} objects with segmentation data")

        # Build UI
        self._build_ui()

        # Bind window close
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        """Build the dialog UI."""
        # Top panel: Background source selection
        top_panel = ttk.Frame(self.dialog)
        top_panel.pack(fill='x', padx=10, pady=10)

        ttk.Label(
            top_panel,
            text="Background Image Source:",
            font=('Segoe UI', 10, 'bold')
        ).pack(side='left', padx=(0, 10))

        self.source_var = tk.StringVar(value='video')

        ttk.Radiobutton(
            top_panel,
            text="📹 Video Stream",
            variable=self.source_var,
            value='video',
            command=self._on_source_changed
        ).pack(side='left', padx=5)

        ttk.Radiobutton(
            top_panel,
            text="📁 Load from File",
            variable=self.source_var,
            value='file',
            command=self._on_source_changed
        ).pack(side='left', padx=5)

        ttk.Button(
            top_panel,
            text="Load Background",
            command=self._load_background
        ).pack(side='left', padx=10)

        # Add Preview Background button
        self.preview_bg_button = ttk.Button(
            top_panel,
            text="Preview Background",
            command=self._preview_background,
            state='disabled'  # Initially disabled
        )
        self.preview_bg_button.pack(side='left', padx=5)

        # Main container (horizontal split)
        main_container = ttk.Frame(self.dialog)
        main_container.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        # LEFT: Image canvas (70% width)
        left_frame = ttk.Frame(main_container)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))

        # Canvas title
        ttk.Label(
            left_frame,
            text="Preview (Background with Placed Objects)",
            font=('Segoe UI', 10, 'bold')
        ).pack(anchor='w', pady=(0, 5))

        # Canvas
        canvas_container = ttk.Frame(left_frame, relief='sunken', borderwidth=2)
        canvas_container.pack(fill='both', expand=True)

        self.canvas = tk.Canvas(
            canvas_container,
            bg='#1e1e1e',
            highlightthickness=0
        )
        self.canvas.pack(fill='both', expand=True)

        # RIGHT: Control panel (30% width) - 2 column layout
        right_frame = ttk.Frame(main_container)
        right_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))

        # Create 2 columns within right frame
        right_left_col = ttk.Frame(right_frame, width=80)
        right_left_col.pack_propagate(False)
        right_left_col.pack(side='left', fill='both', expand=True, padx=(0, 5))

        right_right_col = ttk.Frame(right_frame, width=80)
        right_right_col.pack_propagate(False)
        right_right_col.pack(side='right', fill='both', expand=True, padx=(5, 0))

        # Distribute sections across columns
        # Left column: Object selection
        self._build_object_selection_section(right_left_col)

        # Right column: Generation controls, images list, and action buttons
        self._build_generation_controls_section(right_right_col)
        self._build_generated_images_section(right_right_col)
        self._build_action_buttons_section(right_right_col)

    def _build_object_selection_section(self, parent):
        """Build object selection section with checkboxes and scrolling support."""
        selection_frame = ttk.LabelFrame(parent, text="Select Objects to Place", padding=10)
        selection_frame.pack(fill='both', expand=True, pady=(0, 10))

        # Instruction label
        ttk.Label(
            selection_frame,
            text="Select which objects to randomly place:",
            font=('Segoe UI', 9),
            foreground='gray'
        ).pack(anchor='w', pady=(0, 5))

        # Scrollable list with checkboxes
        list_container = ttk.Frame(selection_frame)
        list_container.pack(fill='both', expand=True)

        # Create canvas for scrolling
        self.objects_scroll_canvas = tk.Canvas(list_container, height=300, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_container, orient='vertical', command=self.objects_scroll_canvas.yview)
        scroll_frame = ttk.Frame(self.objects_scroll_canvas)

        scroll_frame.bind(
            '<Configure>',
            lambda e: self.objects_scroll_canvas.configure(scrollregion=self.objects_scroll_canvas.bbox('all'))
        )

        self.objects_scroll_canvas.create_window((0, 0), window=scroll_frame, anchor='nw')
        self.objects_scroll_canvas.configure(yscrollcommand=scrollbar.set)

        self.objects_scroll_canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Enable mousewheel scrolling when mouse enters/leaves canvas
        self.objects_scroll_canvas.bind(
            '<Enter>',
            lambda e: self.objects_scroll_canvas.bind_all('<MouseWheel>', self._on_objects_mousewheel)
        )
        self.objects_scroll_canvas.bind(
            '<Leave>',
            lambda e: self.objects_scroll_canvas.unbind_all('<MouseWheel>')
        )

        # Create checkboxes for each object
        # Store both BooleanVar and widget reference for proper visual updates
        self.object_checkboxes: Dict[str, Dict[str, any]] = {}

        for obj in self.available_objects:
            var = tk.BooleanVar(value=False)

            # Get object info
            x1, y1, x2, y2 = obj.bbox
            w, h = x2 - x1, y2 - y1
            seg_points = len(obj.segmentation) // 2

            cb = tk.Checkbutton(
                scroll_frame,
                text=f"{obj.label} ({w}x{h}px, {seg_points} seg points)",
                variable=var,
                font=('Segoe UI', 9)
            )
            cb.pack(anchor='w', pady=2)

            # Store both var and widget reference for programmatic control
            self.object_checkboxes[obj.object_id] = {
                'var': var,
                'widget': cb
            }

        # Select All / Deselect All buttons
        button_row = ttk.Frame(selection_frame)
        button_row.pack(fill='x', pady=(5, 0))

        ttk.Button(
            button_row,
            text="Select All",
            command=self._select_all_objects,
            width=15
        ).pack(side='left', padx=(0, 5))

        ttk.Button(
            button_row,
            text="Deselect All",
            command=self._deselect_all_objects,
            width=15
        ).pack(side='left')

    def _build_generation_controls_section(self, parent):
        """Build generation controls section."""
        controls_frame = ttk.LabelFrame(parent, text="Generation Settings", padding=10)
        controls_frame.pack(fill='x', pady=(0, 10))

        # Number of generations
        ttk.Label(
            controls_frame,
            text="Number of Generations:",
            font=('Segoe UI', 9, 'bold')
        ).pack(anchor='w', pady=(0, 5))

        gen_row = ttk.Frame(controls_frame)
        gen_row.pack(fill='x', pady=(0, 10))

        self.num_generations_var = tk.IntVar(value=10)

        gen_spinbox = ttk.Spinbox(
            gen_row,
            from_=1,
            to=100,
            textvariable=self.num_generations_var,
            width=10,
            font=('Segoe UI', 10)
        )
        gen_spinbox.pack(side='left', padx=(0, 5))

        ttk.Label(
            gen_row,
            text="(1-100 images)",
            font=('Segoe UI', 8),
            foreground='gray'
        ).pack(side='left')

        # Random object selection settings
        ttk.Label(
            controls_frame,
            text="Object Selection:",
            font=('Segoe UI', 9, 'bold')
        ).pack(anchor='w', pady=(10, 5))

        # Random selection checkbox
        self.use_random_selection_var = tk.BooleanVar(value=True)
        random_checkbox = ttk.Checkbutton(
            controls_frame,
            text="Random Object Selection",
            variable=self.use_random_selection_var,
            command=self._on_random_selection_toggled
        )
        random_checkbox.pack(anchor='w', pady=(0, 5))

        ttk.Label(
            controls_frame,
            text="(Each selected object type will be placed 0-5 times independently)",
            font=('Segoe UI', 8),
            foreground='gray'
        ).pack(anchor='w', padx=(20, 0), pady=(0, 10))

        # Min objects row
        min_obj_row = ttk.Frame(controls_frame)
        min_obj_row.pack(fill='x', pady=(0, 5))

        ttk.Label(
            min_obj_row,
            text="Min copies per object type:",
            font=('Segoe UI', 9)
        ).pack(side='left', padx=(20, 5))

        self.min_objects_var = tk.IntVar(value=1)
        self.min_objects_spinbox = ttk.Spinbox(
            min_obj_row,
            from_=0,
            to=5,
            textvariable=self.min_objects_var,
            width=8,
            font=('Segoe UI', 9)
        )
        self.min_objects_spinbox.pack(side='left')

        # Max objects row
        max_obj_row = ttk.Frame(controls_frame)
        max_obj_row.pack(fill='x', pady=(0, 10))

        ttk.Label(
            max_obj_row,
            text="Max copies per object type:",
            font=('Segoe UI', 9)
        ).pack(side='left', padx=(20, 5))

        self.max_objects_var = tk.IntVar(value=5)
        self.max_objects_spinbox = ttk.Spinbox(
            max_obj_row,
            from_=0,
            to=5,
            textvariable=self.max_objects_var,
            width=8,
            font=('Segoe UI', 9)
        )
        self.max_objects_spinbox.pack(side='left')

        # Debug mode: Same location checkbox
        # ttk.Label(
        #     controls_frame,
        #     text="Debug Settings:",
        #     font=('Segoe UI', 9, 'bold')
        # ).pack(anchor='w', pady=(10, 5))

        # self.same_location_mode = tk.BooleanVar(value=False)
        # debug_checkbox = ttk.Checkbutton(
        #     controls_frame,
        #     text="🔧 Debug Mode: Place at Original Location",
        #     variable=self.same_location_mode
        # )
        # debug_checkbox.pack(anchor='w', pady=(0, 10))

        # ttk.Label(
        #     controls_frame,
        #     text="(Verifies coordinate transformations are working correctly)",
        #     font=('Segoe UI', 8),
        #     foreground='gray'
        # ).pack(anchor='w', padx=(20, 0), pady=(0, 10))

        # Generate button
        self.generate_button = ttk.Button(
            controls_frame,
            text="🎲 Generate Augmented Images",
            command=self._on_generate,
            width=30
        )
        self.generate_button.pack(fill='x', pady=(0, 5))

        # Status label
        self.generation_status = ttk.Label(
            controls_frame,
            text="",
            font=('Segoe UI', 8),
            foreground='gray'
        )
        self.generation_status.pack(anchor='w')

    def _build_generated_images_section(self, parent):
        """Build generated images list section."""
        images_frame = ttk.LabelFrame(parent, text="Generated Images", padding=10)
        images_frame.pack(fill='both', expand=True, pady=(0, 10))

        # Listbox with scrollbar
        list_container = ttk.Frame(images_frame)
        list_container.pack(fill='both', expand=True)

        self.images_listbox = tk.Listbox(
            list_container,
            font=('Consolas', 9),
            selectmode='single'
        )
        self.images_listbox.pack(side='left', fill='both', expand=True)

        scrollbar = ttk.Scrollbar(list_container, command=self.images_listbox.yview)
        scrollbar.pack(side='right', fill='y')
        self.images_listbox.config(yscrollcommand=scrollbar.set)

        # Bind selection event
        self.images_listbox.bind('<<ListboxSelect>>', self._on_image_selected)

        # Preview hint
        ttk.Label(
            images_frame,
            text="Click to preview on canvas",
            font=('Segoe UI', 8),
            foreground='gray'
        ).pack(pady=(5, 0))

    def _build_action_buttons_section(self, parent):
        """Build action buttons section."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill='x')

        # Save All button
        self.save_button = ttk.Button(
            button_frame,
            text="💾 Save All Images",
            command=self._on_save_all,
            state='disabled',
            width=30
        )
        self.save_button.pack(fill='x', pady=2)

        # Close button
        ttk.Button(
            button_frame,
            text="Close",
            command=self._on_close,
            width=30
        ).pack(fill='x', pady=2)

    def _on_source_changed(self):
        """Handle background source change."""
        logger.info(f"Background source changed to: {self.source_var.get()}")

    def _on_random_selection_toggled(self):
        """Handle random object selection checkbox toggle."""
        use_random = self.use_random_selection_var.get()
        state = 'normal' if use_random else 'disabled'

        self.min_objects_spinbox.config(state=state)
        self.max_objects_spinbox.config(state=state)

        logger.info(f"Random object selection: {'enabled' if use_random else 'disabled'}")

    def _load_background(self):
        """Load background image based on selected source."""
        source = self.source_var.get()

        if source == 'video':
            # Capture from video stream
            frame = self.webcam_service.get_current_frame()
            if frame is None:
                messagebox.showerror(
                    "No Video Stream",
                    "Video stream is not available. Please ensure the camera is connected.",
                    parent=self.dialog
                )
                return

            self.background_image = frame.copy()
            logger.info("Captured background from video stream")

        elif source == 'file':
            # Load from file
            file_path = filedialog.askopenfilename(
                title="Select Background Image",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                    ("All files", "*.*")
                ],
                parent=self.dialog
            )

            if not file_path:
                return

            try:
                img = cv2.imread(file_path)
                if img is None:
                    messagebox.showerror(
                        "Load Error",
                        f"Failed to load image from {file_path}",
                        parent=self.dialog
                    )
                    return

                self.background_image = img
                logger.info(f"Loaded background from file: {file_path}")

            except Exception as e:
                messagebox.showerror(
                    "Load Error",
                    f"Error loading image: {e}",
                    parent=self.dialog
                )
                return

        # Display background on canvas
        if self.background_image is not None:
            # Show background on canvas
            self._display_image(self.background_image)

            # Enable preview button
            self.preview_bg_button.config(state='normal')

            messagebox.showinfo(
                "Background Loaded",
                f"Background image loaded: {self.background_image.shape[1]}x{self.background_image.shape[0]}px",
                parent=self.dialog
            )

    def _preview_background(self):
        """Preview the loaded background image on canvas."""
        if self.background_image is None:
            messagebox.showwarning(
                "No Background",
                "Please load a background image first.",
                parent=self.dialog
            )
            return

        # Display background image
        self._display_image(self.background_image)
        logger.info("Previewing background image")

    def _select_all_objects(self):
        """Select all available objects.

        Uses widget.select() method to ensure both state and visual display update.
        This is critical for checkboxes embedded in Canvas-based scrolling frames.
        """
        count = 0
        for data in self.object_checkboxes.values():
            data['widget'].select()  # Directly select widget (updates both var and visual)
            count += 1
        logger.info(f"Selected all objects ({count} checkboxes updated)")

    def _deselect_all_objects(self):
        """Deselect all objects.

        Uses widget.deselect() method to ensure both state and visual display update.
        This is critical for checkboxes embedded in Canvas-based scrolling frames.
        """
        count = 0
        for data in self.object_checkboxes.values():
            data['widget'].deselect()  # Directly deselect widget (updates both var and visual)
            count += 1
        logger.info(f"Deselected all objects ({count} checkboxes updated)")

    def _on_objects_mousewheel(self, event):
        """Handle mousewheel scrolling for the objects selection canvas.

        Args:
            event: Mouse event containing delta information
        """
        # Scroll the canvas based on mousewheel direction
        # event.delta is positive for scroll up, negative for scroll down
        # Divide by 120 to get scroll units (standard Windows behavior)
        self.objects_scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _create_progress_window(self, total_generations: int) -> tk.Toplevel:
        """Create a progress window to display generation progress.

        This window shows:
        - Current progress (e.g., "Generating 5/10...")
        - Progress bar with percentage
        - Real-time updates as each image is generated

        Args:
            total_generations: Total number of images to generate

        Returns:
            Toplevel window instance with progress tracking widgets
        """
        # Create progress window
        progress_win = tk.Toplevel(self.dialog)
        progress_win.title("Generating Augmented Images")
        progress_win.geometry("450x150")
        progress_win.transient(self.dialog)
        progress_win.grab_set()

        # Center on parent dialog
        progress_win.update_idletasks()
        x = self.dialog.winfo_x() + (self.dialog.winfo_width() - 450) // 2
        y = self.dialog.winfo_y() + (self.dialog.winfo_height() - 150) // 2
        progress_win.geometry(f"+{x}+{y}")

        # Prevent closing during generation
        progress_win.protocol("WM_DELETE_WINDOW", lambda: None)

        # Progress frame
        progress_frame = ttk.Frame(progress_win, padding=20)
        progress_frame.pack(fill='both', expand=True)

        # Status label
        status_label = ttk.Label(
            progress_frame,
            text=f"Generating 0/{total_generations}...",
            font=('Segoe UI', 11, 'bold')
        )
        status_label.pack(pady=(0, 15))

        # Progress bar
        progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            length=400,
            maximum=total_generations
        )
        progress_bar.pack(pady=(0, 10))

        # Percentage label
        percent_label = ttk.Label(
            progress_frame,
            text="0%",
            font=('Segoe UI', 10),
            foreground='gray'
        )
        percent_label.pack()

        # Store widgets as attributes for easy access
        progress_win.status_label = status_label
        progress_win.progress_bar = progress_bar
        progress_win.percent_label = percent_label
        progress_win.total_generations = total_generations

        # Force window to display immediately
        progress_win.update()

        logger.debug(f"Created progress window for {total_generations} generations")

        return progress_win

    def _create_save_progress_window(self, total_images: int) -> tk.Toplevel:
        """Create a progress window to display save operation progress.

        This window shows:
        - Current progress (e.g., "Processing 5/10 images...")
        - Progress bar with percentage
        - Real-time updates as each image is processed

        Args:
            total_images: Total number of images to process

        Returns:
            Toplevel window instance with progress tracking widgets
        """
        # Create progress window
        progress_win = tk.Toplevel(self.dialog)
        progress_win.title("Saving Augmented Objects")
        progress_win.geometry("450x150")
        progress_win.transient(self.dialog)
        progress_win.grab_set()

        # Center on parent dialog
        progress_win.update_idletasks()
        x = self.dialog.winfo_x() + (self.dialog.winfo_width() - 450) // 2
        y = self.dialog.winfo_y() + (self.dialog.winfo_height() - 150) // 2
        progress_win.geometry(f"+{x}+{y}")

        # Prevent closing during save operation
        progress_win.protocol("WM_DELETE_WINDOW", lambda: None)

        # Progress frame
        progress_frame = ttk.Frame(progress_win, padding=20)
        progress_frame.pack(fill='both', expand=True)

        # Status label
        status_label = ttk.Label(
            progress_frame,
            text=f"Processing 0/{total_images} images...",
            font=('Segoe UI', 11, 'bold')
        )
        status_label.pack(pady=(0, 15))

        # Progress bar
        progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            length=400,
            maximum=total_images
        )
        progress_bar.pack(pady=(0, 10))

        # Percentage label
        percent_label = ttk.Label(
            progress_frame,
            text="0%",
            font=('Segoe UI', 10),
            foreground='gray'
        )
        percent_label.pack()

        # Store widgets as attributes for easy access
        progress_win.status_label = status_label
        progress_win.progress_bar = progress_bar
        progress_win.percent_label = percent_label
        progress_win.total_images = total_images

        # Force window to display immediately
        progress_win.update()

        logger.debug(f"Created save progress window for {total_images} images")

        return progress_win

    def _get_selected_objects(self) -> List:
        """Get list of selected training objects.

        Returns:
            List of TrainingObject instances that are selected
        """
        selected = []
        for obj in self.available_objects:
            if self.object_checkboxes[obj.object_id]['var'].get():
                selected.append(obj)
        return selected

    def _yolo_to_contour(self, segmentation: List[float], bbox: Tuple[int, int, int, int],
                        img_w: int, img_h: int) -> Optional[np.ndarray]:
        """Convert YOLO segmentation format to OpenCV contour.

        Args:
            segmentation: YOLO segmentation in normalized format [x1, y1, x2, y2, ...]
            bbox: Original object bbox (x1, y1, x2, y2) in pixels
            img_w: Image width for centering
            img_h: Image height for centering

        Returns:
            Contour as numpy array of shape (N, 1, 2) or None if invalid
        """
        if not segmentation or len(segmentation) < 6:
            return None

        try:
            # Get bbox dimensions
            x1, y1, x2, y2 = bbox
            bbox_w = x2 - x1
            bbox_h = y2 - y1

            # Convert normalized segmentation to pixel coordinates in bbox space
            points = []
            for i in range(0, len(segmentation), 2):
                # Denormalize from [0,1] to bbox pixel space
                x = segmentation[i] * bbox_w
                y = segmentation[i + 1] * bbox_h
                points.append([x, y])

            if not points:
                return None

            # Convert to numpy array
            contour_bbox_space = np.array(points, dtype=np.float32)

            # Center the contour on the background image
            # Calculate center position (centered on canvas)
            center_x = img_w // 2
            center_y = img_h // 2

            # Get contour centroid
            contour_center_x = contour_bbox_space[:, 0].mean()
            contour_center_y = contour_bbox_space[:, 1].mean()

            # Calculate offset to center contour
            offset_x = center_x - contour_center_x
            offset_y = center_y - contour_center_y

            # Translate contour to center
            contour_centered = contour_bbox_space + np.array([offset_x, offset_y])

            # Convert to integer coordinates and reshape for OpenCV
            contour = contour_centered.astype(np.int32).reshape(-1, 1, 2)

            return contour

        except Exception as e:
            logger.error(f"Error converting YOLO segmentation to contour: {e}")
            return None

    def _on_generate(self):
        """Generate augmented images with random object placement."""
        # Validate background
        if self.background_image is None:
            messagebox.showwarning(
                "No Background",
                "Please load a background image first.",
                parent=self.dialog
            )
            return

        # Validate object selection
        selected_objects = self._get_selected_objects()
        if not selected_objects:
            messagebox.showwarning(
                "No Objects Selected",
                "Please select at least one object to place.",
                parent=self.dialog
            )
            return

        # Get number of generations
        num_generations = self.num_generations_var.get()

        # Get random selection settings
        use_random_selection = self.use_random_selection_var.get()
        min_objects = self.min_objects_var.get()
        max_objects = self.max_objects_var.get()

        # Validate min/max settings if random selection is enabled
        if use_random_selection:
            if min_objects > max_objects:
                messagebox.showwarning(
                    "Invalid Settings",
                    "Minimum copies must be less than or equal to maximum copies.",
                    parent=self.dialog
                )
                return

            if min_objects < 0:
                messagebox.showwarning(
                    "Invalid Settings",
                    "Minimum copies must be at least 0.",
                    parent=self.dialog
                )
                return

            if max_objects > 5:
                messagebox.showwarning(
                    "Invalid Settings",
                    "Maximum copies cannot exceed 5.",
                    parent=self.dialog
                )
                return

            if max_objects < 0:
                messagebox.showwarning(
                    "Invalid Settings",
                    "Maximum copies must be at least 0.",
                    parent=self.dialog
                )
                return

        # Generate unique timestamp for this batch to prevent filename conflicts
        self.batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Disable generate button during generation
        self.generate_button.config(state='disabled')
        self.generation_status.config(text="Generating...", foreground='orange')
        self.dialog.update()

        # Create progress window
        progress_window = self._create_progress_window(num_generations)

        try:
            # Generate augmented data with progress tracking
            self.augmented_data = self._place_objects_randomly(
                self.background_image,
                selected_objects,
                num_generations,
                progress_window=progress_window,
                use_random_selection=use_random_selection,
                min_objects=min_objects,
                max_objects=max_objects,
                batch_timestamp=self.batch_timestamp
            )

            # Populate generated images list
            self._populate_generated_list()

            # Update status
            self.generation_status.config(
                text=f"✓ Generated {len(self.augmented_data)} images (batch: {self.batch_timestamp})",
                foreground='green'
            )

            # Enable save button
            self.save_button.config(state='normal')

            logger.info(
                f"Generated {len(self.augmented_data)} augmented images with batch timestamp {self.batch_timestamp}"
            )

        except Exception as e:
            logger.error(f"Error generating augmented images: {e}", exc_info=True)
            messagebox.showerror(
                "Generation Error",
                f"Failed to generate augmented images:\n{e}",
                parent=self.dialog
            )
            self.generation_status.config(text="Generation failed", foreground='red')

        finally:
            # Close progress window
            if progress_window:
                progress_window.destroy()
            self.generate_button.config(state='normal')

    def _place_objects_randomly(self, background: np.ndarray, selected_objects: List,
                                num_generations: int, progress_window: Optional[tk.Toplevel] = None,
                                use_random_selection: bool = False, min_objects: int = 1,
                                max_objects: Optional[int] = None,
                                batch_timestamp: str = "") -> List[Dict]:
        """Generate multiple augmented images with random object placement.

        Args:
            background: Background image
            selected_objects: List of TrainingObject instances to place
            num_generations: Number of augmented images to generate
            progress_window: Optional Toplevel window for progress tracking
            use_random_selection: If True, randomly select subset of objects per image
            min_objects: Minimum objects per image (only used if use_random_selection=True)
            max_objects: Maximum objects per image (None = all selected, only used if use_random_selection=True)
            batch_timestamp: Timestamp string to make filenames unique across batches

        Returns:
            List of dicts with 'image', 'labels', 'filename' keys
        """
        augmented_data = []
        h, w = background.shape[:2]

        # Statistics tracking for per-object-type placement counts
        object_placement_stats = {obj.label: [] for obj in selected_objects}

        # Set default max_objects if not specified
        if max_objects is None:
            max_objects = len(selected_objects)

        # Generate timestamp if not provided (for backward compatibility)
        if not batch_timestamp:
            batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(
            f"Starting batch generation (timestamp: {batch_timestamp}): {num_generations} images, "
            f"{'random placement per object type' if use_random_selection else 'all objects once'}, "
            f"range: {min_objects}-{max_objects} copies per object type"
        )

        for gen_idx in range(num_generations):
            # Determine which objects to place in this image
            if use_random_selection:
                # For each object type, decide how many copies to place
                objects_to_place = []
                placement_summary = {}

                for obj in selected_objects:
                    num_copies = random.randint(min_objects, max_objects)
                    placement_summary[obj.label] = num_copies

                    # Track statistics
                    object_placement_stats[obj.label].append(num_copies)

                    # Add this object num_copies times to the placement list
                    for _ in range(num_copies):
                        objects_to_place.append(obj)

                num_to_place = len(objects_to_place)

                logger.debug(
                    f"Image {gen_idx + 1}: Placing {num_to_place} total objects - "
                    f"{', '.join([f'{label}:{count}' for label, count in placement_summary.items()])}"
                )
            else:
                # Use all selected objects once (original behavior)
                objects_to_place = selected_objects
                num_to_place = len(selected_objects)

            # Update progress window if provided
            if progress_window:
                try:
                    current = gen_idx + 1
                    total_objects_count = len(objects_to_place)
                    progress_window.status_label.config(
                        text=f"Batch {batch_timestamp}: Generating {current}/{num_generations} ({total_objects_count} objects)..."
                    )
                    progress_window.progress_bar['value'] = current
                    percent = int((current / num_generations) * 100)
                    progress_window.percent_label.config(text=f"{percent}%")
                    progress_window.update()
                except Exception as e:
                    logger.warning(f"Error updating progress window: {e}")

            # Create copy of background
            aug_image = background.copy()
            labels = []  # YOLO format labels

            placed_bboxes = []  # Track placed object positions for collision detection

            for obj in objects_to_place:
                # ============================================================
                # TEMPORARILY DISABLED: Rotation for testing purposes
                # To re-enable rotation, uncomment the line below and comment the fixed angle line
                # ============================================================
                angle = random.uniform(0, 360)  # Random rotation (DISABLED FOR TESTING)
                # angle = 45  # Fixed angle - NO ROTATION (TESTING ONLY)

                # Get object's original bbox dimensions (maintain these!)
                obj_x1, obj_y1, obj_x2, obj_y2 = obj.bbox
                obj_w = obj_x2 - obj_x1
                obj_h = obj_y2 - obj_y1

                # Get object's segmentation as contour
                segmentation = np.array(obj.segmentation).reshape(-1, 2)

                # Get original image dimensions
                # obj.image contains the full original source image
                orig_img_h, orig_img_w = obj.image.shape[:2]

                # CRITICAL FIX: Convert to BBOX-RELATIVE coordinates
                # Step 1: Denormalize segmentation to IMAGE-SPACE pixel coordinates
                contour_image_space = segmentation * np.array([orig_img_w, orig_img_h])

                # Step 2: Translate to BBOX-RELATIVE coordinates (subtract bbox origin)
                # This makes the contour relative to (0, 0) at bbox top-left
                contour_bbox_relative = contour_image_space - np.array([obj_x1, obj_y1])
                
                # Calculate centroid
                centroid = contour_bbox_relative.mean(axis=0)

                # Step 3: Rotate contour around its centroid in bbox-relative space
                rotated_contour_bbox_rel = self._rotate_contour(contour_bbox_relative, centroid, angle)

                # Step 4: Get bounding box of rotated contour (in bbox-relative space)
                rotated_bbox_rel = self._get_contour_bbox(rotated_contour_bbox_rel)
                rotated_bbox_w = rotated_bbox_rel[2] - rotated_bbox_rel[0]
                rotated_bbox_h = rotated_bbox_rel[3] - rotated_bbox_rel[1]

                # BOUNDARY FIX: Ensure object stays within background image
                # The rotated bbox dimensions tell us how much space we need
                max_x = max(1, w - int(rotated_bbox_w))
                max_y = max(1, h - int(rotated_bbox_h))

                if max_x <= 0 or max_y <= 0:
                    logger.warning(
                        f"Object {obj.label} too large for background "
                        f"({rotated_bbox_w}x{rotated_bbox_h} vs {w}x{h}), skipping"
                    )
                    continue

                
                # NORMAL MODE: Random placement with collision detection
                # Find random position with no collision (max 100 attempts)
                max_attempts = 100
                placed = False

                for _ in range(max_attempts):
                    # Random position ensuring object fits in bounds
                    x = random.randint(0, max_x)
                    y = random.randint(0, max_y)
                    
                    # Calculate final contour position at random location
                    # Adjust contour position to match the random placement
                    contour_centroid = rotated_contour_bbox_rel.mean(axis=0)
                    target_center_x = x + rotated_bbox_w / 2
                    target_center_y = y + rotated_bbox_h / 2
                    offset_x = target_center_x - contour_centroid[0]
                    offset_y = target_center_y - contour_centroid[1]
                    translated_contour = rotated_contour_bbox_rel + np.array([offset_x, offset_y])

                    # Collision bbox based on rotated dimensions
                    candidate_bbox = self._get_contour_bbox(translated_contour)
                    
                    # Check bounding box is out of image
                    is_out_of_bounds = candidate_bbox[0] < 0 or candidate_bbox[1] < 0 or candidate_bbox[0] >= w or candidate_bbox[1] >= h

                    # Check collision with already placed objects
                    if not is_out_of_bounds and not self._check_collision(candidate_bbox, placed_bboxes):
                        # No collision - place object here
                        placed_bboxes.append(candidate_bbox)

                        # Extract and composite object onto augmented image
                        self._extract_and_composite_object(
                            aug_image=aug_image,
                            obj_image=obj.image,
                            bbox=obj.bbox,
                            segmentation=obj.segmentation,
                            target_contour=translated_contour,
                            angle=angle,
                            apply_augmentation=True  # Apply color/brightness augmentation
                        )

                        # Create YOLO label (using segmentation format)
                        yolo_label = self._create_yolo_segmentation_label(
                            obj.label, translated_contour, w, h
                        )
                        labels.append(yolo_label)

                        placed = True
                        break

                if not placed:
                    logger.warning(
                        f"Could not place object {obj.label} without collision "
                        f"after {max_attempts} attempts (generation {gen_idx + 1})"
                    )

            # Create unique filename with batch timestamp to prevent conflicts
            filename = f'augmented_{batch_timestamp}_{gen_idx:03d}.jpg'

            augmented_data.append({
                'image': aug_image,
                'labels': labels,
                'filename': filename,
                'num_objects': num_to_place  # Store for display in listbox
            })

            # Log progress every 10 generations
            if (gen_idx + 1) % 10 == 0 or (gen_idx + 1) == num_generations:
                logger.info(f"Generated {gen_idx + 1}/{num_generations} augmented images")

        # Log statistics if random selection was used
        if use_random_selection and object_placement_stats:
            logger.info(f"\nBatch {batch_timestamp} generation complete! Per-object-type placement statistics:")
            logger.info(f"Generated {num_generations} augmented images")
            logger.info(f"Object placement counts:")

            for obj_label, counts in object_placement_stats.items():
                if counts:
                    avg_count = sum(counts) / len(counts)
                    min_count = min(counts)
                    max_count = max(counts)
                    total_placed = sum(counts)
                    logger.info(
                        f"  - {obj_label}: avg {avg_count:.1f} copies/image "
                        f"(range {min_count}-{max_count}, total {total_placed} placed)"
                    )

        return augmented_data

    def _check_collision(self, bbox1: Tuple, bboxes: List[Tuple]) -> bool:
        """Check if bbox1 collides with any bbox in bboxes.

        Args:
            bbox1: Bounding box as (x1, y1, x2, y2)
            bboxes: List of bounding boxes to check against

        Returns:
            True if collision detected, False otherwise
        """
        x1, y1, x2, y2 = bbox1

        for bx1, by1, bx2, by2 in bboxes:
            # Check overlap
            if not (x2 < bx1 or x1 > bx2 or y2 < by1 or y1 > by2):
                return True  # Collision detected

        return False

    def _rotate_contour(self, contour: np.ndarray, centroid: np.ndarray, angle: float) -> np.ndarray:
        """Rotate contour points around centroid.

        Args:
            contour: numpy array of shape (N, 2) with [x, y] points
            angle: rotation angle in degrees

        Returns:
            Rotated contour as numpy array of shape (N, 2)
        """
        # Handle edge cases
        if len(contour) == 0:
            return contour.copy()

        if angle == 0:
            return contour.copy()

        # Create rotation matrix
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])

        # Translate to origin, rotate, translate back
        centered = contour - centroid
        rotated = centered @ rotation_matrix.T
        result = rotated + centroid

        return result

    def _translate_contour(self, contour: np.ndarray, x: int, y: int) -> np.ndarray:
        """Translate contour to new position.

        Args:
            contour: Contour points as (N, 2) array
            x: X offset
            y: Y offset

        Returns:
            Translated contour points
        """
        return contour + np.array([x, y])

    def _get_contour_bbox(self, contour: np.ndarray) -> Tuple:
        """Get bounding box of contour.

        Args:
            contour: Contour points as (N, 2) array

        Returns:
            Tuple of (x1, y1, x2, y2)
        """
        if len(contour) == 0:
            return (0, 0, 0, 0)

        x_coords = contour[:, 0]
        y_coords = contour[:, 1]

        return (
            int(x_coords.min()),
            int(y_coords.min()),
            int(x_coords.max()),
            int(y_coords.max())
        )

    def _apply_random_augmentation(self, image: np.ndarray, probability: float = 0.5) -> np.ndarray:
        """Apply random augmentation effects to enhance training data diversity.

        This method applies various image augmentation techniques with configurable probability:
        - Brightness adjustment
        - Contrast adjustment
        - Gaussian blur
        - Gaussian noise
        - Hue/Saturation shifts

        Args:
            image: Input image (BGR format)
            probability: Probability of applying each augmentation (0.0-1.0)

        Returns:
            Augmented image (same shape and type as input)
        """
        result = image.copy()

        try:
            # Random brightness adjustment
            if random.random() < probability:
                brightness_factor = random.uniform(0.9, 1.1)
                result = cv2.convertScaleAbs(result, alpha=brightness_factor, beta=0)

            # Random contrast adjustment
            if random.random() < probability:
                contrast_factor = random.uniform(0.9, 1.1)
                result = cv2.convertScaleAbs(result, alpha=contrast_factor, beta=0)

            # Random Gaussian blur
            if random.random() < probability * 0.3:  # Lower probability for blur
                kernel_size = random.choice([3, 5])
                result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)

            # Random Gaussian noise
            if random.random() < probability * 0.2:  # Lower probability for noise
                noise = np.random.normal(0, 10, result.shape).astype(np.int16)
                result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Random hue/saturation shift
            if random.random() < probability * 0.3:
                hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180  # Hue
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)  # Saturation
                result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        except Exception as e:
            logger.warning(f"Error applying random augmentation: {e}")
            return image  # Return original on error

        return result

    def _extract_and_composite_object(
        self,
        aug_image: np.ndarray,
        obj_image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        segmentation: List[float],
        target_contour: np.ndarray,
        angle: float = 0.0,
        apply_augmentation: bool = False
    ) -> None:
        """Extract object from source image and composite onto target image with rotation and blending.

        This method performs sophisticated object extraction and compositing:
        1. Extracts the object region using the bounding box
        2. Creates a binary mask from YOLO normalized segmentation contours
        3. Applies rotation transformation (if angle != 0)
        4. Composites onto target image with smooth alpha blending
        5. Handles boundary clipping automatically
        6. Optionally applies color/brightness augmentation

        Coordinate Transform Pipeline:
        - YOLO segmentation (normalized [0,1]) → Image pixel coords → Bbox-relative coords
        - Rotation applied in bbox-relative space
        - Final placement calculated to align mask with target_contour

        Args:
            aug_image: Target image to composite onto (modified in-place, BGR format)
            obj_image: Source image containing the object (full frame, BGR format)
            bbox: Bounding box (x1, y1, x2, y2) in source image pixel coordinates
            segmentation: YOLO normalized segmentation [x1, y1, x2, y2, ...] in [0,1] range
            target_contour: Target contour position in augmented image, shape (N, 2), pixel coords
            angle: Rotation angle in degrees (default: 0.0, positive = counter-clockwise)
            apply_augmentation: Whether to apply random color/brightness augmentation

        Returns:
            None (modifies aug_image in-place)

        Raises:
            Logs warnings for invalid inputs but does not raise exceptions

        Example:
            >>> self._extract_and_composite_object(
            ...     aug_image=background,
            ...     obj_image=source_frame,
            ...     bbox=(100, 100, 200, 200),
            ...     segmentation=[0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9],
            ...     target_contour=np.array([[150, 150], [250, 150], [250, 250], [150, 250]]),
            ...     angle=45.0,
            ...     apply_augmentation=True
            ... )
        """
        try:
            # === Step 1: Extract object crop from source image ===
            x1, y1, x2, y2 = bbox

            # Validate bbox bounds
            if x1 < 0 or y1 < 0 or x2 > obj_image.shape[1] or y2 > obj_image.shape[0]:
                logger.warning(f"Bbox {bbox} out of bounds for image shape {obj_image.shape}, skipping")
                return

            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid bbox dimensions {bbox}, skipping")
                return

            obj_crop = obj_image[y1:y2, x1:x2].copy()

            if obj_crop.size == 0:
                logger.warning(f"Empty object crop for bbox {bbox}, skipping")
                return

            bbox_h, bbox_w = obj_crop.shape[:2]

            # === Step 2: Convert YOLO normalized segmentation to bbox-relative pixel coordinates ===
            if not segmentation or len(segmentation) < 6:
                logger.warning("Invalid segmentation data (need at least 3 points), skipping")
                return

            # Reshape to (N, 2) array of [x, y] points
            seg_array = np.array(segmentation).reshape(-1, 2)
            orig_img_h, orig_img_w = obj_image.shape[:2]

            # Step 2a: Denormalize from [0,1] to image-space pixel coordinates
            contour_img_space = seg_array * np.array([orig_img_w, orig_img_h], dtype=np.float32)

            # Step 2b: Convert to bbox-relative coordinates (origin at bbox top-left)
            contour_bbox_rel = contour_img_space - np.array([x1, y1], dtype=np.float32)

            # === Step 3: Create binary mask from contour ===
            mask = np.zeros((bbox_h, bbox_w), dtype=np.uint8)
            contour_int = contour_bbox_rel.astype(np.int32)

            # Validate contour is within bbox bounds
            if np.any(contour_int < 0) or np.any(contour_int[:, 0] >= bbox_w) or np.any(contour_int[:, 1] >= bbox_h):
                logger.debug(f"Contour extends outside bbox bounds, clipping")
                contour_int[:, 0] = np.clip(contour_int[:, 0], 0, bbox_w - 1)
                contour_int[:, 1] = np.clip(contour_int[:, 1], 0, bbox_h - 1)

            cv2.fillPoly(mask, [contour_int], 255)

            # === Step 4: Apply rotation if needed ===
            if abs(angle) > 0.01:  # Only rotate if angle is significant
                # Calculate rotation matrix around bbox center
                center = (bbox_w / 2.0, bbox_h / 2.0)
                rotation_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)

                # Calculate new bounding box size after rotation (to avoid clipping)
                cos = np.abs(rotation_mat[0, 0])
                sin = np.abs(rotation_mat[0, 1])
                new_w = int(np.ceil(bbox_h * sin + bbox_w * cos))
                new_h = int(np.ceil(bbox_h * cos + bbox_w * sin))

                # Adjust rotation matrix to account for new center
                rotation_mat[0, 2] += (new_w / 2.0) - center[0]
                rotation_mat[1, 2] += (new_h / 2.0) - center[1]

                # Apply rotation to both image and mask
                obj_crop = cv2.warpAffine(
                    obj_crop, rotation_mat, (new_w, new_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0)
                )
                mask = cv2.warpAffine(
                    mask, rotation_mat, (new_w, new_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )

                # Also transform the contour points to get their position in rotated bbox space
                # Append 1s for homogeneous coordinates
                ones = np.ones((len(contour_bbox_rel), 1), dtype=np.float32)
                contour_homogeneous = np.hstack([contour_bbox_rel, ones])
                contour_rotated_bbox_space = (rotation_mat @ contour_homogeneous.T).T
            else:
                # No rotation - contour stays the same
                contour_rotated_bbox_space = contour_bbox_rel.copy()

            # === Step 5: Apply color/brightness augmentation if requested ===
            if apply_augmentation:
                obj_crop = self._apply_random_augmentation(obj_crop, probability=0.5)

            # === Step 6: Calculate placement position ===
            # target_contour tells us where the contour should be in the final image
            # contour_rotated_bbox_space tells us where the contour is in the rotated bbox
            # We need to align them by calculating the offset

            target_centroid = target_contour.mean(axis=0)
            rotated_contour_centroid = contour_rotated_bbox_space.mean(axis=0)

            # Top-left corner of rotated bbox in final image coordinates
            # This aligns the rotated contour centroid with the target centroid
            top_left_x = int(np.round(target_centroid[0] - rotated_contour_centroid[0]))
            top_left_y = int(np.round(target_centroid[1] - rotated_contour_centroid[1]))

            # === Step 7: Composite with boundary clipping ===
            aug_h, aug_w = aug_image.shape[:2]
            rotated_h, rotated_w = obj_crop.shape[:2]

            # Calculate source region (in rotated object crop)
            src_x1 = max(0, -top_left_x)
            src_y1 = max(0, -top_left_y)
            src_x2 = min(rotated_w, aug_w - top_left_x)
            src_y2 = min(rotated_h, aug_h - top_left_y)

            # Calculate destination region (in augmented image)
            dst_x1 = max(0, top_left_x)
            dst_y1 = max(0, top_left_y)
            dst_x2 = min(aug_w, top_left_x + rotated_w)
            dst_y2 = min(aug_h, top_left_y + rotated_h)

            # Validate regions
            if src_x2 <= src_x1 or src_y2 <= src_y1 or dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
                logger.warning(
                    f"Object placement completely outside image bounds "
                    f"(top_left: {top_left_x}, {top_left_y}), skipping"
                )
                return

            # Extract overlapping regions
            obj_region = obj_crop[src_y1:src_y2, src_x1:src_x2]
            mask_region = mask[src_y1:src_y2, src_x1:src_x2]

            # === Step 8: Alpha blending for smooth composition ===
            # Convert mask to alpha channel [0, 1]
            alpha = mask_region.astype(np.float32) / 255.0

            # Expand to 3 channels for BGR image
            alpha_3ch = np.expand_dims(alpha, axis=2)

            # Get background and foreground as float32
            background = aug_image[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
            foreground = obj_region.astype(np.float32)

            # Blend: result = background * (1 - alpha) + foreground * alpha
            blended = background * (1.0 - alpha_3ch) + foreground * alpha_3ch

            # Write back to augmented image
            aug_image[dst_y1:dst_y2, dst_x1:dst_x2] = blended.astype(np.uint8)

            logger.debug(
                f"Composited object at ({top_left_x}, {top_left_y}) with "
                f"size ({rotated_w}, {rotated_h}), angle={angle:.1f}°"
            )

        except Exception as e:
            logger.error(f"Error in _extract_and_composite_object: {e}", exc_info=True)
            # Don't raise - just log and continue with other objects

    def _create_yolo_segmentation_label(self, class_name: str, contour: np.ndarray,
                                       img_w: int, img_h: int) -> str:
        """Create YOLO segmentation format label.

        Args:
            class_name: Object class name
            contour: Contour points in image pixel coordinates
            img_w: Image width
            img_h: Image height

        Returns:
            YOLO format label string: "class_id x1 y1 x2 y2 ..."
        """
        # Get class ID (for now, use simple index based on sorted labels)
        # In production, this should match the training dataset's class mapping
        all_objects = self.training_service.get_all_objects()
        class_labels = sorted(list(set(obj.label for obj in all_objects)))

        try:
            class_id = class_labels.index(class_name)
        except ValueError:
            class_id = 0
            logger.warning(f"Class {class_name} not found in class labels, using ID 0")

        # Normalize coordinates
        normalized = []
        for x, y in contour:
            norm_x = x / img_w
            norm_y = y / img_h

            # Clamp to [0, 1]
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))

            normalized.extend([norm_x, norm_y])

        points_str = ' '.join([f"{p:.6f}" for p in normalized])
        return f"{class_id} {points_str}"

    def _populate_generated_list(self):
        """Populate the generated images listbox."""
        self.images_listbox.delete(0, tk.END)

        for data in self.augmented_data:
            # Use the actual number of objects placed (which might differ from labels if placement failed)
            num_objects = data.get('num_objects', len(data['labels']))
            actual_placed = len(data['labels'])

            # Show both intended and actually placed if they differ
            if num_objects != actual_placed:
                display_text = f"{data['filename']} ({actual_placed}/{num_objects} objects placed)"
            else:
                display_text = f"{data['filename']} ({num_objects} objects)"

            self.images_listbox.insert(tk.END, display_text)

    def _on_image_selected(self, event):
        """Handle image selection from listbox - display with contours."""
        selection = self.images_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        self.current_preview_index = idx

        # Get augmented image data
        data = self.augmented_data[idx]

        # Draw contours on the image
        annotated_image = self._draw_contours_on_image(data['image'], data['labels'])

        # Display annotated image
        self._display_image(annotated_image)

        logger.info(f"Previewing augmented image with contours: {data['filename']}")

    def _draw_contours_on_image(self, image: np.ndarray, labels: List[str]) -> np.ndarray:
        """Draw object contours on image based on YOLO segmentation labels.

        Args:
            image: Original augmented image
            labels: List of YOLO segmentation labels (format: "class_id x1 y1 x2 y2 ...")

        Returns:
            Image with contours drawn
        """
        # Create a copy to draw on
        annotated = image.copy()
        h, w = image.shape[:2]

        # Get class names for labeling
        all_objects = self.training_service.get_all_objects()
        class_labels = sorted(list(set(obj.label for obj in all_objects)))

        # Define colors for different objects
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 255),  # Purple
            (255, 128, 0),  # Orange
        ]

        # Draw each object's contour
        for label_idx, label in enumerate(labels):
            try:
                # Parse YOLO segmentation label: "class_id x1 y1 x2 y2 x3 y3 ..."
                parts = label.strip().split()
                if len(parts) < 7:  # At least class_id + 3 points (6 coordinates)
                    continue

                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]

                # Convert normalized coordinates to pixel coordinates
                points = []
                for i in range(0, len(coords), 2):
                    x_norm = coords[i]
                    y_norm = coords[i + 1]

                    x_pixel = int(x_norm * w)
                    y_pixel = int(y_norm * h)

                    points.append([x_pixel, y_pixel])

                if len(points) < 3:
                    continue

                # Convert to numpy array for OpenCV
                contour = np.array(points, dtype=np.int32).reshape(-1, 1, 2)

                # Get color for this object
                color = colors[label_idx % len(colors)]

                # Draw filled semi-transparent contour
                # overlay = annotated.copy()
                # cv2.drawContours(overlay, [contour], -1, color, -1)
                # cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)

                # Draw contour outline (thicker)
                cv2.drawContours(annotated, [contour], -1, color, 3)

                # Get class name
                class_name = class_labels[class_id] if class_id < len(class_labels) else f"Class {class_id}"

                # Draw label near contour
                bbox = cv2.boundingRect(contour)
                label_pos = (bbox[0], max(bbox[1] - 10, 20))

                # Draw text background
                (text_w, text_h), baseline = cv2.getTextSize(
                    class_name,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    2
                )
                cv2.rectangle(
                    annotated,
                    (label_pos[0] - 2, label_pos[1] - text_h - 2),
                    (label_pos[0] + text_w + 2, label_pos[1] + baseline),
                    color,
                    -1
                )

                # Draw text
                cv2.putText(
                    annotated,
                    class_name,
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),  # White text
                    2
                )

            except Exception as e:
                logger.error(f"Error drawing contour for label '{label}': {e}")
                continue

        return annotated

    def _display_image(self, image: np.ndarray):
        """Display image on canvas, scaled to fit.

        Args:
            image: Image to display (BGR format)
        """
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = image_rgb.shape[:2]

            # Get canvas dimensions
            self.dialog.update_idletasks()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # Fallback dimensions
            if canvas_width < 10:
                canvas_width = 1000
            if canvas_height < 10:
                canvas_height = 800

            # Calculate scale factor
            scale_w = canvas_width / orig_w
            scale_h = canvas_height / orig_h
            scale_factor = min(scale_w, scale_h, 1.0)

            # Calculate display dimensions
            display_width = int(orig_w * scale_factor)
            display_height = int(orig_h * scale_factor)

            # Resize image
            display_img = cv2.resize(
                image_rgb,
                (display_width, display_height),
                interpolation=cv2.INTER_AREA
            )

            # Convert to PIL and create PhotoImage
            pil_image = Image.fromarray(display_img)
            self.photo = ImageTk.PhotoImage(image=pil_image)

            # Clear canvas and display image centered
            self.canvas.delete('all')

            x_offset = max(0, (canvas_width - display_width) // 2)
            y_offset = max(0, (canvas_height - display_height) // 2)

            self.canvas.create_image(
                x_offset, y_offset,
                anchor='nw',
                image=self.photo,
                tags='image'
            )

        except Exception as e:
            logger.error(f"Error displaying image: {e}")

    def _on_save_all(self):
        """Save all augmented images and labels to disk.

        After saving, this method automatically adds all generated objects to the
        training service, making them immediately available for model training.

        This method displays a progress window showing real-time updates as
        each image is processed and objects are added to the training service.
        """
        if not self.augmented_data:
            messagebox.showwarning(
                "No Data",
                "No augmented images to save.",
                parent=self.dialog
            )
            return

        # # Ask user to select output directory
        # output_dir = filedialog.askdirectory(
        #     title="Select Output Directory for Augmented Data",
        #     parent=self.dialog
        # )

        # if not output_dir:
        #     return

        # Create save progress window
        progress_window = self._create_save_progress_window(len(self.augmented_data))

        try:
            # # Create subdirectories
            # images_dir = os.path.join(output_dir, 'images')
            # labels_dir = os.path.join(output_dir, 'labels')
            # os.makedirs(images_dir, exist_ok=True)
            # os.makedirs(labels_dir, exist_ok=True)

            # # Save each augmented image and its labels
            # for data in self.augmented_data:
            #     # Save image
            #     img_path = os.path.join(images_dir, data['filename'])
            #     cv2.imwrite(img_path, data['image'])

            #     # Save labels
            #     label_filename = data['filename'].replace('.jpg', '.txt')
            #     label_path = os.path.join(labels_dir, label_filename)

            #     with open(label_path, 'w') as f:
            #         for label in data['labels']:
            #             f.write(label + '\n')

            # logger.info(f"Saved {len(self.augmented_data)} augmented images to {output_dir}")

            # Auto-add to training objects with progress tracking
            total_objects_added = self._add_to_training_objects(progress_window=progress_window)

            # Close progress window before showing success message
            if progress_window:
                progress_window.destroy()
                progress_window = None

            # Show enhanced success message with detailed counts and batch info
            num_images = len(self.augmented_data)
            messagebox.showinfo(
                "Successfully Saved",
                f"Successfully saved!\n\n"
                f"Batch: {self.batch_timestamp}\n"
                f"Added {total_objects_added} objects from {num_images} images to training list.",
                parent=self.dialog
            )

            logger.info(
                f"Save operation completed for batch {self.batch_timestamp}: "
                f"{total_objects_added} objects from {num_images} images added to training service"
            )

        except Exception as e:
            logger.error(f"Error saving augmented data: {e}", exc_info=True)
            messagebox.showerror(
                "Save Error",
                f"Failed to save augmented data:\n{e}",
                parent=self.dialog
            )

        finally:
            # Ensure progress window is always destroyed
            if progress_window:
                try:
                    progress_window.destroy()
                except Exception as e:
                    logger.warning(f"Error destroying save progress window: {e}")

    def _add_to_training_objects(self, progress_window: Optional[tk.Toplevel] = None) -> int:
        """Add all generated augmented objects to the training service.

        This method:
        1. Iterates through all augmented images and their labels
        2. Parses YOLO segmentation format labels
        3. Extracts bbox coordinates from segmentation points
        4. Adds each object to the training service
        5. Updates progress window if provided

        Args:
            progress_window: Optional Toplevel window for progress tracking.
                           If provided, updates will be displayed after processing each image.

        Returns:
            Total number of objects added to training service
        """
        total_objects_added = 0

        try:
            # Get class labels mapping
            all_objects = self.training_service.get_all_objects()
            class_labels = sorted(list(set(obj.label for obj in all_objects)))

            if not class_labels:
                logger.warning("No class labels found in training service")
                return 0

            logger.info(f"Processing augmented images to add to training service...")

            # Iterate through each augmented image
            for data_idx, data in enumerate(self.augmented_data):
                image = data['image']
                labels = data['labels']
                filename = data['filename']

                img_h, img_w = image.shape[:2]

                # Process each object in this image
                for label_idx, label_str in enumerate(labels):
                    try:
                        # Parse YOLO segmentation label: "class_id x1 y1 x2 y2 x3 y3 ..."
                        parts = label_str.strip().split()
                        if len(parts) < 7:  # Need at least class_id + 3 points (6 coords)
                            logger.warning(f"Invalid label format in {filename}: {label_str}")
                            continue

                        # Extract class ID and coordinates
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]

                        # Validate class ID
                        if class_id >= len(class_labels):
                            logger.warning(f"Invalid class ID {class_id} in {filename}")
                            continue

                        class_name = class_labels[class_id]

                        # Convert normalized coordinates to pixel coordinates
                        pixel_coords = []
                        for i in range(0, len(coords), 2):
                            x_pixel = coords[i] * img_w
                            y_pixel = coords[i + 1] * img_h
                            pixel_coords.extend([x_pixel, y_pixel])

                        # Calculate bounding box from segmentation points
                        x_points = [pixel_coords[i] for i in range(0, len(pixel_coords), 2)]
                        y_points = [pixel_coords[i] for i in range(1, len(pixel_coords), 2)]

                        bbox_x1 = int(min(x_points))
                        bbox_y1 = int(min(y_points))
                        bbox_x2 = int(max(x_points))
                        bbox_y2 = int(max(y_points))

                        # Clamp bbox to image bounds
                        bbox_x1 = max(0, bbox_x1)
                        bbox_y1 = max(0, bbox_y1)
                        bbox_x2 = min(img_w - 1, bbox_x2)
                        bbox_y2 = min(img_h - 1, bbox_y2)

                        bbox = (bbox_x1, bbox_y1, bbox_x2, bbox_y2)

                        # Normalize segmentation coordinates back to [0, 1] for storage
                        normalized_segmentation = coords  # Already normalized

                        # Add to training service
                        # Use a unique image_id based on the augmented filename
                        image_id = f"aug_{filename.replace('.jpg', '')}"

                        self.training_service.add_object(
                            image=image,
                            label=class_name,
                            bbox=bbox,
                            background_region=None,  # No background region for augmented images
                            segmentation=normalized_segmentation,
                            threshold=None,  # No threshold for augmented images
                            image_id=image_id
                        )

                        total_objects_added += 1

                        logger.debug(
                            f"Added augmented object {total_objects_added}: "
                            f"{class_name} from {filename} at bbox {bbox}"
                        )

                    except Exception as e:
                        logger.error(
                            f"Error adding object from {filename} label {label_idx}: {e}",
                            exc_info=True
                        )
                        continue

                # Update progress window after processing each image
                if progress_window:
                    try:
                        current = data_idx + 1
                        total = len(self.augmented_data)
                        progress_window.status_label.config(
                            text=f"Processing {current}/{total} images..."
                        )
                        progress_window.progress_bar['value'] = current
                        percent = int((current / total) * 100)
                        progress_window.percent_label.config(text=f"{percent}%")
                        progress_window.update()
                    except Exception as e:
                        logger.warning(f"Error updating save progress window: {e}")

                # Log progress every 10 images
                if (data_idx + 1) % 10 == 0 or (data_idx + 1) == len(self.augmented_data):
                    logger.info(
                        f"Processed {data_idx + 1}/{len(self.augmented_data)} images, "
                        f"added {total_objects_added} objects so far"
                    )

            logger.info(
                f"Successfully added {total_objects_added} objects from "
                f"{len(self.augmented_data)} augmented images to training service"
            )

        except Exception as e:
            logger.error(f"Error in _add_to_training_objects: {e}", exc_info=True)

        return total_objects_added

    def _on_close(self):
        """Handle dialog close and cleanup event bindings."""
        # Unbind mousewheel event to prevent memory leaks
        try:
            if hasattr(self, 'objects_scroll_canvas'):
                self.objects_scroll_canvas.unbind_all('<MouseWheel>')
        except Exception as e:
            logger.debug(f"Error unbinding mousewheel during close: {e}")

        self.dialog.destroy()

    def show(self):
        """Show dialog modally."""
        self.dialog.wait_window()
