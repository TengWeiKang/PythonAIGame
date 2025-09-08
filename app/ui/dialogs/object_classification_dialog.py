"""Object Classification Settings dialog with GrabCut object extraction."""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import threading
import time
import queue
from typing import Optional, List, Tuple

from ...config.settings import Config
from ...services.webcam_service import WebcamService
from ...services.training_service import TrainingService
from ...core.exceptions import WebcamError

class ObjectClassificationDialog:
    """Dialog for object classification with GrabCut extraction."""
    
    def __init__(self, parent, config: Config, webcam_service: WebcamService):
        self.parent = parent
        self.config = config
        self.webcam_service = webcam_service
        self.training_service = TrainingService(config)
        
        # State variables
        self.current_frame = None
        self.capture_frame = None
        self.extracted_objects = []  # List of {name: str, image: np.ndarray, bbox: tuple}
        self.preview_running = False
        self.preview_cap = None
        
        # GrabCut variables
        self.grabcut_mask = None
        self.rect_start = None
        self.rect_end = None
        self.drawing = False
        
        # Create dialog window
        self.window = tk.Toplevel(parent)
        self.window.title("Object Classification Settings")
        self.window.geometry("900x700")
        self.window.resizable(True, True)
        
        # Make it modal
        self.window.transient(parent)
        self.window.grab_set()
        
        # Center on parent
        self._center_window()
        
        # Build UI
        self._build_ui()
        
        # Setup cleanup
        self.window.protocol('WM_DELETE_WINDOW', self._on_close)

    def _center_window(self):
        """Center the dialog on the parent window."""
        self.window.update_idletasks()
        x = self.parent.winfo_x() + (self.parent.winfo_width() // 2) - (self.window.winfo_width() // 2)
        y = self.parent.winfo_y() + (self.parent.winfo_height() // 2) - (self.window.winfo_height() // 2)
        self.window.geometry(f"+{x}+{y}")

    def _build_ui(self):
        """Build the dialog UI."""
        # Main container
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Object Classification Settings", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Create paned window for left/right layout
        paned = ttk.PanedWindow(main_frame, orient='horizontal')
        paned.pack(fill='both', expand=True)
        
        # Left panel - Camera preview and controls
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=2)
        
        self._build_camera_panel(left_frame)
        
        # Right panel - Captured objects and training
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        
        self._build_objects_panel(right_frame)
        
        # Bottom buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(button_frame, text="Train Model", 
                  command=self._train_model).pack(side='left', padx=(0, 10))
        ttk.Button(button_frame, text="Clear All Objects", 
                  command=self._clear_all_objects).pack(side='left', padx=(0, 10))
        ttk.Button(button_frame, text="Close", 
                  command=self._on_close).pack(side='right')

    def _build_camera_panel(self, parent):
        """Build the camera preview panel."""
        # Camera controls
        controls_frame = ttk.LabelFrame(parent, text="Camera Controls", padding="5")
        controls_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Button(controls_frame, text="Start Preview", 
                  command=self._start_preview).pack(side='left', padx=(0, 5))
        ttk.Button(controls_frame, text="Stop Preview", 
                  command=self._stop_preview).pack(side='left', padx=(0, 5))
        ttk.Button(controls_frame, text="Capture Image", 
                  command=self._capture_image).pack(side='left', padx=(0, 5))
        
        # Camera preview
        preview_frame = ttk.LabelFrame(parent, text="Camera Preview", padding="5")
        preview_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Create canvas for camera preview
        self.preview_canvas = tk.Canvas(preview_frame, width=640, height=480, 
                                       bg='black', cursor='crosshair')
        self.preview_canvas.pack(expand=True)
        
        # Bind mouse events for GrabCut rectangle selection
        self.preview_canvas.bind('<Button-1>', self._on_mouse_press)
        self.preview_canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self.preview_canvas.bind('<ButtonRelease-1>', self._on_mouse_release)
        
        # Instructions
        instructions = ttk.Label(preview_frame, 
                                text="1. Start Preview  2. Capture Image  3. Draw rectangle around object  4. Enter object name",
                                font=('Arial', 9))
        instructions.pack(pady=(5, 0))

    def _build_objects_panel(self, parent):
        """Build the captured objects panel."""
        # Captured objects list
        objects_frame = ttk.LabelFrame(parent, text="Captured Objects", padding="5")
        objects_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Listbox with scrollbar
        listbox_frame = ttk.Frame(objects_frame)
        listbox_frame.pack(fill='both', expand=True)
        
        self.objects_listbox = tk.Listbox(listbox_frame, selectmode=tk.SINGLE)
        scrollbar = ttk.Scrollbar(listbox_frame, orient='vertical', 
                                 command=self.objects_listbox.yview)
        self.objects_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.objects_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Object management buttons
        obj_buttons_frame = ttk.Frame(objects_frame)
        obj_buttons_frame.pack(fill='x', pady=(5, 0))
        
        ttk.Button(obj_buttons_frame, text="Delete Selected", 
                  command=self._delete_selected_object).pack(side='left', padx=(0, 5))
        ttk.Button(obj_buttons_frame, text="Preview Object", 
                  command=self._preview_selected_object).pack(side='left')
        
        # Training progress
        progress_frame = ttk.LabelFrame(parent, text="Training Progress", padding="5")
        progress_frame.pack(fill='x')
        
        self.progress_var = tk.StringVar(value="Ready to train")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.pack(fill='x', pady=(5, 0))

    def _start_preview(self):
        """Start camera preview."""
        if self.preview_running:
            return
            
        try:
            # Try to use the existing webcam service or create new capture
            if self.webcam_service.is_opened():
                # Use existing webcam connection
                self.preview_cap = self.webcam_service
            else:
                # Create new capture
                self.preview_cap = WebcamService()
                success = self.preview_cap.open(
                    self.config.last_webcam_index,
                    self.config.camera_width,
                    self.config.camera_height,
                    self.config.camera_fps
                )
                if not success:
                    messagebox.showerror("Error", "Failed to open camera")
                    return
            
            self.preview_running = True
            self._update_preview()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start preview: {e}")

    def _stop_preview(self):
        """Stop camera preview."""
        self.preview_running = False
        if self.preview_cap and self.preview_cap != self.webcam_service:
            self.preview_cap.close()
        self.preview_cap = None
        
        # Clear canvas
        self.preview_canvas.delete('all')

    def _update_preview(self):
        """Update camera preview."""
        if not self.preview_running or not self.preview_cap:
            return
            
        try:
            ret, frame = self.preview_cap.read()
            if ret and frame is not None:
                self.current_frame = frame.copy()
                
                # Resize frame to fit canvas
                canvas_width = self.preview_canvas.winfo_width()
                canvas_height = self.preview_canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    # Calculate scaling to fit canvas
                    h, w = frame.shape[:2]
                    scale = min(canvas_width/w, canvas_height/h)
                    
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    
                    if new_w > 0 and new_h > 0:
                        resized_frame = cv2.resize(frame, (new_w, new_h))
                        
                        # Convert to RGB and create PhotoImage
                        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(rgb_frame)
                        photo = ImageTk.PhotoImage(pil_image)
                        
                        # Update canvas
                        self.preview_canvas.delete('video')
                        self.preview_canvas.create_image(
                            canvas_width//2, canvas_height//2,
                            anchor='center', image=photo, tags='video'
                        )
                        self.preview_canvas.image = photo  # Keep reference
            
            # Schedule next update
            if self.preview_running:
                self.window.after(33, self._update_preview)  # ~30 FPS
                
        except Exception as e:
            print(f"Preview update error: {e}")
            if self.preview_running:
                self.window.after(100, self._update_preview)

    def _capture_image(self):
        """Capture current frame for object extraction."""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No camera preview available. Start preview first.")
            return
            
        self.capture_frame = self.current_frame.copy()
        
        # Display captured frame on canvas
        self._display_captured_frame()
        
        messagebox.showinfo("Captured", "Image captured! Now draw a rectangle around an object to extract it.")

    def _display_captured_frame(self):
        """Display the captured frame on canvas."""
        if self.capture_frame is None:
            return
            
        # Get canvas dimensions
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 640, 480
            
        # Resize frame to fit canvas
        h, w = self.capture_frame.shape[:2]
        scale = min(canvas_width/w, canvas_height/h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        if new_w > 0 and new_h > 0:
            resized_frame = cv2.resize(self.capture_frame, (new_w, new_h))
            
            # Convert to RGB and create PhotoImage
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Store scale for coordinate conversion
            self.canvas_scale = scale
            self.canvas_offset_x = (canvas_width - new_w) // 2
            self.canvas_offset_y = (canvas_height - new_h) // 2
            
            # Update canvas
            self.preview_canvas.delete('all')
            self.preview_canvas.create_image(
                canvas_width//2, canvas_height//2,
                anchor='center', image=photo, tags='captured'
            )
            self.preview_canvas.image = photo  # Keep reference

    def _on_mouse_press(self, event):
        """Handle mouse press for rectangle selection."""
        if self.capture_frame is None:
            return
            
        self.rect_start = (event.x, event.y)
        self.drawing = True
        
        # Remove any existing rectangle
        self.preview_canvas.delete('selection_rect')

    def _on_mouse_drag(self, event):
        """Handle mouse drag for rectangle selection."""
        if not self.drawing or self.rect_start is None:
            return
            
        # Remove previous rectangle
        self.preview_canvas.delete('selection_rect')
        
        # Draw new rectangle
        self.preview_canvas.create_rectangle(
            self.rect_start[0], self.rect_start[1],
            event.x, event.y,
            outline='red', width=2, tags='selection_rect'
        )

    def _on_mouse_release(self, event):
        """Handle mouse release to complete rectangle selection."""
        if not self.drawing or self.rect_start is None:
            return
            
        self.drawing = False
        self.rect_end = (event.x, event.y)
        
        # Check if rectangle is large enough
        if (abs(self.rect_end[0] - self.rect_start[0]) < 10 or 
            abs(self.rect_end[1] - self.rect_start[1]) < 10):
            messagebox.showwarning("Warning", "Rectangle too small. Please draw a larger selection.")
            return
            
        # Convert canvas coordinates to image coordinates
        self._extract_object_with_grabcut()

    def _extract_object_with_grabcut(self):
        """Extract object using GrabCut algorithm."""
        if self.capture_frame is None or self.rect_start is None or self.rect_end is None:
            return
            
        try:
            # Convert canvas coordinates to image coordinates
            x1 = int((self.rect_start[0] - self.canvas_offset_x) / self.canvas_scale)
            y1 = int((self.rect_start[1] - self.canvas_offset_y) / self.canvas_scale)
            x2 = int((self.rect_end[0] - self.canvas_offset_x) / self.canvas_scale)
            y2 = int((self.rect_end[1] - self.canvas_offset_y) / self.canvas_scale)
            
            # Ensure coordinates are within image bounds
            h, w = self.capture_frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            
            # Ensure rectangle is valid
            if x2 <= x1 or y2 <= y1:
                messagebox.showerror("Error", "Invalid selection rectangle")
                return
            
            # Create rectangle for GrabCut (x, y, width, height)
            rect = (x1, y1, x2-x1, y2-y1)
            
            # Initialize mask
            mask = np.zeros(self.capture_frame.shape[:2], np.uint8)
            
            # GrabCut algorithm
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Apply GrabCut
            cv2.grabCut(self.capture_frame, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create final mask
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # Extract object
            result = self.capture_frame * mask2[:, :, np.newaxis]
            
            # Get bounding box of the extracted object
            coords = np.where(mask2)
            if len(coords[0]) == 0 or len(coords[1]) == 0:
                messagebox.showerror("Error", "No object found in selection")
                return
                
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Crop to object bounds
            cropped_object = result[y_min:y_max+1, x_min:x_max+1]
            
            # Ask for object name
            object_name = simpledialog.askstring("Object Name", 
                                                "Enter name for the extracted object:",
                                                parent=self.window)
            
            if object_name and object_name.strip():
                # Store the extracted object
                self.extracted_objects.append({
                    'name': object_name.strip(),
                    'image': cropped_object,
                    'bbox': (x_min, y_min, x_max, y_max),
                    'full_image': self.capture_frame.copy(),
                    'mask': mask2
                })
                
                # Update objects list
                self._update_objects_list()
                
                messagebox.showinfo("Success", f"Object '{object_name}' extracted successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract object: {e}")

    def _update_objects_list(self):
        """Update the objects listbox."""
        self.objects_listbox.delete(0, tk.END)
        
        for i, obj in enumerate(self.extracted_objects):
            self.objects_listbox.insert(tk.END, f"{i+1}. {obj['name']}")

    def _delete_selected_object(self):
        """Delete selected object from the list."""
        selection = self.objects_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select an object to delete")
            return
            
        index = selection[0]
        object_name = self.extracted_objects[index]['name']
        
        if messagebox.askyesno("Confirm Delete", f"Delete object '{object_name}'?"):
            del self.extracted_objects[index]
            self._update_objects_list()

    def _preview_selected_object(self):
        """Preview selected object in a new window."""
        selection = self.objects_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select an object to preview")
            return
            
        index = selection[0]
        obj = self.extracted_objects[index]
        
        # Create preview window
        preview_window = tk.Toplevel(self.window)
        preview_window.title(f"Preview: {obj['name']}")
        preview_window.geometry("400x300")
        
        # Display object image
        if obj['image'] is not None and obj['image'].size > 0:
            # Convert and resize for display
            rgb_image = cv2.cvtColor(obj['image'], cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Resize to fit preview window
            pil_image.thumbnail((350, 250), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_image)
            
            label = tk.Label(preview_window, image=photo)
            label.image = photo  # Keep reference
            label.pack(expand=True)
            
            ttk.Label(preview_window, text=f"Object: {obj['name']}", 
                     font=('Arial', 12, 'bold')).pack(pady=5)

    def _clear_all_objects(self):
        """Clear all extracted objects."""
        if not self.extracted_objects:
            messagebox.showinfo("Info", "No objects to clear")
            return
            
        if messagebox.askyesno("Confirm Clear", "Clear all extracted objects?"):
            self.extracted_objects.clear()
            self._update_objects_list()

    def _train_model(self):
        """Train model with extracted objects."""
        if not self.extracted_objects:
            messagebox.showwarning("Warning", "No objects captured. Capture some objects first.")
            return
            
        # Start training in background thread
        self.progress_var.set("Preparing training data...")
        self.progress_bar.start(10)
        
        # Create training thread
        training_thread = threading.Thread(target=self._train_model_thread, daemon=True)
        training_thread.start()

    def _train_model_thread(self):
        """Train model in background thread."""
        try:
            # Prepare training data
            self._update_progress("Saving training images...")
            
            # Create directories
            data_dir = self.config.data_dir
            images_dir = os.path.join(data_dir, 'images')
            labels_dir = os.path.join(data_dir, 'labels')
            
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            
            # Get unique class names
            class_names = list(set(obj['name'] for obj in self.extracted_objects))
            
            # Save class names
            classes_file = os.path.join(labels_dir, 'classes.json')
            import json
            with open(classes_file, 'w') as f:
                json.dump(class_names, f)
            
            # Save training data
            for i, obj in enumerate(self.extracted_objects):
                timestamp = int(time.time() * 1000)
                filename = f"{obj['name']}_{timestamp}_{i}"
                
                # Save image
                image_path = os.path.join(images_dir, f"{filename}.jpg")
                cv2.imwrite(image_path, obj['full_image'])
                
                # Create YOLO format label
                img_h, img_w = obj['full_image'].shape[:2]
                x_min, y_min, x_max, y_max = obj['bbox']
                
                # Convert to YOLO format (class_id, center_x, center_y, width, height)
                class_id = class_names.index(obj['name'])
                center_x = (x_min + x_max) / 2 / img_w
                center_y = (y_min + y_max) / 2 / img_h
                width = (x_max - x_min) / img_w
                height = (y_max - y_min) / img_h
                
                # Save label
                label_path = os.path.join(labels_dir, f"{filename}.txt")
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
            
            # Start training
            self._update_progress("Starting model training...")
            
            success = self.training_service.train_model(
                dataset_config_path=None,  # Will use data_dir directly
                epochs=self.config.train_epochs,
                batch_size=self.config.batch_size,
                progress_callback=self._training_progress_callback
            )
            
            if success:
                self._update_progress("Training completed successfully!")
                self.window.after(0, lambda: messagebox.showinfo("Success", "Model training completed!"))
            else:
                self._update_progress("Training failed or was interrupted")
                self.window.after(0, lambda: messagebox.showerror("Error", "Training failed"))
                
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            self._update_progress(f"Training error: {str(e)}")
            self.window.after(0, lambda msg=error_msg: messagebox.showerror("Error", msg))
        
        finally:
            # Stop progress bar
            self.window.after(0, self.progress_bar.stop)

    def _training_progress_callback(self, message: str, progress: float):
        """Callback for training progress updates."""
        self._update_progress(message)

    def _update_progress(self, message: str):
        """Update progress message thread-safely."""
        def update():
            self.progress_var.set(message)
        
        try:
            self.window.after(0, update)
        except:
            pass  # Window might be closed

    def _on_close(self):
        """Handle window close."""
        # Stop preview
        self._stop_preview()
        
        # Close window
        self.window.grab_release()
        self.window.destroy()

    def winfo_exists(self):
        """Check if dialog exists."""
        try:
            return self.window.winfo_exists()
        except tk.TclError:
            return False

    def lift(self):
        """Bring dialog to front."""
        self.window.lift()