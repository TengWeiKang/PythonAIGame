"""Webcam settings dialog."""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

from ...config.settings import Config
from ...services.webcam_service import WebcamService

class WebcamDialog:
    """Dialog for webcam configuration."""
    
    def __init__(self, parent, config: Config, webcam_service: WebcamService):
        self.parent = parent
        self.config = config
        self.webcam_service = webcam_service
        
        # Create dialog window
        self.window = tk.Toplevel(parent)
        self.window.title("Webcam Settings")
        self.window.geometry("500x400")
        self.window.resizable(False, False)
        
        # Make it modal
        self.window.transient(parent)
        self.window.grab_set()
        
        # Center on parent
        self._center_window()
        
        # Build UI
        self._build_ui()
        
        # Load current settings
        self._load_settings()

    def _center_window(self):
        """Center the dialog on the parent window."""
        self.window.update_idletasks()
        x = self.parent.winfo_x() + (self.parent.winfo_width() // 2) - (self.window.winfo_width() // 2)
        y = self.parent.winfo_y() + (self.parent.winfo_height() // 2) - (self.window.winfo_height() // 2)
        self.window.geometry(f"+{x}+{y}")

    def _build_ui(self):
        """Build the dialog UI."""
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Device selection
        device_frame = ttk.LabelFrame(main_frame, text="Camera Device", padding="5")
        device_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(device_frame, text="Select Camera:").pack(anchor='w')
        
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(device_frame, textvariable=self.device_var, state='readonly')
        self.device_combo.pack(fill='x', pady=(5, 0))
        
        # Refresh button
        ttk.Button(device_frame, text="Refresh Devices", command=self._refresh_devices).pack(pady=(5, 0))
        
        # Camera properties
        props_frame = ttk.LabelFrame(main_frame, text="Camera Properties", padding="5")
        props_frame.pack(fill='x', pady=(0, 10))
        
        # Resolution
        res_frame = ttk.Frame(props_frame)
        res_frame.pack(fill='x', pady=2)
        
        ttk.Label(res_frame, text="Resolution:").pack(side='left')
        
        self.width_var = tk.StringVar(value=str(self.config.camera_width))
        width_spin = tk.Spinbox(res_frame, from_=320, to=4096, increment=160, 
                               textvariable=self.width_var, width=8)
        width_spin.pack(side='left', padx=(10, 5))
        
        ttk.Label(res_frame, text="x").pack(side='left')
        
        self.height_var = tk.StringVar(value=str(self.config.camera_height))
        height_spin = tk.Spinbox(res_frame, from_=240, to=4096, increment=120, 
                                textvariable=self.height_var, width=8)
        height_spin.pack(side='left', padx=(5, 0))
        
        # FPS
        fps_frame = ttk.Frame(props_frame)
        fps_frame.pack(fill='x', pady=2)
        
        ttk.Label(fps_frame, text="FPS:").pack(side='left')
        
        self.fps_var = tk.StringVar(value=str(self.config.camera_fps))
        fps_spin = tk.Spinbox(fps_frame, from_=5, to=60, increment=5, 
                             textvariable=self.fps_var, width=8)
        fps_spin.pack(side='left', padx=(10, 0))
        
        # Preview frame
        preview_frame = ttk.LabelFrame(main_frame, text="Camera Preview", padding="5")
        preview_frame.pack(fill='x', pady=(0, 10))
        
        # Create canvas for video preview with fixed size
        self.preview_canvas = tk.Canvas(preview_frame, width=400, height=250, 
                                       bg='black', relief='sunken', bd=2)
        self.preview_canvas.pack()
        
        # Display placeholder text
        self.preview_canvas.create_text(
            200, 125, 
            text="Camera Preview\nSelect camera and click 'Test Camera' to preview",
            fill='white', 
            font=('Arial', 12),
            justify='center'
        )
        
        # Initialize preview state
        self.test_running = False
        self.test_cap = None
        
        # Control buttons
        button_frame = ttk.Frame(preview_frame)
        button_frame.pack(fill='x', pady=(5, 0))
        
        ttk.Button(button_frame, text="Test Camera", command=self._test_camera).pack(side='left', padx=(0, 5))
        ttk.Button(button_frame, text="Stop Test", command=self._stop_test).pack(side='left')
        
        # Dialog buttons
        dialog_buttons = ttk.Frame(main_frame)
        dialog_buttons.pack(fill='x', pady=(10, 0))
        
        ttk.Button(dialog_buttons, text="OK", command=self._on_ok).pack(side='right', padx=(5, 0))
        ttk.Button(dialog_buttons, text="Cancel", command=self._on_cancel).pack(side='right')
        ttk.Button(dialog_buttons, text="Apply", command=self._on_apply).pack(side='right', padx=(0, 5))

    def _load_settings(self):
        """Load current webcam settings."""
        self._refresh_devices()
        
        # Set current device if available (convert from 0-based to 1-based display)
        current_device = self.config.last_webcam_index
        devices = self.device_combo['values']
        
        if devices and devices[0] != "No cameras found":
            # Find the device with the matching index
            target_display = f"Camera {current_device + 1}:"
            for i, device_text in enumerate(devices):
                if device_text.startswith(target_display):
                    self.device_combo.current(i)
                    break
            else:
                # If not found, select first device
                if devices:
                    self.device_combo.current(0)

    def _refresh_devices(self):
        """Refresh the list of available webcam devices."""
        try:
            devices = WebcamService.list_devices()
            device_list = []
            
            for idx, name in devices:
                # Show 1-based indexing to user, but store 0-based internally
                display_num = idx + 1
                # Clean up camera name for better display
                if "USB" in name.upper():
                    clean_name = f"USB Camera ({name[:40]}...)" if len(name) > 40 else f"USB Camera ({name})"
                elif "INTEGRATED" in name.upper() or "BUILT" in name.upper():
                    clean_name = "Built-in Camera"
                else:
                    clean_name = name[:50] + "..." if len(name) > 50 else name
                
                device_list.append(f"Camera {display_num}: {clean_name}")
            
            if not device_list:
                device_list = ["No cameras found"]
            
            self.device_combo['values'] = device_list
            
            if device_list and device_list[0] != "No cameras found":
                self.device_combo.current(0)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh devices: {e}")

    def _test_camera(self):
        """Test the selected camera with live video preview."""
        try:
            # Get selected device index
            selection = self.device_var.get()
            if not selection or "No cameras found" in selection:
                messagebox.showwarning("Warning", "No camera selected")
                return
            
            # Parse device index (convert from 1-based display to 0-based internal)
            if "Camera " in selection:
                display_num = int(selection.split("Camera ")[1].split(":")[0])
                device_idx = display_num - 1  # Convert to 0-based
            else:
                device_idx = 0
            
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            fps = int(self.fps_var.get())
            
            # Start test preview
            if not self.test_running:
                self.test_cap = WebcamService()
                if self.test_cap.open(device_idx, width, height, fps):
                    self.test_running = True
                    self._update_test_preview()
                    messagebox.showinfo("Test Started", f"Camera {display_num} test started. Click 'Stop Test' to end.")
                else:
                    messagebox.showerror("Error", f"Failed to open camera {display_num}")
                    self.test_cap = None
            else:
                messagebox.showinfo("Info", "Test already running. Stop current test first.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Camera test failed: {e}")

    def _stop_test(self):
        """Stop camera test."""
        self.test_running = False
        if self.test_cap:
            self.test_cap.close()
            self.test_cap = None
        
        # Clear canvas
        self.preview_canvas.delete('all')
        self.preview_canvas.create_text(
            200, 150, text="Test stopped", 
            fill='white', font=('Arial', 12)
        )

    def _update_test_preview(self):
        """Update the test camera preview."""
        if not self.test_running or not self.test_cap:
            return
            
        try:
            ret, frame = self.test_cap.read()
            if ret and frame is not None:
                # Get canvas dimensions
                canvas_width = self.preview_canvas.winfo_width()
                canvas_height = self.preview_canvas.winfo_height()
                
                if canvas_width <= 1 or canvas_height <= 1:
                    canvas_width, canvas_height = 400, 300
                
                # Resize frame to fit canvas
                h, w = frame.shape[:2]
                scale = min(canvas_width/w, canvas_height/h)
                
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                if new_w > 0 and new_h > 0:
                    resized_frame = cv2.resize(frame, (new_w, new_h))
                    
                    # Convert to RGB and create PhotoImage
                    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                    from PIL import Image, ImageTk
                    pil_image = Image.fromarray(rgb_frame)
                    photo = ImageTk.PhotoImage(pil_image)
                    
                    # Update canvas
                    self.preview_canvas.delete('all')
                    self.preview_canvas.create_image(
                        canvas_width//2, canvas_height//2,
                        anchor='center', image=photo
                    )
                    self.preview_canvas.image = photo  # Keep reference
                    
                    # Add resolution info
                    self.preview_canvas.create_text(
                        10, 10, text=f"Resolution: {w}x{h}", 
                        anchor='nw', fill='yellow', font=('Arial', 10, 'bold')
                    )
            
            # Schedule next update
            if self.test_running:
                self.window.after(33, self._update_test_preview)  # ~30 FPS
                
        except Exception as e:
            print(f"Preview update error: {e}")
            if self.test_running:
                self.window.after(100, self._update_test_preview)

    def _on_apply(self):
        """Apply settings without closing dialog."""
        try:
            self._save_settings()
            messagebox.showinfo("Info", "Settings applied successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply settings: {e}")

    def _on_ok(self):
        """Apply settings and close dialog."""
        try:
            self._save_settings()
            # Stop any running test
            if self.test_running:
                self._stop_test()
            self.window.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    def _on_cancel(self):
        """Close dialog without saving."""
        # Stop any running test
        if self.test_running:
            self._stop_test()
        self.window.destroy()

    def _save_settings(self):
        """Save webcam settings to config."""
        # Get selected device index (convert from 1-based display to 0-based internal)
        selection = self.device_var.get()
        if selection and "No cameras found" not in selection:
            if "Camera " in selection:
                display_num = int(selection.split("Camera ")[1].split(":")[0])
                device_idx = display_num - 1  # Convert to 0-based
            else:
                device_idx = 0
            self.config.last_webcam_index = device_idx
        
        # Save camera properties
        self.config.camera_width = int(self.width_var.get())
        self.config.camera_height = int(self.height_var.get())
        self.config.camera_fps = int(self.fps_var.get())
        
        # Save config to file
        from ...config.settings import save_config
        save_config(self.config)

    def winfo_exists(self):
        """Check if dialog window exists."""
        try:
            return self.window.winfo_exists()
        except tk.TclError:
            return False

    def lift(self):
        """Bring dialog to front."""
        self.window.lift()