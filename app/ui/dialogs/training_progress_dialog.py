"""Training Progress Dialog with real-time updates."""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Optional, Dict, Any, Callable
import threading
import time
from datetime import datetime

from ...services.training_service import TrainingService
from ...core.exceptions import ModelError


class TrainingProgressDialog:
    """Dialog to show training progress with real-time updates."""
    
    def __init__(self, parent, config, object_training_service):
        self.parent = parent
        self.config = config
        self.object_training_service = object_training_service
        self.training_service = TrainingService(config)
        
        # State variables
        self.training_thread = None
        self.is_training = False
        self.training_results = None
        self.cancelled = False
        
        # Create dialog window
        self.window = tk.Toplevel(parent)
        self.window.title("YOLO Model Training")
        self.window.geometry("600x500")
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
        
        # Start training immediately
        self._start_training()

    def _center_window(self):
        """Center the dialog on the parent window."""
        self.window.update_idletasks()
        x = self.parent.winfo_x() + (self.parent.winfo_width() // 2) - (self.window.winfo_width() // 2)
        y = self.parent.winfo_y() + (self.parent.winfo_height() // 2) - (self.window.winfo_height() // 2)
        self.window.geometry(f"+{x}+{y}")

    def _build_ui(self):
        """Build the dialog UI."""
        # Main container
        main_frame = ttk.Frame(self.window, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Training YOLO Model", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Training status section
        status_frame = ttk.LabelFrame(main_frame, text="Training Status", padding="10")
        status_frame.pack(fill='x', pady=(0, 10))
        
        # Status message
        self.status_var = tk.StringVar(value="Preparing to start training...")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                     font=('Arial', 11))
        self.status_label.pack(anchor='w')
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, 
                                           maximum=100, length=400)
        self.progress_bar.pack(fill='x', pady=(10, 0))
        
        # Progress percentage
        self.progress_text_var = tk.StringVar(value="0%")
        self.progress_text_label = ttk.Label(status_frame, textvariable=self.progress_text_var,
                                            font=('Arial', 9))
        self.progress_text_label.pack(anchor='e', pady=(5, 0))
        
        # Configuration section
        config_frame = ttk.LabelFrame(main_frame, text="Training Configuration", padding="10")
        config_frame.pack(fill='x', pady=(0, 10))
        
        # Training parameters
        params_grid = ttk.Frame(config_frame)
        params_grid.pack(fill='x')
        
        # Epochs
        ttk.Label(params_grid, text="Epochs:").grid(row=0, column=0, sticky='w', padx=(0, 10))
        self.epochs_var = tk.IntVar(value=getattr(self.config, 'train_epochs', 50))
        ttk.Label(params_grid, textvariable=self.epochs_var).grid(row=0, column=1, sticky='w')
        
        # Batch size
        ttk.Label(params_grid, text="Batch Size:").grid(row=0, column=2, sticky='w', padx=(20, 10))
        self.batch_size_var = tk.IntVar(value=getattr(self.config, 'batch_size', 16))
        ttk.Label(params_grid, textvariable=self.batch_size_var).grid(row=0, column=3, sticky='w')
        
        # Base model
        ttk.Label(params_grid, text="Base Model:").grid(row=1, column=0, sticky='w', padx=(0, 10), pady=(5, 0))
        self.base_model_var = tk.StringVar(value=getattr(self.config, 'model_size', 'yolo11n.pt'))
        ttk.Label(params_grid, textvariable=self.base_model_var).grid(row=1, column=1, sticky='w', pady=(5, 0))
        
        # Confirmed objects count
        confirmed_count = self.object_training_service.get_confirmed_count()
        ttk.Label(params_grid, text="Confirmed Objects:").grid(row=1, column=2, sticky='w', padx=(20, 10), pady=(5, 0))
        ttk.Label(params_grid, text=str(confirmed_count)).grid(row=1, column=3, sticky='w', pady=(5, 0))
        
        # Model save path section
        save_frame = ttk.LabelFrame(main_frame, text="Model Save Location", padding="10")
        save_frame.pack(fill='x', pady=(0, 10))
        
        save_path_frame = ttk.Frame(save_frame)
        save_path_frame.pack(fill='x')
        
        self.save_path_var = tk.StringVar()
        self.save_path_entry = ttk.Entry(save_path_frame, textvariable=self.save_path_var, 
                                        state='readonly')
        self.save_path_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        ttk.Button(save_path_frame, text="Browse...", 
                  command=self._browse_save_path).pack(side='right')
        
        # Log section
        log_frame = ttk.LabelFrame(main_frame, text="Training Log", padding="10")
        log_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Log text with scrollbar
        log_text_frame = ttk.Frame(log_frame)
        log_text_frame.pack(fill='both', expand=True)
        
        self.log_text = tk.Text(log_text_frame, height=8, wrap=tk.WORD, 
                               font=('Consolas', 9), state='disabled')
        log_scrollbar = ttk.Scrollbar(log_text_frame, orient='vertical', 
                                     command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side='left', fill='both', expand=True)
        log_scrollbar.pack(side='right', fill='y')
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(10, 0))
        
        self.cancel_button = ttk.Button(button_frame, text="Cancel", 
                                       command=self._cancel_training)
        self.cancel_button.pack(side='left')
        
        self.close_button = ttk.Button(button_frame, text="Close", 
                                      command=self._on_close, state='disabled')
        self.close_button.pack(side='right')

    def _browse_save_path(self):
        """Browse for model save location."""
        if self.is_training:
            messagebox.showinfo("Info", "Cannot change save path while training is in progress")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save Trained Model As",
            defaultextension=".pt",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")],
            parent=self.window
        )
        
        if filename:
            self.save_path_var.set(filename)

    def _log_message(self, message: str):
        """Add a message to the training log."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"
            
            def update_log():
                self.log_text.configure(state='normal')
                self.log_text.insert(tk.END, log_entry)
                self.log_text.configure(state='disabled')
                self.log_text.see(tk.END)
            
            if self.window.winfo_exists():
                self.window.after(0, update_log)
        except:
            pass  # Window might be closed

    def _update_progress(self, message: str, progress: float):
        """Update progress display."""
        try:
            def update():
                if progress >= 0:
                    self.progress_var.set(progress * 100)
                    self.progress_text_var.set(f"{progress*100:.1f}%")
                else:
                    # Error state
                    self.progress_var.set(0)
                    self.progress_text_var.set("Error")
                
                self.status_var.set(message)
                self._log_message(message)
            
            if self.window.winfo_exists():
                self.window.after(0, update)
        except:
            pass  # Window might be closed

    def _start_training(self):
        """Start the training process."""
        if self.is_training:
            return
            
        # Check confirmed objects
        confirmed_count = self.object_training_service.get_confirmed_count()
        if confirmed_count == 0:
            messagebox.showerror("Error", 
                               "No confirmed objects found. Please confirm some objects first.")
            self._on_close()
            return
        
        self.is_training = True
        self.cancel_button.configure(state='normal')
        self.close_button.configure(state='disabled')
        
        # Start training in background thread
        self.training_thread = threading.Thread(target=self._training_thread_func, daemon=True)
        self.training_thread.start()

    def _training_thread_func(self):
        """Training thread function."""
        try:
            self._log_message("Starting YOLO model training...")
            
            # Get save path
            save_path = self.save_path_var.get().strip() if self.save_path_var.get().strip() else None
            
            # Start training
            self.training_results = self.training_service.train_model_with_confirmed_objects(
                object_training_service=self.object_training_service,
                base_model=self.base_model_var.get(),
                epochs=self.epochs_var.get(),
                batch_size=self.batch_size_var.get(),
                save_path=save_path,
                progress_callback=self._update_progress
            )
            
            if not self.cancelled:
                self._on_training_complete()
            
        except Exception as e:
            if not self.cancelled:
                self._on_training_error(str(e))
        
        finally:
            self.is_training = False

    def _on_training_complete(self):
        """Handle training completion."""
        def complete():
            self.cancel_button.configure(state='disabled')
            self.close_button.configure(state='normal')
            
            if self.training_results:
                model_path = self.training_results.get('model_path', 'Unknown')
                confirmed_count = self.training_results.get('confirmed_objects_count', 0)
                epochs = self.training_results.get('epochs_completed', 0)
                
                self._log_message(f"Training completed successfully!")
                self._log_message(f"Model saved to: {model_path}")
                self._log_message(f"Trained on {confirmed_count} confirmed objects")
                self._log_message(f"Completed {epochs} epochs")
                
                messagebox.showinfo("Training Complete", 
                                  f"Model training completed successfully!\n\n"
                                  f"Model saved to:\n{model_path}\n\n"
                                  f"Trained on {confirmed_count} confirmed objects over {epochs} epochs.")
            
        try:
            if self.window.winfo_exists():
                self.window.after(0, complete)
        except:
            pass

    def _on_training_error(self, error_message: str):
        """Handle training error."""
        def error():
            self.cancel_button.configure(state='disabled')
            self.close_button.configure(state='normal')
            self.progress_var.set(0)
            self.progress_text_var.set("Error")
            
            self._log_message(f"Training failed: {error_message}")
            messagebox.showerror("Training Failed", f"Model training failed:\n\n{error_message}")
        
        try:
            if self.window.winfo_exists():
                self.window.after(0, error)
        except:
            pass

    def _cancel_training(self):
        """Cancel the training process."""
        if not self.is_training:
            return
            
        if messagebox.askyesno("Cancel Training", 
                              "Are you sure you want to cancel the training?\n\n"
                              "This will stop the current training process."):
            self.cancelled = True
            self._log_message("Training cancelled by user")
            self.status_var.set("Cancelling training...")
            
            # Note: YOLO training cannot be easily stopped mid-training
            # This will prevent the completion handlers from running
            
    def _on_close(self):
        """Handle window close."""
        if self.is_training and not self.cancelled:
            if not messagebox.askyesno("Training in Progress", 
                                     "Training is still in progress. Are you sure you want to close?\n\n"
                                     "The training will continue in the background."):
                return
        
        # Close window
        self.window.grab_release()
        self.window.destroy()

    def get_training_results(self) -> Optional[Dict[str, Any]]:
        """Get the training results."""
        return self.training_results

    def winfo_exists(self):
        """Check if dialog exists."""
        try:
            return self.window.winfo_exists()
        except tk.TclError:
            return False