"""Training progress dialog with real-time updates."""

import tkinter as tk
from tkinter import ttk
import threading
from typing import Optional, Callable


class TrainingProgressDialog:
    """Dialog showing training progress with metrics and progress bar."""

    def __init__(self, parent, title: str = "Training Model"):
        """Initialize training progress dialog.

        Args:
            parent: Parent window
            title: Dialog title
        """
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("500x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Center on parent
        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 500) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 300) // 2
        self.dialog.geometry(f"+{x}+{y}")

        self._build_ui()

        self._training_thread: Optional[threading.Thread] = None
        self._cancelled = False

    def _build_ui(self):
        """Build dialog UI."""
        # Main frame
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill='both', expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Training YOLO Model",
            font=('Segoe UI', 14, 'bold')
        )
        title_label.pack(pady=(0, 20))

        # Status label
        self.status_label = ttk.Label(
            main_frame,
            text="Preparing dataset...",
            font=('Segoe UI', 10)
        )
        self.status_label.pack(pady=(0, 10))

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            main_frame,
            length=400,
            mode='indeterminate'
        )
        self.progress_bar.pack(pady=(0, 20))
        self.progress_bar.start(10)

        # Metrics frame
        metrics_frame = ttk.LabelFrame(main_frame, text="Metrics", padding=10)
        metrics_frame.pack(fill='x', pady=(0, 20))

        # Epoch counter
        epoch_frame = ttk.Frame(metrics_frame)
        epoch_frame.pack(fill='x', pady=5)

        ttk.Label(epoch_frame, text="Epoch:").pack(side='left')
        self.epoch_label = ttk.Label(epoch_frame, text="0/0", font=('Segoe UI', 10, 'bold'))
        self.epoch_label.pack(side='right')

        # ETA
        eta_frame = ttk.Frame(metrics_frame)
        eta_frame.pack(fill='x', pady=5)

        ttk.Label(eta_frame, text="Estimated Time:").pack(side='left')
        self.eta_label = ttk.Label(eta_frame, text="Calculating...", font=('Segoe UI', 10))
        self.eta_label.pack(side='right')

        # Loss
        loss_frame = ttk.Frame(metrics_frame)
        loss_frame.pack(fill='x', pady=5)

        ttk.Label(loss_frame, text="Loss:").pack(side='left')
        self.loss_label = ttk.Label(loss_frame, text="--", font=('Segoe UI', 10))
        self.loss_label.pack(side='right')

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x')

        self.cancel_button = ttk.Button(
            button_frame,
            text="Cancel",
            command=self._on_cancel
        )
        self.cancel_button.pack(side='right')

    def update_status(self, status: str):
        """Update status text.

        Args:
            status: Status message
        """
        self.status_label.config(text=status)

    def update_progress(self, current_epoch: int, total_epochs: int):
        """Update progress bar and epoch counter.

        Args:
            current_epoch: Current epoch number
            total_epochs: Total number of epochs
        """
        # Switch to determinate mode if needed
        if self.progress_bar['mode'] == 'indeterminate':
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate', maximum=total_epochs)

        self.progress_bar['value'] = current_epoch
        self.epoch_label.config(text=f"{current_epoch}/{total_epochs}")

    def update_eta(self, eta_seconds: int):
        """Update estimated time remaining.

        Args:
            eta_seconds: Estimated seconds remaining
        """
        minutes = eta_seconds // 60
        seconds = eta_seconds % 60
        self.eta_label.config(text=f"{minutes}m {seconds}s")

    def update_loss(self, loss: float):
        """Update loss value.

        Args:
            loss: Current loss value
        """
        self.loss_label.config(text=f"{loss:.4f}")

    def set_complete(self, success: bool, message: str = ""):
        """Mark training as complete.

        Args:
            success: Whether training succeeded
            message: Optional completion message
        """
        self.progress_bar.stop()

        if success:
            self.status_label.config(text=message or "Training completed successfully!")
            self.progress_bar['value'] = self.progress_bar['maximum']
        else:
            self.status_label.config(text=message or "Training failed.")

        self.cancel_button.config(text="Close")

    def _on_cancel(self):
        """Handle cancel button click."""
        self._cancelled = True
        self.dialog.destroy()

    def is_cancelled(self) -> bool:
        """Check if training was cancelled.

        Returns:
            True if cancelled, False otherwise
        """
        return self._cancelled

    def show(self):
        """Show dialog modally."""
        self.dialog.wait_window()