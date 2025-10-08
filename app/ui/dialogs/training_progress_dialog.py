"""Training progress dialog with real-time updates."""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from typing import Optional, Callable, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TrainingProgressDialog:
    """Dialog showing training progress with metrics and progress bar."""

    def __init__(self, parent, title: str = "Training Model"):
        """Initialize training progress dialog.

        Args:
            parent: Parent window
            title: Dialog title
        """
        self.parent = parent
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("500x500")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Center on parent
        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 500) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 500) // 2
        self.dialog.geometry(f"+{x}+{y}")

        # State management
        self._training_thread: Optional[threading.Thread] = None
        self._cancelled = False
        self._training_complete = False

        # Bind window close event to handle cancellation
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_window_close)

        self._build_ui()

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
        metrics_frame = ttk.LabelFrame(main_frame, text="Training Metrics", padding=10)
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

        # Training Speed
        speed_frame = ttk.Frame(metrics_frame)
        speed_frame.pack(fill='x', pady=5)

        ttk.Label(speed_frame, text="Speed:").pack(side='left')
        self.speed_label = ttk.Label(speed_frame, text="--", font=('Segoe UI', 10))
        self.speed_label.pack(side='right')

        # Loss
        loss_frame = ttk.Frame(metrics_frame)
        loss_frame.pack(fill='x', pady=5)

        ttk.Label(loss_frame, text="Loss:").pack(side='left')
        self.loss_label = ttk.Label(loss_frame, text="--", font=('Segoe UI', 10))
        self.loss_label.pack(side='right')

        # Precision (optional metric)
        precision_frame = ttk.Frame(metrics_frame)
        precision_frame.pack(fill='x', pady=5)

        ttk.Label(precision_frame, text="Precision:").pack(side='left')
        self.precision_label = ttk.Label(precision_frame, text="--", font=('Segoe UI', 10))
        self.precision_label.pack(side='right')

        # Recall (optional metric)
        recall_frame = ttk.Frame(metrics_frame)
        recall_frame.pack(fill='x', pady=5)

        ttk.Label(recall_frame, text="Recall:").pack(side='left')
        self.recall_label = ttk.Label(recall_frame, text="--", font=('Segoe UI', 10))
        self.recall_label.pack(side='right')

        # mAP50 (optional metric)
        map50_frame = ttk.Frame(metrics_frame)
        map50_frame.pack(fill='x', pady=5)

        ttk.Label(map50_frame, text="mAP50:").pack(side='left')
        self.map50_label = ttk.Label(map50_frame, text="--", font=('Segoe UI', 10))
        self.map50_label.pack(side='right')

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(10, 0))

        self.cancel_button = ttk.Button(
            button_frame,
            text="Cancel Training",
            command=self._on_cancel
        )
        self.cancel_button.pack(side='right')

    def update_status(self, status: str):
        """Update status text (thread-safe).

        Args:
            status: Status message
        """
        try:
            self.parent.after(0, lambda: self.status_label.config(text=status))
        except Exception as e:
            logger.error(f"Error updating status: {e}")

    def update_progress(self, current_epoch: int, total_epochs: int):
        """Update progress bar and epoch counter (thread-safe).

        Args:
            current_epoch: Current epoch number
            total_epochs: Total number of epochs
        """
        def _update():
            try:
                # Switch to determinate mode if needed
                if self.progress_bar['mode'] == 'indeterminate':
                    self.progress_bar.stop()
                    self.progress_bar.config(mode='determinate', maximum=total_epochs)

                self.progress_bar['value'] = current_epoch
                self.epoch_label.config(text=f"{current_epoch}/{total_epochs}")
            except Exception as e:
                logger.error(f"Error updating progress: {e}")

        try:
            self.parent.after(0, _update)
        except Exception as e:
            logger.error(f"Error scheduling progress update: {e}")

    def update_eta(self, eta_seconds: int):
        """Update estimated time remaining (thread-safe).

        Args:
            eta_seconds: Estimated seconds remaining
        """
        def _update():
            try:
                minutes = eta_seconds // 60
                seconds = eta_seconds % 60
                self.eta_label.config(text=f"{minutes}m {seconds}s")
            except Exception as e:
                logger.error(f"Error updating ETA: {e}")

        try:
            self.parent.after(0, _update)
        except Exception as e:
            logger.error(f"Error scheduling ETA update: {e}")

    def update_loss(self, loss: float):
        """Update loss value (thread-safe).

        Args:
            loss: Current loss value
        """
        try:
            self.parent.after(0, lambda: self.loss_label.config(text=f"{loss:.4f}"))
        except Exception as e:
            logger.error(f"Error updating loss: {e}")

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update all metrics from dictionary (thread-safe).

        This is the main method called by the training service callback.

        Args:
            metrics: Dictionary containing training metrics
        """
        def _update():
            try:
                # Update epoch and progress bar
                if 'epoch' in metrics and 'total_epochs' in metrics:
                    current_epoch = metrics['epoch']
                    total_epochs = metrics['total_epochs']

                    # Switch to determinate mode if needed
                    if self.progress_bar['mode'] == 'indeterminate':
                        self.progress_bar.stop()
                        self.progress_bar.config(mode='determinate', maximum=total_epochs)

                    self.progress_bar['value'] = current_epoch
                    self.epoch_label.config(text=f"{current_epoch}/{total_epochs}")

                # Update ETA - use formatted version if available, otherwise calculate from seconds
                if 'eta_formatted' in metrics:
                    # Use the enhanced formatted ETA from training service
                    self.eta_label.config(text=metrics['eta_formatted'])
                elif 'eta_seconds' in metrics:
                    # Fallback to manual formatting for backward compatibility
                    eta_seconds = metrics['eta_seconds']
                    minutes = eta_seconds // 60
                    seconds = eta_seconds % 60
                    self.eta_label.config(text=f"{minutes}m {seconds}s")

                # Update training speed
                if 'epochs_per_minute' in metrics:
                    speed = metrics['epochs_per_minute']
                    self.speed_label.config(text=f"{speed:.2f} epochs/min")

                # Update status with progress information
                if 'epoch' in metrics and 'total_epochs' in metrics:
                    current_epoch = metrics['epoch']
                    total_epochs = metrics['total_epochs']
                    if 'progress_percent' in metrics:
                        # Enhanced status with percentage
                        progress_percent = metrics['progress_percent']
                        self.update_status(f"Training epoch {current_epoch}/{total_epochs} ({progress_percent}%)...")
                    else:
                        # Basic status for backward compatibility
                        self.update_status(f"Training epoch {current_epoch}/{total_epochs}...")

                # Update loss
                if 'loss' in metrics:
                    self.loss_label.config(text=f"{metrics['loss']:.4f}")

                # Update precision
                if 'precision' in metrics:
                    self.precision_label.config(text=f"{metrics['precision']:.4f}")

                # Update recall
                if 'recall' in metrics:
                    self.recall_label.config(text=f"{metrics['recall']:.4f}")

                # Update mAP50
                if 'mAP50' in metrics:
                    self.map50_label.config(text=f"{metrics['mAP50']:.4f}")

                # Handle special status updates
                if 'status' in metrics:
                    status = metrics['status']

                    if status == 'training_started':
                        self.update_status("Training started...")

                    elif status == 'cancelled':
                        # Training was cancelled
                        message = metrics.get('message', 'Training cancelled by user')
                        self.update_status(f"Cancelled: {message}")

                        # Stop progress bar animation
                        try:
                            self.progress_bar.stop()
                        except Exception:
                            pass

                        logger.info(f"Status update: {message}")

                    elif status == 'error':
                        # Training error occurred
                        message = metrics.get('message', 'Training error occurred')
                        self.update_status(f"Error: {message}")

            except Exception as e:
                logger.error(f"Error updating metrics: {e}")

        try:
            self.parent.after(0, _update)
        except Exception as e:
            logger.error(f"Error scheduling metrics update: {e}")

    def set_complete(self, success: bool, message: str = ""):
        """Mark training as complete (thread-safe).

        Args:
            success: Whether training succeeded
            message: Optional completion message
        """
        def _complete():
            try:
                self._training_complete = True
                self.progress_bar.stop()

                if success:
                    self.status_label.config(text=message or "Training completed successfully!")
                    if self.progress_bar['mode'] == 'determinate':
                        self.progress_bar['value'] = self.progress_bar['maximum']
                else:
                    self.status_label.config(text=message or "Training failed.")
                self.cancel_button.config(state='normal')
                self.cancel_button.config(text="Close")
            except Exception as e:
                logger.error(f"Error completing training: {e}")

        try:
            self.parent.after(0, _complete)
        except Exception as e:
            logger.error(f"Error scheduling completion: {e}")

    def _on_cancel(self):
        """Handle cancel button click with confirmation and immediate feedback."""
        if self._training_complete:
            # Training is complete, just close the dialog
            self.dialog.destroy()
            return

        # Prevent multiple cancel requests
        if self._cancelled:
            logger.debug("Cancel already requested, ignoring duplicate click")
            return

        # Show confirmation dialog
        response = messagebox.askyesno(
            "Cancel Training",
            "Are you sure you want to cancel training?\n\n"
            "Training will stop immediately, and progress will be lost.\n"
            "Partial model from the last completed epoch may be saved.",
            icon='warning',
            parent=self.dialog
        )

        if not response:
            # User changed their mind
            logger.info("User cancelled the cancellation request")
            return

        # User confirmed cancellation - set flag and provide immediate feedback
        logger.info("=" * 60)
        logger.info("USER CONFIRMED TRAINING CANCELLATION")
        logger.info("Requesting immediate stop via cancellation flag...")
        logger.info("=" * 60)

        self._cancelled = True

        # IMMEDIATE UI FEEDBACK
        # 1. Update status label
        self.update_status("Cancelling training... Stopping immediately...")

        # 2. Disable cancel button to prevent multiple clicks
        self.cancel_button.config(state='disabled', text="Cancelling...")

        # 3. Switch progress bar to indeterminate mode to show activity
        try:
            self.progress_bar.stop()  # Stop any current animation
            self.progress_bar.config(mode='indeterminate')
            self.progress_bar.start(10)  # Start indeterminate animation
        except Exception as e:
            logger.warning(f"Could not update progress bar during cancellation: {e}")

        logger.info("UI updated with cancellation feedback, waiting for training to stop...")

    def _on_window_close(self):
        """Handle window close event (X button)."""
        if self._training_complete:
            # Training is complete, allow close
            self.dialog.destroy()
        else:
            # Training is still running, treat as cancellation
            self._on_cancel()

    def is_cancelled(self) -> bool:
        """Check if training was cancelled.

        Returns:
            True if cancelled, False otherwise
        """
        return self._cancelled

    def show(self):
        """Show dialog modally."""
        try:
            self.dialog.wait_window()
        except Exception as e:
            logger.error(f"Error showing dialog: {e}")