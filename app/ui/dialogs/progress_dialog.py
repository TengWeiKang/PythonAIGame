"""Settings progress dialog with real-time feedback and cancellation support."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
import threading
import time
from typing import Callable, Optional, List, Dict, Any
from enum import Enum
import math


class ProgressState(Enum):
    """Progress dialog states."""
    VALIDATING = "validating"
    APPLYING = "applying"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class SettingsProgressDialog:
    """Progress dialog for settings application with cancellation support."""

    # Color scheme
    COLORS = {
        'bg_primary': '#1e1e1e',
        'bg_secondary': '#2d2d2d',
        'bg_tertiary': '#3c3c3c',
        'accent_primary': '#007acc',
        'accent_secondary': '#005a9e',
        'text_primary': '#ffffff',
        'text_secondary': '#cccccc',
        'text_muted': '#999999',
        'success': '#4caf50',
        'warning': '#ff9800',
        'error': '#f44336',
        'validating': '#2196f3',
        'applying': '#4caf50',
        'border': '#404040',
    }

    STATE_COLORS = {
        ProgressState.VALIDATING: COLORS['validating'],
        ProgressState.APPLYING: COLORS['applying'],
        ProgressState.WARNING: COLORS['warning'],
        ProgressState.ERROR: COLORS['error'],
        ProgressState.SUCCESS: COLORS['success'],
        ProgressState.ROLLED_BACK: COLORS['warning'],
        ProgressState.CANCELLED: COLORS['text_muted'],
    }

    def __init__(self, parent: tk.Tk, total_steps: int, title: str = "Applying Settings"):
        """Initialize the progress dialog."""
        self.parent = parent
        self.total_steps = total_steps
        self.current_step = 0
        self.cancelled = False
        self.cancel_callback: Optional[Callable] = None
        self.completion_callback: Optional[Callable] = None

        # Progress tracking
        self.start_time = time.time()
        self.step_times: List[float] = []
        self.current_operation = ""
        self.current_substep = ""

        # Animation state
        self.animation_running = False
        self.pulse_value = 0

        # Create dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("500x300")
        self.dialog.resizable(False, False)
        self.dialog.configure(bg=self.COLORS['bg_primary'])

        # Center on parent
        self._center_on_parent()

        # Make modal
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Handle window close
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # Build UI
        self._build_ui()

        # Start animations
        self._start_animations()

    def _center_on_parent(self):
        """Center dialog on parent window."""
        self.dialog.update_idletasks()

        # Get parent geometry
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()

        # Calculate center position
        dialog_width = 500
        dialog_height = 300
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2

        self.dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")

    def _build_ui(self):
        """Build the progress dialog UI."""
        # Main frame
        main_frame = tk.Frame(self.dialog, bg=self.COLORS['bg_primary'])
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Header
        header_frame = tk.Frame(main_frame, bg=self.COLORS['bg_primary'])
        header_frame.pack(fill='x', pady=(0, 20))

        # Title
        self.title_var = tk.StringVar(value="Applying Settings...")
        title_label = tk.Label(
            header_frame,
            textvariable=self.title_var,
            font=('Segoe UI', 14, 'bold'),
            fg=self.COLORS['text_primary'],
            bg=self.COLORS['bg_primary']
        )
        title_label.pack()

        # Status indicator
        self.status_frame = tk.Frame(header_frame, bg=self.COLORS['bg_primary'])
        self.status_frame.pack(pady=(10, 0))

        self.status_canvas = tk.Canvas(
            self.status_frame,
            width=20,
            height=20,
            bg=self.COLORS['bg_primary'],
            highlightthickness=0
        )
        self.status_canvas.pack(side='left', padx=(0, 10))

        self.status_var = tk.StringVar(value="Initializing...")
        status_label = tk.Label(
            self.status_frame,
            textvariable=self.status_var,
            font=('Segoe UI', 10),
            fg=self.COLORS['text_secondary'],
            bg=self.COLORS['bg_primary']
        )
        status_label.pack(side='left')

        # Progress section
        progress_frame = tk.Frame(main_frame, bg=self.COLORS['bg_primary'])
        progress_frame.pack(fill='x', pady=(0, 20))

        # Main progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            length=400,
            mode='determinate'
        )
        self.progress_bar.pack(fill='x', pady=(0, 10))

        # Progress text
        progress_text_frame = tk.Frame(progress_frame, bg=self.COLORS['bg_primary'])
        progress_text_frame.pack(fill='x')

        self.progress_text_var = tk.StringVar(value="0%")
        progress_text_label = tk.Label(
            progress_text_frame,
            textvariable=self.progress_text_var,
            font=('Segoe UI', 9),
            fg=self.COLORS['text_muted'],
            bg=self.COLORS['bg_primary']
        )
        progress_text_label.pack(side='left')

        self.eta_var = tk.StringVar(value="")
        eta_label = tk.Label(
            progress_text_frame,
            textvariable=self.eta_var,
            font=('Segoe UI', 9),
            fg=self.COLORS['text_muted'],
            bg=self.COLORS['bg_primary']
        )
        eta_label.pack(side='right')

        # Current operation
        operation_frame = tk.Frame(main_frame, bg=self.COLORS['bg_primary'])
        operation_frame.pack(fill='x', pady=(0, 20))

        operation_title = tk.Label(
            operation_frame,
            text="Current Operation:",
            font=('Segoe UI', 9, 'bold'),
            fg=self.COLORS['text_secondary'],
            bg=self.COLORS['bg_primary']
        )
        operation_title.pack(anchor='w')

        self.operation_var = tk.StringVar(value="Preparing...")
        operation_label = tk.Label(
            operation_frame,
            textvariable=self.operation_var,
            font=('Segoe UI', 10),
            fg=self.COLORS['text_primary'],
            bg=self.COLORS['bg_primary'],
            wraplength=450,
            justify='left'
        )
        operation_label.pack(anchor='w', pady=(5, 0))

        # Substep
        self.substep_var = tk.StringVar(value="")
        substep_label = tk.Label(
            operation_frame,
            textvariable=self.substep_var,
            font=('Segoe UI', 9),
            fg=self.COLORS['text_muted'],
            bg=self.COLORS['bg_primary'],
            wraplength=450,
            justify='left'
        )
        substep_label.pack(anchor='w', pady=(2, 0))

        # Buttons
        button_frame = tk.Frame(main_frame, bg=self.COLORS['bg_primary'])
        button_frame.pack(fill='x', side='bottom')

        self.cancel_button = tk.Button(
            button_frame,
            text="Cancel",
            command=self._on_cancel,
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_primary'],
            relief='flat',
            padx=20,
            pady=8,
            font=('Segoe UI', 9)
        )
        self.cancel_button.pack(side='right')

        # Details button (initially hidden)
        self.details_button = tk.Button(
            button_frame,
            text="Show Details",
            command=self._show_details,
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_primary'],
            relief='flat',
            padx=20,
            pady=8,
            font=('Segoe UI', 9)
        )
        # Don't pack initially

        # Details frame (initially hidden)
        self.details_frame = tk.Frame(main_frame, bg=self.COLORS['bg_secondary'])
        self.details_text = tk.Text(
            self.details_frame,
            height=6,
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_secondary'],
            font=('Consolas', 8),
            wrap='word'
        )
        scrollbar = ttk.Scrollbar(self.details_frame, orient='vertical', command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=scrollbar.set)

        self.details_text.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        scrollbar.pack(side='right', fill='y', pady=5)

        self.details_visible = False

    def _start_animations(self):
        """Start progress animations."""
        self.animation_running = True
        self._animate_status_indicator()

    def _animate_status_indicator(self):
        """Animate the status indicator."""
        if not self.animation_running:
            return

        # Pulse effect for validating/applying states
        self.pulse_value = (self.pulse_value + 0.1) % (2 * math.pi)
        alpha = (math.sin(self.pulse_value) + 1) / 2  # 0 to 1

        # Update status indicator
        self.status_canvas.delete("all")

        current_state = getattr(self, '_current_state', ProgressState.VALIDATING)
        base_color = self.STATE_COLORS.get(current_state, self.COLORS['validating'])

        if current_state in [ProgressState.VALIDATING, ProgressState.APPLYING]:
            # Pulsing circle
            intensity = 0.3 + 0.7 * alpha
            color = self._blend_color(base_color, intensity)
            self.status_canvas.create_oval(2, 2, 18, 18, fill=color, outline="")
        else:
            # Static circle
            self.status_canvas.create_oval(2, 2, 18, 18, fill=base_color, outline="")

        # Schedule next animation frame
        self.dialog.after(50, self._animate_status_indicator)

    def _blend_color(self, hex_color: str, intensity: float) -> str:
        """Blend color with intensity."""
        # Remove # and convert to RGB
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        # Apply intensity
        r = int(r * intensity)
        g = int(g * intensity)
        b = int(b * intensity)

        return f"#{r:02x}{g:02x}{b:02x}"

    def update_progress(self, step: int, message: str):
        """Update progress to specific step."""
        if self.cancelled:
            return

        self.current_step = step
        progress_percent = (step / self.total_steps) * 100

        # Update progress bar
        self.progress_var.set(progress_percent)
        self.progress_text_var.set(f"{progress_percent:.0f}%")

        # Update operation
        self.operation_var.set(message)
        self.current_operation = message

        # Record step time
        current_time = time.time()
        self.step_times.append(current_time)

        # Calculate ETA
        self._update_eta()

        # Log to details
        self._log_to_details(f"Step {step}/{self.total_steps}: {message}")

    def update_substep(self, message: str):
        """Update substep message."""
        if self.cancelled:
            return

        self.substep_var.set(message)
        self.current_substep = message
        self._log_to_details(f"  → {message}")

    def set_indeterminate(self, message: str):
        """Switch to indeterminate progress mode."""
        if self.cancelled:
            return

        self.progress_bar.configure(mode='indeterminate')
        self.progress_bar.start(10)
        self.operation_var.set(message)
        self.progress_text_var.set("Please wait...")
        self.eta_var.set("")
        self._log_to_details(f"Indeterminate: {message}")

    def set_determinate(self):
        """Switch back to determinate progress mode."""
        self.progress_bar.stop()
        self.progress_bar.configure(mode='determinate')

    def enable_cancel(self, callback: Callable):
        """Enable cancellation with callback."""
        self.cancel_callback = callback
        self.cancel_button.configure(state='normal')

    def disable_cancel(self):
        """Disable cancellation."""
        self.cancel_callback = None
        self.cancel_button.configure(state='disabled')

    def set_state(self, state: ProgressState):
        """Set visual state."""
        self._current_state = state

        # Update title based on state
        state_titles = {
            ProgressState.VALIDATING: "Validating Settings...",
            ProgressState.APPLYING: "Applying Settings...",
            ProgressState.WARNING: "Warning",
            ProgressState.ERROR: "Error",
            ProgressState.SUCCESS: "Success",
            ProgressState.ROLLED_BACK: "Changes Rolled Back",
            ProgressState.CANCELLED: "Cancelled",
        }
        self.title_var.set(state_titles.get(state, "Processing..."))

    def show_completion(self, success: bool, message: str, details: Optional[str] = None):
        """Show completion state."""
        self.animation_running = False

        if success:
            self.set_state(ProgressState.SUCCESS)
            self.progress_var.set(100)
            self.progress_text_var.set("100%")
            self.status_var.set("✓ Completed successfully")

            # Celebration effect
            self._show_celebration()
        else:
            self.set_state(ProgressState.ERROR)
            self.status_var.set("✗ Failed")

        self.operation_var.set(message)
        self.substep_var.set("")
        self.eta_var.set("")

        if details:
            self._log_to_details(details)
            self.details_button.pack(side='left', padx=(0, 10))

        # Update cancel button to close
        self.cancel_button.configure(text="Close", command=self._close_dialog)

        # Auto-close on success after delay
        if success:
            self.dialog.after(2000, self._auto_close)

    def _show_celebration(self):
        """Show success celebration animation."""
        # Simple celebration: brief color flash
        original_bg = self.dialog.cget('bg')
        self.dialog.configure(bg=self.COLORS['success'])
        self.dialog.after(200, lambda: self.dialog.configure(bg=original_bg))

    def _update_eta(self):
        """Update estimated time remaining."""
        if len(self.step_times) < 2:
            return

        # Calculate average time per step
        total_time = self.step_times[-1] - self.step_times[0]
        steps_completed = len(self.step_times) - 1
        avg_time_per_step = total_time / steps_completed

        # Estimate remaining time
        remaining_steps = self.total_steps - self.current_step
        eta_seconds = remaining_steps * avg_time_per_step

        if eta_seconds > 60:
            eta_text = f"ETA: {eta_seconds/60:.1f}m"
        else:
            eta_text = f"ETA: {eta_seconds:.0f}s"

        self.eta_var.set(eta_text)

    def _log_to_details(self, message: str):
        """Log message to details view."""
        timestamp = time.strftime("%H:%M:%S")
        self.details_text.insert('end', f"[{timestamp}] {message}\n")
        self.details_text.see('end')

    def _show_details(self):
        """Toggle details view."""
        if self.details_visible:
            self.details_frame.pack_forget()
            self.details_button.configure(text="Show Details")
            self.dialog.geometry("500x300")
        else:
            self.details_frame.pack(fill='both', expand=True, pady=(10, 0))
            self.details_button.configure(text="Hide Details")
            self.dialog.geometry("500x450")

        self.details_visible = not self.details_visible

    def _on_cancel(self):
        """Handle cancel button click."""
        if self.cancel_callback and not self.cancelled:
            self.cancelled = True
            self.set_state(ProgressState.CANCELLED)
            self.status_var.set("Cancelling...")
            self.cancel_button.configure(state='disabled')

            # Call cancel callback in thread to avoid blocking UI
            threading.Thread(target=self.cancel_callback, daemon=True).start()
        else:
            self._close_dialog()

    def _on_window_close(self):
        """Handle window close event."""
        if not self.cancelled and self.cancel_callback:
            self._on_cancel()
        else:
            self._close_dialog()

    def _auto_close(self):
        """Auto-close dialog after success."""
        if hasattr(self, 'dialog') and self.dialog.winfo_exists():
            self._close_dialog()

    def _close_dialog(self):
        """Close the dialog."""
        self.animation_running = False
        if hasattr(self, 'dialog'):
            self.dialog.grab_release()
            self.dialog.destroy()

        if self.completion_callback:
            self.completion_callback()

    def set_completion_callback(self, callback: Callable):
        """Set callback for dialog completion."""
        self.completion_callback = callback

    def destroy(self):
        """Destroy the dialog."""
        self._close_dialog()


class ProgressDialogManager:
    """Manager for progress dialogs to prevent multiple instances."""

    def __init__(self):
        self.current_dialog: Optional[SettingsProgressDialog] = None

    def show_progress(self, parent: tk.Tk, total_steps: int, title: str = "Applying Settings") -> SettingsProgressDialog:
        """Show progress dialog, closing any existing one."""
        if self.current_dialog:
            self.current_dialog.destroy()

        self.current_dialog = SettingsProgressDialog(parent, total_steps, title)
        self.current_dialog.set_completion_callback(self._on_dialog_completed)
        return self.current_dialog

    def _on_dialog_completed(self):
        """Handle dialog completion."""
        self.current_dialog = None

    def close_current(self):
        """Close current dialog if exists."""
        if self.current_dialog:
            self.current_dialog.destroy()
            self.current_dialog = None


# Global manager instance
progress_manager = ProgressDialogManager()