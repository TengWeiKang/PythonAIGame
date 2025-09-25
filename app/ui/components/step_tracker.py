"""Step tracker widget for showing operation breakdown and progress."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class StepStatus(Enum):
    """Status of individual steps."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Step:
    """Individual step in the operation."""
    name: str
    estimated_ms: int
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    substeps: List[str] = None

    def __post_init__(self):
        if self.substeps is None:
            self.substeps = []

    @property
    def duration_ms(self) -> float:
        """Get step duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0

    @property
    def is_active(self) -> bool:
        """Check if step is currently active."""
        return self.status == StepStatus.IN_PROGRESS

    @property
    def is_complete(self) -> bool:
        """Check if step is completed (success or failure)."""
        return self.status in [StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED]


class StepTracker(tk.Frame):
    """Widget for tracking and displaying operation steps."""

    # Color scheme
    COLORS = {
        'bg_primary': '#1e1e1e',
        'bg_secondary': '#2d2d2d',
        'bg_tertiary': '#3c3c3c',
        'text_primary': '#ffffff',
        'text_secondary': '#cccccc',
        'text_muted': '#999999',
        'pending': '#666666',
        'in_progress': '#2196f3',
        'completed': '#4caf50',
        'failed': '#f44336',
        'skipped': '#ff9800',
        'border': '#404040',
    }

    # Status symbols
    STATUS_SYMBOLS = {
        StepStatus.PENDING: "☐",
        StepStatus.IN_PROGRESS: "◐",
        StepStatus.COMPLETED: "✓",
        StepStatus.FAILED: "✗",
        StepStatus.SKIPPED: "○",
    }

    def __init__(self, parent, show_timing: bool = True, show_substeps: bool = True):
        """Initialize step tracker."""
        super().__init__(parent, bg=self.COLORS['bg_primary'])

        self.show_timing = show_timing
        self.show_substeps = show_substeps
        self.steps: List[Step] = []
        self.step_widgets: List[Dict] = []
        self.current_step_index: Optional[int] = None

        # Timing
        self.operation_start_time: Optional[float] = None
        self.total_estimated_ms = 0

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build the step tracker UI."""
        # Header
        header_frame = tk.Frame(self, bg=self.COLORS['bg_primary'])
        header_frame.pack(fill='x', pady=(0, 10))

        # Title
        self.title_var = tk.StringVar(value="Operation Steps")
        title_label = tk.Label(
            header_frame,
            textvariable=self.title_var,
            font=('Segoe UI', 10, 'bold'),
            fg=self.COLORS['text_primary'],
            bg=self.COLORS['bg_primary']
        )
        title_label.pack(side='left')

        # Overall progress
        if self.show_timing:
            self.progress_var = tk.StringVar(value="")
            progress_label = tk.Label(
                header_frame,
                textvariable=self.progress_var,
                font=('Segoe UI', 9),
                fg=self.COLORS['text_muted'],
                bg=self.COLORS['bg_primary']
            )
            progress_label.pack(side='right')

        # Steps container
        self.steps_frame = tk.Frame(self, bg=self.COLORS['bg_primary'])
        self.steps_frame.pack(fill='both', expand=True)

        # Scrollable if many steps
        self.canvas = tk.Canvas(
            self.steps_frame,
            bg=self.COLORS['bg_primary'],
            highlightthickness=0
        )
        self.scrollbar = ttk.Scrollbar(
            self.steps_frame,
            orient='vertical',
            command=self.canvas.yview
        )
        self.scrollable_frame = tk.Frame(self.canvas, bg=self.COLORS['bg_primary'])

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Pack scrollable components (initially hidden)
        self._setup_scrolling()

    def _setup_scrolling(self):
        """Setup scrolling if needed."""
        # Only show scrollbar if needed
        self.canvas.pack(side="left", fill="both", expand=True)

    def set_title(self, title: str):
        """Set the tracker title."""
        self.title_var.set(title)

    def add_step(self, name: str, estimated_ms: int) -> int:
        """Add a step and return its index."""
        step = Step(name=name, estimated_ms=estimated_ms)
        self.steps.append(step)
        self.total_estimated_ms += estimated_ms

        # Create UI widget for step
        step_index = len(self.steps) - 1
        self._create_step_widget(step_index)

        return step_index

    def add_steps(self, steps: List[Tuple[str, int]]):
        """Add multiple steps at once."""
        for name, estimated_ms in steps:
            self.add_step(name, estimated_ms)

    def _create_step_widget(self, step_index: int):
        """Create UI widget for a step."""
        step = self.steps[step_index]

        # Main step frame
        step_frame = tk.Frame(self.scrollable_frame, bg=self.COLORS['bg_primary'])
        step_frame.pack(fill='x', pady=2, padx=5)

        # Step line frame
        step_line = tk.Frame(step_frame, bg=self.COLORS['bg_primary'])
        step_line.pack(fill='x')

        # Status symbol
        status_var = tk.StringVar(value=self.STATUS_SYMBOLS[step.status])
        status_label = tk.Label(
            step_line,
            textvariable=status_var,
            font=('Segoe UI', 10),
            fg=self.COLORS['pending'],
            bg=self.COLORS['bg_primary'],
            width=2
        )
        status_label.pack(side='left')

        # Step name
        name_var = tk.StringVar(value=step.name)
        name_label = tk.Label(
            step_line,
            textvariable=name_var,
            font=('Segoe UI', 9),
            fg=self.COLORS['text_secondary'],
            bg=self.COLORS['bg_primary'],
            anchor='w'
        )
        name_label.pack(side='left', padx=(5, 0), fill='x', expand=True)

        # Timing info
        timing_var = tk.StringVar(value="")
        timing_label = tk.Label(
            step_line,
            textvariable=timing_var,
            font=('Segoe UI', 8),
            fg=self.COLORS['text_muted'],
            bg=self.COLORS['bg_primary']
        )

        if self.show_timing:
            timing_label.pack(side='right')

        # Substeps frame (initially hidden)
        substeps_frame = tk.Frame(step_frame, bg=self.COLORS['bg_secondary'])

        # Error message frame (initially hidden)
        error_frame = tk.Frame(step_frame, bg=self.COLORS['failed'])

        # Store widget references
        widget_dict = {
            'frame': step_frame,
            'line': step_line,
            'status_var': status_var,
            'status_label': status_label,
            'name_var': name_var,
            'name_label': name_label,
            'timing_var': timing_var,
            'timing_label': timing_label,
            'substeps_frame': substeps_frame,
            'error_frame': error_frame,
            'substep_labels': []
        }

        self.step_widgets.append(widget_dict)
        self._update_step_appearance(step_index)

    def start_step(self, step_index: int) -> bool:
        """Start a specific step."""
        if step_index >= len(self.steps):
            return False

        step = self.steps[step_index]
        if step.status != StepStatus.PENDING:
            return False

        # End any current step
        if self.current_step_index is not None:
            current_step = self.steps[self.current_step_index]
            if current_step.status == StepStatus.IN_PROGRESS:
                self.complete_step(self.current_step_index, success=False)

        # Start new step
        step.status = StepStatus.IN_PROGRESS
        step.start_time = time.time()
        self.current_step_index = step_index

        # Start operation timer if first step
        if self.operation_start_time is None:
            self.operation_start_time = time.time()

        self._update_step_appearance(step_index)
        self._update_progress()

        return True

    def start_next_step(self) -> Optional[int]:
        """Start the next pending step."""
        next_index = self._find_next_pending_step()
        if next_index is not None:
            self.start_step(next_index)
        return next_index

    def complete_step(self, step_index: int, success: bool = True, error_message: Optional[str] = None) -> bool:
        """Complete a step."""
        if step_index >= len(self.steps):
            return False

        step = self.steps[step_index]
        if step.status != StepStatus.IN_PROGRESS:
            return False

        step.end_time = time.time()
        step.status = StepStatus.COMPLETED if success else StepStatus.FAILED
        step.error_message = error_message

        if step_index == self.current_step_index:
            self.current_step_index = None

        self._update_step_appearance(step_index)
        self._update_progress()

        return True

    def skip_step(self, step_index: int, reason: Optional[str] = None) -> bool:
        """Skip a step."""
        if step_index >= len(self.steps):
            return False

        step = self.steps[step_index]
        if step.status not in [StepStatus.PENDING, StepStatus.IN_PROGRESS]:
            return False

        step.status = StepStatus.SKIPPED
        step.error_message = reason

        if step_index == self.current_step_index:
            self.current_step_index = None

        self._update_step_appearance(step_index)
        self._update_progress()

        return True

    def add_substep(self, step_index: int, substep_text: str):
        """Add a substep to a step."""
        if step_index >= len(self.steps):
            return

        step = self.steps[step_index]
        step.substeps.append(substep_text)

        if self.show_substeps:
            self._update_substeps(step_index)

    def clear_substeps(self, step_index: int):
        """Clear all substeps for a step."""
        if step_index >= len(self.steps):
            return

        step = self.steps[step_index]
        step.substeps.clear()

        if self.show_substeps:
            self._update_substeps(step_index)

    def _find_next_pending_step(self) -> Optional[int]:
        """Find the next pending step."""
        for i, step in enumerate(self.steps):
            if step.status == StepStatus.PENDING:
                return i
        return None

    def _update_step_appearance(self, step_index: int):
        """Update visual appearance of a step."""
        if step_index >= len(self.step_widgets):
            return

        step = self.steps[step_index]
        widget = self.step_widgets[step_index]

        # Update status symbol and color
        symbol = self.STATUS_SYMBOLS[step.status]
        color = self.COLORS[step.status.value]

        widget['status_var'].set(symbol)
        widget['status_label'].configure(fg=color)

        # Update text color based on status
        if step.status == StepStatus.IN_PROGRESS:
            text_color = self.COLORS['text_primary']
            font_weight = 'bold'
        elif step.status == StepStatus.COMPLETED:
            text_color = self.COLORS['completed']
            font_weight = 'normal'
        elif step.status == StepStatus.FAILED:
            text_color = self.COLORS['failed']
            font_weight = 'normal'
        else:
            text_color = self.COLORS['text_secondary']
            font_weight = 'normal'

        widget['name_label'].configure(fg=text_color)
        current_font = widget['name_label'].cget('font')
        if isinstance(current_font, str):
            font_family, font_size = current_font.split()[0], current_font.split()[1]
        else:
            font_family, font_size = 'Segoe UI', '9'
        widget['name_label'].configure(font=(font_family, font_size, font_weight))

        # Update timing
        if self.show_timing:
            timing_text = ""
            if step.status == StepStatus.IN_PROGRESS:
                if step.start_time:
                    elapsed = (time.time() - step.start_time) * 1000
                    timing_text = f"{elapsed:.0f}ms"
            elif step.is_complete and step.duration_ms > 0:
                timing_text = f"{step.duration_ms:.0f}ms"
            elif step.status == StepStatus.PENDING:
                timing_text = f"~{step.estimated_ms}ms"

            widget['timing_var'].set(timing_text)

        # Show/hide error message
        if step.status == StepStatus.FAILED and step.error_message:
            self._show_error_message(step_index, step.error_message)
        else:
            self._hide_error_message(step_index)

        # Update substeps
        if self.show_substeps:
            self._update_substeps(step_index)

    def _update_substeps(self, step_index: int):
        """Update substeps display."""
        if step_index >= len(self.step_widgets):
            return

        step = self.steps[step_index]
        widget = self.step_widgets[step_index]

        # Clear existing substep labels
        for label in widget['substep_labels']:
            label.destroy()
        widget['substep_labels'].clear()

        # Hide substeps frame if no substeps
        if not step.substeps:
            widget['substeps_frame'].pack_forget()
            return

        # Show substeps frame
        widget['substeps_frame'].pack(fill='x', padx=(20, 0), pady=(2, 0))

        # Create substep labels
        for substep_text in step.substeps:
            substep_label = tk.Label(
                widget['substeps_frame'],
                text=f"  → {substep_text}",
                font=('Segoe UI', 8),
                fg=self.COLORS['text_muted'],
                bg=self.COLORS['bg_secondary'],
                anchor='w'
            )
            substep_label.pack(fill='x', padx=5, pady=1)
            widget['substep_labels'].append(substep_label)

    def _show_error_message(self, step_index: int, error_message: str):
        """Show error message for a step."""
        if step_index >= len(self.step_widgets):
            return

        widget = self.step_widgets[step_index]

        # Configure error frame
        widget['error_frame'].configure(bg=self.COLORS['bg_primary'])
        widget['error_frame'].pack(fill='x', padx=(20, 0), pady=(2, 0))

        # Error label
        error_label = tk.Label(
            widget['error_frame'],
            text=f"  ✗ {error_message}",
            font=('Segoe UI', 8),
            fg=self.COLORS['failed'],
            bg=self.COLORS['bg_primary'],
            anchor='w',
            wraplength=400
        )
        error_label.pack(fill='x', padx=5, pady=2)

    def _hide_error_message(self, step_index: int):
        """Hide error message for a step."""
        if step_index >= len(self.step_widgets):
            return

        widget = self.step_widgets[step_index]
        widget['error_frame'].pack_forget()

        # Clear error frame children
        for child in widget['error_frame'].winfo_children():
            child.destroy()

    def _update_progress(self):
        """Update overall progress display."""
        if not self.show_timing:
            return

        completed_steps = sum(1 for step in self.steps if step.is_complete)
        total_steps = len(self.steps)

        if total_steps == 0:
            progress_text = ""
        else:
            progress_percent = (completed_steps / total_steps) * 100
            progress_text = f"{completed_steps}/{total_steps} ({progress_percent:.0f}%)"

            # Add ETA if operation is in progress
            if self.operation_start_time and self.current_step_index is not None:
                eta = self._calculate_eta()
                if eta:
                    progress_text += f" • ETA: {eta}"

        self.progress_var.set(progress_text)

    def _calculate_eta(self) -> Optional[str]:
        """Calculate estimated time remaining."""
        if not self.operation_start_time:
            return None

        completed_steps = [s for s in self.steps if s.is_complete]
        if len(completed_steps) < 1:
            return None

        # Calculate average time per completed step
        total_completed_time = sum(s.duration_ms for s in completed_steps)
        avg_time_per_step = total_completed_time / len(completed_steps)

        # Estimate remaining time based on remaining steps
        remaining_steps = [s for s in self.steps if not s.is_complete]
        if not remaining_steps:
            return None

        # Use actual estimates for remaining steps, fall back to average
        estimated_remaining = 0
        for step in remaining_steps:
            if step.status == StepStatus.IN_PROGRESS and step.start_time:
                # For current step, use elapsed + remaining estimate
                elapsed = (time.time() - step.start_time) * 1000
                remaining_for_current = max(0, step.estimated_ms - elapsed)
                estimated_remaining += remaining_for_current
            else:
                # Use estimate or average
                estimated_remaining += step.estimated_ms if step.estimated_ms > 0 else avg_time_per_step

        # Convert to human readable
        eta_seconds = estimated_remaining / 1000
        if eta_seconds > 60:
            return f"{eta_seconds/60:.1f}m"
        else:
            return f"{eta_seconds:.0f}s"

    def get_progress(self) -> float:
        """Get overall progress as percentage (0-100)."""
        if not self.steps:
            return 0.0

        completed_steps = sum(1 for step in self.steps if step.is_complete)
        return (completed_steps / len(self.steps)) * 100

    def get_eta_seconds(self) -> Optional[float]:
        """Get ETA in seconds."""
        if not self.operation_start_time:
            return None

        completed_steps = [s for s in self.steps if s.is_complete]
        if len(completed_steps) < 1:
            return None

        total_completed_time = sum(s.duration_ms for s in completed_steps)
        avg_time_per_step = total_completed_time / len(completed_steps) / 1000

        remaining_steps = len([s for s in self.steps if not s.is_complete])
        return remaining_steps * avg_time_per_step

    def reset(self):
        """Reset all steps to pending state."""
        for step in self.steps:
            step.status = StepStatus.PENDING
            step.start_time = None
            step.end_time = None
            step.error_message = None
            step.substeps.clear()

        self.current_step_index = None
        self.operation_start_time = None

        # Update all widgets
        for i in range(len(self.steps)):
            self._update_step_appearance(i)

        self._update_progress()

    def clear(self):
        """Clear all steps."""
        # Destroy all widgets
        for widget in self.step_widgets:
            widget['frame'].destroy()

        self.steps.clear()
        self.step_widgets.clear()
        self.current_step_index = None
        self.operation_start_time = None
        self.total_estimated_ms = 0

        self._update_progress()

    def get_summary(self) -> Dict:
        """Get operation summary."""
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        failed = sum(1 for s in self.steps if s.status == StepStatus.FAILED)
        skipped = sum(1 for s in self.steps if s.status == StepStatus.SKIPPED)

        total_time = 0
        if self.operation_start_time:
            for step in self.steps:
                if step.is_complete:
                    total_time += step.duration_ms

        return {
            'total_steps': len(self.steps),
            'completed': completed,
            'failed': failed,
            'skipped': skipped,
            'total_time_ms': total_time,
            'success_rate': (completed / len(self.steps) * 100) if self.steps else 0
        }