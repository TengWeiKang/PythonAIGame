"""Success feedback dialog with celebration animations and performance metrics."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
import math
import time
import random
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import threading


@dataclass
class ChangesSummary:
    """Summary of changes applied."""
    category: str
    changes: List[str]
    icon: str = "⚙"


@dataclass
class PerformanceMetric:
    """Performance improvement metric."""
    name: str
    old_value: float
    new_value: float
    unit: str = ""
    improvement_percent: Optional[float] = None
    is_better_higher: bool = True  # True if higher values are better

    def __post_init__(self):
        if self.improvement_percent is None:
            if self.old_value != 0:
                if self.is_better_higher:
                    self.improvement_percent = ((self.new_value - self.old_value) / self.old_value) * 100
                else:
                    self.improvement_percent = ((self.old_value - self.new_value) / self.old_value) * 100
            else:
                self.improvement_percent = 0

    @property
    def improvement_text(self) -> str:
        """Get formatted improvement text."""
        if abs(self.improvement_percent) < 0.1:
            return "No change"

        sign = "+" if self.improvement_percent > 0 else ""
        return f"{sign}{self.improvement_percent:.0f}%"

    @property
    def is_improvement(self) -> bool:
        """Check if this represents an improvement."""
        return self.improvement_percent > 0


@dataclass
class NextStep:
    """Suggested next step for the user."""
    title: str
    description: str
    action: Optional[Callable] = None
    icon: str = "→"


class CelebrationAnimation:
    """Celebration animation with particles."""

    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self.particles = []
        self.animation_running = False

    def start_celebration(self, duration_ms: int = 3000):
        """Start celebration animation."""
        self.animation_running = True
        self._create_particles()
        self._animate_particles()

        # Stop animation after duration
        self.canvas.after(duration_ms, self.stop_celebration)

    def stop_celebration(self):
        """Stop celebration animation."""
        self.animation_running = False
        self._clear_particles()

    def _create_particles(self):
        """Create celebration particles."""
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        # Create confetti particles
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#f0932b', '#eb4d4b', '#6c5ce7']

        for _ in range(20):
            particle = {
                'x': random.uniform(0, width),
                'y': random.uniform(-50, 0),
                'vx': random.uniform(-2, 2),
                'vy': random.uniform(2, 5),
                'color': random.choice(colors),
                'size': random.uniform(3, 8),
                'rotation': random.uniform(0, 360),
                'rotation_speed': random.uniform(-5, 5),
                'id': None
            }
            self.particles.append(particle)

    def _animate_particles(self):
        """Animate particles."""
        if not self.animation_running:
            return

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        for particle in self.particles[:]:
            # Update position
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['vy'] += 0.1  # Gravity
            particle['rotation'] += particle['rotation_speed']

            # Remove particles that are off screen
            if particle['y'] > height + 50:
                if particle['id']:
                    self.canvas.delete(particle['id'])
                self.particles.remove(particle)
                continue

            # Draw particle
            if particle['id']:
                self.canvas.delete(particle['id'])

            # Draw as rotated rectangle (confetti piece)
            x, y = particle['x'], particle['y']
            size = particle['size']
            rotation = math.radians(particle['rotation'])

            # Calculate rotated rectangle corners
            cos_r = math.cos(rotation)
            sin_r = math.sin(rotation)

            corners = [
                (x + size * cos_r - size * sin_r, y + size * sin_r + size * cos_r),
                (x - size * cos_r - size * sin_r, y - size * sin_r + size * cos_r),
                (x - size * cos_r + size * sin_r, y - size * sin_r - size * cos_r),
                (x + size * cos_r + size * sin_r, y + size * sin_r - size * cos_r)
            ]

            particle['id'] = self.canvas.create_polygon(
                corners,
                fill=particle['color'],
                outline=""
            )

        # Continue animation
        self.canvas.after(16, self._animate_particles)  # ~60 FPS

    def _clear_particles(self):
        """Clear all particles."""
        for particle in self.particles:
            if particle['id']:
                self.canvas.delete(particle['id'])
        self.particles.clear()


class SuccessDialog:
    """Success feedback dialog with celebration animations."""

    # Color scheme
    COLORS = {
        'bg_primary': '#1e1e1e',
        'bg_secondary': '#2d2d2d',
        'bg_tertiary': '#3c3c3c',
        'text_primary': '#ffffff',
        'text_secondary': '#cccccc',
        'text_muted': '#999999',
        'success': '#4caf50',
        'success_light': '#81c784',
        'accent': '#007acc',
        'border': '#404040',
        'improvement': '#4caf50',
        'neutral': '#999999',
    }

    def __init__(self, parent: tk.Tk, title: str = "Settings Applied Successfully!",
                 message: str = "", changes: List[ChangesSummary] = None,
                 metrics: List[PerformanceMetric] = None, next_steps: List[NextStep] = None,
                 celebration: bool = True):
        """Initialize success dialog."""
        self.parent = parent
        self.title_text = title
        self.message = message
        self.changes = changes or []
        self.metrics = metrics or []
        self.next_steps = next_steps or []
        self.show_celebration = celebration

        # Create dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Success")
        self.dialog.geometry("700x600")
        self.dialog.configure(bg=self.COLORS['bg_primary'])
        self.dialog.resizable(True, True)

        # Center on parent
        self._center_on_parent()

        # Make modal
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Handle window close
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_close)

        # Build UI
        self._build_ui()

        # Start celebration if enabled
        if self.show_celebration:
            self.dialog.after(500, self._start_celebration)

        # Auto-close after delay
        self.dialog.after(10000, self._auto_close)  # 10 seconds

    def _center_on_parent(self):
        """Center dialog on parent window."""
        self.dialog.update_idletasks()

        # Get parent geometry
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()

        # Calculate center position
        dialog_width = 700
        dialog_height = 600
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2

        self.dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")

    def _build_ui(self):
        """Build the success dialog UI."""
        # Main container
        main_frame = tk.Frame(self.dialog, bg=self.COLORS['bg_primary'])
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Celebration canvas (overlay)
        self.celebration_canvas = tk.Canvas(
            self.dialog,
            bg=self.COLORS['bg_primary'],
            highlightthickness=0
        )
        self.celebration_canvas.place(x=0, y=0, relwidth=1, relheight=1)

        # Make canvas transparent to clicks
        self.celebration_canvas.configure(state='disabled')

        # Header section
        self._build_header(main_frame)

        # Content notebook
        self._build_content_notebook(main_frame)

        # Action buttons
        self._build_action_buttons(main_frame)

        # Initialize celebration animation
        self.celebration = CelebrationAnimation(self.celebration_canvas)

    def _build_header(self, parent):
        """Build header section with success icon and title."""
        header_frame = tk.Frame(parent, bg=self.COLORS['bg_primary'])
        header_frame.pack(fill='x', pady=(0, 20))

        # Success icon with pulse animation
        self.icon_frame = tk.Frame(header_frame, bg=self.COLORS['bg_primary'])
        self.icon_frame.pack(side='left', padx=(0, 20))

        self.icon_canvas = tk.Canvas(
            self.icon_frame,
            width=60,
            height=60,
            bg=self.COLORS['bg_primary'],
            highlightthickness=0
        )
        self.icon_canvas.pack()

        # Draw success checkmark
        self._draw_success_icon()
        self._start_icon_animation()

        # Title and message
        title_frame = tk.Frame(header_frame, bg=self.COLORS['bg_primary'])
        title_frame.pack(side='left', fill='x', expand=True)

        # Success title
        title_label = tk.Label(
            title_frame,
            text=self.title_text,
            font=('Segoe UI', 16, 'bold'),
            fg=self.COLORS['success'],
            bg=self.COLORS['bg_primary'],
            anchor='w'
        )
        title_label.pack(fill='x')

        # Message
        if self.message:
            message_label = tk.Label(
                title_frame,
                text=self.message,
                font=('Segoe UI', 10),
                fg=self.COLORS['text_secondary'],
                bg=self.COLORS['bg_primary'],
                anchor='w',
                wraplength=500,
                justify='left'
            )
            message_label.pack(fill='x', pady=(5, 0))

    def _build_content_notebook(self, parent):
        """Build content notebook with different tabs."""
        # Create notebook
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=self.COLORS['bg_primary'])
        style.configure('TNotebook.Tab', background=self.COLORS['bg_secondary'])

        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill='both', expand=True, pady=(0, 20))

        # Changes summary tab
        if self.changes:
            self._build_changes_tab()

        # Performance metrics tab
        if self.metrics:
            self._build_metrics_tab()

        # Next steps tab
        if self.next_steps:
            self._build_next_steps_tab()

        # If no content, show simple summary
        if not (self.changes or self.metrics or self.next_steps):
            self._build_simple_summary_tab()

    def _build_changes_tab(self):
        """Build changes summary tab."""
        changes_frame = tk.Frame(self.notebook, bg=self.COLORS['bg_primary'])
        self.notebook.add(changes_frame, text="Changes Applied")

        # Scrollable content
        canvas = tk.Canvas(changes_frame, bg=self.COLORS['bg_primary'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(changes_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.COLORS['bg_primary'])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Changes content
        for category_summary in self.changes:
            self._create_change_category(scrollable_frame, category_summary)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _create_change_category(self, parent, category_summary: ChangesSummary):
        """Create a change category section."""
        # Category frame
        category_frame = tk.Frame(parent, bg=self.COLORS['bg_secondary'], relief='raised', bd=1)
        category_frame.pack(fill='x', pady=10, padx=10)

        # Category header
        header_frame = tk.Frame(category_frame, bg=self.COLORS['bg_secondary'])
        header_frame.pack(fill='x', padx=15, pady=10)

        # Icon
        icon_label = tk.Label(
            header_frame,
            text=category_summary.icon,
            font=('Segoe UI', 14),
            fg=self.COLORS['success'],
            bg=self.COLORS['bg_secondary']
        )
        icon_label.pack(side='left', padx=(0, 10))

        # Category name
        category_label = tk.Label(
            header_frame,
            text=category_summary.category,
            font=('Segoe UI', 12, 'bold'),
            fg=self.COLORS['text_primary'],
            bg=self.COLORS['bg_secondary']
        )
        category_label.pack(side='left')

        # Changes list
        for change in category_summary.changes:
            change_frame = tk.Frame(category_frame, bg=self.COLORS['bg_secondary'])
            change_frame.pack(fill='x', padx=25, pady=2)

            # Checkmark
            check_label = tk.Label(
                change_frame,
                text="✓",
                font=('Segoe UI', 10),
                fg=self.COLORS['success'],
                bg=self.COLORS['bg_secondary']
            )
            check_label.pack(side='left', padx=(0, 10))

            # Change text
            change_label = tk.Label(
                change_frame,
                text=change,
                font=('Segoe UI', 9),
                fg=self.COLORS['text_secondary'],
                bg=self.COLORS['bg_secondary'],
                anchor='w',
                wraplength=500,
                justify='left'
            )
            change_label.pack(side='left', fill='x', expand=True)

    def _build_metrics_tab(self):
        """Build performance metrics tab."""
        metrics_frame = tk.Frame(self.notebook, bg=self.COLORS['bg_primary'])
        self.notebook.add(metrics_frame, text="Performance Impact")

        # Metrics container
        metrics_container = tk.Frame(metrics_frame, bg=self.COLORS['bg_primary'])
        metrics_container.pack(fill='both', expand=True, padx=20, pady=20)

        # Title
        title_label = tk.Label(
            metrics_container,
            text="Performance Improvements:",
            font=('Segoe UI', 12, 'bold'),
            fg=self.COLORS['text_primary'],
            bg=self.COLORS['bg_primary']
        )
        title_label.pack(pady=(0, 15))

        # Metrics list
        for metric in self.metrics:
            self._create_metric_widget(metrics_container, metric)

    def _create_metric_widget(self, parent, metric: PerformanceMetric):
        """Create a performance metric widget."""
        # Metric frame
        metric_frame = tk.Frame(parent, bg=self.COLORS['bg_secondary'], relief='raised', bd=1)
        metric_frame.pack(fill='x', pady=5)

        # Content frame
        content_frame = tk.Frame(metric_frame, bg=self.COLORS['bg_secondary'])
        content_frame.pack(fill='x', padx=15, pady=10)

        # Metric header
        header_frame = tk.Frame(content_frame, bg=self.COLORS['bg_secondary'])
        header_frame.pack(fill='x')

        # Metric name
        name_label = tk.Label(
            header_frame,
            text=metric.name,
            font=('Segoe UI', 10, 'bold'),
            fg=self.COLORS['text_primary'],
            bg=self.COLORS['bg_secondary']
        )
        name_label.pack(side='left')

        # Improvement indicator
        improvement_color = self.COLORS['improvement'] if metric.is_improvement else self.COLORS['neutral']
        improvement_icon = "↗" if metric.is_improvement else "→"

        improvement_label = tk.Label(
            header_frame,
            text=f"{improvement_icon} {metric.improvement_text}",
            font=('Segoe UI', 10, 'bold'),
            fg=improvement_color,
            bg=self.COLORS['bg_secondary']
        )
        improvement_label.pack(side='right')

        # Values frame
        values_frame = tk.Frame(content_frame, bg=self.COLORS['bg_secondary'])
        values_frame.pack(fill='x', pady=(5, 0))

        # Old value
        old_text = f"Before: {metric.old_value:.1f}"
        if metric.unit:
            old_text += f" {metric.unit}"

        old_label = tk.Label(
            values_frame,
            text=old_text,
            font=('Segoe UI', 9),
            fg=self.COLORS['text_muted'],
            bg=self.COLORS['bg_secondary']
        )
        old_label.pack(side='left')

        # Arrow
        arrow_label = tk.Label(
            values_frame,
            text=" → ",
            font=('Segoe UI', 9),
            fg=self.COLORS['text_muted'],
            bg=self.COLORS['bg_secondary']
        )
        arrow_label.pack(side='left')

        # New value
        new_text = f"After: {metric.new_value:.1f}"
        if metric.unit:
            new_text += f" {metric.unit}"

        new_label = tk.Label(
            values_frame,
            text=new_text,
            font=('Segoe UI', 9),
            fg=self.COLORS['success'],
            bg=self.COLORS['bg_secondary']
        )
        new_label.pack(side='left')

    def _build_next_steps_tab(self):
        """Build next steps tab."""
        steps_frame = tk.Frame(self.notebook, bg=self.COLORS['bg_primary'])
        self.notebook.add(steps_frame, text="Next Steps")

        # Steps container
        steps_container = tk.Frame(steps_frame, bg=self.COLORS['bg_primary'])
        steps_container.pack(fill='both', expand=True, padx=20, pady=20)

        # Title
        title_label = tk.Label(
            steps_container,
            text="Recommended Next Steps:",
            font=('Segoe UI', 12, 'bold'),
            fg=self.COLORS['text_primary'],
            bg=self.COLORS['bg_primary']
        )
        title_label.pack(pady=(0, 15))

        # Steps list
        for step in self.next_steps:
            self._create_next_step_widget(steps_container, step)

    def _create_next_step_widget(self, parent, step: NextStep):
        """Create a next step widget."""
        # Step frame
        step_frame = tk.Frame(parent, bg=self.COLORS['bg_secondary'], relief='raised', bd=1)
        step_frame.pack(fill='x', pady=5)

        # Make clickable if action is provided
        if step.action:
            step_frame.configure(cursor='hand2')
            step_frame.bind('<Button-1>', lambda e: step.action())

        # Content frame
        content_frame = tk.Frame(step_frame, bg=self.COLORS['bg_secondary'])
        content_frame.pack(fill='x', padx=15, pady=10)

        # Step header
        header_frame = tk.Frame(content_frame, bg=self.COLORS['bg_secondary'])
        header_frame.pack(fill='x')

        # Icon
        icon_label = tk.Label(
            header_frame,
            text=step.icon,
            font=('Segoe UI', 12),
            fg=self.COLORS['accent'],
            bg=self.COLORS['bg_secondary']
        )
        icon_label.pack(side='left', padx=(0, 10))

        # Step title
        title_label = tk.Label(
            header_frame,
            text=step.title,
            font=('Segoe UI', 10, 'bold'),
            fg=self.COLORS['text_primary'],
            bg=self.COLORS['bg_secondary']
        )
        title_label.pack(side='left')

        # Step description
        desc_label = tk.Label(
            content_frame,
            text=step.description,
            font=('Segoe UI', 9),
            fg=self.COLORS['text_secondary'],
            bg=self.COLORS['bg_secondary'],
            anchor='w',
            justify='left',
            wraplength=600
        )
        desc_label.pack(fill='x', pady=(5, 0))

        # Make all child widgets clickable
        if step.action:
            for widget in [content_frame, header_frame, title_label, desc_label]:
                widget.configure(cursor='hand2')
                widget.bind('<Button-1>', lambda e: step.action())

    def _build_simple_summary_tab(self):
        """Build simple summary tab when no detailed content is available."""
        summary_frame = tk.Frame(self.notebook, bg=self.COLORS['bg_primary'])
        self.notebook.add(summary_frame, text="Summary")

        # Simple success message
        message_label = tk.Label(
            summary_frame,
            text="Your settings have been applied successfully!\n\nThe application is now configured according to your preferences.",
            font=('Segoe UI', 11),
            fg=self.COLORS['text_primary'],
            bg=self.COLORS['bg_primary'],
            justify='center'
        )
        message_label.pack(expand=True)

    def _build_action_buttons(self, parent):
        """Build action buttons at bottom of dialog."""
        button_frame = tk.Frame(parent, bg=self.COLORS['bg_primary'])
        button_frame.pack(fill='x', side='bottom')

        # Close button
        close_button = tk.Button(
            button_frame,
            text="Close",
            command=self._on_close,
            bg=self.COLORS['success'],
            fg=self.COLORS['text_primary'],
            relief='flat',
            padx=30,
            pady=10,
            font=('Segoe UI', 10, 'bold')
        )
        close_button.pack(side='right')

        # View details button (if applicable)
        if self.changes or self.metrics:
            details_button = tk.Button(
                button_frame,
                text="View Details",
                command=self._show_details,
                bg=self.COLORS['bg_secondary'],
                fg=self.COLORS['text_primary'],
                relief='flat',
                padx=20,
                pady=10,
                font=('Segoe UI', 9)
            )
            details_button.pack(side='right', padx=(0, 10))

    def _draw_success_icon(self):
        """Draw animated success checkmark icon."""
        self.icon_canvas.delete("all")

        # Background circle
        self.icon_canvas.create_oval(
            5, 5, 55, 55,
            fill=self.COLORS['success'],
            outline=""
        )

        # Checkmark
        self.icon_canvas.create_line(
            18, 30, 26, 38,
            fill=self.COLORS['text_primary'],
            width=4,
            capstyle='round'
        )
        self.icon_canvas.create_line(
            26, 38, 42, 22,
            fill=self.COLORS['text_primary'],
            width=4,
            capstyle='round'
        )

    def _start_icon_animation(self):
        """Start icon pulse animation."""
        self.icon_animation_frame = 0
        self._animate_icon()

    def _animate_icon(self):
        """Animate the success icon."""
        if not hasattr(self, 'dialog') or not self.dialog.winfo_exists():
            return

        self.icon_animation_frame += 1
        pulse_factor = 1 + 0.1 * math.sin(self.icon_animation_frame * 0.2)

        # Clear and redraw with pulse
        self.icon_canvas.delete("all")

        # Pulsing background circle
        size_offset = (pulse_factor - 1) * 5
        self.icon_canvas.create_oval(
            5 - size_offset, 5 - size_offset,
            55 + size_offset, 55 + size_offset,
            fill=self.COLORS['success'],
            outline=""
        )

        # Checkmark
        self.icon_canvas.create_line(
            18, 30, 26, 38,
            fill=self.COLORS['text_primary'],
            width=4,
            capstyle='round'
        )
        self.icon_canvas.create_line(
            26, 38, 42, 22,
            fill=self.COLORS['text_primary'],
            width=4,
            capstyle='round'
        )

        # Continue animation
        self.dialog.after(50, self._animate_icon)

    def _start_celebration(self):
        """Start celebration animation."""
        if self.show_celebration:
            # Update canvas size
            self.celebration_canvas.update_idletasks()
            self.celebration.start_celebration(3000)

    def _show_details(self):
        """Show additional details."""
        # Switch to first available detailed tab
        if self.changes:
            self.notebook.select(0)
        elif self.metrics:
            self.notebook.select(0)

    def _auto_close(self):
        """Auto-close dialog after timeout."""
        if hasattr(self, 'dialog') and self.dialog.winfo_exists():
            self._on_close()

    def _on_close(self):
        """Handle dialog close."""
        if hasattr(self, 'celebration'):
            self.celebration.stop_celebration()

        self.dialog.grab_release()
        self.dialog.destroy()

    def show(self):
        """Show dialog and wait for user action."""
        self.dialog.wait_window()


def show_success_dialog(parent: tk.Tk, title: str = "Success!", message: str = "",
                       changes: List[ChangesSummary] = None, metrics: List[PerformanceMetric] = None,
                       next_steps: List[NextStep] = None, celebration: bool = True):
    """Convenience function to show success dialog."""
    dialog = SuccessDialog(parent, title, message, changes, metrics, next_steps, celebration)
    dialog.show()


def create_settings_success(old_fps: float = None, new_fps: float = None,
                          old_latency: float = None, new_latency: float = None) -> tuple:
    """Create success dialog content for settings application."""
    changes = [
        ChangesSummary(
            category="Camera Settings",
            changes=["Resolution updated", "Frame rate optimized", "Exposure settings adjusted"],
            icon="📹"
        ),
        ChangesSummary(
            category="Detection Settings",
            changes=["Confidence threshold updated", "Model parameters optimized"],
            icon="🎯"
        ),
        ChangesSummary(
            category="Performance Settings",
            changes=["GPU acceleration enabled", "Memory usage optimized"],
            icon="⚡"
        )
    ]

    metrics = []
    if old_fps and new_fps:
        metrics.append(PerformanceMetric(
            name="Frames Per Second",
            old_value=old_fps,
            new_value=new_fps,
            unit="FPS"
        ))

    if old_latency and new_latency:
        metrics.append(PerformanceMetric(
            name="Detection Latency",
            old_value=old_latency,
            new_value=new_latency,
            unit="ms",
            is_better_higher=False
        ))

    next_steps = [
        NextStep(
            title="Test Your Settings",
            description="Try out the new settings with live camera feed to ensure everything works as expected.",
            icon="🧪"
        ),
        NextStep(
            title="Fine-tune Detection",
            description="Adjust detection confidence and other parameters based on your specific use case.",
            icon="🎛"
        ),
        NextStep(
            title="Save Configuration",
            description="Consider saving your current settings as a preset for future use.",
            icon="💾"
        )
    ]

    return changes, metrics, next_steps