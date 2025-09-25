"""Status indicator component with visual states and animations."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
import math
import time
from typing import Optional, Dict, Any
from enum import Enum


class IndicatorState(Enum):
    """Visual states for status indicators."""
    IDLE = "idle"
    VALIDATING = "validating"
    APPLYING = "applying"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class StatusIndicator(tk.Canvas):
    """Animated status indicator with color-coded states."""

    # Color scheme
    COLORS = {
        'bg_primary': '#1e1e1e',
        'bg_secondary': '#2d2d2d',
        'text_primary': '#ffffff',
        'text_secondary': '#cccccc',
        'text_muted': '#999999',
        'idle': '#666666',
        'validating': '#2196f3',
        'applying': '#4caf50',
        'warning': '#ff9800',
        'error': '#f44336',
        'success': '#4caf50',
        'rolled_back': '#ff9800',
        'cancelled': '#999999',
    }

    # Unicode symbols for each state
    SYMBOLS = {
        IndicatorState.IDLE: "○",
        IndicatorState.VALIDATING: "◐",
        IndicatorState.APPLYING: "◑",
        IndicatorState.WARNING: "⚠",
        IndicatorState.ERROR: "✗",
        IndicatorState.SUCCESS: "✓",
        IndicatorState.ROLLED_BACK: "↺",
        IndicatorState.CANCELLED: "⊘",
    }

    def __init__(self, parent, size: int = 20, show_text: bool = True):
        """Initialize status indicator."""
        super().__init__(
            parent,
            width=size,
            height=size,
            bg=self.COLORS['bg_primary'],
            highlightthickness=0
        )

        self.size = size
        self.show_text = show_text
        self.current_state = IndicatorState.IDLE
        self.animation_frame = 0
        self.animation_running = False

        # Text label (optional)
        self.text_label: Optional[tk.Label] = None
        if show_text:
            self.text_var = tk.StringVar(value="Ready")
            self.text_label = tk.Label(
                parent,
                textvariable=self.text_var,
                font=('Segoe UI', 9),
                fg=self.COLORS['text_secondary'],
                bg=self.COLORS['bg_primary']
            )

        # Start with idle state
        self.set_state(IndicatorState.IDLE)

    def set_state(self, state: IndicatorState, text: Optional[str] = None):
        """Set indicator state and update visual."""
        self.current_state = state

        # Update text if provided and text label exists
        if text and self.text_label:
            self.text_var.set(text)

        # Start/stop animation based on state
        if state in [IndicatorState.VALIDATING, IndicatorState.APPLYING]:
            if not self.animation_running:
                self.animation_running = True
                self._animate()
        else:
            self.animation_running = False
            self._draw_static()

    def _animate(self):
        """Animate the indicator for active states."""
        if not self.animation_running:
            return

        self.animation_frame += 1
        self._draw_animated()

        # Schedule next frame
        self.after(50, self._animate)

    def _draw_static(self):
        """Draw static indicator."""
        self.delete("all")

        color = self.COLORS.get(self.current_state.value, self.COLORS['idle'])
        center = self.size // 2

        if self.current_state == IndicatorState.SUCCESS:
            # Draw checkmark
            self._draw_checkmark(center, color)
        elif self.current_state == IndicatorState.ERROR:
            # Draw X
            self._draw_x(center, color)
        elif self.current_state == IndicatorState.WARNING:
            # Draw triangle with exclamation
            self._draw_warning(center, color)
        elif self.current_state == IndicatorState.ROLLED_BACK:
            # Draw curved arrow
            self._draw_rollback(center, color)
        elif self.current_state == IndicatorState.CANCELLED:
            # Draw circle with line through
            self._draw_cancelled(center, color)
        else:
            # Draw simple circle
            self.create_oval(
                2, 2, self.size - 2, self.size - 2,
                fill=color, outline="", width=0
            )

    def _draw_animated(self):
        """Draw animated indicator for active states."""
        self.delete("all")

        center = self.size // 2
        color = self.COLORS.get(self.current_state.value, self.COLORS['validating'])

        if self.current_state == IndicatorState.VALIDATING:
            # Spinning loading indicator
            self._draw_spinner(center, color)
        elif self.current_state == IndicatorState.APPLYING:
            # Pulsing circle
            self._draw_pulse(center, color)

    def _draw_spinner(self, center: int, color: str):
        """Draw spinning loading indicator."""
        angle_offset = (self.animation_frame * 15) % 360

        # Draw multiple arcs to create spinner effect
        for i in range(8):
            angle = (i * 45 + angle_offset) % 360
            arc_extent = 30

            # Calculate alpha based on position
            alpha = 1.0 - (i / 8.0) * 0.7
            arc_color = self._alpha_blend(color, alpha)

            self.create_arc(
                3, 3, self.size - 3, self.size - 3,
                start=angle, extent=arc_extent,
                outline=arc_color, width=2, style='arc'
            )

    def _draw_pulse(self, center: int, color: str):
        """Draw pulsing circle."""
        pulse_phase = (self.animation_frame * 0.2) % (2 * math.pi)
        scale_factor = 0.7 + 0.3 * (math.sin(pulse_phase) + 1) / 2

        radius = (self.size // 2 - 2) * scale_factor

        self.create_oval(
            center - radius, center - radius,
            center + radius, center + radius,
            fill=color, outline="", width=0
        )

    def _draw_checkmark(self, center: int, color: str):
        """Draw checkmark symbol."""
        # Simple checkmark path
        size_factor = self.size / 20
        points = [
            center - 4 * size_factor, center,
            center - 1 * size_factor, center + 3 * size_factor,
            center + 4 * size_factor, center - 2 * size_factor
        ]

        self.create_line(points, fill=color, width=int(2 * size_factor), capstyle='round')

    def _draw_x(self, center: int, color: str):
        """Draw X symbol."""
        size_factor = self.size / 20
        offset = 4 * size_factor

        # Draw X as two lines
        self.create_line(
            center - offset, center - offset,
            center + offset, center + offset,
            fill=color, width=int(2 * size_factor), capstyle='round'
        )
        self.create_line(
            center - offset, center + offset,
            center + offset, center - offset,
            fill=color, width=int(2 * size_factor), capstyle='round'
        )

    def _draw_warning(self, center: int, color: str):
        """Draw warning triangle."""
        size_factor = self.size / 20

        # Triangle outline
        points = [
            center, center - 6 * size_factor,  # top
            center - 5 * size_factor, center + 4 * size_factor,  # bottom left
            center + 5 * size_factor, center + 4 * size_factor,  # bottom right
        ]

        self.create_polygon(points, fill=color, outline="")

        # Exclamation mark
        self.create_oval(
            center - size_factor, center + 2 * size_factor,
            center + size_factor, center + 4 * size_factor,
            fill=self.COLORS['bg_primary'], outline=""
        )
        self.create_line(
            center, center - 3 * size_factor,
            center, center + size_factor,
            fill=self.COLORS['bg_primary'], width=int(2 * size_factor)
        )

    def _draw_rollback(self, center: int, color: str):
        """Draw rollback arrow."""
        size_factor = self.size / 20

        # Curved arrow (simplified as arc with arrow head)
        self.create_arc(
            center - 6 * size_factor, center - 6 * size_factor,
            center + 6 * size_factor, center + 6 * size_factor,
            start=45, extent=270,
            outline=color, width=int(2 * size_factor), style='arc'
        )

        # Arrow head
        arrow_points = [
            center - 3 * size_factor, center - 6 * size_factor,
            center, center - 3 * size_factor,
            center - 6 * size_factor, center - 3 * size_factor
        ]
        self.create_polygon(arrow_points, fill=color, outline="")

    def _draw_cancelled(self, center: int, color: str):
        """Draw cancelled indicator (circle with diagonal line)."""
        size_factor = self.size / 20

        # Circle
        self.create_oval(
            center - 6 * size_factor, center - 6 * size_factor,
            center + 6 * size_factor, center + 6 * size_factor,
            outline=color, width=int(2 * size_factor)
        )

        # Diagonal line
        self.create_line(
            center - 4 * size_factor, center - 4 * size_factor,
            center + 4 * size_factor, center + 4 * size_factor,
            fill=color, width=int(2 * size_factor)
        )

    def _alpha_blend(self, hex_color: str, alpha: float) -> str:
        """Blend color with alpha transparency (simulate with brightness)."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        # Blend with background color based on alpha
        bg_r, bg_g, bg_b = 30, 30, 30  # Dark background RGB

        r = int(r * alpha + bg_r * (1 - alpha))
        g = int(g * alpha + bg_g * (1 - alpha))
        b = int(b * alpha + bg_b * (1 - alpha))

        return f"#{r:02x}{g:02x}{b:02x}"

    def pack_with_text(self, **kwargs):
        """Pack indicator with optional text label."""
        # Create frame to hold both indicator and text
        frame = tk.Frame(self.master, bg=self.COLORS['bg_primary'])

        # Reparent indicator to frame
        self.master = frame
        self.pack(side='left')

        if self.text_label:
            self.text_label.master = frame
            self.text_label.pack(side='left', padx=(5, 0))

        frame.pack(**kwargs)
        return frame

    def destroy(self):
        """Clean up indicator and text label."""
        if self.text_label:
            self.text_label.destroy()
        super().destroy()


class ServiceStatusIndicator(tk.Frame):
    """Compound indicator showing service status with name and state."""

    def __init__(self, parent, service_name: str, initial_state: IndicatorState = IndicatorState.IDLE):
        """Initialize service status indicator."""
        super().__init__(parent, bg=StatusIndicator.COLORS['bg_primary'])

        self.service_name = service_name
        self.current_state = initial_state

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build the service status UI."""
        # Service name label
        self.name_label = tk.Label(
            self,
            text=self.service_name,
            font=('Segoe UI', 9, 'bold'),
            fg=StatusIndicator.COLORS['text_primary'],
            bg=StatusIndicator.COLORS['bg_primary'],
            width=15,
            anchor='w'
        )
        self.name_label.pack(side='left', padx=(0, 10))

        # Status indicator
        self.indicator = StatusIndicator(self, size=16, show_text=False)
        self.indicator.pack(side='left')

        # Status text
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(
            self,
            textvariable=self.status_var,
            font=('Segoe UI', 9),
            fg=StatusIndicator.COLORS['text_secondary'],
            bg=StatusIndicator.COLORS['bg_primary'],
            anchor='w'
        )
        self.status_label.pack(side='left', padx=(5, 0), fill='x', expand=True)

    def set_state(self, state: IndicatorState, status_text: str = ""):
        """Update service state and status text."""
        self.current_state = state
        self.indicator.set_state(state)

        if status_text:
            self.status_var.set(status_text)
        else:
            # Default status text based on state
            default_texts = {
                IndicatorState.IDLE: "Ready",
                IndicatorState.VALIDATING: "Validating...",
                IndicatorState.APPLYING: "Applying changes...",
                IndicatorState.WARNING: "Warning",
                IndicatorState.ERROR: "Error",
                IndicatorState.SUCCESS: "Success",
                IndicatorState.ROLLED_BACK: "Rolled back",
                IndicatorState.CANCELLED: "Cancelled",
            }
            self.status_var.set(default_texts.get(state, "Unknown"))

    def set_error(self, error_message: str):
        """Set error state with message."""
        self.set_state(IndicatorState.ERROR, f"Error: {error_message}")

    def set_success(self, success_message: str = ""):
        """Set success state with optional message."""
        message = success_message or "Operation completed"
        self.set_state(IndicatorState.SUCCESS, message)


class StatusIndicatorPanel(tk.Frame):
    """Panel containing multiple service status indicators."""

    def __init__(self, parent, services: list[str]):
        """Initialize status panel with list of services."""
        super().__init__(parent, bg=StatusIndicator.COLORS['bg_secondary'])

        self.services = services
        self.indicators: Dict[str, ServiceStatusIndicator] = {}

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build the status panel UI."""
        # Title
        title_label = tk.Label(
            self,
            text="Service Status",
            font=('Segoe UI', 10, 'bold'),
            fg=StatusIndicator.COLORS['text_primary'],
            bg=StatusIndicator.COLORS['bg_secondary']
        )
        title_label.pack(pady=(10, 5), padx=10, anchor='w')

        # Separator
        separator = ttk.Separator(self, orient='horizontal')
        separator.pack(fill='x', padx=10, pady=(0, 10))

        # Service indicators
        for service in self.services:
            indicator = ServiceStatusIndicator(self, service)
            indicator.pack(fill='x', padx=10, pady=2)
            self.indicators[service] = indicator

    def set_service_state(self, service: str, state: IndicatorState, status_text: str = ""):
        """Update specific service state."""
        if service in self.indicators:
            self.indicators[service].set_state(state, status_text)

    def set_service_error(self, service: str, error_message: str):
        """Set service error state."""
        if service in self.indicators:
            self.indicators[service].set_error(error_message)

    def set_service_success(self, service: str, success_message: str = ""):
        """Set service success state."""
        if service in self.indicators:
            self.indicators[service].set_success(success_message)

    def reset_all(self):
        """Reset all services to idle state."""
        for indicator in self.indicators.values():
            indicator.set_state(IndicatorState.IDLE)

    def get_service_state(self, service: str) -> Optional[IndicatorState]:
        """Get current state of a service."""
        if service in self.indicators:
            return self.indicators[service].current_state
        return None