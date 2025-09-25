"""Notification manager for toast notifications, status bar updates, and system notifications."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
import threading
import time
import sys
import os
from typing import Optional, Callable, Dict, Any, List
from enum import Enum
from dataclasses import dataclass
import queue


class NotificationType(Enum):
    """Types of notifications."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class SoundEvent(Enum):
    """Sound events for notifications."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    NOTIFICATION = "notification"


@dataclass
class Notification:
    """Notification data structure."""
    message: str
    type: NotificationType
    duration_ms: int = 3000
    title: Optional[str] = None
    callback: Optional[Callable] = None
    show_close: bool = True


class ToastNotification(tk.Toplevel):
    """Toast notification window."""

    # Color scheme
    COLORS = {
        'bg_primary': '#1e1e1e',
        'bg_secondary': '#2d2d2d',
        'text_primary': '#ffffff',
        'text_secondary': '#cccccc',
        'info': '#2196f3',
        'success': '#4caf50',
        'warning': '#ff9800',
        'error': '#f44336',
        'border': '#404040',
    }

    TYPE_COLORS = {
        NotificationType.INFO: COLORS['info'],
        NotificationType.SUCCESS: COLORS['success'],
        NotificationType.WARNING: COLORS['warning'],
        NotificationType.ERROR: COLORS['error'],
    }

    TYPE_SYMBOLS = {
        NotificationType.INFO: "ℹ",
        NotificationType.SUCCESS: "✓",
        NotificationType.WARNING: "⚠",
        NotificationType.ERROR: "✗",
    }

    def __init__(self, parent: tk.Tk, notification: Notification):
        """Initialize toast notification."""
        super().__init__(parent)

        self.notification = notification
        self.fade_in_steps = 10
        self.fade_out_steps = 10
        self.current_alpha = 0.0
        self.target_alpha = 0.9
        self.is_fading_out = False

        # Configure window
        self.configure_window()
        self.build_ui()
        self.position_window()

        # Start fade in animation
        self.fade_in()

        # Schedule auto-close
        if notification.duration_ms > 0:
            self.after(notification.duration_ms, self.start_fade_out)

    def configure_window(self):
        """Configure the toast window."""
        self.withdraw()  # Hide initially
        self.overrideredirect(True)  # Remove window decorations
        self.configure(bg=self.COLORS['bg_primary'])

        # Make window stay on top
        self.attributes('-topmost', True)

        # Set transparency
        self.attributes('-alpha', 0.0)

    def build_ui(self):
        """Build the toast UI."""
        # Main frame with border
        main_frame = tk.Frame(
            self,
            bg=self.TYPE_COLORS[self.notification.type],
            padx=2,
            pady=2
        )
        main_frame.pack(fill='both', expand=True)

        # Content frame
        content_frame = tk.Frame(
            main_frame,
            bg=self.COLORS['bg_secondary'],
            padx=15,
            pady=10
        )
        content_frame.pack(fill='both', expand=True)

        # Icon and message frame
        message_frame = tk.Frame(content_frame, bg=self.COLORS['bg_secondary'])
        message_frame.pack(fill='x')

        # Icon
        icon_label = tk.Label(
            message_frame,
            text=self.TYPE_SYMBOLS[self.notification.type],
            font=('Segoe UI', 12),
            fg=self.TYPE_COLORS[self.notification.type],
            bg=self.COLORS['bg_secondary']
        )
        icon_label.pack(side='left', padx=(0, 10))

        # Message container
        text_frame = tk.Frame(message_frame, bg=self.COLORS['bg_secondary'])
        text_frame.pack(side='left', fill='x', expand=True)

        # Title (if provided)
        if self.notification.title:
            title_label = tk.Label(
                text_frame,
                text=self.notification.title,
                font=('Segoe UI', 9, 'bold'),
                fg=self.COLORS['text_primary'],
                bg=self.COLORS['bg_secondary'],
                anchor='w'
            )
            title_label.pack(fill='x')

        # Message
        message_label = tk.Label(
            text_frame,
            text=self.notification.message,
            font=('Segoe UI', 9),
            fg=self.COLORS['text_secondary'],
            bg=self.COLORS['bg_secondary'],
            anchor='w',
            wraplength=300,
            justify='left'
        )
        message_label.pack(fill='x')

        # Close button (if enabled)
        if self.notification.show_close:
            close_button = tk.Label(
                message_frame,
                text="×",
                font=('Segoe UI', 12, 'bold'),
                fg=self.COLORS['text_secondary'],
                bg=self.COLORS['bg_secondary'],
                cursor='hand2'
            )
            close_button.pack(side='right', padx=(10, 0))
            close_button.bind('<Button-1>', lambda e: self.start_fade_out())
            close_button.bind('<Enter>', lambda e: close_button.configure(fg=self.COLORS['text_primary']))
            close_button.bind('<Leave>', lambda e: close_button.configure(fg=self.COLORS['text_secondary']))

        # Action callback (click entire toast)
        if self.notification.callback:
            self.bind('<Button-1>', lambda e: self._on_click())
            content_frame.bind('<Button-1>', lambda e: self._on_click())
            message_frame.bind('<Button-1>', lambda e: self._on_click())
            self.configure(cursor='hand2')

    def position_window(self):
        """Position the toast window."""
        self.update_idletasks()

        # Get screen dimensions
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Get window dimensions
        window_width = self.winfo_reqwidth()
        window_height = self.winfo_reqheight()

        # Position in bottom-right corner with margin
        margin = 20
        x = screen_width - window_width - margin
        y = screen_height - window_height - margin - 40  # Account for taskbar

        self.geometry(f"+{x}+{y}")

    def fade_in(self):
        """Fade in animation."""
        if self.current_alpha < self.target_alpha:
            self.current_alpha += self.target_alpha / self.fade_in_steps
            self.attributes('-alpha', self.current_alpha)
            self.deiconify()  # Show window
            self.after(30, self.fade_in)

    def start_fade_out(self):
        """Start fade out animation."""
        if not self.is_fading_out:
            self.is_fading_out = True
            self.fade_out()

    def fade_out(self):
        """Fade out animation."""
        if self.current_alpha > 0:
            self.current_alpha -= self.target_alpha / self.fade_out_steps
            self.attributes('-alpha', max(0, self.current_alpha))
            self.after(30, self.fade_out)
        else:
            self.destroy()

    def _on_click(self):
        """Handle toast click."""
        if self.notification.callback:
            self.notification.callback()
        self.start_fade_out()


class NotificationManager:
    """Manager for all notification types."""

    def __init__(self, parent: tk.Tk):
        """Initialize notification manager."""
        self.parent = parent
        self.status_bar: Optional[Any] = None
        self.active_toasts: List[ToastNotification] = []
        self.notification_queue = queue.Queue()
        self.max_concurrent_toasts = 5
        self.toast_spacing = 10

        # Sound settings
        self.sound_enabled = True
        self.sound_callbacks: Dict[SoundEvent, Callable] = {}

        # Start notification processor
        self._start_notification_processor()

    def set_status_bar(self, status_bar):
        """Set reference to status bar component."""
        self.status_bar = status_bar

    def show_toast(self, message: str, type: NotificationType = NotificationType.INFO,
                   title: Optional[str] = None, duration_ms: int = 3000,
                   callback: Optional[Callable] = None, show_close: bool = True):
        """Show toast notification."""
        notification = Notification(
            message=message,
            type=type,
            title=title,
            duration_ms=duration_ms,
            callback=callback,
            show_close=show_close
        )

        # Add to queue for processing
        self.notification_queue.put(('toast', notification))

    def show_success(self, message: str, title: str = "Success", duration_ms: int = 2000):
        """Show success toast."""
        self.show_toast(message, NotificationType.SUCCESS, title, duration_ms)
        self.play_sound(SoundEvent.SUCCESS)

    def show_error(self, message: str, title: str = "Error", duration_ms: int = 5000):
        """Show error toast."""
        self.show_toast(message, NotificationType.ERROR, title, duration_ms)
        self.play_sound(SoundEvent.ERROR)

    def show_warning(self, message: str, title: str = "Warning", duration_ms: int = 4000):
        """Show warning toast."""
        self.show_toast(message, NotificationType.WARNING, title, duration_ms)
        self.play_sound(SoundEvent.WARNING)

    def show_info(self, message: str, title: str = "Information", duration_ms: int = 3000):
        """Show info toast."""
        self.show_toast(message, NotificationType.INFO, title, duration_ms)

    def update_status_bar(self, message: str, type: NotificationType = NotificationType.INFO):
        """Update status bar message."""
        if self.status_bar and hasattr(self.status_bar, 'set_status'):
            self.status_bar.set_status(message)

        # Also queue for processing
        self.notification_queue.put(('status', {'message': message, 'type': type}))

    def show_system_notification(self, title: str, body: str, type: NotificationType = NotificationType.INFO):
        """Show system tray notification (Windows)."""
        try:
            if sys.platform == 'win32':
                import plyer
                plyer.notification.notify(
                    title=title,
                    message=body,
                    timeout=5
                )
        except ImportError:
            # Fallback to toast if plyer not available
            self.show_toast(body, type, title)
        except Exception:
            # Fallback to toast on any error
            self.show_toast(body, type, title)

    def play_sound(self, event: SoundEvent):
        """Play sound for event."""
        if not self.sound_enabled:
            return

        if event in self.sound_callbacks:
            try:
                self.sound_callbacks[event]()
            except Exception:
                pass  # Ignore sound errors
        else:
            # Fallback to system sounds
            try:
                if sys.platform == 'win32':
                    import winsound
                    if event == SoundEvent.ERROR:
                        winsound.MessageBeep(winsound.MB_ICONHAND)
                    elif event == SoundEvent.WARNING:
                        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                    elif event == SoundEvent.SUCCESS:
                        winsound.MessageBeep(winsound.MB_OK)
                    else:
                        winsound.MessageBeep(winsound.MB_ICONASTERISK)
            except Exception:
                pass  # Ignore sound errors

    def set_sound_callback(self, event: SoundEvent, callback: Callable):
        """Set custom sound callback for an event."""
        self.sound_callbacks[event] = callback

    def enable_sound(self, enabled: bool = True):
        """Enable or disable sound notifications."""
        self.sound_enabled = enabled

    def clear_all_toasts(self):
        """Clear all active toast notifications."""
        for toast in self.active_toasts[:]:
            toast.start_fade_out()

    def _start_notification_processor(self):
        """Start the notification processing thread."""
        def process_notifications():
            while True:
                try:
                    notification_type, data = self.notification_queue.get(timeout=1)

                    if notification_type == 'toast':
                        self._process_toast_notification(data)
                    elif notification_type == 'status':
                        self._process_status_update(data)

                    self.notification_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Notification processing error: {e}")

        thread = threading.Thread(target=process_notifications, daemon=True)
        thread.start()

    def _process_toast_notification(self, notification: Notification):
        """Process toast notification on main thread."""
        def show_toast():
            # Limit concurrent toasts
            if len(self.active_toasts) >= self.max_concurrent_toasts:
                self.active_toasts[0].start_fade_out()

            # Create and position toast
            toast = ToastNotification(self.parent, notification)
            self.active_toasts.append(toast)

            # Position relative to other toasts
            self._position_toast(toast)

            # Remove from active list when destroyed
            def on_destroy():
                if toast in self.active_toasts:
                    self.active_toasts.remove(toast)
                self._reposition_toasts()

            toast.bind('<Destroy>', lambda e: on_destroy())

        # Schedule on main thread
        self.parent.after_idle(show_toast)

    def _process_status_update(self, data: Dict):
        """Process status bar update."""
        def update_status():
            if self.status_bar and hasattr(self.status_bar, 'set_status'):
                self.status_bar.set_status(data['message'])

        # Schedule on main thread
        self.parent.after_idle(update_status)

    def _position_toast(self, toast: ToastNotification):
        """Position toast relative to existing toasts."""
        if len(self.active_toasts) <= 1:
            return  # First toast, already positioned

        # Calculate y offset based on existing toasts
        total_offset = 0
        for existing_toast in self.active_toasts[:-1]:  # Exclude current toast
            if existing_toast.winfo_exists():
                total_offset += existing_toast.winfo_reqheight() + self.toast_spacing

        # Get current position
        current_geometry = toast.geometry()
        if '+' in current_geometry:
            # Parse current position
            parts = current_geometry.split('+')
            if len(parts) >= 3:
                x = int(parts[1])
                y = int(parts[2])

                # Adjust y position
                new_y = y - total_offset
                toast.geometry(f"+{x}+{new_y}")

    def _reposition_toasts(self):
        """Reposition all active toasts after one is removed."""
        def reposition():
            screen_height = self.parent.winfo_screenheight()
            margin = 20
            taskbar_height = 40

            current_y = screen_height - margin - taskbar_height

            for toast in reversed(self.active_toasts):  # Bottom to top
                if toast.winfo_exists():
                    toast_height = toast.winfo_reqheight()
                    current_y -= toast_height

                    # Get current x position
                    current_geometry = toast.geometry()
                    if '+' in current_geometry:
                        parts = current_geometry.split('+')
                        if len(parts) >= 2:
                            x = int(parts[1])
                            toast.geometry(f"+{x}+{current_y}")

                    current_y -= self.toast_spacing

        # Schedule on main thread with delay to allow for smooth animation
        self.parent.after(100, reposition)

    def shutdown(self):
        """Shutdown notification manager."""
        self.clear_all_toasts()


# Global notification manager instance
_notification_manager: Optional[NotificationManager] = None


def get_notification_manager(parent: Optional[tk.Tk] = None) -> NotificationManager:
    """Get global notification manager instance."""
    global _notification_manager
    if _notification_manager is None and parent:
        _notification_manager = NotificationManager(parent)
    return _notification_manager


def show_toast(message: str, type: NotificationType = NotificationType.INFO, **kwargs):
    """Convenience function to show toast notification."""
    manager = get_notification_manager()
    if manager:
        manager.show_toast(message, type, **kwargs)


def show_success(message: str, **kwargs):
    """Convenience function to show success notification."""
    manager = get_notification_manager()
    if manager:
        manager.show_success(message, **kwargs)


def show_error(message: str, **kwargs):
    """Convenience function to show error notification."""
    manager = get_notification_manager()
    if manager:
        manager.show_error(message, **kwargs)


def show_warning(message: str, **kwargs):
    """Convenience function to show warning notification."""
    manager = get_notification_manager()
    if manager:
        manager.show_warning(message, **kwargs)


def show_info(message: str, **kwargs):
    """Convenience function to show info notification."""
    manager = get_notification_manager()
    if manager:
        manager.show_info(message, **kwargs)