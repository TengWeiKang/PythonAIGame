"""Live status panel for real-time updates during settings application."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
import threading
import time
import queue
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json


class ServiceStatus(Enum):
    """Service status states."""
    UNKNOWN = "unknown"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    RESTARTING = "restarting"


class LogLevel(Enum):
    """Log message levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ServiceInfo:
    """Service information."""
    name: str
    display_name: str
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_update: float = 0
    error_message: Optional[str] = None
    health_check: Optional[Callable] = None


@dataclass
class LogEntry:
    """Log entry data."""
    timestamp: float
    level: LogLevel
    message: str
    source: str = "System"


@dataclass
class MetricEntry:
    """Performance metric entry."""
    name: str
    value: float
    unit: str = ""
    timestamp: float = 0
    max_value: Optional[float] = None


class LiveStatusPanel(tk.Frame):
    """Panel showing real-time status updates and logs."""

    # Color scheme
    COLORS = {
        'bg_primary': '#1e1e1e',
        'bg_secondary': '#2d2d2d',
        'bg_tertiary': '#3c3c3c',
        'text_primary': '#ffffff',
        'text_secondary': '#cccccc',
        'text_muted': '#999999',
        'success': '#4caf50',
        'warning': '#ff9800',
        'error': '#f44336',
        'info': '#2196f3',
        'border': '#404040',
        'accent': '#007acc',
    }

    # Status colors
    STATUS_COLORS = {
        ServiceStatus.UNKNOWN: COLORS['text_muted'],
        ServiceStatus.STARTING: COLORS['warning'],
        ServiceStatus.RUNNING: COLORS['success'],
        ServiceStatus.STOPPING: COLORS['warning'],
        ServiceStatus.STOPPED: COLORS['text_muted'],
        ServiceStatus.ERROR: COLORS['error'],
        ServiceStatus.RESTARTING: COLORS['info'],
    }

    # Log level colors
    LOG_COLORS = {
        LogLevel.DEBUG: COLORS['text_muted'],
        LogLevel.INFO: COLORS['text_secondary'],
        LogLevel.WARNING: COLORS['warning'],
        LogLevel.ERROR: COLORS['error'],
        LogLevel.CRITICAL: COLORS['error'],
    }

    def __init__(self, parent, services: List[str] = None, show_logs: bool = True,
                 show_metrics: bool = True, max_log_entries: int = 1000):
        """Initialize live status panel."""
        super().__init__(parent, bg=self.COLORS['bg_primary'])

        self.services = services or []
        self.show_logs = show_logs
        self.show_metrics = show_metrics
        self.max_log_entries = max_log_entries

        # Data storage
        self.service_info: Dict[str, ServiceInfo] = {}
        self.log_entries: List[LogEntry] = []
        self.metrics: Dict[str, MetricEntry] = {}

        # Update queues for thread safety
        self.service_queue = queue.Queue()
        self.log_queue = queue.Queue()
        self.metric_queue = queue.Queue()

        # UI state
        self.auto_scroll_logs = True
        self.log_filter_level = LogLevel.DEBUG
        self.is_paused = False

        # Initialize services
        for service_name in self.services:
            self.service_info[service_name] = ServiceInfo(
                name=service_name,
                display_name=service_name.replace('_', ' ').title()
            )

        # Build UI
        self._build_ui()

        # Start update loop
        self._start_update_loop()

    def _build_ui(self):
        """Build the live status panel UI."""
        # Header
        header_frame = tk.Frame(self, bg=self.COLORS['bg_primary'])
        header_frame.pack(fill='x', pady=(0, 10))

        # Title
        title_label = tk.Label(
            header_frame,
            text="Live Status",
            font=('Segoe UI', 12, 'bold'),
            fg=self.COLORS['text_primary'],
            bg=self.COLORS['bg_primary']
        )
        title_label.pack(side='left')

        # Controls
        controls_frame = tk.Frame(header_frame, bg=self.COLORS['bg_primary'])
        controls_frame.pack(side='right')

        # Pause/Resume button
        self.pause_button = tk.Button(
            controls_frame,
            text="Pause",
            command=self._toggle_pause,
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_primary'],
            relief='flat',
            padx=10,
            pady=2,
            font=('Segoe UI', 8)
        )
        self.pause_button.pack(side='right', padx=(5, 0))

        # Clear logs button
        clear_button = tk.Button(
            controls_frame,
            text="Clear",
            command=self._clear_logs,
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_primary'],
            relief='flat',
            padx=10,
            pady=2,
            font=('Segoe UI', 8)
        )
        clear_button.pack(side='right', padx=(5, 0))

        # Create notebook for different views
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)

        # Services tab
        self._build_services_tab()

        # Logs tab (if enabled)
        if self.show_logs:
            self._build_logs_tab()

        # Metrics tab (if enabled)
        if self.show_metrics:
            self._build_metrics_tab()

    def _build_services_tab(self):
        """Build services status tab."""
        services_frame = tk.Frame(self.notebook, bg=self.COLORS['bg_primary'])
        self.notebook.add(services_frame, text="Services")

        # Services list
        self.services_frame = tk.Frame(services_frame, bg=self.COLORS['bg_primary'])
        self.services_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Service widgets
        self.service_widgets: Dict[str, Dict] = {}
        self._create_service_widgets()

    def _create_service_widgets(self):
        """Create widgets for each service."""
        for service_name, service_info in self.service_info.items():
            service_frame = tk.Frame(
                self.services_frame,
                bg=self.COLORS['bg_secondary'],
                relief='raised',
                bd=1
            )
            service_frame.pack(fill='x', pady=5)

            # Service header
            header_frame = tk.Frame(service_frame, bg=self.COLORS['bg_secondary'])
            header_frame.pack(fill='x', padx=10, pady=5)

            # Service name
            name_label = tk.Label(
                header_frame,
                text=service_info.display_name,
                font=('Segoe UI', 10, 'bold'),
                fg=self.COLORS['text_primary'],
                bg=self.COLORS['bg_secondary']
            )
            name_label.pack(side='left')

            # Status indicator
            status_var = tk.StringVar(value="Unknown")
            status_label = tk.Label(
                header_frame,
                textvariable=status_var,
                font=('Segoe UI', 9),
                fg=self.STATUS_COLORS[service_info.status],
                bg=self.COLORS['bg_secondary']
            )
            status_label.pack(side='right')

            # Last update time
            update_var = tk.StringVar(value="")
            update_label = tk.Label(
                header_frame,
                textvariable=update_var,
                font=('Segoe UI', 8),
                fg=self.COLORS['text_muted'],
                bg=self.COLORS['bg_secondary']
            )
            update_label.pack(side='right', padx=(0, 10))

            # Error message (initially hidden)
            error_var = tk.StringVar(value="")
            error_label = tk.Label(
                service_frame,
                textvariable=error_var,
                font=('Segoe UI', 8),
                fg=self.COLORS['error'],
                bg=self.COLORS['bg_secondary'],
                wraplength=400,
                justify='left'
            )

            # Store widget references
            self.service_widgets[service_name] = {
                'frame': service_frame,
                'header': header_frame,
                'name_label': name_label,
                'status_var': status_var,
                'status_label': status_label,
                'update_var': update_var,
                'update_label': update_label,
                'error_var': error_var,
                'error_label': error_label
            }

    def _build_logs_tab(self):
        """Build logs tab."""
        logs_frame = tk.Frame(self.notebook, bg=self.COLORS['bg_primary'])
        self.notebook.add(logs_frame, text="Logs")

        # Log controls
        controls_frame = tk.Frame(logs_frame, bg=self.COLORS['bg_primary'])
        controls_frame.pack(fill='x', padx=10, pady=5)

        # Filter level
        tk.Label(
            controls_frame,
            text="Level:",
            font=('Segoe UI', 9),
            fg=self.COLORS['text_secondary'],
            bg=self.COLORS['bg_primary']
        ).pack(side='left', padx=(0, 5))

        self.level_var = tk.StringVar(value=self.log_filter_level.value)
        level_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.level_var,
            values=[level.value for level in LogLevel],
            state='readonly',
            width=10
        )
        level_combo.pack(side='left', padx=(0, 10))
        level_combo.bind('<<ComboboxSelected>>', self._on_level_filter_change)

        # Auto-scroll checkbox
        self.auto_scroll_var = tk.BooleanVar(value=self.auto_scroll_logs)
        auto_scroll_check = tk.Checkbutton(
            controls_frame,
            text="Auto-scroll",
            variable=self.auto_scroll_var,
            command=self._on_auto_scroll_change,
            fg=self.COLORS['text_secondary'],
            bg=self.COLORS['bg_primary'],
            selectcolor=self.COLORS['bg_secondary']
        )
        auto_scroll_check.pack(side='left')

        # Log text area
        log_frame = tk.Frame(logs_frame, bg=self.COLORS['bg_primary'])
        log_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        self.log_text = tk.Text(
            log_frame,
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_secondary'],
            font=('Consolas', 8),
            wrap='word',
            state='disabled'
        )

        log_scrollbar = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        self.log_text.pack(side='left', fill='both', expand=True)
        log_scrollbar.pack(side='right', fill='y')

        # Configure text tags for different log levels
        for level in LogLevel:
            self.log_text.tag_configure(
                level.value,
                foreground=self.LOG_COLORS[level]
            )

    def _build_metrics_tab(self):
        """Build metrics tab."""
        metrics_frame = tk.Frame(self.notebook, bg=self.COLORS['bg_primary'])
        self.notebook.add(metrics_frame, text="Metrics")

        # Metrics container
        self.metrics_frame = tk.Frame(metrics_frame, bg=self.COLORS['bg_primary'])
        self.metrics_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Metric widgets
        self.metric_widgets: Dict[str, Dict] = {}

    def _start_update_loop(self):
        """Start the UI update loop."""
        def update_ui():
            if not self.is_paused:
                self._process_service_updates()
                self._process_log_updates()
                self._process_metric_updates()

            # Schedule next update
            self.after(100, update_ui)

        update_ui()

    def update_service_status(self, service: str, status: ServiceStatus, error_message: str = None):
        """Update service status (thread-safe)."""
        self.service_queue.put({
            'service': service,
            'status': status,
            'error_message': error_message,
            'timestamp': time.time()
        })

    def append_log(self, message: str, level: LogLevel = LogLevel.INFO, source: str = "System"):
        """Add log entry (thread-safe)."""
        self.log_queue.put(LogEntry(
            timestamp=time.time(),
            level=level,
            message=message,
            source=source
        ))

    def update_metric(self, metric: str, value: float, unit: str = "", max_value: float = None):
        """Update performance metric (thread-safe)."""
        self.metric_queue.put(MetricEntry(
            name=metric,
            value=value,
            unit=unit,
            timestamp=time.time(),
            max_value=max_value
        ))

    def show_resource_usage(self, cpu: float, memory: float, gpu: float = None):
        """Update resource usage metrics."""
        self.update_metric("CPU Usage", cpu, "%", 100)
        self.update_metric("Memory Usage", memory, "%", 100)
        if gpu is not None:
            self.update_metric("GPU Usage", gpu, "%", 100)

    def _process_service_updates(self):
        """Process pending service updates."""
        while not self.service_queue.empty():
            try:
                update = self.service_queue.get_nowait()
                service_name = update['service']

                if service_name in self.service_info:
                    service_info = self.service_info[service_name]
                    service_info.status = update['status']
                    service_info.last_update = update['timestamp']
                    service_info.error_message = update.get('error_message')

                    self._update_service_widget(service_name)

            except queue.Empty:
                break

    def _process_log_updates(self):
        """Process pending log updates."""
        if not self.show_logs:
            return

        new_entries = []
        while not self.log_queue.empty():
            try:
                entry = self.log_queue.get_nowait()
                if entry.level.value >= self.log_filter_level.value or not self._should_filter_log(entry):
                    new_entries.append(entry)
                    self.log_entries.append(entry)
            except queue.Empty:
                break

        # Trim log entries if too many
        if len(self.log_entries) > self.max_log_entries:
            self.log_entries = self.log_entries[-self.max_log_entries:]

        # Update log display
        if new_entries:
            self._update_log_display(new_entries)

    def _process_metric_updates(self):
        """Process pending metric updates."""
        if not self.show_metrics:
            return

        while not self.metric_queue.empty():
            try:
                metric = self.metric_queue.get_nowait()
                self.metrics[metric.name] = metric
                self._update_metric_widget(metric.name)
            except queue.Empty:
                break

    def _update_service_widget(self, service_name: str):
        """Update service widget display."""
        if service_name not in self.service_widgets:
            return

        service_info = self.service_info[service_name]
        widgets = self.service_widgets[service_name]

        # Update status
        status_text = service_info.status.value.replace('_', ' ').title()
        widgets['status_var'].set(status_text)
        widgets['status_label'].configure(fg=self.STATUS_COLORS[service_info.status])

        # Update timestamp
        if service_info.last_update:
            time_str = time.strftime("%H:%M:%S", time.localtime(service_info.last_update))
            widgets['update_var'].set(f"Updated: {time_str}")

        # Show/hide error message
        if service_info.error_message and service_info.status == ServiceStatus.ERROR:
            widgets['error_var'].set(f"Error: {service_info.error_message}")
            widgets['error_label'].pack(fill='x', padx=10, pady=(0, 5))
        else:
            widgets['error_label'].pack_forget()

    def _update_log_display(self, new_entries: List[LogEntry]):
        """Update log text display with new entries."""
        self.log_text.configure(state='normal')

        for entry in new_entries:
            timestamp_str = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))
            log_line = f"[{timestamp_str}] [{entry.level.value.upper()}] [{entry.source}] {entry.message}\n"

            self.log_text.insert('end', log_line, entry.level.value)

        # Auto-scroll if enabled
        if self.auto_scroll_logs:
            self.log_text.see('end')

        self.log_text.configure(state='disabled')

    def _update_metric_widget(self, metric_name: str):
        """Update or create metric widget."""
        if metric_name not in self.metric_widgets:
            self._create_metric_widget(metric_name)

        metric = self.metrics[metric_name]
        widgets = self.metric_widgets[metric_name]

        # Update value
        value_text = f"{metric.value:.1f}"
        if metric.unit:
            value_text += f" {metric.unit}"
        widgets['value_var'].set(value_text)

        # Update progress bar if max value is set
        if metric.max_value and 'progress_bar' in widgets:
            progress_percent = (metric.value / metric.max_value) * 100
            widgets['progress_var'].set(progress_percent)

    def _create_metric_widget(self, metric_name: str):
        """Create widget for a metric."""
        metric = self.metrics[metric_name]

        # Metric frame
        metric_frame = tk.Frame(self.metrics_frame, bg=self.COLORS['bg_secondary'])
        metric_frame.pack(fill='x', pady=5)

        # Metric header
        header_frame = tk.Frame(metric_frame, bg=self.COLORS['bg_secondary'])
        header_frame.pack(fill='x', padx=10, pady=5)

        # Metric name
        name_label = tk.Label(
            header_frame,
            text=metric_name,
            font=('Segoe UI', 9, 'bold'),
            fg=self.COLORS['text_primary'],
            bg=self.COLORS['bg_secondary']
        )
        name_label.pack(side='left')

        # Metric value
        value_var = tk.StringVar()
        value_label = tk.Label(
            header_frame,
            textvariable=value_var,
            font=('Segoe UI', 9),
            fg=self.COLORS['text_secondary'],
            bg=self.COLORS['bg_secondary']
        )
        value_label.pack(side='right')

        widgets = {
            'frame': metric_frame,
            'header': header_frame,
            'name_label': name_label,
            'value_var': value_var,
            'value_label': value_label
        }

        # Add progress bar if metric has max value
        if metric.max_value:
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(
                metric_frame,
                variable=progress_var,
                maximum=100,
                length=200
            )
            progress_bar.pack(fill='x', padx=10, pady=(0, 5))

            widgets['progress_var'] = progress_var
            widgets['progress_bar'] = progress_bar

        self.metric_widgets[metric_name] = widgets

    def _should_filter_log(self, entry: LogEntry) -> bool:
        """Check if log entry should be filtered out."""
        # Custom filtering logic can be added here
        return False

    def _toggle_pause(self):
        """Toggle pause state."""
        self.is_paused = not self.is_paused
        self.pause_button.configure(text="Resume" if self.is_paused else "Pause")

    def _clear_logs(self):
        """Clear all log entries."""
        self.log_entries.clear()
        if self.show_logs:
            self.log_text.configure(state='normal')
            self.log_text.delete('1.0', 'end')
            self.log_text.configure(state='disabled')

    def _on_level_filter_change(self, event):
        """Handle log level filter change."""
        self.log_filter_level = LogLevel(self.level_var.get())
        # Refresh log display
        self._refresh_log_display()

    def _on_auto_scroll_change(self):
        """Handle auto-scroll setting change."""
        self.auto_scroll_logs = self.auto_scroll_var.get()

    def _refresh_log_display(self):
        """Refresh log display with current filter."""
        if not self.show_logs:
            return

        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', 'end')

        # Re-add filtered entries
        filtered_entries = [
            entry for entry in self.log_entries
            if entry.level.value >= self.log_filter_level.value or not self._should_filter_log(entry)
        ]

        self._update_log_display(filtered_entries)

    def set_service_health_check(self, service: str, health_check: Callable):
        """Set health check function for a service."""
        if service in self.service_info:
            self.service_info[service].health_check = health_check

    def run_health_checks(self):
        """Run health checks for all services."""
        def check_services():
            for service_name, service_info in self.service_info.items():
                if service_info.health_check:
                    try:
                        is_healthy = service_info.health_check()
                        if is_healthy:
                            self.update_service_status(service_name, ServiceStatus.RUNNING)
                        else:
                            self.update_service_status(service_name, ServiceStatus.ERROR, "Health check failed")
                    except Exception as e:
                        self.update_service_status(service_name, ServiceStatus.ERROR, str(e))

        threading.Thread(target=check_services, daemon=True).start()

    def get_service_status(self, service: str) -> Optional[ServiceStatus]:
        """Get current status of a service."""
        if service in self.service_info:
            return self.service_info[service].status
        return None

    def export_logs(self, file_path: str):
        """Export logs to file."""
        try:
            with open(file_path, 'w') as f:
                for entry in self.log_entries:
                    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.timestamp))
                    f.write(f"[{timestamp_str}] [{entry.level.value.upper()}] [{entry.source}] {entry.message}\n")
        except Exception as e:
            self.append_log(f"Failed to export logs: {str(e)}", LogLevel.ERROR)

    def shutdown(self):
        """Shutdown the live status panel."""
        self.is_paused = True