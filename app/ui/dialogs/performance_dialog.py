"""Performance monitoring dialog."""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PerformanceDialog:
    """Dialog for displaying performance monitoring information."""

    def __init__(self, parent, performance_report: Dict[str, Any],
                 cache_manager=None, memory_manager=None, performance_monitor=None):
        """Initialize the performance dialog.

        Args:
            parent: Parent window
            performance_report: Dictionary containing performance metrics
            cache_manager: Cache manager instance (optional)
            memory_manager: Memory manager instance (optional)
            performance_monitor: Performance monitor instance (optional)
        """
        self.parent = parent
        self.performance_report = performance_report
        self.cache_manager = cache_manager
        self.memory_manager = memory_manager
        self.performance_monitor = performance_monitor

        # Create dialog window
        self.window = tk.Toplevel(parent)
        self.window.title("Performance Monitor")
        self.window.geometry("800x600")
        self.window.resizable(True, True)

        # Make it modal
        self.window.transient(parent)
        self.window.grab_set()

        # Center on parent
        self._center_window()

        # Build UI
        self._build_ui()

        # Load performance data
        self._load_performance_data()

        # Setup auto-refresh if performance monitor is available
        if self.performance_monitor:
            self._setup_auto_refresh()

    def _center_window(self):
        """Center the dialog on the parent window."""
        self.window.update_idletasks()
        x = self.parent.winfo_x() + (self.parent.winfo_width() // 2) - (self.window.winfo_width() // 2)
        y = self.parent.winfo_y() + (self.parent.winfo_height() // 2) - (self.window.winfo_height() // 2)
        self.window.geometry(f"+{x}+{y}")

    def _build_ui(self):
        """Build the dialog UI."""
        # Main frame
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True, pady=(0, 10))

        # Memory tab
        self._build_memory_tab()

        # Threading tab
        self._build_threading_tab()

        # Cache tab
        self._build_cache_tab()

        # Performance metrics tab
        self._build_performance_tab()

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(10, 0))

        # Refresh button
        refresh_btn = ttk.Button(button_frame, text="Refresh", command=self._refresh_data)
        refresh_btn.pack(side='left', padx=(0, 10))

        # Auto-refresh toggle
        self.auto_refresh_var = tk.BooleanVar(value=False)
        auto_refresh_cb = ttk.Checkbutton(
            button_frame, text="Auto-refresh (5s)",
            variable=self.auto_refresh_var,
            command=self._toggle_auto_refresh
        )
        auto_refresh_cb.pack(side='left', padx=(0, 10))

        # Close button
        close_btn = ttk.Button(button_frame, text="Close", command=self.window.destroy)
        close_btn.pack(side='right')

    def _build_memory_tab(self):
        """Build the memory monitoring tab."""
        memory_frame = ttk.Frame(self.notebook)
        self.notebook.add(memory_frame, text="Memory")

        # Memory info text widget
        self.memory_text = tk.Text(memory_frame, wrap=tk.WORD, font=('Consolas', 10))
        memory_scrollbar = ttk.Scrollbar(memory_frame, orient='vertical', command=self.memory_text.yview)
        self.memory_text.configure(yscrollcommand=memory_scrollbar.set)

        self.memory_text.pack(side='left', fill='both', expand=True)
        memory_scrollbar.pack(side='right', fill='y')

    def _build_threading_tab(self):
        """Build the threading monitoring tab."""
        threading_frame = ttk.Frame(self.notebook)
        self.notebook.add(threading_frame, text="Threading")

        # Threading info text widget
        self.threading_text = tk.Text(threading_frame, wrap=tk.WORD, font=('Consolas', 10))
        threading_scrollbar = ttk.Scrollbar(threading_frame, orient='vertical', command=self.threading_text.yview)
        self.threading_text.configure(yscrollcommand=threading_scrollbar.set)

        self.threading_text.pack(side='left', fill='both', expand=True)
        threading_scrollbar.pack(side='right', fill='y')

    def _build_cache_tab(self):
        """Build the cache monitoring tab."""
        cache_frame = ttk.Frame(self.notebook)
        self.notebook.add(cache_frame, text="Cache")

        # Cache info text widget
        self.cache_text = tk.Text(cache_frame, wrap=tk.WORD, font=('Consolas', 10))
        cache_scrollbar = ttk.Scrollbar(cache_frame, orient='vertical', command=self.cache_text.yview)
        self.cache_text.configure(yscrollcommand=cache_scrollbar.set)

        self.cache_text.pack(side='left', fill='both', expand=True)
        cache_scrollbar.pack(side='right', fill='y')

    def _build_performance_tab(self):
        """Build the performance metrics tab."""
        perf_frame = ttk.Frame(self.notebook)
        self.notebook.add(perf_frame, text="Performance")

        # Performance info text widget
        self.performance_text = tk.Text(perf_frame, wrap=tk.WORD, font=('Consolas', 10))
        perf_scrollbar = ttk.Scrollbar(perf_frame, orient='vertical', command=self.performance_text.yview)
        self.performance_text.configure(yscrollcommand=perf_scrollbar.set)

        self.performance_text.pack(side='left', fill='both', expand=True)
        perf_scrollbar.pack(side='right', fill='y')

    def _load_performance_data(self):
        """Load and display performance data."""
        try:
            # Memory data
            self._display_memory_data()

            # Threading data
            self._display_threading_data()

            # Cache data
            self._display_cache_data()

            # Performance data
            self._display_performance_data()

        except Exception as e:
            logger.error(f"Error loading performance data: {e}")

    def _display_memory_data(self):
        """Display memory usage information."""
        self.memory_text.delete(1.0, tk.END)

        memory_info = self.performance_report.get('memory', {})

        content = f"Memory Usage Report - {datetime.now().strftime('%H:%M:%S')}\n"
        content += "=" * 50 + "\n\n"

        # Current memory usage
        current = memory_info.get('current', {})
        if current:
            content += "Current Memory Usage:\n"
            content += f"  Memory Percent: {current.get('memory_percent', 0):.1f}%\n"
            content += f"  Used Memory: {current.get('used_memory_mb', 0):.1f} MB\n"
            content += f"  Available Memory: {current.get('available_memory_mb', 0):.1f} MB\n"
            content += f"  Total Memory: {current.get('total_memory_mb', 0):.1f} MB\n\n"

        # Process memory
        process = memory_info.get('process', {})
        if process:
            content += "Process Memory Usage:\n"
            content += f"  RSS (Resident Set Size): {process.get('rss_mb', 0):.1f} MB\n"
            content += f"  VMS (Virtual Memory Size): {process.get('vms_mb', 0):.1f} MB\n"
            content += f"  Memory Percent: {process.get('memory_percent', 0):.2f}%\n\n"

        # Memory history
        history = memory_info.get('history', [])
        if history:
            content += "Recent Memory History:\n"
            for i, entry in enumerate(history[-10:]):  # Show last 10 entries
                content += f"  [{i+1:2d}] {entry.get('memory_percent', 0):5.1f}% "
                content += f"({entry.get('used_memory_mb', 0):6.1f} MB)\n"

        self.memory_text.insert(1.0, content)

    def _display_threading_data(self):
        """Display threading information."""
        self.threading_text.delete(1.0, tk.END)

        threading_info = self.performance_report.get('threading', {})

        content = f"Threading Report - {datetime.now().strftime('%H:%M:%S')}\n"
        content += "=" * 50 + "\n\n"

        # System threads
        system = threading_info.get('system', {})
        if system:
            content += "System Threading Info:\n"
            content += f"  Total Threads: {system.get('total_threads', 0)}\n"
            content += f"  Active Threads: {system.get('active_threads', 0)}\n"
            content += f"  Daemon Threads: {system.get('daemon_threads', 0)}\n\n"

        # Application threads
        app = threading_info.get('application', {})
        if app:
            content += "Application Threads:\n"
            content += f"  Main Thread: {app.get('main_thread', 'Unknown')}\n"
            content += f"  Background Threads: {app.get('background_threads', 0)}\n\n"

        # Thread details
        threads = threading_info.get('threads', [])
        if threads:
            content += "Thread Details:\n"
            for thread in threads:
                content += f"  - {thread.get('name', 'Unknown')}: "
                content += f"{'Alive' if thread.get('is_alive') else 'Dead'} "
                content += f"({'Daemon' if thread.get('daemon') else 'Non-daemon'})\n"

        self.threading_text.insert(1.0, content)

    def _display_cache_data(self):
        """Display cache performance information."""
        self.cache_text.delete(1.0, tk.END)

        cache_info = self.performance_report.get('cache', {})

        content = f"Cache Performance Report - {datetime.now().strftime('%H:%M:%S')}\n"
        content += "=" * 50 + "\n\n"

        if not cache_info:
            content += "No cache data available.\n"
        else:
            for cache_name, stats in cache_info.items():
                content += f"{cache_name.replace('_', ' ').title()} Cache:\n"

                hits = stats.get('hits', 0)
                misses = stats.get('misses', 0)
                total = hits + misses
                hit_rate = (hits / total * 100) if total > 0 else 0

                content += f"  Hit Rate: {hit_rate:.1f}%\n"
                content += f"  Hits: {hits}\n"
                content += f"  Misses: {misses}\n"
                content += f"  Total Requests: {total}\n"

                if 'evictions' in stats:
                    content += f"  Evictions: {stats['evictions']}\n"
                if 'size' in stats:
                    content += f"  Current Size: {stats['size']}\n"
                if 'max_size' in stats:
                    content += f"  Max Size: {stats['max_size']}\n"

                content += "\n"

        self.cache_text.insert(1.0, content)

    def _display_performance_data(self):
        """Display performance metrics."""
        self.performance_text.delete(1.0, tk.END)

        perf_info = self.performance_report.get('performance', {})

        content = f"Performance Metrics Report - {datetime.now().strftime('%H:%M:%S')}\n"
        content += "=" * 50 + "\n\n"

        # Current metrics
        current = perf_info.get('current_metrics', {})
        if current:
            content += "Current Performance Metrics:\n"
            for key, value in current.items():
                if isinstance(value, float):
                    content += f"  {key.replace('_', ' ').title()}: {value:.3f}\n"
                else:
                    content += f"  {key.replace('_', ' ').title()}: {value}\n"
            content += "\n"

        # Operation statistics
        op_stats = perf_info.get('operation_stats', {})
        if op_stats:
            content += "Operation Statistics:\n"
            for operation, stats in op_stats.items():
                if stats:
                    content += f"  {operation.replace('_', ' ').title()}:\n"
                    content += f"    Average Time: {stats.get('avg_time', 0):.3f}s\n"
                    content += f"    Min Time: {stats.get('min_time', 0):.3f}s\n"
                    content += f"    Max Time: {stats.get('max_time', 0):.3f}s\n"
                    content += f"    Total Calls: {stats.get('call_count', 0)}\n"
                    content += "\n"

        self.performance_text.insert(1.0, content)

    def _refresh_data(self):
        """Refresh performance data."""
        try:
            # Get fresh performance report if performance monitor is available
            if self.performance_monitor:
                # Update the performance report with fresh data
                fresh_report = {
                    'memory': self.memory_manager.get_memory_report() if self.memory_manager else {},
                    'cache': self.cache_manager.get_cache_stats() if self.cache_manager else {},
                    'performance': {
                        'current_metrics': self.performance_monitor.get_current_metrics(),
                        'operation_stats': {
                            op: self.performance_monitor.get_operation_stats(op)
                            for op in ['webcam_read', 'ui_display_image_on_canvas', 'gemini_send_message']
                        }
                    }
                }
                self.performance_report = fresh_report

            # Reload the data
            self._load_performance_data()

        except Exception as e:
            logger.error(f"Error refreshing performance data: {e}")

    def _setup_auto_refresh(self):
        """Setup automatic refresh of data."""
        self._auto_refresh_job = None

    def _toggle_auto_refresh(self):
        """Toggle automatic refresh on/off."""
        if self.auto_refresh_var.get():
            self._start_auto_refresh()
        else:
            self._stop_auto_refresh()

    def _start_auto_refresh(self):
        """Start automatic refresh."""
        if hasattr(self, '_auto_refresh_job') and self._auto_refresh_job:
            self.window.after_cancel(self._auto_refresh_job)

        def auto_refresh():
            if self.auto_refresh_var.get():
                self._refresh_data()
                self._auto_refresh_job = self.window.after(5000, auto_refresh)  # 5 seconds

        self._auto_refresh_job = self.window.after(5000, auto_refresh)

    def _stop_auto_refresh(self):
        """Stop automatic refresh."""
        if hasattr(self, '_auto_refresh_job') and self._auto_refresh_job:
            self.window.after_cancel(self._auto_refresh_job)
            self._auto_refresh_job = None