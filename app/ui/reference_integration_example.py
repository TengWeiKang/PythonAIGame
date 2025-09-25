"""
Integration example for ReferenceImageManager with ModernMainWindow.

This module demonstrates how to integrate the ReferenceImageManager into the
existing UI for capturing and comparing reference images with live webcam frames.
"""

import asyncio
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from typing import Optional
import logging

from ..services.reference_manager import ReferenceImageManager, ComparisonResult
from ..backends.yolo_backend import YoloBackend
from ..core.entities import Detection

logger = logging.getLogger(__name__)


class ReferenceImagePanel:
    """
    UI panel for reference image management integrated into ModernMainWindow.

    This panel provides:
    - Capture reference button
    - Reference selection dropdown
    - Live comparison toggle
    - Comparison results display
    """

    def __init__(self, parent_frame: tk.Frame, yolo_backend: YoloBackend, config: dict):
        """
        Initialize the reference image panel.

        Args:
            parent_frame: Parent tkinter frame to add the panel to
            yolo_backend: YOLO backend for object detection
            config: Application configuration dictionary
        """
        self.parent_frame = parent_frame
        self.yolo_backend = yolo_backend
        self.config = config

        # Initialize ReferenceImageManager
        data_dir = config.get('data_dir', './data')
        self.reference_manager = ReferenceImageManager(
            yolo_backend=yolo_backend,
            data_dir=data_dir,
            max_references=config.get('max_references', 100),
            max_memory_mb=config.get('reference_max_memory_mb', 50),
            auto_cleanup_days=config.get('reference_cleanup_days', 7),
            enable_compression=True
        )

        # UI state
        self.selected_reference_id = None
        self.live_comparison_enabled = False
        self.last_comparison_result: Optional[ComparisonResult] = None

        # Create UI components
        self._create_ui()

        # Load existing references
        self._refresh_reference_list()

    def _create_ui(self):
        """Create the UI components for the reference panel."""
        # Main container
        self.container = ttk.LabelFrame(
            self.parent_frame,
            text="Reference Image Analysis",
            padding="10"
        )
        self.container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Capture section
        capture_frame = ttk.Frame(self.container)
        capture_frame.pack(fill=tk.X, pady=(0, 10))

        self.capture_button = ttk.Button(
            capture_frame,
            text="📸 Capture Reference",
            command=self._on_capture_reference
        )
        self.capture_button.pack(side=tk.LEFT, padx=(0, 5))

        self.capture_status = ttk.Label(capture_frame, text="Ready")
        self.capture_status.pack(side=tk.LEFT, padx=5)

        # Reference selection section
        selection_frame = ttk.Frame(self.container)
        selection_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(selection_frame, text="Reference:").pack(side=tk.LEFT, padx=(0, 5))

        self.reference_combo = ttk.Combobox(
            selection_frame,
            state="readonly",
            width=30
        )
        self.reference_combo.pack(side=tk.LEFT, padx=5)
        self.reference_combo.bind("<<ComboboxSelected>>", self._on_reference_selected)

        self.refresh_button = ttk.Button(
            selection_frame,
            text="🔄",
            width=3,
            command=self._refresh_reference_list
        )
        self.refresh_button.pack(side=tk.LEFT, padx=2)

        self.delete_button = ttk.Button(
            selection_frame,
            text="🗑️",
            width=3,
            command=self._on_delete_reference,
            state=tk.DISABLED
        )
        self.delete_button.pack(side=tk.LEFT, padx=2)

        # Live comparison section
        comparison_frame = ttk.Frame(self.container)
        comparison_frame.pack(fill=tk.X, pady=(0, 10))

        self.live_comparison_var = tk.BooleanVar(value=False)
        self.live_comparison_check = ttk.Checkbutton(
            comparison_frame,
            text="Enable Live Comparison",
            variable=self.live_comparison_var,
            command=self._on_toggle_live_comparison
        )
        self.live_comparison_check.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(comparison_frame, text="Threshold:").pack(side=tk.LEFT, padx=(0, 5))

        self.confidence_var = tk.DoubleVar(value=0.5)
        self.confidence_scale = ttk.Scale(
            comparison_frame,
            from_=0.1,
            to=0.9,
            variable=self.confidence_var,
            orient=tk.HORIZONTAL,
            length=100
        )
        self.confidence_scale.pack(side=tk.LEFT, padx=5)

        self.confidence_label = ttk.Label(comparison_frame, text="0.50")
        self.confidence_label.pack(side=tk.LEFT, padx=5)
        self.confidence_var.trace('w', self._on_confidence_changed)

        # Results display
        results_frame = ttk.LabelFrame(self.container, text="Comparison Results", padding="5")
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Create text widget for results
        self.results_text = tk.Text(
            results_frame,
            height=8,
            wrap=tk.WORD,
            font=("Consolas", 9)
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Scrollbar for results
        scrollbar = ttk.Scrollbar(self.results_text, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)

        # Performance metrics
        metrics_frame = ttk.Frame(self.container)
        metrics_frame.pack(fill=tk.X, pady=(10, 0))

        self.metrics_label = ttk.Label(
            metrics_frame,
            text="Performance: -- | Cache Hit Rate: --%",
            font=("Consolas", 9)
        )
        self.metrics_label.pack(side=tk.LEFT)

    def _on_capture_reference(self):
        """Handle capture reference button click."""
        self.capture_button.config(state=tk.DISABLED)
        self.capture_status.config(text="Capturing...")

        # Run capture in background thread
        thread = threading.Thread(target=self._capture_reference_thread, daemon=True)
        thread.start()

    def _capture_reference_thread(self):
        """Background thread for capturing reference."""
        try:
            # Get current frame from parent window
            frame = self._get_current_frame()
            if frame is None:
                raise ValueError("No frame available")

            # Create async event loop for capture
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Capture reference
            confidence = self.confidence_var.get()
            reference_id = loop.run_until_complete(
                self.reference_manager.capture_reference(
                    frame,
                    confidence_threshold=confidence
                )
            )

            # Update UI in main thread
            self.parent_frame.after(0, self._on_capture_complete, reference_id, None)

        except Exception as e:
            logger.error(f"Reference capture failed: {e}")
            self.parent_frame.after(0, self._on_capture_complete, None, str(e))

    def _on_capture_complete(self, reference_id: Optional[str], error: Optional[str]):
        """Handle capture completion in main thread."""
        self.capture_button.config(state=tk.NORMAL)

        if error:
            self.capture_status.config(text=f"Error: {error[:30]}")
            messagebox.showerror("Capture Failed", error)
        else:
            self.capture_status.config(text=f"Captured: {reference_id}")
            self._refresh_reference_list()

            # Select the new reference
            refs = self.reference_manager.get_all_references()
            if refs:
                self.reference_combo.set(f"{reference_id} ({refs[-1]['detection_count']} objects)")
                self.selected_reference_id = reference_id
                self.delete_button.config(state=tk.NORMAL)

    def _on_reference_selected(self, event=None):
        """Handle reference selection from dropdown."""
        selection = self.reference_combo.get()
        if selection:
            # Extract reference ID from selection string
            self.selected_reference_id = selection.split(" (")[0]
            self.delete_button.config(state=tk.NORMAL)

            # Display reference info
            self._display_reference_info()

    def _on_delete_reference(self):
        """Handle delete reference button click."""
        if not self.selected_reference_id:
            return

        if messagebox.askyesno("Delete Reference",
                              f"Delete reference {self.selected_reference_id}?"):
            self.reference_manager._delete_reference(self.selected_reference_id)
            self.selected_reference_id = None
            self.delete_button.config(state=tk.DISABLED)
            self._refresh_reference_list()
            self.results_text.delete(1.0, tk.END)

    def _on_toggle_live_comparison(self):
        """Toggle live comparison mode."""
        self.live_comparison_enabled = self.live_comparison_var.get()

        if self.live_comparison_enabled and not self.selected_reference_id:
            messagebox.showwarning("No Reference",
                                 "Please select a reference image first")
            self.live_comparison_var.set(False)
            self.live_comparison_enabled = False

    def _on_confidence_changed(self, *args):
        """Handle confidence threshold change."""
        value = self.confidence_var.get()
        self.confidence_label.config(text=f"{value:.2f}")

    def _refresh_reference_list(self):
        """Refresh the reference dropdown list."""
        refs = self.reference_manager.get_all_references()
        items = []
        for ref in refs:
            ref_id = ref['reference_id']
            count = ref['detection_count']
            items.append(f"{ref_id} ({count} objects)")

        self.reference_combo['values'] = items
        if not items:
            self.reference_combo.set("")
            self.delete_button.config(state=tk.DISABLED)

    def _display_reference_info(self):
        """Display information about selected reference."""
        if not self.selected_reference_id:
            return

        try:
            ref_data = self.reference_manager.get_reference(self.selected_reference_id)
            metadata = ref_data['metadata']

            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Reference: {self.selected_reference_id}\n")
            self.results_text.insert(tk.END, f"Timestamp: {metadata['timestamp']}\n")
            self.results_text.insert(tk.END, f"Dimensions: {metadata['dimensions']}\n")
            self.results_text.insert(tk.END, f"Detections: {metadata['detection_count']}\n")
            self.results_text.insert(tk.END, f"File Size: {metadata['file_size_bytes'] / 1024:.1f} KB\n")
            self.results_text.insert(tk.END, f"Analysis Time: {metadata['analysis_time_ms']:.1f} ms\n")

        except Exception as e:
            logger.error(f"Failed to display reference info: {e}")

    def process_frame(self, frame, detections: list) -> Optional[ComparisonResult]:
        """
        Process a frame for live comparison with selected reference.

        This method should be called from the main window's frame processing loop.

        Args:
            frame: Current video frame
            detections: Current frame detections

        Returns:
            ComparisonResult if live comparison is enabled, None otherwise
        """
        if not self.live_comparison_enabled or not self.selected_reference_id:
            return None

        try:
            # Perform comparison
            result = self.reference_manager.compare_with_reference(
                detections,
                self.selected_reference_id,
                use_cache=True
            )

            # Update display
            self._update_comparison_display(result)

            # Update metrics
            self._update_metrics()

            self.last_comparison_result = result
            return result

        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return None

    def _update_comparison_display(self, result: ComparisonResult):
        """Update the comparison results display."""
        self.results_text.delete(1.0, tk.END)

        # Display comparison summary
        self.results_text.insert(tk.END, f"=== Live Comparison Results ===\n")
        self.results_text.insert(tk.END, f"Reference: {result.reference_id}\n")
        self.results_text.insert(tk.END, f"Similarity: {result.overall_similarity:.1%}\n")
        self.results_text.insert(tk.END, f"Scene Change: {result.scene_change_score:.1%}\n\n")

        # Object statistics
        self.results_text.insert(tk.END, "Object Changes:\n")
        self.results_text.insert(tk.END, f"  • Matched: {len([m for m in result.object_matches if m.match_type == 'match'])}\n")
        self.results_text.insert(tk.END, f"  • Moved: {len([m for m in result.object_matches if m.match_type == 'moved'])}\n")
        self.results_text.insert(tk.END, f"  • Added: {len(result.objects_added)}\n")
        self.results_text.insert(tk.END, f"  • Missing: {len(result.objects_missing)}\n")

        # Performance
        self.results_text.insert(tk.END, f"\nComparison Time: {result.comparison_time_ms:.1f}ms")
        if result.cache_hit:
            self.results_text.insert(tk.END, " (cached)")

    def _update_metrics(self):
        """Update performance metrics display."""
        try:
            stats = self.reference_manager.get_memory_usage()
            comparison_cache_hit = stats['comparison_cache_hit_rate'] * 100

            if self.last_comparison_result:
                perf_text = f"Performance: {self.last_comparison_result.comparison_time_ms:.1f}ms"
            else:
                perf_text = "Performance: --"

            self.metrics_label.config(
                text=f"{perf_text} | Cache Hit Rate: {comparison_cache_hit:.0f}%"
            )
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")

    def _get_current_frame(self):
        """
        Get current frame from parent window.
        This should be implemented to access the parent window's current frame.
        """
        # This is a placeholder - in actual integration, this would access
        # the ModernMainWindow's self._current_frame or webcam service
        parent = self.parent_frame.winfo_toplevel()
        if hasattr(parent, '_current_frame'):
            return parent._current_frame
        elif hasattr(parent, 'webcam_service'):
            return parent.webcam_service.get_frame()
        return None

    def shutdown(self):
        """Shutdown the reference manager and cleanup resources."""
        self.reference_manager.shutdown()


def integrate_with_main_window(main_window, yolo_backend):
    """
    Helper function to integrate ReferenceImagePanel into ModernMainWindow.

    Usage in ModernMainWindow.__init__:
        from .reference_integration_example import integrate_with_main_window
        self.reference_panel = integrate_with_main_window(self, self.yolo_backend)

    Then in the frame processing loop:
        if self.reference_panel:
            comparison_result = self.reference_panel.process_frame(frame, detections)
            if comparison_result:
                # Use comparison results for display or further processing
                pass
    """
    # Create a frame for the reference panel in the main window
    reference_frame = ttk.Frame(main_window.root)
    reference_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)

    # Create the reference panel
    panel = ReferenceImagePanel(
        reference_frame,
        yolo_backend,
        main_window.config.__dict__ if hasattr(main_window.config, '__dict__') else {}
    )

    # Add shutdown hook
    original_shutdown = main_window.shutdown if hasattr(main_window, 'shutdown') else None

    def shutdown_with_reference():
        panel.shutdown()
        if original_shutdown:
            original_shutdown()

    main_window.shutdown = shutdown_with_reference

    return panel