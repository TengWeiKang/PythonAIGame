"""Gemini API settings dialog for configuration management."""

import tkinter as tk
from tkinter import ttk, messagebox
import os
from typing import Callable, Optional

from ...config.settings import Config, save_config
from ...services.gemini_service import GeminiService


class GeminiSettingsDialog:
    """Dialog for configuring Gemini API settings."""
    
    # Color scheme matching the main window
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
        'border': '#404040',
    }
    
    def __init__(self, parent: tk.Tk, config: Config, gemini_service: GeminiService, 
                 callback: Optional[Callable] = None):
        """Initialize the settings dialog."""
        self.parent = parent
        self.config = config
        self.gemini_service = gemini_service
        self.callback = callback
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Gemini AI Settings")
        self.dialog.geometry("500x400")
        self.dialog.resizable(False, False)
        self.dialog.configure(bg=self.COLORS['bg_primary'])
        
        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center dialog on parent
        self._center_dialog()
        
        # Variables for form fields
        self.api_key_var = tk.StringVar()
        self.test_result_var = tk.StringVar()
        
        # Load current settings
        self._load_current_settings()
        
        # Build UI
        self._build_ui()
        
        # Focus on dialog
        self.dialog.focus_set()
    
    def _center_dialog(self):
        """Center the dialog on the parent window."""
        self.dialog.update_idletasks()
        
        # Get parent position and size
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        # Calculate dialog position
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        x = parent_x + (parent_width // 2) - (dialog_width // 2)
        y = parent_y + (parent_height // 2) - (dialog_height // 2)
        
        self.dialog.geometry(f"+{x}+{y}")
    
    def _load_current_settings(self):
        """Load current settings from config."""
        # Load API key if available
        api_key = getattr(self.config, 'gemini_api_key', '')
        self.api_key_var.set(api_key)
    
    def _build_ui(self):
        """Build the dialog user interface."""
        # Main container
        main_frame = tk.Frame(self.dialog, bg=self.COLORS['bg_primary'])
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="🤖 Gemini AI Configuration",
            bg=self.COLORS['bg_primary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 14, 'bold')
        )
        title_label.pack(pady=(0, 20))
        
        # API Key Section
        self._build_api_key_section(main_frame)
        
        # Test Section
        self._build_test_section(main_frame)
        
        # Information Section
        self._build_info_section(main_frame)
        
        # Buttons
        self._build_buttons(main_frame)
    
    def _build_api_key_section(self, parent):
        """Build the API key configuration section."""
        # API Key frame
        api_frame = tk.Frame(parent, bg=self.COLORS['bg_secondary'], relief='solid', bd=1)
        api_frame.pack(fill='x', pady=(0, 15))
        
        # Section header
        header_frame = tk.Frame(api_frame, bg=self.COLORS['bg_tertiary'], height=30)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame,
            text="API Key Configuration",
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 10, 'bold')
        ).pack(side='left', padx=10, pady=5)
        
        # Content frame
        content_frame = tk.Frame(api_frame, bg=self.COLORS['bg_secondary'])
        content_frame.pack(fill='x', padx=15, pady=15)
        
        # API Key label and entry
        tk.Label(
            content_frame,
            text="Google Gemini API Key:",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 9)
        ).pack(anchor='w', pady=(0, 5))
        
        # API Key entry with show/hide functionality
        entry_frame = tk.Frame(content_frame, bg=self.COLORS['bg_secondary'])
        entry_frame.pack(fill='x', pady=(0, 10))
        
        self.api_key_entry = tk.Entry(
            entry_frame,
            textvariable=self.api_key_var,
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 9),
            show='*',
            borderwidth=0,
            insertbackground=self.COLORS['text_primary']
        )
        self.api_key_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        # Show/Hide button
        self.show_hide_button = tk.Button(
            entry_frame,
            text="👁",
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_primary'],
            borderwidth=0,
            command=self._toggle_api_key_visibility,
            font=('Segoe UI', 8),
            width=3
        )
        self.show_hide_button.pack(side='right')
        
        # Help text
        help_text = ("Get your free API key from Google AI Studio:\\n"
                    "https://makersuite.google.com/app/apikey")
        
        help_label = tk.Label(
            content_frame,
            text=help_text,
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8),
            justify='left',
            wraplength=400
        )
        help_label.pack(anchor='w')
    
    def _build_test_section(self, parent):
        """Build the API key test section."""
        test_frame = tk.Frame(parent, bg=self.COLORS['bg_secondary'], relief='solid', bd=1)
        test_frame.pack(fill='x', pady=(0, 15))
        
        # Section header
        header_frame = tk.Frame(test_frame, bg=self.COLORS['bg_tertiary'], height=30)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame,
            text="Connection Test",
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 10, 'bold')
        ).pack(side='left', padx=10, pady=5)
        
        # Content frame
        content_frame = tk.Frame(test_frame, bg=self.COLORS['bg_secondary'])
        content_frame.pack(fill='x', padx=15, pady=15)
        
        # Test button
        self.test_button = tk.Button(
            content_frame,
            text="🧪 Test API Connection",
            bg=self.COLORS['accent_primary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 9, 'bold'),
            borderwidth=0,
            padx=15,
            pady=8,
            command=self._test_api_key
        )
        self.test_button.pack(pady=(0, 10))
        
        # Test result label
        self.test_result_label = tk.Label(
            content_frame,
            textvariable=self.test_result_var,
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 9),
            wraplength=400,
            justify='left'
        )
        self.test_result_label.pack(anchor='w')
    
    def _build_info_section(self, parent):
        """Build the information section."""
        info_frame = tk.Frame(parent, bg=self.COLORS['bg_secondary'], relief='solid', bd=1)
        info_frame.pack(fill='x', pady=(0, 15))
        
        # Section header
        header_frame = tk.Frame(info_frame, bg=self.COLORS['bg_tertiary'], height=30)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame,
            text="Information",
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 10, 'bold')
        ).pack(side='left', padx=10, pady=5)
        
        # Content frame
        content_frame = tk.Frame(info_frame, bg=self.COLORS['bg_secondary'])
        content_frame.pack(fill='x', padx=15, pady=15)
        
        info_text = (
            "• The Gemini API is used for AI-powered image analysis\\n"
            "• Your API key is stored locally in the configuration\\n"
            "• An internet connection is required for analysis\\n"
            "• Free tier includes generous usage limits\\n"
            "• All image analysis is processed securely by Google"
        )
        
        info_label = tk.Label(
            content_frame,
            text=info_text,
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_secondary'],
            font=('Segoe UI', 8),
            justify='left',
            wraplength=400
        )
        info_label.pack(anchor='w')
    
    def _build_buttons(self, parent):
        """Build the dialog buttons."""
        button_frame = tk.Frame(parent, bg=self.COLORS['bg_primary'])
        button_frame.pack(fill='x', pady=(10, 0))
        
        # Cancel button
        cancel_button = tk.Button(
            button_frame,
            text="Cancel",
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 9),
            borderwidth=0,
            padx=20,
            pady=8,
            command=self._on_cancel
        )
        cancel_button.pack(side='right', padx=(10, 0))
        
        # Save button
        save_button = tk.Button(
            button_frame,
            text="💾 Save Settings",
            bg=self.COLORS['success'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 9, 'bold'),
            borderwidth=0,
            padx=20,
            pady=8,
            command=self._on_save
        )
        save_button.pack(side='right')
    
    def _toggle_api_key_visibility(self):
        """Toggle API key visibility in the entry field."""
        if self.api_key_entry.cget('show') == '*':
            self.api_key_entry.configure(show='')
            self.show_hide_button.configure(text="🙈")
        else:
            self.api_key_entry.configure(show='*')
            self.show_hide_button.configure(text="👁")
    
    def _test_api_key(self):
        """Test the API key connection."""
        api_key = self.api_key_var.get().strip()
        
        if not api_key:
            self.test_result_var.set("❌ Please enter an API key first")
            self.test_result_label.configure(fg=self.COLORS['error'])
            return
        
        # Disable test button during test
        self.test_button.configure(state='disabled', text="⏳ Testing...")
        self.test_result_var.set("Testing connection...")
        self.test_result_label.configure(fg=self.COLORS['text_muted'])
        
        # Create temporary service for testing
        test_service = GeminiService(api_key)
        
        # Create a simple test image (small black square)
        import numpy as np
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        def on_test_result(result, error):
            """Handle test result."""
            self.dialog.after(0, self._handle_test_result, result, error)
        
        # Run test asynchronously
        import threading
        def test_worker():
            try:
                result = test_service.analyze_single_image(
                    test_image, 
                    "This is a test. Please respond with 'Test successful' if you can see this."
                )
                on_test_result(result, None)
            except Exception as e:
                on_test_result(None, str(e))
        
        threading.Thread(target=test_worker, daemon=True).start()
    
    def _handle_test_result(self, result, error):
        """Handle the API test result."""
        # Re-enable test button
        self.test_button.configure(state='normal', text="🧪 Test API Connection")
        
        if error:
            self.test_result_var.set(f"❌ Connection failed: {error}")
            self.test_result_label.configure(fg=self.COLORS['error'])
        else:
            self.test_result_var.set("✅ Connection successful! API key is working.")
            self.test_result_label.configure(fg=self.COLORS['success'])
    
    def _on_save(self):
        """Save the settings."""
        api_key = self.api_key_var.get().strip()
        
        if not api_key:
            messagebox.showwarning("Warning", "Please enter a valid API key")
            return
        
        try:
            # Update config
            self.config.extra['gemini_api_key'] = api_key
            
            # Save config to file
            save_config(self.config)
            
            # Update the service
            self.gemini_service.set_api_key(api_key)
            
            messagebox.showinfo("Success", "Settings saved successfully!")
            
            # Call callback if provided
            if self.callback:
                self.callback()
            
            # Close dialog
            self._on_cancel()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")
    
    def _on_cancel(self):
        """Cancel and close the dialog."""
        self.dialog.grab_release()
        self.dialog.destroy()
    
    def winfo_exists(self):
        """Check if dialog window exists."""
        try:
            return self.dialog.winfo_exists()
        except tk.TclError:
            return False
    
    def lift(self):
        """Bring dialog to front."""
        if self.winfo_exists():
            self.dialog.lift()
            self.dialog.focus_set()