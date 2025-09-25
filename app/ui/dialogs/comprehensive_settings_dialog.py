"""Comprehensive settings dialog with tabbed interface for all application settings."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import json
import threading
import logging
from typing import Callable, Optional, Dict, Any, List
from dataclasses import asdict
import cv2
from PIL import Image, ImageTk
import time

logger = logging.getLogger(__name__)

from ...config.settings import Config, save_config
from ...config.settings_manager import get_settings_manager, SettingsManager
from ...config.validation import SettingsValidator, SettingsPresets, ValidationResult
from ...services.gemini_service import GeminiService
from ...services.webcam_service import WebcamService


class ComprehensiveSettingsDialog:
    """Comprehensive settings dialog with professional tabbed interface."""
    
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
    
    def __init__(self, parent: tk.Tk, config: Config, services: Dict[str, Any], 
                 callback: Optional[Callable] = None):
        """Initialize the comprehensive settings dialog."""
        self.parent = parent
        self.config = config
        self.original_config = config  # Keep reference to original for comparison
        self.services = services
        self.callback = callback
        
        # Initialize settings manager and register services
        self.settings_manager = get_settings_manager()
        self._register_services()
        
        # Track changes for Apply button
        self._changes_made = False
        self._validation_results = {}
        
        # Store original values for change detection
        self.original_values = {}
        self.has_changes = False
        
        # Camera preview state
        self._test_service = None
        self._preview_running = False
        self._preview_thread = None
        self._preview_canvas = None
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Settings - Vision Analysis System")
        self.dialog.geometry("800x700")
        self.dialog.minsize(700, 600)
        self.dialog.configure(bg=self.COLORS['bg_primary'])
        
        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center dialog on parent
        self._center_dialog()
        
        # Variables for form fields - organized by category
        self._init_variables()
        
        # Load current settings
        self._load_current_settings()
        
        # Build UI
        self._build_ui()
        
        # Initialize change tracking after UI is built
        self._init_change_tracking()
        
        # Focus on dialog
        self.dialog.focus_set()
        
        # Bind close event
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_cancel)
    
    def _register_services(self):
        """Register available services with the settings manager for real-time updates."""
        try:
            if 'webcam' in self.services:
                self.settings_manager.register_service('webcam', self.services['webcam'])
            
            if 'gemini' in self.services:
                self.settings_manager.register_service('gemini', self.services['gemini'])
            
            if 'detection' in self.services:
                self.settings_manager.register_service('detection', self.services['detection'])
            
            if 'main_window' in self.services:
                self.settings_manager.register_service('main_window', self.services['main_window'])
                
        except Exception as e:
            logger.warning(f"Failed to register some services with settings manager: {e}")
    
    def _center_dialog(self):
        """Center the dialog on the parent window."""
        self.dialog.update_idletasks()
        
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        x = parent_x + (parent_width // 2) - (dialog_width // 2)
        y = parent_y + (parent_height // 2) - (dialog_height // 2)
        
        self.dialog.geometry(f"+{x}+{y}")
    
    def _init_variables(self):
        """Initialize all tkinter variables for settings."""
        # General settings
        self.theme_var = tk.StringVar()
        self.language_var = tk.StringVar()
        self.auto_save_var = tk.BooleanVar()
        self.debug_var = tk.BooleanVar()
        self.startup_fullscreen_var = tk.BooleanVar()
        self.remember_window_var = tk.BooleanVar()
        self.remember_window_state_var = tk.BooleanVar()
        self.performance_mode_var = tk.StringVar()
        self.memory_limit_var = tk.IntVar()
        self.enable_logging_var = tk.BooleanVar()
        self.log_level_var = tk.StringVar()
        self.auto_save_interval_var = tk.IntVar()
        # Removed: check_updates_var, update_interval_var, backup_on_change_var
        
        # Webcam settings
        self.camera_index_var = tk.IntVar()
        self.camera_width_var = tk.IntVar()
        self.camera_height_var = tk.IntVar()
        self.camera_fps_var = tk.IntVar()
        self.auto_exposure_var = tk.BooleanVar()
        self.auto_focus_var = tk.BooleanVar()
        self.brightness_var = tk.IntVar()
        self.contrast_var = tk.IntVar()
        self.saturation_var = tk.IntVar()
        self.recording_format_var = tk.StringVar()
        self.buffer_size_var = tk.IntVar()
        self.preview_enabled_var = tk.BooleanVar()
        self.camera_device_name_var = tk.StringVar()
        
        # Image Analysis settings
        self.confidence_threshold_var = tk.DoubleVar()
        self.iou_threshold_var = tk.DoubleVar()
        # Removed: noise_reduction_var, contrast_enhancement_var 
        self.enable_roi_var = tk.BooleanVar()
        self.roi_x_var = tk.IntVar()
        self.roi_y_var = tk.IntVar()
        self.roi_width_var = tk.IntVar()
        self.roi_height_var = tk.IntVar()
        self.preferred_model_var = tk.StringVar()
        # Removed: export_format_var (now hardcoded to PNG)
        self.export_quality_var = tk.IntVar()  # Always 100%
        self.difference_sensitivity_var = tk.DoubleVar()
        self.highlight_differences_var = tk.BooleanVar()
        self.reference_image_path_var = tk.StringVar()
        self.analysis_history_days_var = tk.IntVar()
        self.export_metadata_var = tk.BooleanVar()
        self.default_data_dir_var = tk.StringVar()
        self.default_models_dir_var = tk.StringVar()
        self.default_results_dir_var = tk.StringVar()
        
        # Chatbot settings
        self.api_key_var = tk.StringVar()
        self.gemini_model_var = tk.StringVar()
        self.timeout_var = tk.IntVar()
        self.temperature_var = tk.DoubleVar()
        self.max_tokens_var = tk.IntVar()
        self.enable_ai_var = tk.BooleanVar()
        self.chat_history_limit_var = tk.IntVar()
        self.chat_auto_save_var = tk.BooleanVar()
        self.response_format_var = tk.StringVar()
        self.enable_rate_limiting_var = tk.BooleanVar()
        self.requests_per_minute_var = tk.IntVar()
        self.context_window_var = tk.IntVar()
        self.conversation_memory_var = tk.BooleanVar()
        self.chat_export_format_var = tk.StringVar()
        self.chatbot_persona_var = tk.StringVar()
        
        # Test results
        self.test_result_var = tk.StringVar()
    
    def _load_current_settings(self):
        """Load current settings from config into variables."""
        # General settings
        self.theme_var.set(getattr(self.config, 'app_theme', 'Dark'))
        self.language_var.set(getattr(self.config, 'language', 'en'))
        self.auto_save_var.set(getattr(self.config, 'auto_save_config', True))
        self.debug_var.set(getattr(self.config, 'debug', False))
        self.startup_fullscreen_var.set(getattr(self.config, 'startup_fullscreen', False))
        self.remember_window_var.set(getattr(self.config, 'remember_window_state', True))
        self.remember_window_state_var.set(getattr(self.config, 'remember_window_state', True))
        self.performance_mode_var.set(getattr(self.config, 'performance_mode', 'Balanced'))
        self.memory_limit_var.set(getattr(self.config, 'max_memory_usage_mb', 2048))
        self.enable_logging_var.set(getattr(self.config, 'enable_logging', True))
        self.log_level_var.set(getattr(self.config, 'log_level', 'INFO'))
        self.auto_save_interval_var.set(getattr(self.config, 'auto_save_interval_minutes', 5))
        # Removed: check_updates_var, update_interval_var, backup_on_change_var settings
        
        # Webcam settings
        self.camera_index_var.set(getattr(self.config, 'last_webcam_index', 0))
        self.camera_width_var.set(getattr(self.config, 'camera_width', 1280))
        self.camera_height_var.set(getattr(self.config, 'camera_height', 720))
        self.camera_fps_var.set(getattr(self.config, 'camera_fps', 30))
        self.auto_exposure_var.set(getattr(self.config, 'camera_auto_exposure', True))
        self.auto_focus_var.set(getattr(self.config, 'camera_auto_focus', True))
        self.brightness_var.set(getattr(self.config, 'camera_brightness', 0))
        self.contrast_var.set(getattr(self.config, 'camera_contrast', 0))
        self.saturation_var.set(getattr(self.config, 'camera_saturation', 0))
        self.recording_format_var.set(getattr(self.config, 'camera_recording_format', 'MP4'))
        self.buffer_size_var.set(getattr(self.config, 'camera_buffer_size', 5))
        self.preview_enabled_var.set(getattr(self.config, 'camera_preview_enabled', True))
        self.camera_device_name_var.set(getattr(self.config, 'camera_device_name', ''))
        
        # Image Analysis settings
        self.confidence_threshold_var.set(getattr(self.config, 'detection_confidence_threshold', 0.5))
        self.iou_threshold_var.set(getattr(self.config, 'detection_iou_threshold', 0.45))
        # Removed: noise_reduction_var, contrast_enhancement_var settings
        self.enable_roi_var.set(getattr(self.config, 'enable_roi', False))
        self.roi_x_var.set(getattr(self.config, 'roi_x', 0))
        self.roi_y_var.set(getattr(self.config, 'roi_y', 0))
        self.roi_width_var.set(getattr(self.config, 'roi_width', 0))
        self.roi_height_var.set(getattr(self.config, 'roi_height', 0))
        self.preferred_model_var.set(getattr(self.config, 'preferred_model', 'yolo12n'))
        # Removed: export_format_var (always PNG)
        self.export_quality_var.set(100)  # Always 100% quality
        self.difference_sensitivity_var.set(getattr(self.config, 'difference_sensitivity', 0.1))
        self.highlight_differences_var.set(getattr(self.config, 'highlight_differences', True))
        self.reference_image_path_var.set(getattr(self.config, 'reference_image_path', ''))
        self.analysis_history_days_var.set(getattr(self.config, 'analysis_history_days', 30))
        self.export_metadata_var.set(getattr(self.config, 'export_include_metadata', True))
        self.default_data_dir_var.set(getattr(self.config, 'default_data_dir', 'data'))
        self.default_models_dir_var.set(getattr(self.config, 'default_models_dir', 'data/models'))
        self.default_results_dir_var.set(getattr(self.config, 'default_results_dir', 'data/results'))
        
        # Chatbot settings
        self.api_key_var.set(getattr(self.config, 'gemini_api_key', ''))
        self.gemini_model_var.set(getattr(self.config, 'gemini_model', 'gemini-1.5-flash'))
        self.timeout_var.set(getattr(self.config, 'gemini_timeout', 30))
        self.temperature_var.set(getattr(self.config, 'gemini_temperature', 0.7))
        self.max_tokens_var.set(getattr(self.config, 'gemini_max_tokens', 2048))
        self.enable_ai_var.set(getattr(self.config, 'enable_ai_analysis', False))
        self.chat_history_limit_var.set(getattr(self.config, 'chat_history_limit', 100))
        self.chat_auto_save_var.set(getattr(self.config, 'chat_auto_save', True))
        self.response_format_var.set(getattr(self.config, 'response_format', 'Detailed'))
        self.enable_rate_limiting_var.set(getattr(self.config, 'enable_rate_limiting', True))
        self.requests_per_minute_var.set(getattr(self.config, 'requests_per_minute', 15))
        self.context_window_var.set(getattr(self.config, 'context_window_size', 4000))
        self.conversation_memory_var.set(getattr(self.config, 'enable_conversation_memory', True))
        self.chat_export_format_var.set(getattr(self.config, 'chat_export_format', 'JSON'))
        self.chatbot_persona_var.set(getattr(self.config, 'chatbot_persona', ''))
    
    def _init_change_tracking(self):
        """Initialize comprehensive change tracking for all settings variables."""
        # Store original values for comparison
        self.original_values = {
            # General settings
            'theme': self.theme_var.get(),
            'language': self.language_var.get(),
            'auto_save': self.auto_save_var.get(),
            'debug': self.debug_var.get(),
            'startup_fullscreen': self.startup_fullscreen_var.get(),
            'remember_window': self.remember_window_var.get(),
            'performance_mode': self.performance_mode_var.get(),
            'memory_limit': self.memory_limit_var.get(),
            'enable_logging': self.enable_logging_var.get(),
            'log_level': self.log_level_var.get(),
            'auto_save_interval': self.auto_save_interval_var.get(),
            
            # Webcam settings
            'camera_index': self.camera_index_var.get(),
            'camera_width': self.camera_width_var.get(),
            'camera_height': self.camera_height_var.get(),
            'camera_fps': self.camera_fps_var.get(),
            'auto_exposure': self.auto_exposure_var.get(),
            'auto_focus': self.auto_focus_var.get(),
            'brightness': self.brightness_var.get(),
            'contrast': self.contrast_var.get(),
            'saturation': self.saturation_var.get(),
            'recording_format': self.recording_format_var.get(),
            'buffer_size': self.buffer_size_var.get(),
            'preview_enabled': self.preview_enabled_var.get(),
            'camera_device_name': self.camera_device_name_var.get(),
            
            # Image Analysis settings
            'confidence_threshold': self.confidence_threshold_var.get(),
            'iou_threshold': self.iou_threshold_var.get(),
            'enable_roi': self.enable_roi_var.get(),
            'roi_x': self.roi_x_var.get(),
            'roi_y': self.roi_y_var.get(),
            'roi_width': self.roi_width_var.get(),
            'roi_height': self.roi_height_var.get(),
            'preferred_model': self.preferred_model_var.get(),
            'export_quality': self.export_quality_var.get(),
            'difference_sensitivity': self.difference_sensitivity_var.get(),
            'highlight_differences': self.highlight_differences_var.get(),
            'reference_image_path': self.reference_image_path_var.get(),
            'analysis_history_days': self.analysis_history_days_var.get(),
            'export_metadata': self.export_metadata_var.get(),
            'default_data_dir': self.default_data_dir_var.get(),
            'default_models_dir': self.default_models_dir_var.get(),
            'default_results_dir': self.default_results_dir_var.get(),
            
            # Chatbot settings
            'api_key': self.api_key_var.get(),
            'gemini_model': self.gemini_model_var.get(),
            'timeout': self.timeout_var.get(),
            'temperature': self.temperature_var.get(),
            'max_tokens': self.max_tokens_var.get(),
            'enable_ai': self.enable_ai_var.get(),
            'chat_history_limit': self.chat_history_limit_var.get(),
            'chat_auto_save': self.chat_auto_save_var.get(),
            'response_format': self.response_format_var.get(),
            'enable_rate_limiting': self.enable_rate_limiting_var.get(),
            'requests_per_minute': self.requests_per_minute_var.get(),
            'context_window': self.context_window_var.get(),
            'conversation_memory': self.conversation_memory_var.get(),
            'chat_export_format': self.chat_export_format_var.get(),
            'chatbot_persona': self.chatbot_persona_var.get(),
        }
        
        # Setup variable traces for change detection
        self._setup_variable_traces()
        
        # Initialize as no changes
        self.has_changes = False
        if hasattr(self, 'apply_button'):
            self.apply_button.configure(state='disabled')
    
    def _setup_variable_traces(self):
        """Setup trace callbacks for all variables to detect changes."""
        # General settings
        self.theme_var.trace('w', lambda *args: self._on_setting_changed('theme'))
        self.language_var.trace('w', lambda *args: self._on_setting_changed('language'))
        self.auto_save_var.trace('w', lambda *args: self._on_setting_changed('auto_save'))
        self.debug_var.trace('w', lambda *args: self._on_setting_changed('debug'))
        self.startup_fullscreen_var.trace('w', lambda *args: self._on_setting_changed('startup_fullscreen'))
        self.remember_window_var.trace('w', lambda *args: self._on_setting_changed('remember_window'))
        self.remember_window_state_var.trace('w', lambda *args: self._on_setting_changed('remember_window_state'))
        self.performance_mode_var.trace('w', lambda *args: self._on_setting_changed('performance_mode'))
        self.memory_limit_var.trace('w', lambda *args: self._on_setting_changed('memory_limit'))
        self.enable_logging_var.trace('w', lambda *args: self._on_setting_changed('enable_logging'))
        self.log_level_var.trace('w', lambda *args: self._on_setting_changed('log_level'))
        self.auto_save_interval_var.trace('w', lambda *args: self._on_setting_changed('auto_save_interval'))
        
        # Webcam settings
        self.camera_index_var.trace('w', lambda *args: self._on_setting_changed('camera_index'))
        self.camera_width_var.trace('w', lambda *args: self._on_setting_changed('camera_width'))
        self.camera_height_var.trace('w', lambda *args: self._on_setting_changed('camera_height'))
        self.camera_fps_var.trace('w', lambda *args: self._on_setting_changed('camera_fps'))
        self.auto_exposure_var.trace('w', lambda *args: self._on_setting_changed('auto_exposure'))
        self.auto_focus_var.trace('w', lambda *args: self._on_setting_changed('auto_focus'))
        self.brightness_var.trace('w', lambda *args: self._on_setting_changed('brightness'))
        self.contrast_var.trace('w', lambda *args: self._on_setting_changed('contrast'))
        self.saturation_var.trace('w', lambda *args: self._on_setting_changed('saturation'))
        self.recording_format_var.trace('w', lambda *args: self._on_setting_changed('recording_format'))
        self.buffer_size_var.trace('w', lambda *args: self._on_setting_changed('buffer_size'))
        self.preview_enabled_var.trace('w', lambda *args: self._on_setting_changed('preview_enabled'))
        self.camera_device_name_var.trace('w', lambda *args: self._on_setting_changed('camera_device_name'))
        
        # Image Analysis settings
        self.confidence_threshold_var.trace('w', lambda *args: self._on_setting_changed('confidence_threshold'))
        self.iou_threshold_var.trace('w', lambda *args: self._on_setting_changed('iou_threshold'))
        self.enable_roi_var.trace('w', lambda *args: self._on_setting_changed('enable_roi'))
        self.roi_x_var.trace('w', lambda *args: self._on_setting_changed('roi_x'))
        self.roi_y_var.trace('w', lambda *args: self._on_setting_changed('roi_y'))
        self.roi_width_var.trace('w', lambda *args: self._on_setting_changed('roi_width'))
        self.roi_height_var.trace('w', lambda *args: self._on_setting_changed('roi_height'))
        self.preferred_model_var.trace('w', lambda *args: self._on_setting_changed('preferred_model'))
        self.export_quality_var.trace('w', lambda *args: self._on_setting_changed('export_quality'))
        self.difference_sensitivity_var.trace('w', lambda *args: self._on_setting_changed('difference_sensitivity'))
        self.highlight_differences_var.trace('w', lambda *args: self._on_setting_changed('highlight_differences'))
        self.reference_image_path_var.trace('w', lambda *args: self._on_setting_changed('reference_image_path'))
        self.analysis_history_days_var.trace('w', lambda *args: self._on_setting_changed('analysis_history_days'))
        self.export_metadata_var.trace('w', lambda *args: self._on_setting_changed('export_metadata'))
        self.default_data_dir_var.trace('w', lambda *args: self._on_setting_changed('default_data_dir'))
        self.default_models_dir_var.trace('w', lambda *args: self._on_setting_changed('default_models_dir'))
        self.default_results_dir_var.trace('w', lambda *args: self._on_setting_changed('default_results_dir'))
        
        # Chatbot settings
        self.api_key_var.trace('w', lambda *args: self._on_setting_changed('api_key'))
        self.gemini_model_var.trace('w', lambda *args: self._on_setting_changed('gemini_model'))
        self.timeout_var.trace('w', lambda *args: self._on_setting_changed('timeout'))
        self.temperature_var.trace('w', lambda *args: self._on_setting_changed('temperature'))
        self.max_tokens_var.trace('w', lambda *args: self._on_setting_changed('max_tokens'))
        self.enable_ai_var.trace('w', lambda *args: self._on_setting_changed('enable_ai'))
        self.chat_history_limit_var.trace('w', lambda *args: self._on_setting_changed('chat_history_limit'))
        self.chat_auto_save_var.trace('w', lambda *args: self._on_setting_changed('chat_auto_save'))
        self.response_format_var.trace('w', lambda *args: self._on_setting_changed('response_format'))
        self.enable_rate_limiting_var.trace('w', lambda *args: self._on_setting_changed('enable_rate_limiting'))
        self.requests_per_minute_var.trace('w', lambda *args: self._on_setting_changed('requests_per_minute'))
        self.context_window_var.trace('w', lambda *args: self._on_setting_changed('context_window'))
        self.conversation_memory_var.trace('w', lambda *args: self._on_setting_changed('conversation_memory'))
        self.chat_export_format_var.trace('w', lambda *args: self._on_setting_changed('chat_export_format'))
        self.chatbot_persona_var.trace('w', lambda *args: self._on_setting_changed('chatbot_persona'))
    
    def _on_setting_changed(self, setting_name):
        """Handle when a setting value changes."""
        try:
            # Get current value from the appropriate variable
            var_name = self._get_variable_name_for_setting(setting_name)
            if hasattr(self, var_name):
                current_value = getattr(self, var_name).get()
                original_value = self.original_values.get(setting_name)
                
                # Check if value actually changed from original
                if current_value != original_value:
                    self._mark_as_changed()
                else:
                    # Check if any other changes exist
                    self._check_for_changes()
            
        except Exception as e:
            logger.warning(f"Error tracking change for {setting_name}: {e}")
    
    def _get_variable_name_for_setting(self, setting_name):
        """Get the tkinter variable name for a setting."""
        variable_mapping = {
            'theme': 'theme_var',
            'language': 'language_var',
            'auto_save': 'auto_save_var',
            'debug': 'debug_var',
            'startup_fullscreen': 'startup_fullscreen_var',
            'remember_window': 'remember_window_var',
            'performance_mode': 'performance_mode_var',
            'memory_limit': 'memory_limit_var',
            'enable_logging': 'enable_logging_var',
            'log_level': 'log_level_var',
            'auto_save_interval': 'auto_save_interval_var',
            'camera_index': 'camera_index_var',
            'camera_width': 'camera_width_var',
            'camera_height': 'camera_height_var',
            'camera_fps': 'camera_fps_var',
            'auto_exposure': 'auto_exposure_var',
            'auto_focus': 'auto_focus_var',
            'brightness': 'brightness_var',
            'contrast': 'contrast_var',
            'saturation': 'saturation_var',
            'recording_format': 'recording_format_var',
            'buffer_size': 'buffer_size_var',
            'preview_enabled': 'preview_enabled_var',
            'camera_device_name': 'camera_device_name_var',
            'confidence_threshold': 'confidence_threshold_var',
            'iou_threshold': 'iou_threshold_var',
            'enable_roi': 'enable_roi_var',
            'roi_x': 'roi_x_var',
            'roi_y': 'roi_y_var',
            'roi_width': 'roi_width_var',
            'roi_height': 'roi_height_var',
            'preferred_model': 'preferred_model_var',
            'export_quality': 'export_quality_var',
            'difference_sensitivity': 'difference_sensitivity_var',
            'highlight_differences': 'highlight_differences_var',
            'reference_image_path': 'reference_image_path_var',
            'analysis_history_days': 'analysis_history_days_var',
            'export_metadata': 'export_metadata_var',
            'default_data_dir': 'default_data_dir_var',
            'default_models_dir': 'default_models_dir_var',
            'default_results_dir': 'default_results_dir_var',
            'api_key': 'api_key_var',
            'gemini_model': 'gemini_model_var',
            'timeout': 'timeout_var',
            'temperature': 'temperature_var',
            'max_tokens': 'max_tokens_var',
            'enable_ai': 'enable_ai_var',
            'chat_history_limit': 'chat_history_limit_var',
            'chat_auto_save': 'chat_auto_save_var',
            'response_format': 'response_format_var',
            'enable_rate_limiting': 'enable_rate_limiting_var',
            'requests_per_minute': 'requests_per_minute_var',
            'context_window': 'context_window_var',
            'conversation_memory': 'conversation_memory_var',
            'chat_export_format': 'chat_export_format_var',
            'chatbot_persona': 'chatbot_persona_var',
        }
        return variable_mapping.get(setting_name, f'{setting_name}_var')
    
    def _check_for_changes(self):
        """Check if any settings have been changed from their original values."""
        has_changes = False
        
        # Check all variables against their original values
        for setting_name, original_value in self.original_values.items():
            var_name = self._get_variable_name_for_setting(setting_name)
            if hasattr(self, var_name):
                current_value = getattr(self, var_name).get()
                if current_value != original_value:
                    has_changes = True
                    break
        
        if has_changes:
            self._mark_as_changed()
        else:
            self._mark_as_unchanged()
    
    def _mark_as_changed(self):
        """Mark dialog as having changes and enable Apply button."""
        if not self.has_changes:
            self.has_changes = True
            if hasattr(self, 'apply_button'):
                self.apply_button.configure(state='normal')
            if hasattr(self, 'validation_status_label'):
                self.validation_status_label.configure(
                    text="Changes pending...", 
                    fg=self.COLORS['warning']
                )
    
    def _mark_as_unchanged(self):
        """Mark dialog as having no changes and disable Apply button."""
        if self.has_changes:
            self.has_changes = False
            if hasattr(self, 'apply_button'):
                self.apply_button.configure(state='disabled')
            if hasattr(self, 'validation_status_label'):
                self.validation_status_label.configure(
                    text="No changes to apply", 
                    fg=self.COLORS['text_muted']
                )
    
    def _reset_change_tracking(self):
        """Reset change tracking after successful apply operation."""
        # Update original values to current values
        for setting_name in self.original_values.keys():
            var_name = self._get_variable_name_for_setting(setting_name)
            if hasattr(self, var_name):
                self.original_values[setting_name] = getattr(self, var_name).get()
        
        # Mark as no changes
        self._mark_as_unchanged()
    
    def _create_checkbox(self, parent, text, variable, **kwargs):
        """Create a properly configured checkbox with fixed styling."""
        default_config = {
            'bg': self.COLORS['bg_secondary'],
            'fg': self.COLORS['text_primary'],
            'activebackground': self.COLORS['bg_tertiary'],
            'activeforeground': self.COLORS['text_primary'],
            'selectcolor': self.COLORS['bg_tertiary'],  # Background when checked
            'indicatoron': True,  # Ensure checkbox indicator is shown
            'font': ('Segoe UI', 9),
            'anchor': 'w',
            'relief': 'flat',
            'borderwidth': 0,
            'highlightthickness': 0,
            'disabledforeground': self.COLORS['text_muted'],
        }
        
        # Update with any custom kwargs
        default_config.update(kwargs)
        
        checkbox = tk.Checkbutton(
            parent, 
            text=text, 
            variable=variable, 
            **default_config
        )
        
        # Force immediate update of display to ensure proper rendering
        checkbox.update_idletasks()
        
        return checkbox
    
    def _build_ui(self):
        """Build the complete dialog user interface."""
        # Main container
        main_frame = tk.Frame(self.dialog, bg=self.COLORS['bg_primary'])
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="⚙️ Comprehensive Settings",
            bg=self.COLORS['bg_primary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 16, 'bold')
        )
        title_label.pack(pady=(0, 15))
        
        # Settings notebook
        self._build_settings_notebook(main_frame)
        
        # Buttons frame
        self._build_buttons(main_frame)
    
    def _build_settings_notebook(self, parent):
        """Build the tabbed settings notebook."""
        notebook_frame = tk.Frame(parent, bg=self.COLORS['bg_primary'])
        notebook_frame.pack(fill='both', expand=True, pady=(0, 15))
        
        # Create notebook
        self.notebook = ttk.Notebook(notebook_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Configure notebook style
        style = ttk.Style()
        style.configure('Settings.TNotebook', background=self.COLORS['bg_primary'])
        style.configure('Settings.TNotebook.Tab',
                       background=self.COLORS['bg_tertiary'],
                       foreground=self.COLORS['text_primary'],
                       padding=(15, 10))
        style.map('Settings.TNotebook.Tab',
                  background=[('selected', self.COLORS['accent_primary']),
                             ('active', self.COLORS['bg_secondary'])])
        
        # Add tabs
        general_tab = self._build_general_tab()
        self.notebook.add(general_tab, text="🎛️ General")
        
        webcam_tab = self._build_webcam_tab()
        self.notebook.add(webcam_tab, text="📹 Webcam")
        
        analysis_tab = self._build_analysis_tab()
        self.notebook.add(analysis_tab, text="🔍 Analysis")
        
        chatbot_tab = self._build_chatbot_tab()
        self.notebook.add(chatbot_tab, text="🤖 Chatbot")
        
        advanced_tab = self._build_advanced_tab()
        self.notebook.add(advanced_tab, text="⚡ Advanced")
    
    def _create_section_frame(self, parent, title: str) -> tuple[tk.Frame, tk.Frame]:
        """Create a styled section frame with title and content area."""
        section_frame = tk.Frame(parent, bg=self.COLORS['bg_secondary'], relief='solid', bd=1)
        
        # Section header
        header_frame = tk.Frame(section_frame, bg=self.COLORS['bg_tertiary'], height=35)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame,
            text=title,
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 10, 'bold')
        ).pack(side='left', padx=15, pady=8)
        
        # Content frame
        content_frame = tk.Frame(section_frame, bg=self.COLORS['bg_secondary'])
        content_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        return section_frame, content_frame
    
    def _build_general_tab(self) -> tk.Frame:
        """Build the General settings tab."""
        tab = tk.Frame(self.notebook, bg=self.COLORS['bg_primary'])
        
        # Scrollable frame
        canvas = tk.Canvas(tab, bg=self.COLORS['bg_primary'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.COLORS['bg_primary'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Theme Settings Section
        theme_section, theme_content = self._create_section_frame(scrollable_frame, "🎨 Appearance")
        theme_section.pack(fill='x', pady=(0, 10))
        
        # Theme selection
        tk.Label(theme_content, text="Theme:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).grid(row=0, column=0, sticky='w', pady=5)
        
        theme_combo = ttk.Combobox(theme_content, textvariable=self.theme_var, 
                                  values=SettingsValidator.VALID_THEMES, state='readonly')
        theme_combo.grid(row=0, column=1, sticky='w', padx=(10, 0), pady=5)
        
        # Language selection
        tk.Label(theme_content, text="Language:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).grid(row=1, column=0, sticky='w', pady=5)
        
        lang_combo = ttk.Combobox(theme_content, textvariable=self.language_var,
                                 values=SettingsValidator.VALID_LANGUAGES, state='readonly')
        lang_combo.grid(row=1, column=1, sticky='w', padx=(10, 0), pady=5)
        
        # Application Behavior Section
        behavior_section, behavior_content = self._create_section_frame(scrollable_frame, "🏃 Behavior")
        behavior_section.pack(fill='x', pady=(0, 10))
        
        # Auto-save config
        self._create_checkbox(behavior_content, "Auto-save configuration changes",
                             self.auto_save_var).pack(anchor='w', pady=2)
        
        # Debug mode
        self._create_checkbox(behavior_content, "Enable debug mode",
                             self.debug_var).pack(anchor='w', pady=2)
        
        # Startup fullscreen
        self._create_checkbox(behavior_content, "Start in fullscreen mode",
                             self.startup_fullscreen_var).pack(anchor='w', pady=2)
        
        # Remember window state
        self._create_checkbox(behavior_content, "Remember window size and position",
                             self.remember_window_var).pack(anchor='w', pady=2)
        
        # Auto-save interval
        save_frame = tk.Frame(behavior_content, bg=self.COLORS['bg_secondary'])
        save_frame.pack(fill='x', pady=5)
        
        tk.Label(save_frame, text="Auto-save interval (minutes):", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        save_spin = tk.Spinbox(save_frame, from_=1, to=60, textvariable=self.auto_save_interval_var,
                              bg=self.COLORS['bg_tertiary'], fg=self.COLORS['text_primary'], width=6)
        save_spin.pack(side='left', padx=(10, 0))
        
        # Backup on change - REMOVED as per requirements
        
        # Performance Section
        perf_section, perf_content = self._create_section_frame(scrollable_frame, "⚡ Performance")
        perf_section.pack(fill='x', pady=(0, 10))
        
        # Performance mode
        tk.Label(perf_content, text="Performance Mode:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).grid(row=0, column=0, sticky='w', pady=5)
        
        perf_combo = ttk.Combobox(perf_content, textvariable=self.performance_mode_var,
                                 values=SettingsValidator.VALID_PERFORMANCE_MODES, state='readonly')
        perf_combo.grid(row=0, column=1, sticky='w', padx=(10, 0), pady=5)
        
        # Memory limit
        tk.Label(perf_content, text="Memory Limit (MB):", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).grid(row=1, column=0, sticky='w', pady=5)
        
        memory_spin = tk.Spinbox(perf_content, from_=512, to=16384, increment=256,
                               textvariable=self.memory_limit_var, bg=self.COLORS['bg_tertiary'],
                               fg=self.COLORS['text_primary'], width=10)
        memory_spin.grid(row=1, column=1, sticky='w', padx=(10, 0), pady=5)
        
        # Logging Section
        log_section, log_content = self._create_section_frame(scrollable_frame, "📝 Logging")
        log_section.pack(fill='x')
        
        # Enable logging
        self._create_checkbox(log_content, "Enable logging",
                             self.enable_logging_var).pack(anchor='w', pady=2)
        
        # Log level
        log_frame = tk.Frame(log_content, bg=self.COLORS['bg_secondary'])
        log_frame.pack(fill='x', pady=5)
        
        tk.Label(log_frame, text="Log Level:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        log_combo = ttk.Combobox(log_frame, textvariable=self.log_level_var,
                                values=SettingsValidator.VALID_LOG_LEVELS, state='readonly', width=15)
        log_combo.pack(side='left', padx=(10, 0))
        
        # Default Directories Section
        dirs_section, dirs_content = self._create_section_frame(scrollable_frame, "📁 Default Directories")
        dirs_section.pack(fill='x', pady=(10, 0))
        
        # Default directories
        dirs_frame = tk.Frame(dirs_content, bg=self.COLORS['bg_secondary'])
        dirs_frame.pack(fill='x', pady=5)
        
        # Data directory
        data_dir_frame = tk.Frame(dirs_frame, bg=self.COLORS['bg_secondary'])
        data_dir_frame.pack(fill='x', pady=2)
        
        tk.Label(data_dir_frame, text="Data Directory:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        data_entry = tk.Entry(data_dir_frame, textvariable=self.default_data_dir_var,
                             bg=self.COLORS['bg_tertiary'], fg=self.COLORS['text_primary'],
                             font=('Segoe UI', 9), width=30)
        data_entry.pack(side='left', padx=(10, 5))
        
        ttk.Button(data_dir_frame, text="Browse", 
                  command=lambda: self._browse_directory(self.default_data_dir_var, "Select Data Directory")).pack(side='left')
        
        # Models directory
        models_dir_frame = tk.Frame(dirs_frame, bg=self.COLORS['bg_secondary'])
        models_dir_frame.pack(fill='x', pady=2)
        
        tk.Label(models_dir_frame, text="Models Directory:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        models_entry = tk.Entry(models_dir_frame, textvariable=self.default_models_dir_var,
                               bg=self.COLORS['bg_tertiary'], fg=self.COLORS['text_primary'],
                               font=('Segoe UI', 9), width=30)
        models_entry.pack(side='left', padx=(10, 5))
        
        ttk.Button(models_dir_frame, text="Browse", 
                  command=lambda: self._browse_directory(self.default_models_dir_var, "Select Models Directory")).pack(side='left')
        
        # Results directory
        results_dir_frame = tk.Frame(dirs_frame, bg=self.COLORS['bg_secondary'])
        results_dir_frame.pack(fill='x', pady=2)
        
        tk.Label(results_dir_frame, text="Results Directory:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        results_entry = tk.Entry(results_dir_frame, textvariable=self.default_results_dir_var,
                                bg=self.COLORS['bg_tertiary'], fg=self.COLORS['text_primary'],
                                font=('Segoe UI', 9), width=30)
        results_entry.pack(side='left', padx=(10, 5))
        
        ttk.Button(results_dir_frame, text="Browse", 
                  command=lambda: self._browse_directory(self.default_results_dir_var, "Select Results Directory")).pack(side='left')
        
        return tab
    
    def _build_webcam_tab(self) -> tk.Frame:
        """Build the Webcam settings tab."""
        tab = tk.Frame(self.notebook, bg=self.COLORS['bg_primary'])
        
        # Scrollable frame
        canvas = tk.Canvas(tab, bg=self.COLORS['bg_primary'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.COLORS['bg_primary'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Device Selection Section
        device_section, device_content = self._create_section_frame(scrollable_frame, "📱 Camera Device")
        device_section.pack(fill='x', pady=(0, 10))
        
        # Camera index selection with detection
        cam_frame = tk.Frame(device_content, bg=self.COLORS['bg_secondary'])
        cam_frame.pack(fill='x', pady=5)
        
        tk.Label(cam_frame, text="Camera:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        camera_spin = tk.Spinbox(cam_frame, from_=0, to=10, textvariable=self.camera_index_var,
                               bg=self.COLORS['bg_tertiary'], fg=self.COLORS['text_primary'], width=5)
        camera_spin.pack(side='left', padx=(10, 5))
        
        # Detect cameras button
        ttk.Button(cam_frame, text="Detect Cameras", command=self._detect_cameras).pack(side='left', padx=(5, 0))
        
        # Camera name display
        self.camera_name_label = tk.Label(device_content, text="", bg=self.COLORS['bg_secondary'], 
                                         fg=self.COLORS['text_muted'], font=('Segoe UI', 8))
        self.camera_name_label.pack(anchor='w', pady=(0, 5))
        
        # Live Camera Preview Section - moved here per requirements
        preview_section, preview_content = self._create_section_frame(scrollable_frame, "📹 Live Camera Preview")
        preview_section.pack(fill='x', pady=(0, 10))
        
        # Preview canvas
        canvas_frame = tk.Frame(preview_content, bg=self.COLORS['bg_primary'], relief='sunken', bd=1)
        canvas_frame.pack(fill='x', pady=5)
        
        self._preview_canvas = tk.Canvas(
            canvas_frame,
            width=400,
            height=240,
            bg='black',
            highlightthickness=1,
            highlightbackground=self.COLORS['border']
        )
        self._preview_canvas.pack()
        
        # Display placeholder text
        self._preview_canvas.create_text(
            200, 120, 
            text="Camera Preview\nSelect camera and click 'Test' to preview",
            fill='white', 
            font=('Segoe UI', 12),
            justify='center'
        )
        
        # Preview controls
        preview_controls = tk.Frame(preview_content, bg=self.COLORS['bg_secondary'])
        preview_controls.pack(fill='x', pady=(5, 0))
        
        self.test_button = ttk.Button(
            preview_controls,
            text="🎥 Test Camera",
            command=self._test_camera
        )
        self.test_button.pack(side='left', padx=(0, 5))
        
        self.stop_test_button = ttk.Button(
            preview_controls,
            text="⏹ Stop Test",
            command=self._stop_test,
            state='disabled'
        )
        self.stop_test_button.pack(side='left', padx=5)
        
        # Camera info display
        self.camera_info_var = tk.StringVar(value="Select a camera to test")
        info_label = tk.Label(
            preview_content,
            textvariable=self.camera_info_var,
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8)
        )
        info_label.pack(pady=(5, 0))
        
        # Buffer settings
        buffer_frame = tk.Frame(preview_content, bg=self.COLORS['bg_secondary'])
        buffer_frame.pack(fill='x', pady=5)
        
        tk.Label(buffer_frame, text="Buffer Size (frames):", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        buffer_spin = tk.Spinbox(buffer_frame, from_=1, to=30, textvariable=self.buffer_size_var,
                               bg=self.COLORS['bg_tertiary'], fg=self.COLORS['text_primary'], width=6)
        buffer_spin.pack(side='left', padx=(10, 0))
        
        # Resolution and Performance Section
        res_section, res_content = self._create_section_frame(scrollable_frame, "🎥 Resolution & Performance")
        res_section.pack(fill='x', pady=(0, 10))
        
        # Resolution presets and custom
        res_preset_frame = tk.Frame(res_content, bg=self.COLORS['bg_secondary'])
        res_preset_frame.pack(fill='x', pady=5)
        
        tk.Label(res_preset_frame, text="Preset:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        preset_combo = ttk.Combobox(res_preset_frame, values=["720p (1280x720)", "1080p (1920x1080)", "4K (3840x2160)", "Custom"],
                                   state='readonly', width=20)
        preset_combo.pack(side='left', padx=(10, 0))
        preset_combo.bind('<<ComboboxSelected>>', self._on_resolution_preset_changed)
        
        # Custom resolution
        custom_res_frame = tk.Frame(res_content, bg=self.COLORS['bg_secondary'])
        custom_res_frame.pack(fill='x', pady=5)
        
        tk.Label(custom_res_frame, text="Width:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        width_spin = tk.Spinbox(custom_res_frame, from_=240, to=4096, increment=80,
                              textvariable=self.camera_width_var, bg=self.COLORS['bg_tertiary'],
                              fg=self.COLORS['text_primary'], width=8)
        width_spin.pack(side='left', padx=(5, 10))
        
        tk.Label(custom_res_frame, text="Height:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        height_spin = tk.Spinbox(custom_res_frame, from_=240, to=4096, increment=60,
                               textvariable=self.camera_height_var, bg=self.COLORS['bg_tertiary'],
                               fg=self.COLORS['text_primary'], width=8)
        height_spin.pack(side='left', padx=(5, 10))
        
        tk.Label(custom_res_frame, text="FPS:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        fps_spin = tk.Spinbox(custom_res_frame, from_=1, to=120, increment=5,
                            textvariable=self.camera_fps_var, bg=self.COLORS['bg_tertiary'],
                            fg=self.COLORS['text_primary'], width=6)
        fps_spin.pack(side='left', padx=(5, 0))
        
        # Camera Controls Section
        controls_section, controls_content = self._create_section_frame(scrollable_frame, "🎛️ Camera Controls")
        controls_section.pack(fill='x', pady=(0, 10))
        
        # Auto settings
        auto_frame = tk.Frame(controls_content, bg=self.COLORS['bg_secondary'])
        auto_frame.pack(fill='x', pady=5)
        
        self._create_checkbox(auto_frame, "Auto Exposure",
                             self.auto_exposure_var).pack(side='left', padx=(0, 20))
        
        self._create_checkbox(auto_frame, "Auto Focus",
                             self.auto_focus_var).pack(side='left')
        
        # Manual adjustments
        adj_frame = tk.Frame(controls_content, bg=self.COLORS['bg_secondary'])
        adj_frame.pack(fill='x', pady=5)
        
        # Brightness
        brightness_frame = tk.Frame(adj_frame, bg=self.COLORS['bg_secondary'])
        brightness_frame.pack(fill='x', pady=2)
        
        tk.Label(brightness_frame, text="Brightness:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9), width=12, anchor='w').pack(side='left')
        
        brightness_scale = tk.Scale(brightness_frame, from_=-100, to=100, orient='horizontal',
                                  variable=self.brightness_var, bg=self.COLORS['bg_secondary'],
                                  fg=self.COLORS['text_primary'], highlightthickness=0, length=200)
        brightness_scale.pack(side='left', padx=(5, 0))
        
        # Contrast
        contrast_frame = tk.Frame(adj_frame, bg=self.COLORS['bg_secondary'])
        contrast_frame.pack(fill='x', pady=2)
        
        tk.Label(contrast_frame, text="Contrast:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9), width=12, anchor='w').pack(side='left')
        
        contrast_scale = tk.Scale(contrast_frame, from_=-100, to=100, orient='horizontal',
                                variable=self.contrast_var, bg=self.COLORS['bg_secondary'],
                                fg=self.COLORS['text_primary'], highlightthickness=0, length=200)
        contrast_scale.pack(side='left', padx=(5, 0))
        
        # Saturation
        saturation_frame = tk.Frame(adj_frame, bg=self.COLORS['bg_secondary'])
        saturation_frame.pack(fill='x', pady=2)
        
        tk.Label(saturation_frame, text="Saturation:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9), width=12, anchor='w').pack(side='left')
        
        saturation_scale = tk.Scale(saturation_frame, from_=-100, to=100, orient='horizontal',
                                  variable=self.saturation_var, bg=self.COLORS['bg_secondary'],
                                  fg=self.COLORS['text_primary'], highlightthickness=0, length=200)
        saturation_scale.pack(side='left', padx=(5, 0))
        
        return tab
    
    def _build_analysis_tab(self) -> tk.Frame:
        """Build the Image Analysis settings tab."""
        tab = tk.Frame(self.notebook, bg=self.COLORS['bg_primary'])
        
        # Scrollable frame
        canvas = tk.Canvas(tab, bg=self.COLORS['bg_primary'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.COLORS['bg_primary'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Detection Settings Section
        detection_section, detection_content = self._create_section_frame(scrollable_frame, "🎯 Detection Settings")
        detection_section.pack(fill='x', pady=(0, 10))
        
        # Confidence threshold
        conf_frame = tk.Frame(detection_content, bg=self.COLORS['bg_secondary'])
        conf_frame.pack(fill='x', pady=5)
        
        tk.Label(conf_frame, text="Confidence Threshold:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9), width=20, anchor='w').pack(side='left')
        
        conf_scale = tk.Scale(conf_frame, from_=0.0, to=1.0, resolution=0.01, orient='horizontal',
                            variable=self.confidence_threshold_var, bg=self.COLORS['bg_secondary'],
                            fg=self.COLORS['text_primary'], highlightthickness=0, length=200)
        conf_scale.pack(side='left', padx=(5, 0))
        
        # IoU threshold
        iou_frame = tk.Frame(detection_content, bg=self.COLORS['bg_secondary'])
        iou_frame.pack(fill='x', pady=5)
        
        tk.Label(iou_frame, text="IoU Threshold:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9), width=20, anchor='w').pack(side='left')
        
        iou_scale = tk.Scale(iou_frame, from_=0.0, to=1.0, resolution=0.01, orient='horizontal',
                           variable=self.iou_threshold_var, bg=self.COLORS['bg_secondary'],
                           fg=self.COLORS['text_primary'], highlightthickness=0, length=200)
        iou_scale.pack(side='left', padx=(5, 0))
        
        # Image Processing Section - REMOVED as per requirements
        
        # Region of Interest Section
        roi_section, roi_content = self._create_section_frame(scrollable_frame, "📐 Region of Interest")
        roi_section.pack(fill='x', pady=(0, 10))
        
        # Enable ROI
        self._create_checkbox(roi_content, "Enable Region of Interest (ROI)",
                             self.enable_roi_var).pack(anchor='w', pady=5)
        
        # ROI coordinates
        roi_coord_frame = tk.Frame(roi_content, bg=self.COLORS['bg_secondary'])
        roi_coord_frame.pack(fill='x', pady=5)
        
        # X and Y
        tk.Label(roi_coord_frame, text="X:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).grid(row=0, column=0, padx=(0, 5))
        
        tk.Spinbox(roi_coord_frame, from_=0, to=4096, textvariable=self.roi_x_var,
                  bg=self.COLORS['bg_tertiary'], fg=self.COLORS['text_primary'], width=8).grid(row=0, column=1, padx=(0, 10))
        
        tk.Label(roi_coord_frame, text="Y:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).grid(row=0, column=2, padx=(0, 5))
        
        tk.Spinbox(roi_coord_frame, from_=0, to=4096, textvariable=self.roi_y_var,
                  bg=self.COLORS['bg_tertiary'], fg=self.COLORS['text_primary'], width=8).grid(row=0, column=3)
        
        # Width and Height
        tk.Label(roi_coord_frame, text="Width:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).grid(row=1, column=0, padx=(0, 5), pady=(5, 0))
        
        tk.Spinbox(roi_coord_frame, from_=0, to=4096, textvariable=self.roi_width_var,
                  bg=self.COLORS['bg_tertiary'], fg=self.COLORS['text_primary'], width=8).grid(row=1, column=1, padx=(0, 10), pady=(5, 0))
        
        tk.Label(roi_coord_frame, text="Height:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).grid(row=1, column=2, padx=(0, 5), pady=(5, 0))
        
        tk.Spinbox(roi_coord_frame, from_=0, to=4096, textvariable=self.roi_height_var,
                  bg=self.COLORS['bg_tertiary'], fg=self.COLORS['text_primary'], width=8).grid(row=1, column=3, pady=(5, 0))
        
        # Model and Export Section
        model_section, model_content = self._create_section_frame(scrollable_frame, "⚙️ Model & Export")
        model_section.pack(fill='x', pady=(0, 10))
        
        # Model selection
        model_frame = tk.Frame(model_content, bg=self.COLORS['bg_secondary'])
        model_frame.pack(fill='x', pady=5)
        
        tk.Label(model_frame, text="Preferred Model:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        model_combo = ttk.Combobox(model_frame, textvariable=self.preferred_model_var,
                                  values=SettingsValidator.VALID_MODEL_SIZES, state='readonly', width=15)
        model_combo.pack(side='left', padx=(10, 0))
        
        # Model info label
        model_info_frame = tk.Frame(model_content, bg=self.COLORS['bg_secondary'])
        model_info_frame.pack(fill='x', pady=(0, 5))
        
        model_info = tk.Label(
            model_info_frame,
            text="YOLOv12 models (recommended) offer improved accuracy and speed over previous versions.\nModel sizes: n(nano) < s(small) < m(medium) < l(large) < x(extra-large)",
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 8),
            justify='left'
        )
        model_info.pack(anchor='w')
        
        # Export settings - Format removed, quality fixed at 100%
        export_frame = tk.Frame(model_content, bg=self.COLORS['bg_secondary'])
        export_frame.pack(fill='x', pady=5)
        
        tk.Label(export_frame, text="Export Quality: 100% (PNG format)", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        # Difference Detection Section
        diff_section, diff_content = self._create_section_frame(scrollable_frame, "🔍 Difference Detection")
        diff_section.pack(fill='x')
        
        # Sensitivity
        sens_frame = tk.Frame(diff_content, bg=self.COLORS['bg_secondary'])
        sens_frame.pack(fill='x', pady=5)
        
        tk.Label(sens_frame, text="Sensitivity:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9), width=15, anchor='w').pack(side='left')
        
        sens_scale = tk.Scale(sens_frame, from_=0.0, to=1.0, resolution=0.01, orient='horizontal',
                            variable=self.difference_sensitivity_var, bg=self.COLORS['bg_secondary'],
                            fg=self.COLORS['text_primary'], highlightthickness=0, length=200)
        sens_scale.pack(side='left', padx=(5, 0))
        
        # Highlight differences
        self._create_checkbox(diff_content, "Highlight differences automatically",
                             self.highlight_differences_var).pack(anchor='w', pady=5)
        
        return tab
    
    def _build_chatbot_tab(self) -> tk.Frame:
        """Build the Chatbot settings tab."""
        tab = tk.Frame(self.notebook, bg=self.COLORS['bg_primary'])
        
        # Scrollable frame
        canvas = tk.Canvas(tab, bg=self.COLORS['bg_primary'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.COLORS['bg_primary'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # API Configuration Section
        api_section, api_content = self._create_section_frame(scrollable_frame, "🔑 API Configuration")
        api_section.pack(fill='x', pady=(0, 10))
        
        # API Key
        api_frame = tk.Frame(api_content, bg=self.COLORS['bg_secondary'])
        api_frame.pack(fill='x', pady=5)
        
        tk.Label(api_frame, text="Gemini API Key:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(anchor='w', pady=(0, 5))
        
        key_entry_frame = tk.Frame(api_frame, bg=self.COLORS['bg_secondary'])
        key_entry_frame.pack(fill='x')
        
        self.api_key_entry = tk.Entry(key_entry_frame, textvariable=self.api_key_var,
                                     bg=self.COLORS['bg_tertiary'], fg=self.COLORS['text_primary'],
                                     font=('Segoe UI', 9), show='*', borderwidth=0,
                                     insertbackground=self.COLORS['text_primary'])
        self.api_key_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        # Show/Hide API key button
        self.show_key_button = tk.Button(key_entry_frame, text="👁", bg=self.COLORS['bg_tertiary'],
                                        fg=self.COLORS['text_primary'], borderwidth=0,
                                        command=self._toggle_api_key_visibility, font=('Segoe UI', 8), width=3)
        self.show_key_button.pack(side='right')
        
        # API help text
        help_label = tk.Label(api_content, text="Get your free API key from Google AI Studio:\nhttps://makersuite.google.com/app/apikey",
                             bg=self.COLORS['bg_secondary'], fg=self.COLORS['text_muted'],
                             font=('Segoe UI', 8), justify='left')
        help_label.pack(anchor='w', pady=(5, 0))
        
        # Model Configuration Section
        model_section, model_content = self._create_section_frame(scrollable_frame, "🤖 Model Configuration")
        model_section.pack(fill='x', pady=(0, 10))
        
        # Model selection
        model_frame = tk.Frame(model_content, bg=self.COLORS['bg_secondary'])
        model_frame.pack(fill='x', pady=5)
        
        tk.Label(model_frame, text="Model:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        gemini_model_combo = ttk.Combobox(model_frame, textvariable=self.gemini_model_var,
                                         values=SettingsValidator.VALID_GEMINI_MODELS, state='readonly', width=20)
        gemini_model_combo.pack(side='left', padx=(10, 0))
        
        # Temperature
        temp_frame = tk.Frame(model_content, bg=self.COLORS['bg_secondary'])
        temp_frame.pack(fill='x', pady=5)
        
        tk.Label(temp_frame, text="Creativity (Temperature):", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9), width=20, anchor='w').pack(side='left')
        
        temp_scale = tk.Scale(temp_frame, from_=0.0, to=1.0, resolution=0.1, orient='horizontal',
                            variable=self.temperature_var, bg=self.COLORS['bg_secondary'],
                            fg=self.COLORS['text_primary'], highlightthickness=0, length=200)
        temp_scale.pack(side='left', padx=(5, 0))
        
        # Max tokens
        tokens_frame = tk.Frame(model_content, bg=self.COLORS['bg_secondary'])
        tokens_frame.pack(fill='x', pady=5)
        
        tk.Label(tokens_frame, text="Max Tokens:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        tokens_spin = tk.Spinbox(tokens_frame, from_=100, to=8192, increment=256,
                               textvariable=self.max_tokens_var, bg=self.COLORS['bg_tertiary'],
                               fg=self.COLORS['text_primary'], width=8)
        tokens_spin.pack(side='left', padx=(10, 20))
        
        # Timeout
        tk.Label(tokens_frame, text="Timeout (s):", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        timeout_spin = tk.Spinbox(tokens_frame, from_=5, to=300, increment=5,
                                textvariable=self.timeout_var, bg=self.COLORS['bg_tertiary'],
                                fg=self.COLORS['text_primary'], width=6)
        timeout_spin.pack(side='left', padx=(10, 0))
        
        # Chat Behavior Section
        chat_section, chat_content = self._create_section_frame(scrollable_frame, "💬 Chat Behavior")
        chat_section.pack(fill='x', pady=(0, 10))
        
        # Enable AI analysis
        self._create_checkbox(chat_content, "Enable AI analysis features",
                             self.enable_ai_var).pack(anchor='w', pady=2)
        
        # Response format
        format_frame = tk.Frame(chat_content, bg=self.COLORS['bg_secondary'])
        format_frame.pack(fill='x', pady=5)
        
        tk.Label(format_frame, text="Response Format:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        response_combo = ttk.Combobox(format_frame, textvariable=self.response_format_var,
                                     values=SettingsValidator.VALID_RESPONSE_FORMATS, state='readonly', width=15)
        response_combo.pack(side='left', padx=(10, 0))
        
        # Chat history limit
        history_frame = tk.Frame(chat_content, bg=self.COLORS['bg_secondary'])
        history_frame.pack(fill='x', pady=5)
        
        tk.Label(history_frame, text="History Limit:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        history_spin = tk.Spinbox(history_frame, from_=10, to=1000, increment=10,
                                textvariable=self.chat_history_limit_var, bg=self.COLORS['bg_tertiary'],
                                fg=self.COLORS['text_primary'], width=8)
        history_spin.pack(side='left', padx=(10, 0))
        
        # Auto-save and memory
        self._create_checkbox(chat_content, "Auto-save chat history",
                             self.chat_auto_save_var).pack(anchor='w', pady=2)
        
        self._create_checkbox(chat_content, "Enable conversation memory",
                             self.conversation_memory_var).pack(anchor='w', pady=2)
        
        # Chat export format
        export_format_frame = tk.Frame(chat_content, bg=self.COLORS['bg_secondary'])
        export_format_frame.pack(fill='x', pady=5)
        
        tk.Label(export_format_frame, text="Export Format:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        export_format_combo = ttk.Combobox(export_format_frame, textvariable=self.chat_export_format_var,
                                          values=SettingsValidator.VALID_CHAT_EXPORT_FORMATS, state='readonly', width=10)
        export_format_combo.pack(side='left', padx=(10, 0))
        
        # Custom Persona Section
        persona_section, persona_content = self._create_section_frame(scrollable_frame, "🎭 Custom Role/Persona")
        persona_section.pack(fill='x', pady=(0, 10))
        
        # Persona label
        persona_label_frame = tk.Frame(persona_content, bg=self.COLORS['bg_secondary'])
        persona_label_frame.pack(fill='x', pady=(0, 5))
        
        tk.Label(persona_label_frame, text="Custom Role/Persona:", 
                bg=self.COLORS['bg_secondary'], fg=self.COLORS['text_primary'], 
                font=('Segoe UI', 9)).pack(anchor='w')
        
        tk.Label(persona_label_frame, text="Define how the AI should behave and respond (leave empty for default):",
                bg=self.COLORS['bg_secondary'], fg=self.COLORS['text_muted'], 
                font=('Segoe UI', 8)).pack(anchor='w', pady=(2, 0))
        
        # Persona text widget with scrollbar
        persona_frame = tk.Frame(persona_content, bg=self.COLORS['bg_secondary'])
        persona_frame.pack(fill='x', pady=5)
        
        # Create text widget container
        text_container = tk.Frame(persona_frame, bg=self.COLORS['bg_tertiary'], relief='sunken', bd=1)
        text_container.pack(fill='x', padx=2, pady=2)
        
        # Scrollable text widget
        self.persona_text = tk.Text(
            text_container,
            height=6,  # 6 lines visible
            width=60,
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_primary'],
            font=('Segoe UI', 9),
            wrap='word',
            borderwidth=0,
            insertbackground=self.COLORS['text_primary'],
            selectbackground=self.COLORS['accent_primary'],
            selectforeground=self.COLORS['text_primary']
        )
        
        persona_scrollbar = ttk.Scrollbar(text_container, command=self.persona_text.yview)
        self.persona_text.configure(yscrollcommand=persona_scrollbar.set)
        
        self.persona_text.pack(side='left', fill='both', expand=True)
        persona_scrollbar.pack(side='right', fill='y')
        
        # Load current persona text
        current_persona = getattr(self.config, 'chatbot_persona', '')
        # Don't set default display text - persona should only define behavior, not display content
        # Leave empty if no persona is configured
        self.persona_text.insert('1.0', current_persona)
        
        # Bind text changes to update the variable (for change tracking)
        def on_persona_change(*args):
            try:
                text_content = self.persona_text.get('1.0', tk.END).strip()
                self.chatbot_persona_var.set(text_content)
            except tk.TclError:
                pass  # Widget might be destroyed
        
        self.persona_text.bind('<KeyRelease>', on_persona_change)
        self.persona_text.bind('<FocusOut>', on_persona_change)
        
        # Character count label
        char_count_frame = tk.Frame(persona_content, bg=self.COLORS['bg_secondary'])
        char_count_frame.pack(fill='x', pady=(5, 0))
        
        self.persona_char_count = tk.Label(char_count_frame, 
                                          text=f"Characters: {len(current_persona)}/5000",
                                          bg=self.COLORS['bg_secondary'], 
                                          fg=self.COLORS['text_muted'], 
                                          font=('Segoe UI', 8))
        self.persona_char_count.pack(anchor='e')
        
        # Update character count when text changes
        def update_char_count(*args):
            try:
                text_content = self.persona_text.get('1.0', tk.END).strip()
                char_count = len(text_content)
                
                # Color coding for character count
                if char_count > 5000:
                    color = self.COLORS['error']
                elif char_count > 4500:
                    color = self.COLORS['warning']
                else:
                    color = self.COLORS['text_muted']
                
                self.persona_char_count.config(
                    text=f"Characters: {char_count}/5000",
                    fg=color
                )
            except tk.TclError:
                pass  # Widget might be destroyed
        
        self.persona_text.bind('<KeyRelease>', lambda e: (on_persona_change(), update_char_count()))
        self.persona_text.bind('<FocusOut>', lambda e: (on_persona_change(), update_char_count()))
        
        # Rate Limiting Section
        rate_section, rate_content = self._create_section_frame(scrollable_frame, "⏱️ Rate Limiting")
        rate_section.pack(fill='x', pady=(0, 10))
        
        # Enable rate limiting
        self._create_checkbox(rate_content, "Enable rate limiting",
                             self.enable_rate_limiting_var).pack(anchor='w', pady=2)
        
        # Requests per minute
        rate_frame = tk.Frame(rate_content, bg=self.COLORS['bg_secondary'])
        rate_frame.pack(fill='x', pady=5)
        
        tk.Label(rate_frame, text="Requests/minute:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        rate_spin = tk.Spinbox(rate_frame, from_=1, to=60, textvariable=self.requests_per_minute_var,
                             bg=self.COLORS['bg_tertiary'], fg=self.COLORS['text_primary'], width=6)
        rate_spin.pack(side='left', padx=(10, 20))
        
        # Context window size
        tk.Label(rate_frame, text="Context Window:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        context_spin = tk.Spinbox(rate_frame, from_=1000, to=32000, increment=1000,
                                textvariable=self.context_window_var, bg=self.COLORS['bg_tertiary'],
                                fg=self.COLORS['text_primary'], width=8)
        context_spin.pack(side='left', padx=(10, 0))
        
        # Test Connection Section
        test_section, test_content = self._create_section_frame(scrollable_frame, "🧪 Connection Test")
        test_section.pack(fill='x')
        
        # Test button
        test_frame = tk.Frame(test_content, bg=self.COLORS['bg_secondary'])
        test_frame.pack(fill='x', pady=5)
        
        self.test_button = ttk.Button(test_frame, text="🧪 Test API Connection", command=self._test_api_connection)
        self.test_button.pack(side='left')
        
        # Test result
        self.test_result_label = tk.Label(test_content, textvariable=self.test_result_var,
                                         bg=self.COLORS['bg_secondary'], fg=self.COLORS['text_muted'],
                                         font=('Segoe UI', 9), wraplength=500, justify='left')
        self.test_result_label.pack(anchor='w', pady=(5, 0))
        
        return tab
    
    def _build_advanced_tab(self) -> tk.Frame:
        """Build the Advanced settings tab."""
        tab = tk.Frame(self.notebook, bg=self.COLORS['bg_primary'])
        
        # Scrollable frame
        canvas = tk.Canvas(tab, bg=self.COLORS['bg_primary'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.COLORS['bg_primary'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Configuration Presets Section
        preset_section, preset_content = self._create_section_frame(scrollable_frame, "🎛️ Configuration Presets")
        preset_section.pack(fill='x', pady=(0, 10))
        
        # Preset selection and apply
        preset_frame = tk.Frame(preset_content, bg=self.COLORS['bg_secondary'])
        preset_frame.pack(fill='x', pady=5)
        
        tk.Label(preset_frame, text="Load Preset:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        self.preset_var = tk.StringVar()
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var,
                                   values=SettingsPresets.get_available_presets(), state='readonly', width=15)
        preset_combo.pack(side='left', padx=(10, 10))
        
        ttk.Button(preset_frame, text="Apply Preset", command=self._apply_preset).pack(side='left')
        
        # Import/Export Section
        io_section, io_content = self._create_section_frame(scrollable_frame, "📁 Import/Export")
        io_section.pack(fill='x', pady=(0, 10))
        
        # Import/Export buttons
        io_frame = tk.Frame(io_content, bg=self.COLORS['bg_secondary'])
        io_frame.pack(fill='x', pady=5)
        
        ttk.Button(io_frame, text="📥 Import Settings", command=self._import_settings).pack(side='left', padx=(0, 10))
        ttk.Button(io_frame, text="📤 Export Settings", command=self._export_settings).pack(side='left', padx=(0, 10))
        ttk.Button(io_frame, text="🔄 Reset to Defaults", command=self._reset_to_defaults).pack(side='left')
        
        # Settings Search Section - REMOVED as per requirements
        
        # Validation Status Section - REMOVED as per requirements
        # Backup and Recovery Section - REMOVED as per requirements
        
        # Memory and Performance Section
        memory_section, memory_content = self._create_section_frame(scrollable_frame, "⚡ Memory & Performance")
        memory_section.pack(fill='x')
        
        # Memory limits
        memory_frame = tk.Frame(memory_content, bg=self.COLORS['bg_secondary'])
        memory_frame.pack(fill='x', pady=5)
        
        tk.Label(memory_frame, text="Max Memory Usage (MB):", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        memory_spin = tk.Spinbox(memory_frame, from_=512, to=16384, increment=256,
                               textvariable=self.memory_limit_var, bg=self.COLORS['bg_tertiary'],
                               fg=self.COLORS['text_primary'], width=8)
        memory_spin.pack(side='left', padx=(10, 0))
        
        # Performance tuning options
        perf_frame = tk.Frame(memory_content, bg=self.COLORS['bg_secondary'])
        perf_frame.pack(fill='x', pady=5)
        
        tk.Label(perf_frame, text="Performance Mode:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')
        
        perf_combo = ttk.Combobox(perf_frame, textvariable=self.performance_mode_var,
                                 values=SettingsValidator.VALID_PERFORMANCE_MODES, state='readonly', width=15)
        perf_combo.pack(side='left', padx=(10, 0))
        
        # Logging configuration 
        log_frame = tk.Frame(memory_content, bg=self.COLORS['bg_secondary'])
        log_frame.pack(fill='x', pady=5)
        
        self._create_checkbox(log_frame, "Enable detailed logging",
                             self.enable_logging_var).pack(side='left')
        
        tk.Label(log_frame, text="Log Level:", bg=self.COLORS['bg_secondary'], 
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left', padx=(20, 5))
        
        log_combo = ttk.Combobox(log_frame, textvariable=self.log_level_var,
                                values=SettingsValidator.VALID_LOG_LEVELS, state='readonly', width=10)
        log_combo.pack(side='left')

        # Window Management Section
        window_section, window_content = self._create_section_frame(scrollable_frame, "🖼️ Window Management")
        window_section.pack(fill='x', pady=(10, 0))

        # Window state options
        window_options_frame = tk.Frame(window_content, bg=self.COLORS['bg_secondary'])
        window_options_frame.pack(fill='x', pady=5)

        self._create_checkbox(window_options_frame, "Remember window state and position",
                             self.remember_window_state_var).pack(anchor='w', pady=2)

        self._create_checkbox(window_options_frame, "Start in fullscreen mode",
                             self.startup_fullscreen_var).pack(anchor='w', pady=2)

        # Auto-save settings
        auto_save_frame = tk.Frame(window_content, bg=self.COLORS['bg_secondary'])
        auto_save_frame.pack(fill='x', pady=5)

        tk.Label(auto_save_frame, text="Auto-save interval (minutes):", bg=self.COLORS['bg_secondary'],
                fg=self.COLORS['text_primary'], font=('Segoe UI', 9)).pack(side='left')

        autosave_spin = tk.Spinbox(auto_save_frame, from_=1, to=60, increment=1,
                                 textvariable=self.auto_save_interval_var, bg=self.COLORS['bg_tertiary'],
                                 fg=self.COLORS['text_primary'], width=8)
        autosave_spin.pack(side='left', padx=(10, 0))

        return tab
    
    def _build_buttons(self, parent):
        """Build the dialog action buttons."""
        button_frame = tk.Frame(parent, bg=self.COLORS['bg_primary'])
        button_frame.pack(fill='x', pady=(10, 0))
        
        # Left side - Validation status
        left_frame = tk.Frame(button_frame, bg=self.COLORS['bg_primary'])
        left_frame.pack(side='left')
        
        self.validation_status_label = tk.Label(
            left_frame,
            text="Ready",
            bg=self.COLORS['bg_primary'],
            fg=self.COLORS['text_muted'],
            font=('Segoe UI', 9)
        )
        self.validation_status_label.pack()
        
        # Right side - Action buttons
        right_frame = tk.Frame(button_frame, bg=self.COLORS['bg_primary'])
        right_frame.pack(side='right')
        
        # Cancel button
        cancel_button = ttk.Button(
            right_frame,
            text="Cancel",
            command=self._on_cancel
        )
        cancel_button.pack(side='right', padx=(10, 0))
        
        # Apply button
        self.apply_button = ttk.Button(
            right_frame,
            text="Apply",
            command=self._on_apply,
            state='disabled'
        )
        self.apply_button.pack(side='right', padx=(10, 0))
        
        # OK button
        ok_button = ttk.Button(
            right_frame,
            text="OK",
            command=self._on_ok
        )
        ok_button.pack(side='right')
        
        # Configure button styles
        style = ttk.Style()
        style.configure('Action.TButton', padding=(15, 8))
    
    # Event handlers and utility methods
    
    def _toggle_api_key_visibility(self):
        """Toggle API key visibility in the entry field."""
        if self.api_key_entry.cget('show') == '*':
            self.api_key_entry.configure(show='')
            self.show_key_button.configure(text="🙈")
        else:
            self.api_key_entry.configure(show='*')
            self.show_key_button.configure(text="👁")
    
    def _detect_cameras(self):
        """Detect available cameras and update the camera list."""
        def detect_worker():
            cameras = []
            try:
                # Use WebcamService to detect cameras with descriptive names
                detected_devices = WebcamService.list_devices()
                
                for idx, name in detected_devices:
                    # Test if camera actually works
                    test_cap = cv2.VideoCapture(idx)
                    if test_cap.isOpened():
                        ret, _ = test_cap.read()
                        if ret:
                            # Clean up camera name for display
                            if "USB" in name.upper():
                                clean_name = f"USB Camera ({name[:30]}...)" if len(name) > 30 else f"USB Camera ({name})"
                            elif "INTEGRATED" in name.upper() or "BUILT" in name.upper():
                                clean_name = "Built-in Camera"
                            else:
                                clean_name = name[:40] + "..." if len(name) > 40 else name
                            
                            cameras.append((idx, f"Camera {idx+1}: {clean_name}"))
                    test_cap.release()
                    
            except Exception as e:
                print(f"Camera detection error: {e}")
            
            # Update UI on main thread
            self.dialog.after(0, lambda: self._update_camera_list(cameras))
        
        # Show detection in progress
        self.camera_name_label.configure(text="Detecting cameras...", fg=self.COLORS['warning'])
        
        # Run detection in background thread
        threading.Thread(target=detect_worker, daemon=True).start()
    
    def _update_camera_list(self, cameras: list):
        """Update the camera list display."""
        if cameras:
            # Update camera spinbox max value based on detected cameras
            if cameras:
                max_index = max(cam[0] for cam in cameras)
                # Update the camera index spinbox range
                
            camera_names = [cam[1] for cam in cameras]
            camera_text = f"Found: {', '.join(camera_names[:3])}"
            if len(cameras) > 3:
                camera_text += f" and {len(cameras) - 3} more"
            self.camera_name_label.configure(text=camera_text, fg=self.COLORS['success'])
        else:
            self.camera_name_label.configure(text="No cameras detected", fg=self.COLORS['error'])
    
    def _on_resolution_preset_changed(self, event=None):
        """Handle resolution preset selection."""
        preset = event.widget.get()
        
        if preset == "720p (1280x720)":
            self.camera_width_var.set(1280)
            self.camera_height_var.set(720)
        elif preset == "1080p (1920x1080)":
            self.camera_width_var.set(1920)
            self.camera_height_var.set(1080)
        elif preset == "4K (3840x2160)":
            self.camera_width_var.set(3840)
            self.camera_height_var.set(2160)
        # "Custom" doesn't change values
    
    def _test_api_connection(self):
        """Test the Gemini API connection."""
        api_key = self.api_key_var.get().strip()
        
        if not api_key:
            self.test_result_var.set("❌ Please enter an API key first")
            self.test_result_label.configure(fg=self.COLORS['error'])
            return
        
        # Disable test button during test
        self.test_button.configure(state='disabled', text="⏳ Testing...")
        self.test_result_var.set("Testing connection...")
        self.test_result_label.configure(fg=self.COLORS['text_muted'])
        
        def test_worker():
            try:
                # Create temporary service for testing
                test_service = GeminiService(api_key)
                
                # Create a simple test image
                import numpy as np
                test_image = np.zeros((100, 100, 3), dtype=np.uint8)
                
                # Test API call
                result = test_service.analyze_single_image(
                    test_image, 
                    "This is a test. Please respond with 'Test successful' if you can see this."
                )
                
                # Update UI on main thread
                self.dialog.after(0, lambda: self._handle_test_result(None, None))
                
            except Exception as e:
                # Update UI on main thread
                self.dialog.after(0, lambda: self._handle_test_result(None, str(e)))
        
        # Run test in background
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
    
    def _apply_preset(self):
        """Apply the selected configuration preset."""
        preset_name = self.preset_var.get()
        if not preset_name:
            messagebox.showwarning("Warning", "Please select a preset to apply")
            return
        
        preset_settings = SettingsPresets.get_preset(preset_name)
        if not preset_settings:
            messagebox.showerror("Error", f"Preset '{preset_name}' not found")
            return
        
        # Apply preset settings to variables
        for key, value in preset_settings.items():
            var_name = f"{key}_var"
            if hasattr(self, var_name):
                var = getattr(self, var_name)
                var.set(value)
        
        self._mark_changed()
        messagebox.showinfo("Preset Applied", f"Applied '{preset_name}' preset settings")
    
    def _import_settings(self):
        """Import settings from a JSON file with validation and backup."""
        file_path = filedialog.askopenfilename(
            title="Import Settings",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    import_data = json.load(f)
                
                # Handle both old format (direct settings) and new format (with metadata)
                if isinstance(import_data, dict) and "settings" in import_data:
                    # New format with metadata
                    imported_settings = import_data["settings"]
                    metadata = import_data.get("metadata", {})
                    
                    # Show import info
                    import_info = f"Import from: {os.path.basename(file_path)}\n"
                    if "export_timestamp" in metadata:
                        import_info += f"Exported: {metadata['export_timestamp']}\n"
                    if "settings_count" in metadata:
                        import_info += f"Settings count: {metadata['settings_count']}"
                else:
                    # Old format (direct settings)
                    imported_settings = import_data
                    import_info = f"Import from: {os.path.basename(file_path)}\nLegacy format detected"
                
                # Create backup before importing
                if messagebox.askyesno("Import Settings", 
                                     f"{import_info}\n\nThis will replace current settings. Continue?"):
                    
                    # Auto-backup current settings
                    try:
                        self._create_auto_backup("before_import")
                    except Exception as backup_error:
                        if not messagebox.askyesno("Backup Failed", 
                                                  f"Failed to create backup: {backup_error}\nContinue import anyway?"):
                            return
                    
                    # Validate and apply imported settings
                    applied_count = 0
                    errors = []
                    
                    for key, value in imported_settings.items():
                        var_name = f"{key}_var"
                        if hasattr(self, var_name):
                            try:
                                var = getattr(self, var_name)
                                var.set(value)
                                applied_count += 1
                            except Exception as e:
                                errors.append(f"{key}: {e}")
                        else:
                            errors.append(f"{key}: Setting not found")
                    
                    self._mark_changed()
                    
                    # Show results
                    result_msg = f"Import completed\nApplied: {applied_count} settings"
                    if errors:
                        result_msg += f"\nErrors: {len(errors)}"
                        if len(errors) <= 5:
                            result_msg += "\n" + "\n".join(errors)
                        else:
                            result_msg += f"\nFirst 5 errors:\n" + "\n".join(errors[:5])
                    
                    if errors:
                        messagebox.showwarning("Import Completed with Errors", result_msg)
                    else:
                        messagebox.showinfo("Import Successful", result_msg)
                
            except Exception as e:
                messagebox.showerror("Import Error", f"Failed to import settings: {e}")
    
    def _export_settings(self):
        """Export current settings to a JSON file with metadata."""
        file_path = filedialog.asksaveasfilename(
            title="Export Settings",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfilename=f"settings_export_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if file_path:
            try:
                # Collect all current settings
                settings = self._collect_current_settings()
                
                # Add export metadata
                export_data = {
                    "metadata": {
                        "export_version": "1.0",
                        "export_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "application": "Vision Analysis System",
                        "settings_count": len(settings)
                    },
                    "settings": settings
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                messagebox.showinfo("Export Successful", 
                                   f"Settings exported to {os.path.basename(file_path)}\n"
                                   f"Contains {len(settings)} settings")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export settings: {e}")
    
    def _reset_to_defaults(self):
        """Reset all settings to their default values."""
        if messagebox.askyesno("Reset Settings", "Are you sure you want to reset all settings to defaults?"):
            # Load defaults
            from ...config.defaults import DEFAULT_CONFIG
            
            # Apply defaults to variables
            for key, value in DEFAULT_CONFIG.items():
                var_name = f"{key}_var"
                if hasattr(self, var_name):
                    var = getattr(self, var_name)
                    var.set(value)
            
            self._mark_changed()
            messagebox.showinfo("Reset Complete", "All settings have been reset to defaults")
    
    def _on_search_changed(self, event=None):
        """Handle search text changes."""
        search_term = self.search_var.get().lower()
        
        if not search_term:
            # Show all tabs
            for i in range(self.notebook.index("end")):
                self.notebook.tab(i, state="normal")
        else:
            # Hide tabs that don't match search
            tab_names = ["general", "webcam", "analysis", "chatbot", "advanced"]
            for i, tab_name in enumerate(tab_names):
                if search_term in tab_name.lower():
                    self.notebook.tab(i, state="normal")
                else:
                    self.notebook.tab(i, state="disabled")
    
    def _clear_search(self):
        """Clear the search field and show all tabs."""
        self.search_var.set("")
        for i in range(self.notebook.index("end")):
            self.notebook.tab(i, state="normal")
    
    def _validate_all_settings(self):
        """Validate all current settings and display results."""
        settings = self._collect_current_settings()
        validation_results = SettingsValidator.validate_all_settings(settings)
        
        # Display validation results
        self.validation_text.configure(state='normal')
        self.validation_text.delete(1.0, tk.END)
        
        valid_count = 0
        error_count = 0
        warning_count = 0
        
        for key, result in validation_results.items():
            if result.is_valid:
                if result.corrected_value is not None:
                    self.validation_text.insert(tk.END, f"⚠️  {key}: Corrected to {result.corrected_value}\n")
                    warning_count += 1
                else:
                    valid_count += 1
            else:
                self.validation_text.insert(tk.END, f"❌ {key}: {result.error_message}\n")
                error_count += 1
        
        # Summary
        summary = f"\nValidation Summary:\n✅ Valid: {valid_count}\n⚠️  Corrected: {warning_count}\n❌ Errors: {error_count}"
        self.validation_text.insert(tk.END, summary)
        
        self.validation_text.configure(state='disabled')
        
        # Update status
        if error_count > 0:
            self.validation_status_label.configure(text=f"{error_count} validation errors", fg=self.COLORS['error'])
        elif warning_count > 0:
            self.validation_status_label.configure(text=f"{warning_count} values corrected", fg=self.COLORS['warning'])
        else:
            self.validation_status_label.configure(text="All settings valid", fg=self.COLORS['success'])
    
    def _create_backup(self):
        """Create a manual backup of current settings."""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_filename = f"manual_backup_{timestamp}.json"
            
            # Create backups directory if it doesn't exist
            backup_dir = "backups"
            os.makedirs(backup_dir, exist_ok=True)
            
            backup_path = os.path.join(backup_dir, backup_filename)
            
            # Create backup with metadata
            self._save_backup_file(backup_path, "manual")
            
            messagebox.showinfo("Backup Created", f"Settings backed up to {backup_filename}")
            
        except Exception as e:
            messagebox.showerror("Backup Error", f"Failed to create backup: {e}")
    
    def _create_auto_backup(self, reason: str = "auto") -> str:
        """Create an automatic backup and return the file path."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{reason}_backup_{timestamp}.json"
        
        # Create backups directory if it doesn't exist
        backup_dir = "backups"
        os.makedirs(backup_dir, exist_ok=True)
        
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Create backup with metadata
        self._save_backup_file(backup_path, reason)
        
        return backup_path
    
    def _save_backup_file(self, file_path: str, reason: str):
        """Save a backup file with metadata."""
        settings = self._collect_current_settings()
        
        backup_data = {
            "metadata": {
                "backup_version": "1.0",
                "backup_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "backup_reason": reason,
                "application": "Vision Analysis System",
                "settings_count": len(settings)
            },
            "settings": settings
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
    
    def _restore_backup(self):
        """Restore settings from a backup file with validation."""
        backup_dir = "backups"
        if not os.path.exists(backup_dir):
            messagebox.showwarning("No Backups", "No backup directory found")
            return
        
        file_path = filedialog.askopenfilename(
            title="Restore Backup",
            initialdir=backup_dir,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)
                
                # Handle both old and new backup formats
                if isinstance(backup_data, dict) and "settings" in backup_data:
                    # New format with metadata
                    backup_settings = backup_data["settings"]
                    metadata = backup_data.get("metadata", {})
                    
                    # Show backup info
                    backup_info = f"Restore from: {os.path.basename(file_path)}\n"
                    if "backup_timestamp" in metadata:
                        backup_info += f"Created: {metadata['backup_timestamp']}\n"
                    if "backup_reason" in metadata:
                        backup_info += f"Reason: {metadata['backup_reason']}\n"
                    if "settings_count" in metadata:
                        backup_info += f"Settings: {metadata['settings_count']}"
                else:
                    # Old format (direct settings)
                    backup_settings = backup_data
                    backup_info = f"Restore from: {os.path.basename(file_path)}\nLegacy backup format"
                
                # Confirm restore
                if messagebox.askyesno("Restore Backup", 
                                     f"{backup_info}\n\nThis will replace all current settings. Continue?"):
                    
                    # Create safety backup before restoring
                    try:
                        safety_backup = self._create_auto_backup("before_restore")
                        print(f"Safety backup created: {safety_backup}")
                    except Exception as backup_error:
                        if not messagebox.askyesno("Safety Backup Failed", 
                                                  f"Failed to create safety backup: {backup_error}\nContinue restore anyway?"):
                            return
                    
                    # Apply backup settings
                    applied_count = 0
                    errors = []
                    
                    for key, value in backup_settings.items():
                        var_name = f"{key}_var"
                        if hasattr(self, var_name):
                            try:
                                var = getattr(self, var_name)
                                var.set(value)
                                applied_count += 1
                            except Exception as e:
                                errors.append(f"{key}: {e}")
                        else:
                            errors.append(f"{key}: Setting not found")
                    
                    self._mark_changed()
                    
                    # Show results
                    result_msg = f"Restore completed\nApplied: {applied_count} settings"
                    if errors:
                        result_msg += f"\nErrors: {len(errors)}"
                        if len(errors) <= 3:
                            result_msg += "\n" + "\n".join(errors)
                    
                    if errors:
                        messagebox.showwarning("Restore Completed with Errors", result_msg)
                    else:
                        messagebox.showinfo("Restore Complete", result_msg)
                
            except Exception as e:
                messagebox.showerror("Restore Error", f"Failed to restore backup: {e}")
    
    def _manage_backups(self):
        """Open backup management dialog."""
        # This would open a separate dialog to manage backup files
        # For now, just show info about backups directory
        backup_dir = "backups"
        if os.path.exists(backup_dir):
            backups = [f for f in os.listdir(backup_dir) if f.endswith('.json')]
            if backups:
                backup_list = "\n".join(backups)
                messagebox.showinfo("Available Backups", f"Backup files in {backup_dir}:\n\n{backup_list}")
            else:
                messagebox.showinfo("No Backups", f"No backup files found in {backup_dir}")
        else:
            messagebox.showinfo("No Backup Directory", "No backups have been created yet")
    
    def _collect_current_settings(self) -> Dict[str, Any]:
        """Collect all current settings from the UI variables.
        
        Returns:
            Dict[str, Any]: Complete dictionary of all settings from UI components.
            
        Raises:
            Exception: If any variable access fails or settings are invalid.
        """
        try:
            settings = {}
            
            # General settings
            settings['app_theme'] = self.theme_var.get()
            settings['language'] = self.language_var.get()
            settings['auto_save_config'] = self.auto_save_var.get()
            settings['debug'] = self.debug_var.get()
            settings['startup_fullscreen'] = self.startup_fullscreen_var.get()
            settings['remember_window_state'] = self.remember_window_state_var.get()
            settings['performance_mode'] = self.performance_mode_var.get()
            settings['max_memory_usage_mb'] = self.memory_limit_var.get()
            settings['enable_logging'] = self.enable_logging_var.get()
            settings['log_level'] = self.log_level_var.get()
            settings['auto_save_interval_minutes'] = self.auto_save_interval_var.get()
            settings['default_data_dir'] = self.default_data_dir_var.get()
            settings['default_models_dir'] = self.default_models_dir_var.get()
            settings['default_results_dir'] = self.default_results_dir_var.get()
            
            # Webcam settings
            settings['last_webcam_index'] = self.camera_index_var.get()
            settings['camera_width'] = self.camera_width_var.get()
            settings['camera_height'] = self.camera_height_var.get()
            settings['camera_fps'] = self.camera_fps_var.get()
            settings['camera_auto_exposure'] = self.auto_exposure_var.get()
            settings['camera_auto_focus'] = self.auto_focus_var.get()
            settings['camera_brightness'] = self.brightness_var.get()
            settings['camera_contrast'] = self.contrast_var.get()
            settings['camera_saturation'] = self.saturation_var.get()
            settings['camera_recording_format'] = self.recording_format_var.get()
            settings['camera_buffer_size'] = self.buffer_size_var.get()
            settings['camera_preview_enabled'] = self.preview_enabled_var.get()
            settings['camera_device_name'] = self.camera_device_name_var.get()
            
            # Image Analysis settings
            settings['detection_confidence_threshold'] = self.confidence_threshold_var.get()
            settings['detection_iou_threshold'] = self.iou_threshold_var.get()
            settings['enable_roi'] = self.enable_roi_var.get()
            settings['roi_x'] = self.roi_x_var.get()
            settings['roi_y'] = self.roi_y_var.get()
            settings['roi_width'] = self.roi_width_var.get()
            settings['roi_height'] = self.roi_height_var.get()
            settings['preferred_model'] = self.preferred_model_var.get()
            settings['export_quality'] = self.export_quality_var.get()
            settings['difference_sensitivity'] = self.difference_sensitivity_var.get()
            settings['highlight_differences'] = self.highlight_differences_var.get()
            settings['reference_image_path'] = self.reference_image_path_var.get()
            settings['analysis_history_days'] = self.analysis_history_days_var.get()
            settings['export_include_metadata'] = self.export_metadata_var.get()
            
            # Chatbot settings
            settings['gemini_api_key'] = self.api_key_var.get()
            settings['gemini_model'] = self.gemini_model_var.get()
            settings['gemini_timeout'] = self.timeout_var.get()
            settings['gemini_temperature'] = self.temperature_var.get()
            settings['gemini_max_tokens'] = self.max_tokens_var.get()
            settings['enable_ai_analysis'] = self.enable_ai_var.get()
            settings['chat_history_limit'] = self.chat_history_limit_var.get()
            settings['chat_auto_save'] = self.chat_auto_save_var.get()
            settings['response_format'] = self.response_format_var.get()
            settings['enable_rate_limiting'] = self.enable_rate_limiting_var.get()
            settings['requests_per_minute'] = self.requests_per_minute_var.get()
            settings['context_window_size'] = self.context_window_var.get()
            settings['enable_conversation_memory'] = self.conversation_memory_var.get()
            settings['chat_export_format'] = self.chat_export_format_var.get()
            settings['chatbot_persona'] = self.chatbot_persona_var.get()
            
            # Validate collected settings before returning
            self._validate_collected_settings(settings)
            
            logger.info(f"Successfully collected {len(settings)} settings from UI")
            return settings
            
        except Exception as e:
            logger.error(f"Failed to collect settings from UI: {e}")
            raise Exception(f"Settings collection failed: {e}") from e
    
    def _mark_changed(self):
        """Mark that changes have been made and enable Apply button."""
        self._changes_made = True
        self._mark_as_changed()  # Use the new comprehensive change tracking
    
    def _validate_collected_settings(self, settings: Dict[str, Any]) -> None:
        """Validate collected settings for basic consistency and completeness.
        
        Args:
            settings: Dictionary of collected settings to validate.
            
        Raises:
            ValueError: If settings are invalid or incomplete.
        """
        required_keys = [
            'app_theme', 'language', 'auto_save_config', 'debug',
            'last_webcam_index', 'camera_width', 'camera_height', 'camera_fps',
            'detection_confidence_threshold', 'detection_iou_threshold',
            'gemini_api_key', 'gemini_model'
        ]
        
        missing_keys = [key for key in required_keys if key not in settings]
        if missing_keys:
            raise ValueError(f"Missing required settings: {missing_keys}")
        
        # Validate value ranges
        if not (0 <= settings.get('detection_confidence_threshold', 0) <= 1):
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        if not (0 <= settings.get('detection_iou_threshold', 0) <= 1):
            raise ValueError("IoU threshold must be between 0 and 1")
        
        if settings.get('camera_width', 0) <= 0 or settings.get('camera_height', 0) <= 0:
            raise ValueError("Camera dimensions must be positive")
    
    def _create_settings_backup(self) -> str:
        """Create a backup of current settings before applying changes.
        
        Returns:
            str: Path to the backup file created.
            
        Raises:
            OSError: If backup creation fails.
        """
        try:
            backup_dir = os.path.join(os.getcwd(), 'data', 'backups')
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_filename = f"config_backup_{timestamp}.json"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            # Use the config object's to_dict method for consistency
            backup_data = self.config.to_dict() if hasattr(self.config, 'to_dict') else asdict(self.config)
            
            # Include extra settings
            if hasattr(self.config, 'extra') and self.config.extra:
                backup_data.update(self.config.extra)
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Settings backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to create settings backup: {e}")
            raise OSError(f"Backup creation failed: {e}") from e
    
    def _apply_settings_immediately(self, settings: Dict[str, Any]) -> None:
        """Apply settings to running application immediately without requiring restart.
        
        Args:
            settings: Dictionary of settings to apply
        """
        try:
            # Apply theme changes immediately if theme changed
            if 'app_theme' in settings:
                new_theme = settings['app_theme']
                original_theme = self.original_values.get('app_theme', 'Dark')
                
                if new_theme != original_theme:
                    logger.info(f"Applying theme change immediately: {original_theme} -> {new_theme}")
                    self._update_dialog_theme(new_theme)
                    
                    # Apply theme to main window if available
                    main_window = self.services.get('main_window')
                    if main_window and hasattr(main_window, 'apply_theme'):
                        main_window.apply_theme(new_theme)
                        logger.info(f"Applied theme {new_theme} to main window")
            
            # Apply webcam settings immediately if they changed
            webcam_settings = ['last_webcam_index', 'camera_width', 'camera_height', 
                              'camera_fps', 'camera_brightness', 'camera_contrast', 
                              'camera_saturation', 'camera_auto_exposure', 'camera_auto_focus']
            
            if any(setting in settings for setting in webcam_settings):
                webcam_service = self.services.get('webcam')
                if webcam_service:
                    logger.info("Applying webcam settings immediately")
                    self._apply_webcam_changes_immediately(webcam_service, settings)
            
            # Apply Gemini settings immediately if they changed  
            gemini_settings = ['gemini_api_key', 'gemini_model', 'gemini_temperature',
                             'gemini_max_tokens', 'gemini_timeout', 'chatbot_persona',
                             'enable_rate_limiting', 'requests_per_minute']
                             
            if any(setting in settings for setting in gemini_settings):
                gemini_service = self.services.get('gemini')
                if gemini_service:
                    logger.info("Applying Gemini settings immediately")
                    self._apply_gemini_changes_immediately(gemini_service, settings)
            
            # Apply analysis settings immediately
            analysis_settings = ['detection_confidence_threshold', 'detection_iou_threshold',
                                'enable_roi', 'roi_x', 'roi_y', 'roi_width', 'roi_height']
                                
            if any(setting in settings for setting in analysis_settings):
                detection_service = self.services.get('detection')
                if detection_service:
                    logger.info("Applying analysis settings immediately")
                    self._apply_analysis_changes_immediately(detection_service, settings)
                    
            logger.info("All applicable settings applied immediately to running services")
            
        except Exception as e:
            logger.error(f"Error applying settings immediately: {e}", exc_info=True)
            # Don't fail the entire operation, just log the error
    
    def _update_dialog_theme(self, theme: str) -> None:
        """Update dialog appearance based on theme immediately.
        
        Args:
            theme: Theme name ('Dark', 'Light', 'Auto')
        """
        try:
            if theme == "Light":
                self.COLORS.update({
                    'bg_primary': '#ffffff',
                    'bg_secondary': '#f5f5f5',
                    'bg_tertiary': '#eeeeee',
                    'text_primary': '#000000',
                    'text_secondary': '#333333',
                    'text_muted': '#666666',
                    'border': '#cccccc',
                })
            else:  # Dark theme (default)
                self.COLORS.update({
                    'bg_primary': '#1e1e1e',
                    'bg_secondary': '#2d2d2d',
                    'bg_tertiary': '#3c3c3c',
                    'text_primary': '#ffffff',
                    'text_secondary': '#cccccc',
                    'text_muted': '#999999',
                    'border': '#404040',
                })
            
            # Update dialog window colors
            self.dialog.configure(bg=self.COLORS['bg_primary'])
            
            # Update all widget colors (this is a simplified implementation)
            # In a full implementation, you would recursively update all widgets
            self._refresh_widget_colors()
            
            logger.info(f"Dialog theme updated to {theme}")
            
        except Exception as e:
            logger.error(f"Failed to update dialog theme: {e}")
    
    def _refresh_widget_colors(self) -> None:
        """Refresh all widget colors to match current theme."""
        # This is a simplified implementation
        # In practice, you would need to traverse the widget tree and update colors
        try:
            # Update persona text widget if it exists
            if hasattr(self, 'persona_text'):
                self.persona_text.configure(
                    bg=self.COLORS['bg_tertiary'],
                    fg=self.COLORS['text_primary'],
                    insertbackground=self.COLORS['text_primary'],
                    selectbackground=self.COLORS['accent_primary']
                )
                
            logger.debug("Widget colors refreshed")
            
        except Exception as e:
            logger.error(f"Error refreshing widget colors: {e}")
    
    def _apply_webcam_changes_immediately(self, webcam_service, settings: Dict[str, Any]) -> None:
        """Apply webcam settings to service immediately.
        
        Args:
            webcam_service: Webcam service instance
            settings: Settings dictionary
        """
        try:
            # Update camera device if changed
            if 'last_webcam_index' in settings:
                if hasattr(webcam_service, 'set_camera'):
                    webcam_service.set_camera(settings['last_webcam_index'])
            
            # Update resolution if changed
            if 'camera_width' in settings and 'camera_height' in settings:
                if hasattr(webcam_service, 'set_resolution'):
                    webcam_service.set_resolution(
                        settings['camera_width'], 
                        settings['camera_height']
                    )
            
            # Update FPS if changed
            if 'camera_fps' in settings:
                if hasattr(webcam_service, 'set_fps'):
                    webcam_service.set_fps(settings['camera_fps'])
            
            # Update camera controls if available
            if 'camera_brightness' in settings and hasattr(webcam_service, 'set_brightness'):
                webcam_service.set_brightness(settings['camera_brightness'])
                
            if 'camera_contrast' in settings and hasattr(webcam_service, 'set_contrast'):
                webcam_service.set_contrast(settings['camera_contrast'])
                
            if 'camera_saturation' in settings and hasattr(webcam_service, 'set_saturation'):
                webcam_service.set_saturation(settings['camera_saturation'])
            
            logger.debug("Webcam settings applied immediately")
            
        except Exception as e:
            logger.error(f"Error applying webcam settings immediately: {e}")
    
    def _apply_gemini_changes_immediately(self, gemini_service, settings: Dict[str, Any]) -> None:
        """Apply Gemini settings to service immediately.
        
        Args:
            gemini_service: Gemini service instance
            settings: Settings dictionary
        """
        try:
            # Build update configuration
            update_config = {}
            
            if 'gemini_api_key' in settings:
                update_config['api_key'] = settings['gemini_api_key']
            if 'gemini_model' in settings:
                update_config['model'] = settings['gemini_model']
            if 'gemini_temperature' in settings:
                update_config['temperature'] = settings['gemini_temperature']
            if 'gemini_max_tokens' in settings:
                update_config['max_tokens'] = settings['gemini_max_tokens']
            if 'gemini_timeout' in settings:
                update_config['timeout'] = settings['gemini_timeout']
            if 'chatbot_persona' in settings:
                update_config['persona'] = settings['chatbot_persona']
            
            # Apply configuration updates
            if update_config and hasattr(gemini_service, 'update_configuration'):
                gemini_service.update_configuration(**update_config)
            
            # Apply rate limiting settings
            if ('enable_rate_limiting' in settings and 'requests_per_minute' in settings 
                and hasattr(gemini_service, 'update_configuration')):
                gemini_service.update_configuration(
                    enable_rate_limiting=settings['enable_rate_limiting'],
                    requests_per_minute=settings['requests_per_minute']
                )
            
            logger.debug("Gemini settings applied immediately")
            
        except Exception as e:
            logger.error(f"Error applying Gemini settings immediately: {e}")
    
    def _apply_analysis_changes_immediately(self, detection_service, settings: Dict[str, Any]) -> None:
        """Apply analysis settings to service immediately.
        
        Args:
            detection_service: Detection service instance
            settings: Settings dictionary
        """
        try:
            # Update detection thresholds
            if 'detection_confidence_threshold' in settings and hasattr(detection_service, 'set_confidence_threshold'):
                detection_service.set_confidence_threshold(settings['detection_confidence_threshold'])
                
            if 'detection_iou_threshold' in settings and hasattr(detection_service, 'set_iou_threshold'):
                detection_service.set_iou_threshold(settings['detection_iou_threshold'])
            
            # Update ROI settings
            if hasattr(detection_service, 'set_roi') and hasattr(detection_service, 'clear_roi'):
                if settings.get('enable_roi', False):
                    detection_service.set_roi(
                        settings.get('roi_x', 0),
                        settings.get('roi_y', 0),
                        settings.get('roi_width', 0),
                        settings.get('roi_height', 0)
                    )
                else:
                    detection_service.clear_roi()
            
            logger.debug("Analysis settings applied immediately")
            
        except Exception as e:
            logger.error(f"Error applying analysis settings immediately: {e}")

    def _apply_changes(self) -> bool:
        """Apply current settings with enhanced error handling and validation.
        
        Returns:
            bool: True if settings were successfully applied and saved, False otherwise.
        """
        backup_path = None
        
        try:
            logger.info("Starting settings application process")
            
            # Create backup before making changes
            try:
                backup_path = self._create_settings_backup()
            except OSError as e:
                logger.warning(f"Could not create backup: {e}")
                # Continue without backup if user chooses to
                if not messagebox.askyesno(
                    "Backup Warning", 
                    f"Could not create settings backup: {e}\n\nContinue without backup?"
                ):
                    return False
            
            # Collect all settings
            settings = self._collect_current_settings()
            
            # Validate settings using the enhanced validator
            validation_results = SettingsValidator.validate_all_settings(settings)
            
            # Check for validation errors
            errors = [result.error_message for result in validation_results.values() 
                     if not result.is_valid and result.error_message]
            
            if errors:
                # Show detailed validation errors with better UX
                self._show_validation_errors(errors, validation_results)
                self._show_status_message("Settings validation failed. Please fix errors above.", "error")
                return False
            
            # Apply validated settings to config object
            config_updated = False
            for key, value in settings.items():
                # Apply corrections if any
                if key in validation_results and validation_results[key].corrected_value is not None:
                    value = validation_results[key].corrected_value
                    logger.info(f"Applied correction for {key}: {settings[key]} -> {value}")
                
                # Update config object
                if hasattr(self.config, key):
                    old_value = getattr(self.config, key)
                    if old_value != value:
                        setattr(self.config, key, value)
                        config_updated = True
                        logger.debug(f"Updated config.{key}: {old_value} -> {value}")
                else:
                    # Store in extra for unknown keys
                    if key not in self.config.extra or self.config.extra[key] != value:
                        self.config.extra[key] = value
                        config_updated = True
                        logger.debug(f"Set extra config.{key}: {value}")
            
            # Force export quality to 100% for best image quality
            if hasattr(self.config, 'export_quality') and self.config.export_quality != 100:
                self.config.export_quality = 100
                logger.info("Forced export quality to 100% for optimal image quality")
            
            if not config_updated:
                logger.info("No configuration changes detected")
                self._show_status_message("No changes to apply", "info", duration=2000)
                return True
            
            # Apply settings to running application immediately (real-time application)
            logger.info("Applying settings to running application immediately")
            self._apply_settings_immediately(settings)
            
            # Save settings using the enhanced settings manager with atomic operations
            logger.info("Saving settings to file")
            success = self.settings_manager.save_settings(self.config)
            
            if not success:
                error_msg = "Failed to save settings to file. Settings may not persist after restart."
                logger.error(error_msg)
                messagebox.showerror("Save Error", error_msg)
                return False
            
            # Verify the save was successful by attempting to read back the config file
            config_path = "config.json"
            try:
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        saved_data = json.load(f)
                    logger.info(f"Settings successfully saved and verified ({len(saved_data)} entries)")
                else:
                    logger.warning(f"Config file {config_path} not found after save")
            except Exception as e:
                logger.warning(f"Could not verify saved settings: {e}")
            
            # Reset change tracking using the new comprehensive system
            self._changes_made = False
            self._reset_change_tracking()
            self.validation_status_label.configure(
                text=f"Settings applied successfully at {time.strftime('%H:%M:%S')}", 
                fg=self.COLORS['success']
            )
            
            # Clean up old backups (keep only last 10)
            try:
                if backup_path:
                    self._cleanup_old_backups(os.path.dirname(backup_path))
            except Exception as e:
                logger.warning(f"Could not clean up old backups: {e}")
            
            logger.info("Settings application completed successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to apply settings: {e}"
            logger.error(error_msg, exc_info=True)
            messagebox.showerror("Settings Error", error_msg)
            
            # If we have a backup, offer to restore
            if backup_path and os.path.exists(backup_path):
                if messagebox.askyesno(
                    "Restore Backup?",
                    "Settings application failed. Would you like to restore the backup?"
                ):
                    try:
                        self._restore_backup(backup_path)
                    except Exception as restore_error:
                        logger.error(f"Failed to restore backup: {restore_error}")
                        messagebox.showerror("Restore Failed", f"Could not restore backup: {restore_error}")
            
            return False
    
    def _cleanup_old_backups(self, backup_dir: str, keep_count: int = 10) -> None:
        """Clean up old backup files, keeping only the most recent ones.
        
        Args:
            backup_dir: Directory containing backup files.
            keep_count: Number of recent backups to keep.
        """
        try:
            if not os.path.exists(backup_dir):
                return
            
            backup_files = [
                os.path.join(backup_dir, f) for f in os.listdir(backup_dir)
                if f.startswith('config_backup_') and f.endswith('.json')
            ]
            
            if len(backup_files) <= keep_count:
                return
            
            # Sort by modification time (newest first)
            backup_files.sort(key=os.path.getmtime, reverse=True)
            
            # Remove old backups
            for old_backup in backup_files[keep_count:]:
                try:
                    os.remove(old_backup)
                    logger.debug(f"Removed old backup: {old_backup}")
                except OSError as e:
                    logger.warning(f"Could not remove old backup {old_backup}: {e}")
                    
        except Exception as e:
            logger.warning(f"Error during backup cleanup: {e}")
    
    def _restore_backup(self, backup_path: str) -> None:
        """Restore settings from a backup file.
        
        Args:
            backup_path: Path to the backup file to restore.
            
        Raises:
            Exception: If backup restoration fails.
        """
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_settings = json.load(f)
            
            # Apply backup settings to config
            for key, value in backup_settings.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    self.config.extra[key] = value
            
            # Save restored settings
            success = self.settings_manager.save_settings(self.config)
            if not success:
                raise Exception("Failed to save restored settings")
            
            # Reload UI with restored settings
            self._load_current_settings()
            
            logger.info(f"Successfully restored settings from backup: {backup_path}")
            messagebox.showinfo("Backup Restored", "Settings have been restored from backup.")
            
        except Exception as e:
            logger.error(f"Failed to restore backup {backup_path}: {e}")
            raise Exception(f"Backup restoration failed: {e}") from e
    
    def _on_apply(self) -> None:
        """Handle Apply button click with enhanced user feedback and validation."""
        # Show applying status
        self.apply_button.configure(text="Applying...", state='disabled')
        self.dialog.update_idletasks()

        try:
            success = self._apply_changes()
            if success:
                # Provide immediate success feedback
                self._show_status_message("Settings applied successfully!", "success")

                # Reset change tracking after successful application
                self._changes_made = False
                self.apply_button.configure(state='disabled')

                # Update original values to current values for future change detection
                self._update_original_values()

                # Log successful application
                logger.info("Settings applied successfully through Apply button")
            else:
                # Error feedback is handled within _apply_changes
                logger.warning("Settings application failed through Apply button")
        except Exception as e:
            logger.error(f"Unexpected error in Apply button handler: {e}")
            self._show_status_message(f"Unexpected error: {str(e)}", "error")
        finally:
            # Always restore button state
            self.apply_button.configure(text="Apply")
            if self._changes_made:
                self.apply_button.configure(state='normal')
    
    def _on_ok(self) -> None:
        """Handle OK button click with comprehensive validation and state management."""
        # If no changes, close immediately
        if not self._changes_made:
            logger.info("Closing settings dialog - no changes made")
            self._close_dialog()
            return

        # Show processing status
        original_text = "OK"
        self._update_button_states(applying=True)

        try:
            # Apply changes with enhanced validation
            success = self._apply_changes()

            if success:
                # Show brief success message before closing
                self._show_status_message("Settings saved successfully!", "success", duration=1000)

                # Wait briefly for user to see success message
                self.dialog.after(1200, self._close_dialog)

                logger.info("Settings applied and dialog will close")
            else:
                # Don't close if apply failed - let user fix issues
                self._show_status_message("Please fix validation errors before closing", "warning")
                logger.warning("Settings application failed - dialog remains open")

        except Exception as e:
            logger.error(f"Unexpected error in OK button handler: {e}")
            self._show_status_message(f"Unexpected error: {str(e)}", "error")
        finally:
            # Restore button states if we're not closing
            if self._changes_made:  # Only restore if we didn't succeed
                self._update_button_states(applying=False)
    
    def _test_camera(self):
        """Test the selected camera with live preview."""
        if self._preview_running:
            messagebox.showinfo("Info", "Camera test is already running. Stop the current test first.")
            return
        
        try:
            camera_index = self.camera_index_var.get()
            width = self.camera_width_var.get()
            height = self.camera_height_var.get()
            fps = self.camera_fps_var.get()
            
            # Create test webcam service
            self._test_service = WebcamService()
            
            if self._test_service.open(camera_index, width, height, fps):
                self._preview_running = True
                
                # Update UI
                self.test_button.configure(state='disabled')
                self.stop_test_button.configure(state='normal')
                self.camera_info_var.set(f"Testing Camera {camera_index + 1} at {width}x{height}@{fps}fps")
                
                # Start preview in background thread
                self._preview_thread = threading.Thread(target=self._preview_worker, daemon=True)
                self._preview_thread.start()
                
            else:
                messagebox.showerror("Camera Error", f"Failed to open camera {camera_index + 1}")
                self._test_service = None
                
        except Exception as e:
            messagebox.showerror("Error", f"Camera test failed: {e}")
            self._test_service = None
    
    def _stop_test(self):
        """Stop the camera test."""
        self._preview_running = False
        
        # Wait for preview thread to finish
        if self._preview_thread and self._preview_thread.is_alive():
            self._preview_thread.join(timeout=1.0)
        
        # Clean up webcam service
        if self._test_service:
            self._test_service.close()
            self._test_service = None
        
        # Update UI
        self.test_button.configure(state='normal')
        self.stop_test_button.configure(state='disabled')
        self.camera_info_var.set("Camera test stopped")
        
        # Clear preview canvas
        if self._preview_canvas:
            self._preview_canvas.delete('all')
            self._preview_canvas.create_text(
                200, 120, 
                text="Camera Preview\nClick 'Test Camera' to start",
                fill='white', 
                font=('Segoe UI', 12),
                justify='center'
            )
    
    def _preview_worker(self):
        """Background worker for camera preview."""
        fps_counter = 0
        last_fps_time = time.time()
        
        while self._preview_running and self._test_service:
            try:
                ret, frame = self._test_service.read()
                if ret and frame is not None:
                    # Update preview on main thread
                    self.dialog.after(0, self._update_preview_display, frame)
                    
                    # Calculate FPS
                    fps_counter += 1
                    current_time = time.time()
                    if current_time - last_fps_time >= 1.0:
                        actual_fps = fps_counter / (current_time - last_fps_time)
                        fps_counter = 0
                        last_fps_time = current_time
                        
                        # Update FPS display on main thread
                        camera_index = self.camera_index_var.get()
                        width = self.camera_width_var.get()
                        height = self.camera_height_var.get()
                        info_text = f"Camera {camera_index + 1}: {width}x{height} @ {actual_fps:.1f}fps"
                        self.dialog.after(0, lambda: self.camera_info_var.set(info_text))
                
                time.sleep(1.0 / 30.0)  # Target 30 FPS
                
            except Exception as e:
                print(f"Preview worker error: {e}")
                break
    
    def _update_preview_display(self, frame):
        """Update the preview canvas with the latest frame."""
        try:
            if not self._preview_canvas:
                return
                
            # Get canvas dimensions
            canvas_width = self._preview_canvas.winfo_width()
            canvas_height = self._preview_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                return
            
            # Calculate scaling to fit canvas while maintaining aspect ratio
            frame_height, frame_width = frame.shape[:2]
            scale = min(canvas_width / frame_width, canvas_height / frame_height)
            
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
            
            if new_width > 0 and new_height > 0:
                # Resize frame
                resized_frame = cv2.resize(frame, (new_width, new_height))
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                
                # Create PhotoImage
                pil_image = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update canvas
                self._preview_canvas.delete('all')
                self._preview_canvas.create_image(
                    canvas_width // 2,
                    canvas_height // 2,
                    anchor='center',
                    image=photo
                )
                self._preview_canvas.image = photo  # Keep reference
                
                # Add resolution overlay
                self._preview_canvas.create_text(
                    10, 10, 
                    text=f"Resolution: {frame_width}x{frame_height}",
                    anchor='nw', 
                    fill='yellow', 
                    font=('Segoe UI', 9, 'bold')
                )
                
        except Exception as e:
            print(f"Preview display error: {e}")
    
    def _browse_directory(self, var: tk.StringVar, title: str):
        """Browse for a directory and update the variable."""
        directory = filedialog.askdirectory(title=title, initialdir=var.get() if var.get() else ".")
        if directory:
            var.set(directory)
            self._mark_changed()
    
    def _browse_file(self, var: tk.StringVar, title: str, filetypes: list):
        """Browse for a file and update the variable."""
        filename = filedialog.askopenfilename(title=title, filetypes=filetypes, initialdir=os.path.dirname(var.get()) if var.get() else ".")
        if filename:
            var.set(filename)
            self._mark_changed()
    
    def _on_cancel(self) -> None:
        """Handle Cancel button click with improved user experience."""
        # Stop any running camera test
        if self._preview_running:
            self._stop_test()

        # Handle unsaved changes with clear options
        if self._changes_made:
            # Create custom dialog for better UX
            result = self._show_unsaved_changes_dialog()

            if result == "save":
                # Try to save changes
                if self._apply_changes():
                    self._show_status_message("Settings saved!", "success", duration=800)
                    self.dialog.after(1000, self._close_dialog)
                else:
                    # Don't close if save failed
                    self._show_status_message("Please fix errors before closing", "warning")
                    return
            elif result == "discard":
                # User chose to discard changes
                logger.info("User discarded settings changes")
                self._close_dialog()
            else:  # result == "cancel" or dialog was closed
                # User wants to continue editing
                return
        else:
            # No changes, close immediately
            self._close_dialog()

    def _close_dialog(self) -> None:
        """Safely close the dialog with proper cleanup."""
        try:
            # Stop any running camera test
            if self._preview_running:
                self._stop_test()

            # Release modal grab and destroy
            self.dialog.grab_release()
            self.dialog.destroy()

            logger.info("Settings dialog closed successfully")
        except Exception as e:
            logger.error(f"Error closing settings dialog: {e}")

    def _update_button_states(self, applying: bool) -> None:
        """Update button states during apply operations."""
        if applying:
            # Disable all buttons during application
            if hasattr(self, 'apply_button'):
                self.apply_button.configure(text="Applying...", state='disabled')
            if hasattr(self, 'ok_button'):
                self.ok_button.configure(text="Applying...", state='disabled')
            if hasattr(self, 'cancel_button'):
                self.cancel_button.configure(state='disabled')
        else:
            # Restore normal button states
            if hasattr(self, 'apply_button'):
                self.apply_button.configure(text="Apply")
                self.apply_button.configure(state='normal' if self._changes_made else 'disabled')
            if hasattr(self, 'ok_button'):
                self.ok_button.configure(text="OK", state='normal')
            if hasattr(self, 'cancel_button'):
                self.cancel_button.configure(state='normal')

    def _show_status_message(self, message: str, message_type: str = "info", duration: int = 3000) -> None:
        """Show status message with appropriate styling and auto-hide."""
        try:
            # Create status frame if it doesn't exist
            if not hasattr(self, 'status_frame'):
                self._create_status_frame()

            # Update status message with color coding
            colors = {
                'success': self.COLORS['success'],
                'error': self.COLORS['error'],
                'warning': self.COLORS['warning'],
                'info': self.COLORS['accent_primary']
            }

            color = colors.get(message_type, self.COLORS['text_primary'])

            if hasattr(self, 'status_label'):
                self.status_label.configure(text=message, fg=color)
                self.status_frame.pack(fill='x', pady=(5, 0))

                # Auto-hide after duration
                if duration > 0:
                    self.dialog.after(duration, self._hide_status_message)

        except Exception as e:
            logger.error(f"Error showing status message: {e}")
            # Fallback to messagebox if status frame fails
            messagebox.showinfo("Status", message)

    def _create_status_frame(self) -> None:
        """Create status message frame if it doesn't exist."""
        try:
            # Find the main frame (should be the first child)
            main_frame = None
            for child in self.dialog.winfo_children():
                if isinstance(child, tk.Frame):
                    main_frame = child
                    break

            if main_frame:
                self.status_frame = tk.Frame(main_frame, bg=self.COLORS['bg_secondary'],
                                           relief='solid', bd=1)
                self.status_label = tk.Label(
                    self.status_frame,
                    text="",
                    bg=self.COLORS['bg_secondary'],
                    fg=self.COLORS['text_primary'],
                    font=('Segoe UI', 9),
                    pady=8
                )
                self.status_label.pack(fill='x')

        except Exception as e:
            logger.error(f"Error creating status frame: {e}")

    def _hide_status_message(self) -> None:
        """Hide the status message frame."""
        try:
            if hasattr(self, 'status_frame'):
                self.status_frame.pack_forget()
        except Exception as e:
            logger.error(f"Error hiding status message: {e}")

    def _show_unsaved_changes_dialog(self) -> str:
        """Show enhanced dialog for unsaved changes with clear options."""
        try:
            # Custom dialog with better UX than messagebox
            from tkinter import simpledialog

            dialog = tk.Toplevel(self.dialog)
            dialog.title("Unsaved Changes")
            dialog.geometry("400x150")
            dialog.configure(bg=self.COLORS['bg_primary'])
            dialog.transient(self.dialog)
            dialog.grab_set()

            # Center on parent
            dialog.update_idletasks()
            x = self.dialog.winfo_rootx() + (self.dialog.winfo_width() // 2) - (dialog.winfo_width() // 2)
            y = self.dialog.winfo_rooty() + (self.dialog.winfo_height() // 2) - (dialog.winfo_height() // 2)
            dialog.geometry(f"+{x}+{y}")

            result = {"value": "cancel"}

            # Message
            message_frame = tk.Frame(dialog, bg=self.COLORS['bg_primary'])
            message_frame.pack(fill='x', expand=True, padx=20, pady=20)

            icon_label = tk.Label(message_frame, text="⚠️", bg=self.COLORS['bg_primary'],
                                fg=self.COLORS['warning'], font=('Segoe UI', 16))
            icon_label.pack(side='left', padx=(0, 10))

            text_label = tk.Label(message_frame,
                                text="You have unsaved changes.\nWhat would you like to do?",
                                bg=self.COLORS['bg_primary'], fg=self.COLORS['text_primary'],
                                font=('Segoe UI', 10), justify='left')
            text_label.pack(side='left')

            # Buttons
            button_frame = tk.Frame(dialog, bg=self.COLORS['bg_primary'])
            button_frame.pack(fill='x', padx=20, pady=(0, 20))

            def set_result(value):
                result["value"] = value
                dialog.destroy()

            save_btn = tk.Button(button_frame, text="Save Changes",
                               command=lambda: set_result("save"),
                               bg=self.COLORS['success'], fg='white', relief='flat',
                               font=('Segoe UI', 9), pady=8, padx=15)
            save_btn.pack(side='right', padx=(5, 0))

            discard_btn = tk.Button(button_frame, text="Discard Changes",
                                  command=lambda: set_result("discard"),
                                  bg=self.COLORS['error'], fg='white', relief='flat',
                                  font=('Segoe UI', 9), pady=8, padx=15)
            discard_btn.pack(side='right', padx=(5, 0))

            cancel_btn = tk.Button(button_frame, text="Continue Editing",
                                 command=lambda: set_result("cancel"),
                                 bg=self.COLORS['bg_tertiary'], fg=self.COLORS['text_primary'],
                                 relief='flat', font=('Segoe UI', 9), pady=8, padx=15)
            cancel_btn.pack(side='right', padx=(5, 0))

            # Wait for dialog
            dialog.wait_window()
            return result["value"]

        except Exception as e:
            logger.error(f"Error showing unsaved changes dialog: {e}")
            # Fallback to standard messagebox
            result = messagebox.askyesnocancel("Unsaved Changes",
                                              "You have unsaved changes. Save before closing?")
            if result is True:
                return "save"
            elif result is False:
                return "discard"
            else:
                return "cancel"

    def _update_original_values(self) -> None:
        """Update original values after successful settings application."""
        try:
            self.original_values = self._collect_current_settings()
            logger.debug("Original values updated after successful settings application")
        except Exception as e:
            logger.error(f"Error updating original values: {e}")

    def _show_validation_errors(self, errors: List[str], validation_results: Dict[str, Any]) -> None:
        """Show comprehensive validation errors with improved user experience."""
        try:
            # Create validation error dialog
            error_dialog = tk.Toplevel(self.dialog)
            error_dialog.title("Settings Validation Errors")
            error_dialog.geometry("600x400")
            error_dialog.configure(bg=self.COLORS['bg_primary'])
            error_dialog.transient(self.dialog)
            error_dialog.grab_set()

            # Center on parent
            error_dialog.update_idletasks()
            x = self.dialog.winfo_rootx() + (self.dialog.winfo_width() // 2) - (error_dialog.winfo_width() // 2)
            y = self.dialog.winfo_rooty() + (self.dialog.winfo_height() // 2) - (error_dialog.winfo_height() // 2)
            error_dialog.geometry(f"+{x}+{y}")

            # Header
            header_frame = tk.Frame(error_dialog, bg=self.COLORS['bg_primary'])
            header_frame.pack(fill='x', padx=20, pady=(20, 10))

            error_icon = tk.Label(header_frame, text="⚠️", bg=self.COLORS['bg_primary'],
                                fg=self.COLORS['error'], font=('Segoe UI', 20))
            error_icon.pack(side='left', padx=(0, 10))

            title_label = tk.Label(header_frame,
                                 text="Settings Validation Failed",
                                 bg=self.COLORS['bg_primary'], fg=self.COLORS['text_primary'],
                                 font=('Segoe UI', 14, 'bold'))
            title_label.pack(side='left')

            subtitle_label = tk.Label(header_frame,
                                    text=f"Please fix {len(errors)} error(s) before applying settings:",
                                    bg=self.COLORS['bg_primary'], fg=self.COLORS['text_secondary'],
                                    font=('Segoe UI', 10))
            subtitle_label.pack(side='left', padx=(10, 0))

            # Scrollable error list
            list_frame = tk.Frame(error_dialog, bg=self.COLORS['bg_primary'])
            list_frame.pack(fill='both', expand=True, padx=20, pady=10)

            # Create scrollable text widget for errors
            text_frame = tk.Frame(list_frame, bg=self.COLORS['bg_secondary'])
            text_frame.pack(fill='both', expand=True)

            scrollbar = tk.Scrollbar(text_frame)
            scrollbar.pack(side='right', fill='y')

            error_text = tk.Text(text_frame,
                               bg=self.COLORS['bg_secondary'],
                               fg=self.COLORS['text_primary'],
                               font=('Segoe UI', 9),
                               wrap='word',
                               yscrollcommand=scrollbar.set,
                               height=15)
            error_text.pack(side='left', fill='both', expand=True)
            scrollbar.config(command=error_text.yview)

            # Populate errors with formatting
            for i, error in enumerate(errors, 1):
                error_text.insert('end', f"{i}. {error}\n\n")

            # Make text read-only
            error_text.config(state='disabled')

            # Buttons
            button_frame = tk.Frame(error_dialog, bg=self.COLORS['bg_primary'])
            button_frame.pack(fill='x', padx=20, pady=(10, 20))

            def close_error_dialog():
                error_dialog.destroy()

            close_btn = tk.Button(button_frame, text="OK",
                                command=close_error_dialog,
                                bg=self.COLORS['accent_primary'], fg='white',
                                relief='flat', font=('Segoe UI', 10),
                                pady=8, padx=20)
            close_btn.pack(side='right')

            # Focus on dialog
            error_dialog.focus_set()

        except Exception as e:
            logger.error(f"Error showing validation errors dialog: {e}")
            # Fallback to simple messagebox
            error_msg = "Settings validation failed:\n\n" + "\n".join(errors[:5])
            if len(errors) > 5:
                error_msg += f"\n... and {len(errors) - 5} more errors"
            messagebox.showerror("Validation Error", error_msg)

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

    # Public API methods - delegate to private implementations
    def apply_settings(self) -> bool:
        """Apply settings immediately while keeping dialog open.

        Returns:
            bool: True if settings were applied successfully, False otherwise.
        """
        try:
            self._on_apply()
            return not self._changes_made  # True if changes were successfully applied
        except Exception as e:
            logger.error(f"Error in apply_settings: {e}")
            return False

    def ok_pressed(self) -> bool:
        """Apply settings and close dialog if successful.

        Returns:
            bool: True if dialog was closed (settings applied or no changes), False if dialog remains open due to errors.
        """
        try:
            dialog_exists_before = self.winfo_exists()
            self._on_ok()
            # Check if dialog still exists after _on_ok - if not, it was closed successfully
            return not self.winfo_exists() if dialog_exists_before else True
        except Exception as e:
            logger.error(f"Error in ok_pressed: {e}")
            return False

    def cancel_pressed(self) -> bool:
        """Handle cancel with unsaved changes confirmation.

        Returns:
            bool: True if dialog was closed, False if user chose to continue editing.
        """
        try:
            dialog_exists_before = self.winfo_exists()
            self._on_cancel()
            # Check if dialog still exists after _on_cancel - if not, it was closed
            return not self.winfo_exists() if dialog_exists_before else True
        except Exception as e:
            logger.error(f"Error in cancel_pressed: {e}")
            return False

    def validate_settings(self) -> Dict[str, Any]:
        """Validate all settings and return detailed results.

        Returns:
            Dict[str, Any]: Validation results with detailed information about each setting.
        """
        try:
            settings = self._collect_current_settings()
            validation_results = SettingsValidator.validate_all_settings(settings)

            # Update the validation display
            self._validate_all_settings()

            return validation_results
        except Exception as e:
            logger.error(f"Error in validate_settings: {e}")
            return {}

    def reset_to_defaults(self) -> bool:
        """Reset all settings to default values.

        Returns:
            bool: True if reset was successful, False otherwise.
        """
        try:
            self._reset_to_defaults()
            return True
        except Exception as e:
            logger.error(f"Error in reset_to_defaults: {e}")
            return False

    def load_settings(self) -> bool:
        """Load current settings into the UI.

        Returns:
            bool: True if settings were loaded successfully, False otherwise.
        """
        try:
            self._load_current_settings()
            return True
        except Exception as e:
            logger.error(f"Error in load_settings: {e}")
            return False

    def save_settings(self) -> bool:
        """Save current settings from the UI to configuration.

        Returns:
            bool: True if settings were saved successfully, False otherwise.
        """
        try:
            return self._apply_changes()
        except Exception as e:
            logger.error(f"Error in save_settings: {e}")
            return False

    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes in the dialog.

        Returns:
            bool: True if there are unsaved changes, False otherwise.
        """
        return self._changes_made

    def get_current_settings(self) -> Dict[str, Any]:
        """Get the current settings from the UI without saving.

        Returns:
            Dict[str, Any]: Current settings from the UI components.
        """
        try:
            return self._collect_current_settings()
        except Exception as e:
            logger.error(f"Error in get_current_settings: {e}")
            return {}


__all__ = ["ComprehensiveSettingsDialog"]