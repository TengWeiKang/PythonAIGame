# Settings Usage Throughout the Application

This document explains how settings are saved, loaded, and used throughout the Python Game Detection System.

## Settings Architecture

### Core Components

1. **Config Dataclass** (`app/config/settings.py`)
   - Strongly-typed configuration object
   - Contains all application settings with type hints
   - Supports validation and serialization

2. **Settings Manager** (`app/config/settings_manager.py`)
   - Centralized settings management
   - Atomic save operations with backup/restore
   - Real-time application to all services
   - Configuration validation and migration

3. **Comprehensive Settings Dialog** (`app/ui/dialogs/comprehensive_settings_dialog.py`)
   - User interface for modifying settings
   - Real-time validation and preview
   - Organized into logical tabs

## Settings Storage and Management

### Save Settings Functionality

```python
from app.config.settings_manager import get_settings_manager

# Get the global settings manager
manager = get_settings_manager()

# Save settings (atomic operation with backup)
success = manager.save_settings(config)

# Load settings with migration support
config = manager.load_settings()

# Apply settings to all registered services
manager.apply_settings(config)
```

### Key Features:
- **Atomic Operations**: Settings are saved to a temporary file first, then atomically replaced
- **Automatic Backups**: Creates backup before each save (keeps last 10 backups)
- **Validation**: All settings are validated before saving
- **Migration**: Automatically migrates old configuration formats
- **Error Recovery**: Restores from backup if save fails

## Settings Usage Throughout the Program

### 1. Webcam Service Integration

**Settings Applied:**
- Camera device index (`last_webcam_index`)
- Resolution (`camera_width`, `camera_height`)
- FPS (`camera_fps`)
- Camera controls (`camera_brightness`, `camera_contrast`, `camera_saturation`)
- Auto settings (`camera_auto_exposure`, `camera_auto_focus`)

**Implementation:**
```python
def _apply_webcam_settings(self, service: WebcamService, config: Config) -> None:
    # Update camera device
    if config.last_webcam_index != service.current_camera_index:
        service.set_camera(config.last_webcam_index)
    
    # Update camera properties
    service.set_resolution(config.camera_width, config.camera_height)
    service.set_fps(config.camera_fps)
    
    # Apply camera controls
    service.set_brightness(config.camera_brightness)
    service.set_contrast(config.camera_contrast)
    service.set_saturation(config.camera_saturation)
```

### 2. Gemini AI Service Integration

**Settings Applied:**
- API key and model selection (`gemini_api_key`, `gemini_model`)
- Request parameters (`gemini_temperature`, `gemini_max_tokens`, `gemini_timeout`)
- Rate limiting (`enable_rate_limiting`, `requests_per_minute`)
- Context management (`context_window_size`)

**Implementation:**
```python
def _apply_gemini_settings(self, service: GeminiService, config: Config) -> None:
    # Update API configuration
    service.update_config(
        api_key=config.gemini_api_key,
        model=config.gemini_model,
        temperature=config.gemini_temperature,
        max_tokens=config.gemini_max_tokens,
        timeout=config.gemini_timeout
    )
    
    # Update rate limiting
    service.set_rate_limit(
        enabled=config.enable_rate_limiting,
        requests_per_minute=config.requests_per_minute
    )
```

### 3. UI Theme Application

**Settings Applied:**
- Theme selection (`app_theme`: Dark/Light)
- Window state management (`remember_window_state`)
- Startup preferences (`startup_fullscreen`)

**Implementation:**
```python
def _apply_theme_settings(self, main_window, config: Config) -> None:
    # Apply theme to all UI components
    main_window.apply_theme(config.app_theme)
    
    # Update window state preferences
    main_window.set_remember_state(config.remember_window_state)
```

### 4. Image Analysis Engine Integration

**Settings Applied:**
- Detection thresholds (`detection_confidence_threshold`, `detection_iou_threshold`)
- Region of Interest (`enable_roi`, `roi_x`, `roi_y`, `roi_width`, `roi_height`)
- Analysis parameters (`difference_sensitivity`, `highlight_differences`)

**Implementation:**
```python
def _apply_analysis_settings(self, service, config: Config) -> None:
    # Update detection thresholds
    service.set_confidence_threshold(config.detection_confidence_threshold)
    service.set_iou_threshold(config.detection_iou_threshold)
    
    # Update ROI settings
    if config.enable_roi:
        service.set_roi(config.roi_x, config.roi_y, 
                       config.roi_width, config.roi_height)
    else:
        service.clear_roi()
```

### 5. Auto-Save and Performance Management

**Settings Applied:**
- Auto-save intervals (`auto_save_interval_minutes`)
- Performance modes (`performance_mode`: Performance/Balanced/Power Saving)
- Memory limits (`memory_limit_mb`)
- Logging configuration (`enable_logging`, `log_level`)

**Usage in Application:**
```python
# Auto-save timer based on settings
if config.auto_save_config:
    auto_save_timer = threading.Timer(
        config.auto_save_interval_minutes * 60,
        lambda: manager.save_settings(config)
    )
    auto_save_timer.start()

# Performance mode adjustments
if config.performance_mode == "Performance":
    # Enable high-performance settings
    cv2.setNumThreads(-1)
    torch.set_num_threads(os.cpu_count())
elif config.performance_mode == "Power Saving":
    # Enable power-saving settings
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
```

## Real-Time Settings Application

### Service Registration System

```python
# Register services to receive settings updates
settings_manager = get_settings_manager()

# Register all services
settings_manager.register_service('webcam', webcam_service)
settings_manager.register_service('gemini', gemini_service)
settings_manager.register_service('detection', detection_service)
settings_manager.register_service('main_window', main_window)

# Settings are automatically applied to all registered services
# when save_settings() or apply_settings() is called
```

### Change Callbacks

```python
# Add callbacks to be notified of settings changes
def on_settings_changed(config: Config):
    print(f"Settings updated: Theme={config.app_theme}")
    # Perform additional actions

settings_manager.add_change_callback(on_settings_changed)
```

## Settings Validation and Migration

### Automatic Validation

All settings are validated before saving:

```python
def _validate_config(self, config: Config) -> bool:
    # Check numeric ranges
    if not (0.0 <= config.detection_confidence_threshold <= 1.0):
        logger.error("Detection confidence threshold must be between 0.0 and 1.0")
        return False
    
    # Check required fields
    if not config.gemini_api_key.strip() and config.enable_ai_analysis:
        logger.warning("Gemini API key required when AI analysis is enabled")
    
    # Validate directory paths
    for path_attr in ['data_dir', 'models_dir', 'results_export_dir']:
        path_value = getattr(config, path_attr)
        if path_value:
            Path(path_value).mkdir(parents=True, exist_ok=True)
    
    return True
```

### Configuration Migration

Old settings are automatically migrated:

```python
def _migrate_configuration(self, data: Dict[str, Any]) -> Dict[str, Any]:
    # Remove deprecated settings
    deprecated_settings = {
        'check_for_updates',           # Update section removed
        'backup_settings_on_change',   # Backup settings removed
        'export_format',               # Format forced to PNG
        'enable_noise_reduction',      # Image processing removed
    }
    
    migrated = {k: v for k, v in data.items() if k not in deprecated_settings}
    
    # Force export quality to 100%
    migrated['export_quality'] = 100
    
    return migrated
```

## Settings Import/Export

### Export Settings

```python
# Export current settings to file
success = settings_manager.export_settings("my_settings.json")

# Export format includes metadata
{
    "exported_at": "2024-01-15T10:30:00",
    "version": "1.0",
    "settings": {
        "app_theme": "Dark",
        "camera_width": 1920,
        "camera_height": 1080,
        // ... all settings
    }
}
```

### Import Settings

```python
# Import settings from file
success = settings_manager.import_settings("backup_settings.json")

# Settings are automatically validated and migrated during import
# All registered services receive the updated settings immediately
```

## Error Handling and Recovery

### Automatic Backup System

- Creates backup before each save operation
- Keeps last 10 backups automatically
- Restores from backup if save fails

### Graceful Degradation

- Falls back to defaults if configuration is corrupted
- Validates all settings before application
- Logs all errors for debugging

### Thread Safety

All settings operations are thread-safe using `threading.RLock()`:

```python
with self._lock:
    # All settings operations are protected
    self._config = new_config
    self._apply_settings_to_services(config)
```

## Usage Examples in Application Code

### Main Application Startup

```python
# app/main.py
from app.config.settings_manager import get_settings_manager

def main():
    # Initialize settings manager
    settings_manager = get_settings_manager()
    config = settings_manager.load_settings()
    
    # Create services
    webcam_service = WebcamService()
    gemini_service = GeminiService()
    
    # Register services for real-time updates
    settings_manager.register_service('webcam', webcam_service)
    settings_manager.register_service('gemini', gemini_service)
    
    # Apply initial settings
    settings_manager.apply_settings(config)
    
    # Create UI
    app = ModernMainWindow(root, config)
    settings_manager.register_service('main_window', app)
```

### Settings Dialog Usage

```python
# app/ui/dialogs/comprehensive_settings_dialog.py
def _on_apply(self):
    """Apply settings without closing dialog."""
    try:
        # Collect current settings from UI
        settings = self._collect_current_settings()
        
        # Update config object
        for key, value in settings.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Save and apply settings (automatic service updates)
        success = self.settings_manager.save_settings(self.config)
        
        if success:
            messagebox.showinfo("Settings Applied", 
                              "Settings have been applied successfully.")
        else:
            messagebox.showerror("Error", "Failed to save settings.")
            
    except Exception as e:
        messagebox.showerror("Error", f"Failed to apply settings: {e}")
```

This comprehensive settings system ensures that all configuration changes are:
1. **Validated** before being applied
2. **Safely stored** with atomic operations and backups
3. **Immediately applied** to all relevant services
4. **Persistent** across application restarts
5. **Migrated** automatically when upgrading versions

The settings manager acts as the central hub for all configuration management, providing a clean, type-safe interface for the entire application.