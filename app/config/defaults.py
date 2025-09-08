"""Default configuration values."""

from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    # Legacy/core settings
    "python_version": "3.13.7",
    "img_size": 640,
    "target_fps": 30,
    "iou_match_threshold": 0.5,
    "master_tolerance_px": 40,
    "angle_tolerance_deg": 20,
    "use_gpu": True,
    "data_dir": "data",
    "models_dir": "data/models",
    "master_dir": "data/master",
    "locales_dir": "locales",
    "default_locale": "en",
    "results_export_dir": "data/results",
    "default_data_dir": "data",
    "default_models_dir": "data/models",
    "default_results_dir": "data/results",
    "model_size": "yolo12n",
    "train_epochs": 10,
    "batch_size": 8,
    "debug": False,
    "last_webcam_index": 0,
    "preview_max_width": 960,
    "preview_max_height": 720,
    "camera_width": 1280,
    "camera_height": 720,
    "camera_fps": 30,
    
    # General Application Settings
    "app_theme": "Dark",  # Dark, Light, Auto
    "language": "en",
    "performance_mode": "Performance",  # Performance, Balanced, Power_Saving
    "max_memory_usage_mb": 2048,
    "enable_logging": True,
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    
    # Enhanced Webcam Settings
    "camera_auto_exposure": True,
    "camera_auto_focus": True,
    "camera_brightness": 0,  # -100 to 100
    "camera_contrast": 0,    # -100 to 100
    "camera_saturation": 0,  # -100 to 100
    "camera_recording_format": "MP4",  # MP4, AVI, MOV
    "camera_buffer_size": 5,  # frames
    "camera_preview_enabled": True,
    "camera_device_name": "",  # Auto-detected
    
    # Enhanced Image Analysis Settings
    "detection_confidence_threshold": 0.5,  # 0.0 to 1.0
    "detection_iou_threshold": 0.45,  # 0.0 to 1.0
    "roi_x": 0,
    "roi_y": 0,
    "roi_width": 0,  # 0 = full width
    "roi_height": 0,  # 0 = full height
    "enable_roi": False,
    "preferred_model": "yolo12n",
    "export_quality": 100,  # Always 100% for best quality
    "difference_sensitivity": 0.1,  # 0.0 to 1.0
    "highlight_differences": True,
    
    # Enhanced Chatbot Settings
    "gemini_api_key": "",
    "gemini_model": "gemini-1.5-flash",  # gemini-1.5-flash, gemini-1.5-pro
    "gemini_timeout": 30,
    "gemini_temperature": 0.7,  # 0.0 to 1.0
    "gemini_max_tokens": 2048,
    "enable_ai_analysis": False,
    "chat_history_limit": 100,
    "chat_auto_save": True,
    "response_format": "Detailed",  # Brief, Detailed, Technical
    "enable_rate_limiting": True,
    "requests_per_minute": 15,
    "context_window_size": 4000,
    "enable_conversation_memory": True,
    "chatbot_persona": """You are a helpful AI assistant for image analysis. Your role is to:
1. Compare reference images with captured images
2. Identify differences and changes
3. Provide clear, detailed explanations
4. Help users understand what has changed""",
    
    # Additional Enhanced Settings
    "export_include_metadata": True,
    "reference_image_path": "",
    "analysis_history_days": 30,
    "chat_export_format": "JSON",  # JSON, TXT, CSV
}