"""Default configuration values."""

from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    # Legacy/core settings
    "python_version": "3.13.7",
    "img_size": 640,
    "target_fps": 30,
    "iou_match_threshold": 0.5,
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
    "last_webcam_index": 0,
    "camera_width": 1280,
    "camera_height": 720,
    "camera_fps": 30,
    
    # General Application Settings
    "language": "en",
    
    # Enhanced Webcam Settings
    "camera_device_name": "",  # Auto-detected
    
    # Enhanced Image Analysis Settings
    "detection_confidence_threshold": 0.5,  # 0.0 to 1.0
    "min_detection_confidence": 0.5,  # Alias for detection_confidence_threshold for workflow compatibility
    "detection_iou_threshold": 0.45,  # 0.0 to 1.0
    "preferred_model": "yolo12n",
    "reference_image_path": "",  # Path to currently loaded reference image
    
    # Enhanced Chatbot Settings
    "gemini_api_key": "",
    "gemini_model": "gemini-2.5-pro",  # gemini-1.5-flash, gemini-1.5-pro
    "gemini_timeout": 30,
    "gemini_temperature": 0.7,  # 0.0 to 1.0
    "gemini_max_tokens": 2048,
    "chatbot_persona": """You are a helpful AI assistant for image analysis. Your role is to:
1. Compare reference images with captured images
2. Identify differences and changes
3. Provide clear, detailed explanations
4. Help users understand what has changed""",

    # AI Analysis Settings
    "enable_ai_analysis": False,  # Enabled automatically when API key is provided
    "enable_rate_limiting": True,
    "requests_per_minute": 15,

    # Debug and Logging Settings
    "debug": False,
    "log_level": "INFO",
    "log_dir": "logs",
    "structured_logging": False,
}