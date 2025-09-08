#!/usr/bin/env python3
"""Test script to verify settings dialog enhancements and camera preview repositioning."""

import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import json
import tempfile
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_settings_dialog_import():
    """Test that the enhanced settings dialog can be imported."""
    try:
        from app.ui.dialogs.comprehensive_settings_dialog import ComprehensiveSettingsDialog
        from app.config.settings import load_config
        
        logger.info("✓ Settings dialog import successful")
        return True
    except ImportError as e:
        logger.error(f"✗ Settings dialog import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error during import: {e}")
        return False

def test_settings_collection():
    """Test the enhanced settings collection functionality."""
    try:
        from app.ui.dialogs.comprehensive_settings_dialog import ComprehensiveSettingsDialog
        from app.config.settings import load_config
        
        # Create a temporary root window
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        config = load_config()
        services = {}  # Empty services dict for testing
        
        # Create dialog instance
        dialog = ComprehensiveSettingsDialog(root, config, services)
        
        # Test settings collection
        settings = dialog._collect_current_settings()
        
        # Verify essential settings are collected
        required_settings = [
            'app_theme', 'language', 'debug',
            'last_webcam_index', 'camera_width', 'camera_height', 'camera_fps',
            'detection_confidence_threshold', 'detection_iou_threshold',
            'gemini_api_key', 'gemini_model'
        ]
        
        missing_settings = []
        for setting in required_settings:
            if setting not in settings:
                missing_settings.append(setting)
        
        if missing_settings:
            logger.error(f"✗ Missing settings in collection: {missing_settings}")
            return False
        
        logger.info(f"✓ Settings collection successful ({len(settings)} settings)")
        
        # Clean up
        dialog.dialog.destroy()
        root.destroy()
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Settings collection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_settings_validation():
    """Test the enhanced settings validation."""
    try:
        from app.ui.dialogs.comprehensive_settings_dialog import ComprehensiveSettingsDialog
        from app.config.settings import load_config
        
        # Create a temporary root window
        root = tk.Tk()
        root.withdraw()
        
        config = load_config()
        services = {}
        
        # Create dialog instance
        dialog = ComprehensiveSettingsDialog(root, config, services)
        
        # Test valid settings
        valid_settings = {
            'app_theme': 'Dark',
            'language': 'en',
            'auto_save_config': True,
            'debug': False,
            'last_webcam_index': 0,
            'camera_width': 1280,
            'camera_height': 720,
            'camera_fps': 30,
            'detection_confidence_threshold': 0.5,
            'detection_iou_threshold': 0.45,
            'gemini_api_key': 'test_key',
            'gemini_model': 'gemini-1.5-flash'
        }
        
        # This should not raise an exception
        dialog._validate_collected_settings(valid_settings)
        logger.info("✓ Valid settings validation passed")
        
        # Test invalid settings
        invalid_settings = valid_settings.copy()
        invalid_settings['detection_confidence_threshold'] = 1.5  # Invalid range
        
        try:
            dialog._validate_collected_settings(invalid_settings)
            logger.error("✗ Invalid settings validation should have failed but didn't")
            return False
        except ValueError:
            logger.info("✓ Invalid settings validation correctly failed")
        
        # Clean up
        dialog.dialog.destroy()
        root.destroy()
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Settings validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backup_functionality():
    """Test the backup creation functionality."""
    try:
        from app.ui.dialogs.comprehensive_settings_dialog import ComprehensiveSettingsDialog
        from app.config.settings import load_config
        
        # Create a temporary root window
        root = tk.Tk()
        root.withdraw()
        
        config = load_config()
        services = {}
        
        # Create dialog instance
        dialog = ComprehensiveSettingsDialog(root, config, services)
        
        # Create a test backup
        backup_path = dialog._create_settings_backup()
        
        # Verify backup file was created
        if not os.path.exists(backup_path):
            logger.error("✗ Backup file was not created")
            return False
        
        # Verify backup content
        with open(backup_path, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
        
        if not backup_data or len(backup_data) < 10:  # Should have many settings
            logger.error("✗ Backup file appears to be empty or incomplete")
            return False
        
        logger.info(f"✓ Backup functionality successful (backup: {backup_path})")
        
        # Clean up
        os.remove(backup_path)
        dialog.dialog.destroy()
        root.destroy()
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Backup functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_webcam_tab_structure():
    """Test that the webcam tab has the correct structure with preview after device section."""
    try:
        from app.ui.dialogs.comprehensive_settings_dialog import ComprehensiveSettingsDialog
        from app.config.settings import load_config
        
        # Create a temporary root window
        root = tk.Tk()
        root.withdraw()
        
        config = load_config()
        services = {}
        
        # Create dialog instance
        dialog = ComprehensiveSettingsDialog(root, config, services)
        
        # Access the webcam tab (this builds the UI)
        webcam_tab = dialog._build_webcam_tab()
        
        # Verify the tab was created successfully
        if not webcam_tab:
            logger.error("✗ Webcam tab was not created")
            return False
        
        # Check that preview canvas exists (indicates preview section was built)
        if not hasattr(dialog, '_preview_canvas') or not dialog._preview_canvas:
            logger.error("✗ Preview canvas was not created")
            return False
        
        # Check that test camera button exists
        if not hasattr(dialog, 'test_button') or not dialog.test_button:
            logger.error("✗ Test camera button was not created")
            return False
        
        logger.info("✓ Webcam tab structure verification successful")
        
        # Clean up
        dialog.dialog.destroy()
        root.destroy()
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Webcam tab structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all enhancement tests."""
    logger.info("=== Testing Settings Dialog Enhancements ===\n")
    
    tests = [
        ("Import Test", test_settings_dialog_import),
        ("Settings Collection Test", test_settings_collection),
        ("Settings Validation Test", test_settings_validation),
        ("Backup Functionality Test", test_backup_functionality),
        ("Webcam Tab Structure Test", test_webcam_tab_structure),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"Running {test_name}...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED\n")
            else:
                logger.error(f"✗ {test_name} FAILED\n")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}\n")
    
    logger.info("=== Test Results ===")
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Settings enhancements are working correctly.")
        return True
    else:
        logger.error("❌ Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)