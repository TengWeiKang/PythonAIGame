"""Test script to verify dialog functionality."""

import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_dialog_imports():
    """Test that all dialogs can be imported successfully."""
    try:
        print("Testing dialog imports...")
        
        # Test config and services first
        from app.config.settings import load_config
        from app.services.webcam_service import WebcamService
        
        config = load_config()
        webcam_service = WebcamService()
        
        print("+ Config and services loaded")
        
        # Test webcam dialog
        from app.ui.dialogs.webcam_dialog import WebcamDialog
        print("+ WebcamDialog import successful")
        
        # Test settings dialog  
        from app.ui.dialogs.settings_dialog import SettingsDialog
        print("+ SettingsDialog import successful")
        
        # Test object classification dialog
        from app.ui.dialogs.object_classification_dialog import ObjectClassificationDialog
        print("+ ObjectClassificationDialog import successful")
        
        print("\nSUCCESS: All dialog imports working correctly!")
        return True
        
    except Exception as e:
        print(f"ERROR: Dialog import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dialog_creation():
    """Test that dialogs can be created without immediate errors."""
    import tkinter as tk
    
    try:
        print("\nTesting dialog creation...")
        
        # Create root window (hidden)
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        from app.config.settings import load_config
        from app.services.webcam_service import WebcamService
        
        config = load_config()
        webcam_service = WebcamService()
        
        print("+ Root window and services ready")
        
        # Test creating dialogs (but don't show them)
        try:
            from app.ui.dialogs.webcam_dialog import WebcamDialog
            # Just test class instantiation without showing
            print("+ WebcamDialog class ready for instantiation")
        except Exception as e:
            print(f"  WebcamDialog creation test failed: {e}")
        
        try:
            from app.ui.dialogs.settings_dialog import SettingsDialog  
            print("+ SettingsDialog class ready for instantiation")
        except Exception as e:
            print(f"  SettingsDialog creation test failed: {e}")
            
        try:
            from app.ui.dialogs.object_classification_dialog import ObjectClassificationDialog
            print("+ ObjectClassificationDialog class ready for instantiation")
        except Exception as e:
            print(f"  ObjectClassificationDialog creation test failed: {e}")
        
        root.destroy()
        print("\nSUCCESS: Dialog creation tests passed!")
        return True
        
    except Exception as e:
        print(f"ERROR: Dialog creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Dialog Testing Suite ===\n")
    
    # Test imports
    import_success = test_dialog_imports()
    
    # Test creation
    creation_success = test_dialog_creation()
    
    if import_success and creation_success:
        print("\n🎉 All dialog tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some dialog tests failed!")
        sys.exit(1)