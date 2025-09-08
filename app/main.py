"""Main entry point for the Python Game Detection System with Modern UI."""

import tkinter as tk
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import the app package
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.settings import load_config
from app.utils.geometry import ensure_dirs
from app.ui.modern_main_window import ModernMainWindow

def setup_directories(config):
    """Ensure required directories exist."""
    ensure_dirs(
        config.data_dir,
        config.models_dir,
        config.master_dir,
        config.results_export_dir
    )

def main():
    """Modern application entry point with enhanced UI/UX."""
    try:
        # Load configuration
        config = load_config()
        
        # Setup directories
        setup_directories(config)
        
        # Create main window with modern UI
        root = tk.Tk()
        app = ModernMainWindow(root, config)
        
        # Setup window icon if available
        try:
            # You can add an icon file here if needed
            # root.iconbitmap('app_icon.ico')
            pass
        except:
            pass
        
        # Center window on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        pos_x = (root.winfo_screenwidth() // 2) - (width // 2)
        pos_y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f"{width}x{height}+{pos_x}+{pos_y}")
        
        # Handle window closing with proper cleanup
        def on_closing():
            try:
                # Cleanup resources
                if hasattr(app, '_is_streaming'):
                    app._is_streaming = False
                if hasattr(app, 'webcam_service'):
                    app.webcam_service.close()
                if hasattr(app, 'gemini_service'):
                    app.gemini_service.cleanup_threads()
            except:
                pass
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Display startup information
        print("🚀 Starting Python Game Detection System with Modern UI...")
        print("Features enabled:")
        print("  ✓ Live webcam streaming with enhanced controls")
        print("  ✓ Reference image management")
        print("  ✓ AI-powered image analysis (Gemini API)")
        print("  ✓ Real-time difference detection")
        print("  ✓ Interactive chat interface")
        print("  ✓ Professional modern UI/UX")
        print()
        print("💡 Configure your Gemini API key in Settings to enable AI features.")
        
        # Start the application
        root.mainloop()
        
    except Exception as e:
        print(f"Failed to start application: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())