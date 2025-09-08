"""Demo script to showcase the modern UI implementation."""

import tkinter as tk
import sys
import os
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from app.config.settings import load_config
    from app.ui.modern_main_window import ModernMainWindow
    from app.utils.geometry import ensure_dirs
    
    def demo_main():
        """Demo the modern UI."""
        print("🚀 Vision Analysis System - Modern UI Demo")
        print("=" * 50)
        
        try:
            # Load configuration
            config = load_config()
            print("✓ Configuration loaded")
            
            # Setup directories
            ensure_dirs(
                config.data_dir,
                config.models_dir,
                config.master_dir,
                config.results_export_dir
            )
            print("✓ Directories created")
            
            # Create the modern interface
            root = tk.Tk()
            print("✓ Tkinter root created")
            
            app = ModernMainWindow(root, config)
            print("✓ Modern main window initialized")
            
            print("\n🎯 Features Available:")
            print("  • Professional dark theme with blue accents")
            print("  • Live webcam streaming with controls")
            print("  • Reference image management")
            print("  • Current image capture and saving")
            print("  • AI-powered analysis (requires API key)")
            print("  • Real-time difference detection")
            print("  • Interactive chat interface")
            print("  • Tabbed organization for better workflow")
            
            print("\n📝 Quick Start:")
            print("  1. Click 'Settings' to configure Gemini API key")
            print("  2. Click 'Start Stream' to begin video capture")
            print("  3. Use the tabs to navigate different functions")
            print("  4. Capture reference and current images")
            print("  5. Use AI analysis for intelligent comparison")
            
            print("\n🖥️  Starting application...")
            print("   Close the window to exit the demo.")
            
            # Center window on screen
            root.update_idletasks()
            width = root.winfo_width()
            height = root.winfo_height()
            pos_x = (root.winfo_screenwidth() // 2) - (width // 2)
            pos_y = (root.winfo_screenheight() // 2) - (height // 2)
            root.geometry(f"{width}x{height}+{pos_x}+{pos_y}")
            
            # Handle window closing
            def on_closing():
                print("\n👋 Demo completed. Thank you!")
                try:
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
            
            # Start the application
            root.mainloop()
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("\n🔧 Please install required dependencies:")
            print("   pip install opencv-python pillow numpy requests")
            return 1
        except Exception as e:
            print(f"❌ Error starting demo: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        return 0
    
    if __name__ == "__main__":
        sys.exit(demo_main())
        
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("\n🔧 This appears to be a dependency issue.")
    print("   Please ensure all required packages are installed:")
    print("   pip install opencv-python pillow numpy requests")
    print(f"\n📁 Current working directory: {os.getcwd()}")
    print(f"🐍 Python path: {sys.executable}")
    sys.exit(1)