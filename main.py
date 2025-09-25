"""Unified main entry point for the Python Game Detection System.

This is the single entry point that consolidates all application functionality
with modern UI, comprehensive error handling, and proper resource management.
"""
import sys
import os
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
import tkinter as tk

# Add the current directory to path for app imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import application modules
from app.config.settings import load_config, Config
from app.utils.geometry import ensure_dirs
from app.core.exceptions import ApplicationError, ConfigError
from app.core.performance import PerformanceMonitor
from app.core.logging_config import configure_logging, get_logger, set_correlation_id, CorrelationContext, log_security_event
from app.core.health_monitor import get_health_monitor, setup_default_health_checks


def setup_logging(config: Config) -> None:
    """Set up production-ready structured logging configuration."""
    try:
        log_level = getattr(config, 'log_level', 'INFO')
        log_dir = getattr(config, 'log_dir', 'logs')
        structured_logs = getattr(config, 'structured_logging', False)

        configure_logging(
            log_level=log_level,
            log_dir=log_dir,
            enable_file_logging=True,
            enable_console_logging=True,
            structured_logging=structured_logs,
            application_name='python-game-detection'
        )

        # Set initial correlation ID for application startup
        startup_correlation_id = set_correlation_id()

        logger = get_logger(__name__)
        logger.info("Structured logging configured successfully", extra={
            'startup_correlation_id': startup_correlation_id,
            'log_level': log_level,
            'structured': structured_logs
        })

    except Exception as e:
        # Fallback to basic logging if structured logging fails
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.error(f"Failed to configure structured logging, using basic logging: {e}")


def setup_directories(config: Config) -> None:
    """Ensure required directories exist."""
    try:
        directories = [
            config.data_dir,
            config.models_dir,
            config.master_dir,
            getattr(config, 'results_export_dir', 'results')
        ]

        ensure_dirs(*directories)
        logging.info(f"Directories created/verified: {directories}")

    except Exception as e:
        raise ApplicationError(f"Failed to setup directories: {e}")


def create_main_window(config: Config) -> tk.Tk:
    """Create and configure the main application window."""
    try:
        from app.ui.modern_main_window import ModernMainWindow

        root = tk.Tk()
        root.title("Python Game Detection System")

        # Create main application window
        app = ModernMainWindow(root, config)

        # Setup window icon if available
        icon_path = current_dir / "assets" / "app_icon.ico"
        if icon_path.exists():
            try:
                root.iconbitmap(str(icon_path))
            except Exception as e:
                logging.debug(f"Could not set application icon: {e}")

        # Center window on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        pos_x = (root.winfo_screenwidth() // 2) - (width // 2)
        pos_y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f"{width}x{height}+{pos_x}+{pos_y}")

        # Setup graceful shutdown
        def on_closing():
            """Handle application shutdown with proper resource cleanup."""
            try:
                logging.info("Application shutdown initiated")

                # Stop performance monitoring
                try:
                    PerformanceMonitor.instance().stop()
                except Exception as e:
                    logger = get_logger(__name__)
                    logger.warning(f"Failed to stop performance monitoring: {e}")

                # Cleanup application resources
                if hasattr(app, '_is_streaming'):
                    app._is_streaming = False

                if hasattr(app, 'webcam_service') and app.webcam_service:
                    app.webcam_service.close()
                    logging.info("Webcam service closed")

                if hasattr(app, 'gemini_service') and app.gemini_service:
                    if hasattr(app.gemini_service, 'cleanup_threads'):
                        app.gemini_service.cleanup_threads()
                    logging.info("AI service cleaned up")

                # Final cleanup
                logging.info("Resource cleanup completed")

            except AttributeError as e:
                logging.warning(f"Cleanup failed - missing attribute: {e}")
            except Exception as e:
                logging.error(f"Error during application cleanup: {e}")
                traceback.print_exc()
            finally:
                root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)

        return root, app

    except ImportError as e:
        raise ApplicationError(f"Failed to import UI components: {e}")
    except Exception as e:
        raise ApplicationError(f"Failed to create main window: {e}")


def setup_console_encoding() -> None:
    """Set up console encoding for Windows compatibility."""
    if sys.platform == 'win32':
        try:
            # Try to set console to UTF-8 mode on Windows 10+
            import subprocess
            # SECURITY FIX: Remove shell=True to prevent injection attacks
            subprocess.run(['chcp', '65001'], capture_output=True, timeout=5)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            # Command not available or failed - continue without UTF-8 setup
            pass
        except Exception:
            # Ignore other encoding setup failures - will fall back to ASCII-safe output
            pass


def display_startup_info(config: Config) -> None:
    """Display startup information with Windows-compatible ASCII characters."""
    setup_console_encoding()

    # Use ASCII-safe characters for Windows compatibility
    try:
        print(">>> Python Game Detection System")
    except UnicodeEncodeError:
        print(">>> Python Game Detection System")

    print("=" * 50)
    print("Features:")
    print("  + Live webcam streaming with enhanced controls")
    print("  + Reference image management")
    print("  + AI-powered image analysis (Gemini API)")
    print("  + Real-time difference detection")
    print("  + Interactive chat interface")
    print("  + Modern professional UI/UX")
    print("  + Comprehensive test coverage")
    print("  + Production-ready logging")
    print()

    # Display configuration info
    print("Configuration:")
    print(f"  * Data directory: {config.data_dir}")
    print(f"  * Models directory: {config.models_dir}")
    print(f"  * Debug mode: {getattr(config, 'debug', False)}")
    print()

    # AI service status - check both config and environment
    ai_configured = (
        bool(getattr(config, 'gemini_api_key', '')) or
        bool(getattr(config, '_has_secure_api_key', False))
    )
    if ai_configured:
        print(">>> AI features enabled - Gemini API configured")
    else:
        print(">>> AI features disabled - Configure GEMINI_API_KEY environment variable or add to Settings")
        print("    Create a .env file with your API key or set the environment variable to enable AI analysis")

    print("=" * 50)


def validate_system_requirements() -> bool:
    """Validate system requirements and dependencies."""
    try:
        import numpy
        import cv2
        import PIL
        logging.info("Core dependencies validated")

        # Check Python version
        if sys.version_info < (3, 8):
            logging.error("Python 3.8+ is required")
            return False

        return True

    except ImportError as e:
        logging.error(f"Missing required dependency: {e}")
        return False


def setup_windows_console() -> None:
    """Setup Windows console for better Unicode support."""
    if sys.platform == 'win32':
        try:
            # Set console output encoding to UTF-8 if possible
            if hasattr(sys.stdout, 'reconfigure'):
                try:
                    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
                    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
                except Exception:
                    pass

            # Set environment variable for Python to use UTF-8 encoding
            os.environ.setdefault('PYTHONIOENCODING', 'utf-8:replace')

        except Exception:
            # If all else fails, we'll rely on ASCII-safe output in display functions
            pass


def main() -> int:
    """Main application entry point with comprehensive error handling."""
    # Setup console encoding first thing
    setup_windows_console()

    exit_code = 0

    with CorrelationContext() as correlation_id:
        try:
            # Validate system requirements
            if not validate_system_requirements():
                print("System requirements not met. Please check the installation.")
                return 1

            # Load configuration
            config = load_config()

            # Setup structured logging
            setup_logging(config)
            logger = get_logger(__name__)
            logger.info("Application startup initiated", extra={
                'startup_correlation_id': correlation_id,
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'platform': sys.platform
            })

            # Setup directories
            setup_directories(config)

            # Start performance monitoring
            try:
                PerformanceMonitor.instance().start()
                logger.info("Performance monitoring started")
            except Exception as e:
                logger.warning(f"Failed to start performance monitoring: {e}")

            # Setup health monitoring
            setup_default_health_checks(config)
            health_monitor = get_health_monitor()
            health_monitor.start()
            logger.info("Health monitoring started")

            # Display startup information
            display_startup_info(config)

            # Create and run main window
            root, app = create_main_window(config)

            logger.info("Starting main application loop", extra={
                'ui_backend': 'tkinter',
                'config_loaded': True
            })
            root.mainloop()

            logger.info("Application terminated normally")

        except ConfigError as e:
            logger = get_logger(__name__)
            logger.error("Configuration error occurred", extra={
                'error_type': 'ConfigError',
                'error_message': str(e),
                'correlation_id': correlation_id
            })

            log_security_event(
                event_type='application_error',
                description=f'Configuration error on startup: {str(e)}',
                severity='ERROR',
                additional_data={'error_type': 'ConfigError', 'correlation_id': correlation_id}
            )

            print(f"Configuration Error: {e}")
            print("Please check your configuration file and try again.")
            exit_code = 2

        except ApplicationError as e:
            logger = get_logger(__name__)
            logger.error("Application error occurred", extra={
                'error_type': 'ApplicationError',
                'error_message': str(e),
                'correlation_id': correlation_id
            })

            log_security_event(
                event_type='application_error',
                description=f'Application error on startup: {str(e)}',
                severity='ERROR',
                additional_data={'error_type': 'ApplicationError', 'correlation_id': correlation_id}
            )

            print(f"Application Error: {e}")
            exit_code = 3

        except KeyboardInterrupt:
            logger = get_logger(__name__)
            logger.info("Application interrupted by user", extra={
                'interrupt_type': 'KeyboardInterrupt',
                'correlation_id': correlation_id
            })

            log_security_event(
                event_type='application_shutdown',
                description='Application interrupted by user',
                severity='INFO',
                additional_data={'shutdown_type': 'keyboard_interrupt', 'correlation_id': correlation_id}
            )

            print("\\nApplication interrupted by user")
            exit_code = 130

        except Exception as e:
            logger = get_logger(__name__)
            logger.critical("Unexpected error occurred", extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'correlation_id': correlation_id
            }, exc_info=True)

            log_security_event(
                event_type='application_error',
                description=f'Unexpected error on startup: {str(e)}',
                severity='CRITICAL',
                additional_data={'error_type': type(e).__name__, 'correlation_id': correlation_id}
            )

            print(f"Unexpected Error: {e}")
            print("Stack trace:")
            traceback.print_exc()
            exit_code = 1

        finally:
            # Ensure cleanup happens
            try:
                # Stop health monitoring
                health_monitor = get_health_monitor()
                if health_monitor:
                    health_monitor.stop()
                logger = get_logger(__name__)
                logger.info("Health monitoring stopped")
            except Exception as cleanup_error:
                logger = get_logger(__name__)
                logger.warning("Error stopping health monitoring", extra={
                    'cleanup_error': str(cleanup_error),
                    'correlation_id': correlation_id
                })

            try:
                PerformanceMonitor.instance().stop()
                logger = get_logger(__name__)
                logger.info("Performance monitoring stopped")
            except Exception as cleanup_error:
                logger = get_logger(__name__)
                logger.warning("Error stopping performance monitoring", extra={
                    'cleanup_error': str(cleanup_error),
                    'correlation_id': correlation_id
                })

            logger = get_logger(__name__)
            logger.info("Application exiting", extra={
                'exit_code': exit_code,
                'correlation_id': correlation_id
            })

            # Log application shutdown event
            log_security_event(
                event_type='application_shutdown',
                description=f'Application shutdown with exit code {exit_code}',
                severity='INFO',
                additional_data={'exit_code': exit_code, 'correlation_id': correlation_id}
            )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())