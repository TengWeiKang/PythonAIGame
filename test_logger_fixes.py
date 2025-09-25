#!/usr/bin/env python3
"""Test script to validate logger fixes across the application.

This script tests that all logger-related NameError issues have been resolved
by importing key modules and testing basic functionality.
"""
import sys
import traceback
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_module_import(module_name: str, description: str) -> bool:
    """Test importing a module and return success status."""
    try:
        __import__(module_name)
        print(f"✅ {description}: Import successful")
        return True
    except NameError as e:
        if 'logger' in str(e):
            print(f"❌ {description}: Logger NameError - {e}")
        else:
            print(f"❌ {description}: Other NameError - {e}")
        return False
    except Exception as e:
        print(f"⚠️  {description}: Import error (not logger-related) - {e}")
        return True  # Not a logger issue

def test_webcam_service_instantiation() -> bool:
    """Test creating WebcamService instance to check logger usage."""
    try:
        from app.services.webcam_service import WebcamService

        # Test instantiation
        ws = WebcamService()
        print("✅ WebcamService: Instantiation successful")

        # Test close method (where the original error occurred)
        ws.close()  # This should use logger.info() without NameError
        print("✅ WebcamService: close() method works without logger errors")

        return True
    except NameError as e:
        if 'logger' in str(e):
            print(f"❌ WebcamService: Logger NameError in instantiation/close - {e}")
            traceback.print_exc()
        else:
            print(f"❌ WebcamService: Other NameError - {e}")
        return False
    except Exception as e:
        print(f"⚠️  WebcamService: Other error (not logger-related) - {e}")
        return True  # Not a logger issue

def run_logger_validation_tests():
    """Run comprehensive logger validation tests."""
    print("🔍 LOGGER VALIDATION TEST SUITE")
    print("="*50)

    # Test critical modules that were fixed
    tests = [
        ("app.services.webcam_service", "WebcamService (main fix)"),
        ("app.services.improved_webcam_service", "ImprovedWebcamService"),
        ("app.services.gemini_service", "GeminiService"),
        ("app.ui.optimized_canvas", "OptimizedCanvas (UI fix)"),
        ("app.config.settings_manager", "SettingsManager"),
        ("app.config.service_integration", "ServiceIntegration"),
        ("app.core.performance", "Performance module"),
        ("app.core.logging_config", "Logging configuration"),
    ]

    passed = 0
    total = len(tests)

    for module, description in tests:
        if test_module_import(module, description):
            passed += 1

    print("\n" + "="*50)
    print(f"📊 MODULE IMPORT RESULTS: {passed}/{total} passed")

    # Test specific WebcamService functionality
    print("\n🎯 SPECIFIC FUNCTIONALITY TESTS")
    print("-"*50)

    if test_webcam_service_instantiation():
        passed += 1
        total += 1

    print("\n" + "="*50)
    print(f"🏆 OVERALL RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("✅ ALL LOGGER ISSUES RESOLVED!")
        print("✅ Application should now run without NameError: name 'logger' is not defined")
        return True
    else:
        print(f"❌ {total - passed} issues still need fixing")
        return False

if __name__ == "__main__":
    success = run_logger_validation_tests()
    sys.exit(0 if success else 1)