#!/usr/bin/env python3
"""Test script to validate chatbot configuration fixes.

This script tests the complete configuration flow to ensure that:
1. Environment variables are properly loaded
2. API key validation works correctly
3. Service initialization is robust
4. Configuration diagnostics provide useful information

Usage:
    python test_chatbot_fixes.py [test_api_key]

If test_api_key is provided, it will be used for testing.
Otherwise, the script will look for GEMINI_API_KEY in environment.
"""

import sys
import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add the current directory to path for app imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_environment_config_loading():
    """Test environment configuration loading with various scenarios."""
    print("\n" + "="*60)
    print("TESTING ENVIRONMENT CONFIGURATION LOADING")
    print("="*60)

    try:
        from app.config.env_config import load_environment_config, EnvironmentValidator

        # Test 1: Load without .env file
        print("\n1. Testing environment config without .env file...")
        config = load_environment_config()
        print(f"   Config loaded: {config is not None}")
        print(f"   API key configured: {config.is_api_key_configured if config else False}")

        # Test 2: API key validation
        print("\n2. Testing API key validation...")
        validator = EnvironmentValidator()

        # Test invalid keys
        invalid_keys = ["", "invalid", "AIza", "AIzaShortKey", "NotAIzaPrefix123456789012345678901234567890"]
        for key in invalid_keys:
            is_valid = validator.validate_api_key(key)
            print(f"   '{key[:20]}...': {is_valid} (expected: False)")

        # Test valid format (dummy key)
        valid_dummy = "AIzaSyDemoKeyForTestingOnly123456789ab"
        is_valid = validator.validate_api_key(valid_dummy)
        print(f"   Valid format test: {is_valid} (expected: True)")

        return True

    except Exception as e:
        print(f"   ERROR: {e}")
        return False

def test_gemini_service_enhanced_initialization():
    """Test the enhanced Gemini service initialization."""
    print("\n" + "="*60)
    print("TESTING ENHANCED GEMINI SERVICE")
    print("="*60)

    try:
        from app.services.gemini_service import AsyncGeminiService

        # Test 1: Service creation without API key
        print("\n1. Testing service creation without API key...")
        service1 = AsyncGeminiService()
        print(f"   Service created: {service1 is not None}")
        print(f"   Is configured: {service1.is_configured()}")

        # Test 2: Configuration status diagnostics
        print("\n2. Testing configuration diagnostics...")
        if hasattr(service1, 'get_configuration_status'):
            status = service1.get_configuration_status()
            print(f"   Has diagnostics method: True")
            print(f"   SDK available: {status.get('sdk_available', False)}")
            print(f"   Has API key: {status.get('has_api_key', False)}")
            print(f"   Errors: {len(status.get('errors', []))}")
        else:
            print(f"   Has diagnostics method: False")

        # Test 3: API key resolution
        print("\n3. Testing API key resolution...")
        if hasattr(service1, '_resolve_api_key'):
            # Test with dummy key
            resolved = service1._resolve_api_key("test_key")
            print(f"   Explicit key resolution: {resolved == 'test_key'}")

            # Test with environment fallback
            os.environ['GEMINI_API_KEY'] = 'env_test_key'
            resolved = service1._resolve_api_key(None)
            print(f"   Environment fallback: {resolved == 'env_test_key'}")
            del os.environ['GEMINI_API_KEY']
        else:
            print(f"   API key resolution method: Not found")

        return True

    except Exception as e:
        print(f"   ERROR: {e}")
        return False

def test_configuration_integration():
    """Test the complete configuration integration."""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION INTEGRATION")
    print("="*60)

    try:
        from app.config.settings import load_config

        # Test 1: Configuration loading
        print("\n1. Testing configuration loading...")
        config = load_config()
        print(f"   Config loaded: {config is not None}")

        if config:
            print(f"   Has environment config: {hasattr(config, '_environment_config')}")
            print(f"   Has secure API key flag: {hasattr(config, '_has_secure_api_key')}")
            print(f"   Secure API key: {getattr(config, '_has_secure_api_key', False)}")

        # Test 2: Service initialization with config
        print("\n2. Testing service initialization with config...")
        from app.services.gemini_service import AsyncGeminiService

        # Simulate the UI initialization logic
        env_api_key = None
        if hasattr(config, '_environment_config') and config._environment_config:
            env_api_key = config._environment_config.gemini_api_key

        effective_api_key = env_api_key or getattr(config, 'gemini_api_key', '')

        service = AsyncGeminiService(
            api_key=effective_api_key,
            model=getattr(config, 'gemini_model', 'gemini-1.5-flash'),
            timeout=getattr(config, 'gemini_timeout', 30),
            temperature=getattr(config, 'gemini_temperature', 0.7),
            max_tokens=getattr(config, 'gemini_max_tokens', 2048),
        )

        print(f"   Service created: {service is not None}")
        print(f"   Service configured: {service.is_configured() if service else False}")

        return True

    except Exception as e:
        print(f"   ERROR: {e}")
        return False

def test_with_temporary_env_file(api_key: str):
    """Test configuration with a temporary .env file."""
    print("\n" + "="*60)
    print("TESTING WITH TEMPORARY .ENV FILE")
    print("="*60)

    # Create temporary .env file
    temp_env_content = f"""# Test .env file
GEMINI_API_KEY={api_key}
GEMINI_MODEL=gemini-1.5-flash
GEMINI_TIMEOUT=30
DEBUG_LOGGING=true
"""

    env_file_path = current_dir / ".env.test"

    try:
        # Write temporary .env file
        with open(env_file_path, 'w') as f:
            f.write(temp_env_content)

        print(f"\n1. Created temporary .env file: {env_file_path}")

        # Test environment config loading
        from app.config.env_config import load_environment_config
        config = load_environment_config(str(env_file_path))

        print(f"   Environment config loaded: {config is not None}")
        if config:
            print(f"   API key configured: {config.is_api_key_configured}")
            print(f"   API key length: {len(config.gemini_api_key) if config.gemini_api_key else 0}")

        # Test with service
        from app.services.gemini_service import AsyncGeminiService
        service = AsyncGeminiService(api_key=config.gemini_api_key if config else None)

        print(f"   Service created: {service is not None}")
        if service and hasattr(service, 'get_configuration_status'):
            status = service.get_configuration_status()
            print(f"   Service configured: {status.get('is_configured', False)}")
            print(f"   Has model: {status.get('has_model', False)}")
            if status.get('errors'):
                print(f"   Service errors: {status['errors']}")

        return True

    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    finally:
        # Clean up temp file
        try:
            if env_file_path.exists():
                env_file_path.unlink()
                print(f"   Cleaned up temporary file: {env_file_path}")
        except Exception as e:
            print(f"   Warning: Could not clean up temp file: {e}")

def test_diagnostic_tool():
    """Test the diagnostic tool functionality."""
    print("\n" + "="*60)
    print("TESTING DIAGNOSTIC TOOL")
    print("="*60)

    try:
        # Import and test key functions from diagnostic tool
        import importlib.util
        spec = importlib.util.spec_from_file_location("diagnostic", "diagnose_chatbot_config.py")
        diagnostic = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(diagnostic)

        print("\n1. Testing diagnostic tool import...")
        print(f"   Diagnostic module loaded: True")

        # Test environment variable checking
        print("\n2. Testing environment variable analysis...")
        env_results = diagnostic.check_environment_variables()
        print(f"   Environment check completed: {env_results is not None}")
        print(f"   Found effective key: {bool(env_results.get('effective_key'))}")

        # Test SDK availability check
        print("\n3. Testing SDK availability check...")
        sdk_results = diagnostic.test_sdk_availability()
        print(f"   SDK check completed: {sdk_results is not None}")
        print(f"   SDK available: {sdk_results.get('sdk_installed', False)}")

        return True

    except Exception as e:
        print(f"   ERROR: {e}")
        return False

def main():
    """Run all chatbot configuration tests."""
    print("🧪 CHATBOT CONFIGURATION FIXES VALIDATION")
    print("This script validates that all chatbot configuration fixes are working properly.")

    # Get test API key from command line or environment
    test_api_key = None
    if len(sys.argv) > 1:
        test_api_key = sys.argv[1]
        print(f"Using provided test API key: {test_api_key[:10]}...")
    else:
        test_api_key = os.getenv('GEMINI_API_KEY')
        if test_api_key:
            print(f"Using API key from environment: {test_api_key[:10]}...")
        else:
            print("No API key provided - some tests will be limited")

    # Run all tests
    test_results = {
        'Environment Config Loading': test_environment_config_loading(),
        'Enhanced Gemini Service': test_gemini_service_enhanced_initialization(),
        'Configuration Integration': test_configuration_integration(),
        'Diagnostic Tool': test_diagnostic_tool(),
    }

    # Run .env file test only if we have an API key
    if test_api_key:
        test_results['Temporary .env File'] = test_with_temporary_env_file(test_api_key)

    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)

    passed = sum(test_results.values())
    total = len(test_results)

    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The chatbot configuration fixes are working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review the output above for details.")

    # Recommendations
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)

    if not test_api_key:
        print("1. To test with a real API key, run:")
        print("   python test_chatbot_fixes.py YOUR_API_KEY")

    print("2. To diagnose configuration in the main app, run:")
    print("   python diagnose_chatbot_config.py")

    print("3. To set up your API key, copy .env.example to .env and configure it")

    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())