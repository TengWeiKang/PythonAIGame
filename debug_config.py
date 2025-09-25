#!/usr/bin/env python3
"""Debug script to analyze configuration loading and Gemini service initialization.

This script traces the complete configuration flow to identify why the chatbot
shows as "not configured" despite having a valid API key.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def debug_environment_config():
    """Test environment configuration loading."""
    print("=" * 60)
    print("1. ENVIRONMENT CONFIGURATION ANALYSIS")
    print("=" * 60)

    # Check environment variables
    gemini_api_key_env = os.getenv('GEMINI_API_KEY')
    print(f"Environment variable GEMINI_API_KEY: {gemini_api_key_env[:10] + '...' if gemini_api_key_env else 'NOT SET'}")

    # Check .env file
    env_file_path = current_dir / ".env"
    print(f".env file exists: {env_file_path.exists()}")

    if env_file_path.exists():
        try:
            with open(env_file_path, 'r') as f:
                content = f.read()
                has_api_key = 'GEMINI_API_KEY' in content
                print(f".env file contains GEMINI_API_KEY: {has_api_key}")
        except Exception as e:
            print(f"Error reading .env file: {e}")

    # Test environment config loading
    try:
        from app.config.env_config import load_environment_config
        env_config = load_environment_config()
        print(f"Environment config loaded successfully")
        print(f"  - has API key: {env_config.is_api_key_configured}")
        print(f"  - API key value: {env_config.gemini_api_key[:10] + '...' if env_config.gemini_api_key else 'None'}")
        print(f"  - model: {env_config.gemini_model}")
        print(f"  - has secure config: {env_config.has_secure_config}")
        return env_config
    except Exception as e:
        print(f"ERROR loading environment config: {e}")
        import traceback
        traceback.print_exc()
        return None


def debug_config_json():
    """Test config.json loading."""
    print("\n" + "=" * 60)
    print("2. CONFIG.JSON ANALYSIS")
    print("=" * 60)

    config_path = current_dir / "config.json"
    print(f"config.json path: {config_path}")
    print(f"config.json exists: {config_path.exists()}")

    if config_path.exists():
        try:
            import json
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            api_key = config_data.get('gemini_api_key', '')
            print(f"config.json gemini_api_key: {api_key[:10] + '...' if api_key else 'NOT SET'}")
            print(f"config.json enable_ai_analysis: {config_data.get('enable_ai_analysis', False)}")
            print(f"config.json _has_secure_api_key: {config_data.get('_has_secure_api_key', False)}")

            env_config_in_json = config_data.get('_environment_config', {})
            if env_config_in_json:
                print(f"Environment config in JSON:")
                print(f"  - gemini_api_key: {env_config_in_json.get('gemini_api_key', 'None')}")
                print(f"  - is_api_key_configured: {env_config_in_json.get('is_api_key_configured', False)}")

            return config_data
        except Exception as e:
            print(f"ERROR reading config.json: {e}")
            return None
    else:
        print("config.json does not exist")
        return None


def debug_config_loading():
    """Test full configuration loading."""
    print("\n" + "=" * 60)
    print("3. FULL CONFIGURATION LOADING ANALYSIS")
    print("=" * 60)

    try:
        from app.config.settings import load_config
        config = load_config()

        print(f"Configuration loaded successfully")
        print(f"  - gemini_api_key: {config.gemini_api_key[:10] + '...' if config.gemini_api_key else 'NOT SET'}")
        print(f"  - enable_ai_analysis: {config.enable_ai_analysis}")
        print(f"  - _has_secure_api_key: {config._has_secure_api_key}")

        if hasattr(config, '_environment_config') and config._environment_config:
            env_config = config._environment_config
            print(f"Environment config attached:")
            print(f"  - gemini_api_key: {env_config.gemini_api_key}")
            print(f"  - is_api_key_configured: {env_config.is_api_key_configured}")
        else:
            print("No environment config attached")

        return config
    except Exception as e:
        print(f"ERROR loading configuration: {e}")
        import traceback
        traceback.print_exc()
        return None


def debug_gemini_service_init(config):
    """Test Gemini service initialization."""
    print("\n" + "=" * 60)
    print("4. GEMINI SERVICE INITIALIZATION ANALYSIS")
    print("=" * 60)

    try:
        from app.services.gemini_service import AsyncGeminiService

        # Test with different initialization approaches
        print("Testing AsyncGeminiService initialization...")

        # Approach 1: No API key provided (should auto-resolve)
        print("\n--- Test 1: Auto-resolve API key ---")
        service1 = AsyncGeminiService(api_key=None)
        print(f"Service 1 API key: {service1.api_key[:10] + '...' if service1.api_key else 'None'}")
        print(f"Service 1 is_configured(): {service1.is_configured()}")

        if hasattr(service1, 'get_configuration_status'):
            status1 = service1.get_configuration_status()
            print(f"Service 1 status: {status1}")

        # Approach 2: Explicit API key from config
        print("\n--- Test 2: Explicit API key from config ---")
        api_key_from_config = getattr(config, 'gemini_api_key', '')
        if api_key_from_config:
            service2 = AsyncGeminiService(api_key=api_key_from_config)
            print(f"Service 2 API key: {service2.api_key[:10] + '...' if service2.api_key else 'None'}")
            print(f"Service 2 is_configured(): {service2.is_configured()}")

            if hasattr(service2, 'get_configuration_status'):
                status2 = service2.get_configuration_status()
                print(f"Service 2 status: {status2}")
        else:
            print("No API key in config to test with")

        # Approach 3: UI initialization approach
        print("\n--- Test 3: UI initialization approach ---")
        env_api_key = None
        if hasattr(config, '_environment_config') and config._environment_config:
            env_api_key = config._environment_config.gemini_api_key

        effective_api_key = env_api_key or getattr(config, 'gemini_api_key', '')
        print(f"Effective API key (UI approach): {effective_api_key[:10] + '...' if effective_api_key else 'None'}")

        if effective_api_key:
            service3 = AsyncGeminiService(api_key=effective_api_key)
            print(f"Service 3 API key: {service3.api_key[:10] + '...' if service3.api_key else 'None'}")
            print(f"Service 3 is_configured(): {service3.is_configured()}")

            if hasattr(service3, 'get_configuration_status'):
                status3 = service3.get_configuration_status()
                print(f"Service 3 status: {status3}")

        return service1

    except Exception as e:
        print(f"ERROR initializing Gemini service: {e}")
        import traceback
        traceback.print_exc()
        return None


def debug_api_key_validation():
    """Test API key validation."""
    print("\n" + "=" * 60)
    print("5. API KEY VALIDATION ANALYSIS")
    print("=" * 60)

    try:
        from app.config.env_config import EnvironmentValidator

        # Get API key from config.json
        config_path = current_dir / "config.json"
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            api_key = config_data.get('gemini_api_key', '')

            if api_key:
                print(f"Testing API key validation: {api_key[:10]}...")
                is_valid = EnvironmentValidator.validate_api_key(api_key)
                print(f"API key is valid format: {is_valid}")

                # Test pattern matching
                import re
                pattern = EnvironmentValidator.GEMINI_API_KEY_PATTERN
                matches = pattern.match(api_key)
                print(f"API key matches pattern: {bool(matches)}")
                print(f"API key length: {len(api_key)}")
                print(f"API key starts with 'AIza': {api_key.startswith('AIza')}")

            else:
                print("No API key found in config.json")
        else:
            print("config.json not found")

    except Exception as e:
        print(f"ERROR validating API key: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run comprehensive configuration debugging."""
    print("GEMINI CHATBOT CONFIGURATION DIAGNOSIS")
    print("=" * 60)
    print("This script will analyze why the chatbot shows as 'not configured'")
    print("despite having a valid API key.\n")

    # Step 1: Environment configuration
    env_config = debug_environment_config()

    # Step 2: Config.json analysis
    config_data = debug_config_json()

    # Step 3: Full configuration loading
    config = debug_config_loading()

    # Step 4: Gemini service initialization
    if config:
        service = debug_gemini_service_init(config)

    # Step 5: API key validation
    debug_api_key_validation()

    # Summary and recommendations
    print("\n" + "=" * 60)
    print("6. DIAGNOSIS SUMMARY")
    print("=" * 60)

    # Check if we found the issue
    has_env_var = bool(os.getenv('GEMINI_API_KEY'))
    has_env_file = (current_dir / ".env").exists()
    has_config_key = bool(config_data and config_data.get('gemini_api_key', ''))

    print(f"Environment variable GEMINI_API_KEY: {'✓' if has_env_var else '✗'}")
    print(f".env file exists: {'✓' if has_env_file else '✗'}")
    print(f"config.json has API key: {'✓' if has_config_key else '✗'}")

    if env_config:
        env_configured = env_config.is_api_key_configured
        print(f"Environment config reports API key configured: {'✓' if env_configured else '✗'}")

        if not env_configured and has_config_key:
            print("\n🔍 POTENTIAL ISSUE IDENTIFIED:")
            print("   - API key exists in config.json")
            print("   - But environment config reports it as not configured")
            print("   - This suggests the environment system is not falling back to config.json")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()