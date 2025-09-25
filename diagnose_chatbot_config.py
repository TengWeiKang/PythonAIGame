#!/usr/bin/env python3
"""Comprehensive Chatbot Configuration Diagnostic Tool.

This tool performs a thorough analysis of the chatbot configuration to identify
and resolve issues with API key detection and service initialization.

Usage:
    python diagnose_chatbot_config.py

The tool will:
1. Check environment variable sources
2. Validate API key format and sources
3. Test Gemini SDK availability and initialization
4. Provide specific recommendations for fixing issues
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add the current directory to path for app imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Setup basic logging for diagnostics
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")

def check_environment_variables() -> Dict[str, Any]:
    """Check all possible sources of environment variables."""
    print_section("ENVIRONMENT VARIABLE ANALYSIS")

    results = {
        'system_env': None,
        'env_file': None,
        'env_file_exists': False,
        'env_file_readable': False,
        'env_file_content': {},
        'effective_key': None,
        'errors': []
    }

    # 1. Check system environment variable
    print_subsection("System Environment Variable")
    system_key = os.getenv('GEMINI_API_KEY')
    if system_key:
        print(f"✓ GEMINI_API_KEY found in system environment")
        print(f"  Length: {len(system_key)} characters")
        print(f"  Starts with: {system_key[:10]}..." if len(system_key) > 10 else f"  Value: {system_key}")
        results['system_env'] = system_key
    else:
        print("✗ GEMINI_API_KEY not found in system environment")

    # 2. Check .env file
    print_subsection(".env File Analysis")
    env_file_path = current_dir / ".env"
    results['env_file_exists'] = env_file_path.exists()

    if env_file_path.exists():
        print(f"✓ .env file found at: {env_file_path}")
        try:
            with open(env_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                results['env_file_readable'] = True
                print(f"✓ .env file is readable ({len(content)} characters)")

                # Parse the .env file manually
                env_vars = {}
                for line_num, line in enumerate(content.splitlines(), 1):
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        env_vars[key] = value

                        if key == 'GEMINI_API_KEY':
                            print(f"✓ GEMINI_API_KEY found in .env file (line {line_num})")
                            print(f"  Length: {len(value)} characters")
                            print(f"  Starts with: {value[:10]}..." if len(value) > 10 else f"  Value: {value}")
                            results['env_file'] = value

                results['env_file_content'] = env_vars

                if 'GEMINI_API_KEY' not in env_vars:
                    print("✗ GEMINI_API_KEY not found in .env file")
                    print("  Available variables:", list(env_vars.keys()))

        except Exception as e:
            print(f"✗ Error reading .env file: {e}")
            results['errors'].append(f"Failed to read .env file: {e}")
    else:
        print(f"✗ .env file not found at: {env_file_path}")
        print("  You can create one with: GEMINI_API_KEY=your_api_key_here")

    # 3. Determine effective API key
    print_subsection("Effective API Key Resolution")
    effective_key = system_key or results.get('env_file')
    if effective_key:
        print(f"✓ Effective API key resolved")
        print(f"  Source: {'System Environment' if system_key else '.env file'}")
        print(f"  Length: {len(effective_key)} characters")
        results['effective_key'] = effective_key
    else:
        print("✗ No API key found in any source")
        results['errors'].append("No GEMINI_API_KEY found in environment or .env file")

    return results

def validate_api_key_format(api_key: str) -> Dict[str, Any]:
    """Validate API key format according to Gemini requirements."""
    print_section("API KEY FORMAT VALIDATION")

    results = {
        'is_valid': False,
        'length_check': False,
        'prefix_check': False,
        'character_check': False,
        'errors': []
    }

    if not api_key:
        print("✗ No API key provided for validation")
        results['errors'].append("No API key to validate")
        return results

    print(f"Validating API key: {api_key[:10]}...{api_key[-5:] if len(api_key) > 15 else api_key}")

    # Check length (Gemini keys are typically 39 characters)
    if len(api_key) == 39:
        print("✓ API key length is correct (39 characters)")
        results['length_check'] = True
    else:
        print(f"✗ API key length is {len(api_key)}, expected 39 characters")
        results['errors'].append(f"Invalid length: {len(api_key)} (expected 39)")

    # Check prefix
    if api_key.startswith('AIza'):
        print("✓ API key has correct prefix (AIza)")
        results['prefix_check'] = True
    else:
        print(f"✗ API key should start with 'AIza', found: {api_key[:4]}")
        results['errors'].append(f"Invalid prefix: {api_key[:4]} (expected 'AIza')")

    # Check character composition
    import re
    pattern = re.compile(r'^AIza[0-9A-Za-z-_]{35}$')
    if pattern.match(api_key):
        print("✓ API key character composition is valid")
        results['character_check'] = True
    else:
        print("✗ API key contains invalid characters")
        results['errors'].append("Invalid characters (should be alphanumeric, dash, underscore only)")

    results['is_valid'] = all([results['length_check'], results['prefix_check'], results['character_check']])

    if results['is_valid']:
        print("\n✓ API key format validation PASSED")
    else:
        print("\n✗ API key format validation FAILED")

    return results

def test_sdk_availability() -> Dict[str, Any]:
    """Test Google Generative AI SDK availability and functionality."""
    print_section("GOOGLE GENERATIVE AI SDK TEST")

    results = {
        'sdk_installed': False,
        'sdk_version': None,
        'import_successful': False,
        'configuration_test': False,
        'errors': []
    }

    # Test SDK import
    print_subsection("SDK Import Test")
    try:
        import google.generativeai as genai
        from google.generativeai import GenerativeModel
        print("✓ Google Generative AI SDK imported successfully")
        results['sdk_installed'] = True
        results['import_successful'] = True

        try:
            version = genai.__version__
            print(f"✓ SDK version: {version}")
            results['sdk_version'] = version
        except AttributeError:
            print("? SDK version information not available")

    except ImportError as e:
        print(f"✗ Failed to import Google Generative AI SDK: {e}")
        print("  Install with: pip install google-generativeai")
        results['errors'].append(f"SDK import failed: {e}")
        return results

    # Test SDK configuration (without actual API call)
    print_subsection("SDK Configuration Test")
    try:
        # Test with a dummy key to check configuration mechanism
        genai.configure(api_key="AIzaTest_dummy_key_for_config_test_only")
        print("✓ SDK configuration mechanism works")
        results['configuration_test'] = True
    except Exception as e:
        print(f"✗ SDK configuration failed: {e}")
        results['errors'].append(f"SDK configuration failed: {e}")

    return results

def test_application_config_loading() -> Dict[str, Any]:
    """Test application configuration loading."""
    print_section("APPLICATION CONFIGURATION LOADING TEST")

    results = {
        'config_loading': False,
        'env_config_loading': False,
        'config_has_api_key': False,
        'env_config_has_api_key': False,
        'config_object': None,
        'env_config_object': None,
        'errors': []
    }

    # Test main configuration loading
    print_subsection("Main Configuration Loading")
    try:
        from app.config.settings import load_config
        config = load_config()
        print("✓ Application configuration loaded successfully")
        results['config_loading'] = True
        results['config_object'] = config

        # Check if config has API key
        api_key = getattr(config, 'gemini_api_key', '')
        if api_key:
            print(f"✓ Configuration contains API key ({len(api_key)} characters)")
            results['config_has_api_key'] = True
        else:
            print("✗ Configuration does not contain API key")

        # Check environment configuration
        if hasattr(config, '_environment_config') and config._environment_config:
            print("✓ Environment configuration object found")
            results['env_config_loading'] = True
            results['env_config_object'] = config._environment_config

            env_api_key = config._environment_config.gemini_api_key
            if env_api_key:
                print(f"✓ Environment configuration contains API key ({len(env_api_key)} characters)")
                results['env_config_has_api_key'] = True
            else:
                print("✗ Environment configuration does not contain API key")
        else:
            print("✗ Environment configuration object not found")

    except Exception as e:
        print(f"✗ Failed to load application configuration: {e}")
        results['errors'].append(f"Config loading failed: {e}")

    return results

def test_gemini_service_initialization(api_key: Optional[str] = None) -> Dict[str, Any]:
    """Test Gemini service initialization."""
    print_section("GEMINI SERVICE INITIALIZATION TEST")

    results = {
        'service_creation': False,
        'service_configuration': False,
        'service_status': {},
        'errors': []
    }

    if not api_key:
        print("✗ No API key provided for service test")
        results['errors'].append("No API key for service test")
        return results

    print_subsection("Service Creation Test")
    try:
        from app.services.gemini_service import AsyncGeminiService
        service = AsyncGeminiService(api_key=api_key)
        print("✓ AsyncGeminiService created successfully")
        results['service_creation'] = True

        # Test configuration status
        print_subsection("Service Configuration Status")
        if hasattr(service, 'get_configuration_status'):
            status = service.get_configuration_status()
            results['service_status'] = status

            print(f"  Is configured: {status.get('is_configured', False)}")
            print(f"  Has API key: {status.get('has_api_key', False)}")
            print(f"  Has model: {status.get('has_model', False)}")
            print(f"  SDK available: {status.get('sdk_available', False)}")
            print(f"  API key valid format: {status.get('api_key_valid_format', False)}")

            if status.get('errors'):
                print("  Errors:")
                for error in status['errors']:
                    print(f"    - {error}")

            if status.get('config_sources'):
                print("  Configuration sources:")
                for source in status['config_sources']:
                    print(f"    - {source}")

            if status.get('is_configured'):
                print("✓ Service is properly configured")
                results['service_configuration'] = True
            else:
                print("✗ Service is not properly configured")
        else:
            # Fallback to basic is_configured check
            if service.is_configured():
                print("✓ Service reports as configured")
                results['service_configuration'] = True
            else:
                print("✗ Service reports as not configured")

    except Exception as e:
        print(f"✗ Failed to create or test Gemini service: {e}")
        results['errors'].append(f"Service initialization failed: {e}")

    return results

def generate_recommendations(all_results: Dict[str, Any]) -> List[str]:
    """Generate specific recommendations based on diagnostic results."""
    recommendations = []

    env_results = all_results.get('environment', {})
    format_results = all_results.get('format', {})
    sdk_results = all_results.get('sdk', {})
    config_results = all_results.get('config', {})
    service_results = all_results.get('service', {})

    # API Key recommendations
    if not env_results.get('effective_key'):
        recommendations.append(
            "🔑 SET UP API KEY: Create a .env file in the project directory with:\n"
            "   GEMINI_API_KEY=your_actual_api_key_here\n"
            "   Get your API key from: https://makersuite.google.com/app/apikey"
        )
    elif not format_results.get('is_valid'):
        recommendations.append(
            "🔑 FIX API KEY FORMAT: Your API key format is invalid.\n"
            "   Gemini API keys should:\n"
            "   - Be exactly 39 characters long\n"
            "   - Start with 'AIza'\n"
            "   - Contain only letters, numbers, dashes, and underscores"
        )

    # SDK recommendations
    if not sdk_results.get('sdk_installed'):
        recommendations.append(
            "📦 INSTALL SDK: Install the Google Generative AI SDK:\n"
            "   pip install google-generativeai"
        )

    # Configuration recommendations
    if config_results.get('config_loading') and not config_results.get('env_config_has_api_key'):
        recommendations.append(
            "⚙️ CONFIGURATION ISSUE: Environment variables not being loaded properly.\n"
            "   Try restarting the application after setting the environment variable."
        )

    # Service recommendations
    if service_results.get('service_creation') and not service_results.get('service_configuration'):
        service_errors = service_results.get('service_status', {}).get('errors', [])
        if service_errors:
            recommendations.append(
                f"🔧 SERVICE CONFIGURATION: Fix these issues:\n" +
                "\n".join(f"   - {error}" for error in service_errors)
            )

    # General recommendations
    if not recommendations:
        recommendations.append(
            "✅ CONFIGURATION LOOKS GOOD: If you're still having issues, try:\n"
            "   1. Restart the application\n"
            "   2. Check your internet connection\n"
            "   3. Verify your API key works at https://makersuite.google.com/"
        )
    else:
        recommendations.append(
            "🔄 AFTER FIXING: Restart the application to apply changes."
        )

    return recommendations

def main():
    """Run comprehensive chatbot configuration diagnostics."""
    print("🔍 CHATBOT CONFIGURATION DIAGNOSTIC TOOL")
    print("This tool will analyze your chatbot configuration and identify any issues.")

    # Store all results for final analysis
    all_results = {}

    # Run all diagnostic tests
    all_results['environment'] = check_environment_variables()

    # Only validate format if we have a key
    if all_results['environment'].get('effective_key'):
        all_results['format'] = validate_api_key_format(all_results['environment']['effective_key'])
    else:
        all_results['format'] = {'is_valid': False, 'errors': ['No API key to validate']}

    all_results['sdk'] = test_sdk_availability()
    all_results['config'] = test_application_config_loading()

    # Only test service if we have a valid key and SDK
    if (all_results['environment'].get('effective_key') and
        all_results['sdk'].get('sdk_installed')):
        all_results['service'] = test_gemini_service_initialization(
            all_results['environment']['effective_key']
        )
    else:
        all_results['service'] = {'service_creation': False, 'errors': ['Prerequisites not met']}

    # Generate and display recommendations
    print_section("RECOMMENDATIONS")
    recommendations = generate_recommendations(all_results)

    for i, recommendation in enumerate(recommendations, 1):
        print(f"\n{i}. {recommendation}")

    # Summary
    print_section("DIAGNOSTIC SUMMARY")

    issues_found = []
    for test_name, results in all_results.items():
        if results.get('errors'):
            issues_found.extend(results['errors'])

    if issues_found:
        print(f"❌ {len(issues_found)} issues found:")
        for issue in issues_found:
            print(f"   • {issue}")
    else:
        print("✅ No major issues detected!")

    print(f"\n📋 Full diagnostic results saved to: chatbot_diagnostic_results.json")

    # Save detailed results
    with open('chatbot_diagnostic_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

if __name__ == "__main__":
    main()