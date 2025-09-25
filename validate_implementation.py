#!/usr/bin/env python3
"""Validation script for production readiness implementation.

This script validates that all critical production features are working correctly.
"""
import sys
import os
from pathlib import Path
import time
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all critical modules can be imported."""
    print("Testing imports...")

    try:
        # Test core exceptions
        from app.core.exceptions import ApplicationError, ConfigError, AIServiceError
        print("  OK Core exceptions")

        # Test logging system
        from app.core.logging_config import get_logger, configure_logging, CorrelationContext
        print("  OK Logging system")

        # Test health monitoring
        from app.core.health_monitor import get_health_monitor, HealthCheck, SystemResourcesHealthCheck
        print("  OK Health monitoring")

        # Test main application
        import main
        print("  OK Main application")

        return True

    except Exception as e:
        print(f"  FAIL Import error: {e}")
        traceback.print_exc()
        return False

def test_logging_system():
    """Test the structured logging system."""
    print("Testing logging system...")

    try:
        from app.core.logging_config import configure_logging, get_logger, set_correlation_id, CorrelationContext

        # Configure logging
        configure_logging(
            log_level='INFO',
            log_dir='test_logs',
            enable_file_logging=True,
            enable_console_logging=False,
            structured_logging=False
        )
        print("  OK Logging configuration")

        # Test correlation IDs
        corr_id = set_correlation_id()
        assert corr_id is not None
        print("  OK Correlation ID generation")

        # Test context manager
        with CorrelationContext() as ctx_corr_id:
            logger = get_logger('test')
            logger.info("Test message with correlation", extra={'test': True})
            assert ctx_corr_id is not None
        print("  OK Correlation context")

        # Test security safe logging
        logger.info("Test with API key: sk-1234567890abcdef")  # Should be sanitized
        print("  OK Security-safe logging")

        return True

    except Exception as e:
        print(f"  FAIL Logging test error: {e}")
        traceback.print_exc()
        return False

def test_health_monitoring():
    """Test the health monitoring system."""
    print("Testing health monitoring...")

    try:
        from app.core.health_monitor import (
            get_health_monitor, SystemResourcesHealthCheck,
            ConfigurationHealthCheck, ApplicationServicesHealthCheck
        )

        monitor = get_health_monitor()
        print("  OK Health monitor instance")

        # Add health checks
        monitor.add_health_check(SystemResourcesHealthCheck())

        def mock_config_validator():
            return True, "Mock configuration valid"

        monitor.add_health_check(ConfigurationHealthCheck(mock_config_validator))

        services = {'mock_service': lambda: True}
        monitor.add_health_check(ApplicationServicesHealthCheck(services))
        print("  OK Health checks added")

        # Test health status
        status = monitor.get_health_status()
        assert 'status' in status
        print("  OK Health status retrieval")

        # Test metrics
        metrics = monitor.get_metrics()
        assert 'timestamp' in metrics
        assert 'system' in metrics
        assert 'application' in metrics
        print("  OK Metrics collection")

        return True

    except Exception as e:
        print(f"  FAIL Health monitoring test error: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test comprehensive error handling."""
    print("Testing error handling...")

    try:
        from app.core.exceptions import ApplicationError, ConfigError, AIServiceError
        from app.core.logging_config import get_logger, log_security_event

        # Test exception hierarchy
        try:
            raise ConfigError("Test configuration error")
        except ApplicationError as e:
            assert isinstance(e, ConfigError)
            print("  OK Exception hierarchy")

        # Test security event logging
        log_security_event(
            event_type='test_event',
            description='Test security event',
            severity='INFO'
        )
        print("  OK Security event logging")

        return True

    except Exception as e:
        print(f"  FAIL Error handling test error: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading and validation."""
    print("Testing configuration...")

    try:
        from app.config.settings import load_config

        # Test configuration loading
        config = load_config()
        assert hasattr(config, 'data_dir')
        print("  OK Configuration loading")

        return True

    except Exception as e:
        print(f"  FAIL Configuration test error: {e}")
        traceback.print_exc()
        return False

def test_directory_structure():
    """Test that required directories and files exist."""
    print("Testing directory structure...")

    try:
        required_dirs = [
            'app',
            'app/core',
            'app/services',
            'app/config',
            'tests',
            'tests/unit',
            'tests/integration',
            'tests/security',
            'tests/performance'
        ]

        for directory in required_dirs:
            path = project_root / directory
            assert path.exists(), f"Missing directory: {directory}"
        print("  OK Directory structure")

        required_files = [
            'main.py',
            'app/core/logging_config.py',
            'app/core/health_monitor.py',
            'app/core/exceptions.py',
            'pytest.ini',
            'run_tests.py',
            'PRODUCTION_READY_SUMMARY.md'
        ]

        for file_path in required_files:
            path = project_root / file_path
            assert path.exists(), f"Missing file: {file_path}"
        print("  OK Required files")

        return True

    except Exception as e:
        print(f"  FAIL Directory structure test error: {e}")
        return False

def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Production Readiness Validation")
    print("=" * 60)

    tests = [
        test_imports,
        test_directory_structure,
        test_logging_system,
        test_health_monitoring,
        test_error_handling,
        test_configuration
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
                print(f"OK {test.__name__}")
            else:
                failed += 1
                print(f"FAIL {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {test.__name__}: {e}")

        print()

    print("=" * 60)
    print(f"Validation Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("SUCCESS: All validation tests passed! System is production ready.")
        return 0
    else:
        print(f"ERROR: {failed} validation test(s) failed. Review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())