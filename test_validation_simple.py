#!/usr/bin/env python3
"""Simple validation test to verify the ValidationEngine is working correctly."""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_import():
    """Test that we can import the validation classes."""
    try:
        from app.utils.validation import ValidationEngine, ValidationReport
        print("✓ Successfully imported ValidationEngine and ValidationReport")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_validation_report():
    """Test ValidationReport functionality."""
    try:
        from app.utils.validation import ValidationReport

        # Test initialization
        report = ValidationReport(is_valid=True)

        # Test adding errors
        report.add_error("test_field", "Test error")
        assert not report.is_valid, "Report should be invalid after adding error"
        assert "test_field" in report.errors, "Error field should be recorded"

        # Test adding warnings
        report.add_warning("warning_field", "Test warning")
        assert "warning_field" in report.warnings, "Warning field should be recorded"

        # Test auto-corrections
        report.suggest_correction("fix_field", "corrected_value")
        assert report.auto_corrections["fix_field"] == "corrected_value"

        # Test field status
        assert report.get_field_status("test_field") == "error"
        assert report.get_field_status("warning_field") == "warning"
        assert report.get_field_status("valid_field") == "valid"

        print("✓ ValidationReport tests passed")
        return True
    except Exception as e:
        print(f"✗ ValidationReport test failed: {e}")
        return False

def test_validation_engine():
    """Test ValidationEngine basic functionality."""
    try:
        from app.utils.validation import ValidationEngine

        # Create engine
        engine = ValidationEngine()

        # Test with valid config
        valid_config = {
            'detection_confidence_threshold': 0.5,
            'camera_width': 1280,
            'enable_roi': False
        }

        report = engine.validate_config_changes(valid_config)

        assert isinstance(report, ValidationReport), "Should return ValidationReport"
        assert len(report.stages_completed) == 5, "Should complete all 5 stages"

        # Test with invalid config
        invalid_config = {
            'detection_confidence_threshold': "invalid",  # Wrong type
            'camera_fps': 200  # Out of range
        }

        report = engine.validate_config_changes(invalid_config)

        assert not report.is_valid, "Should be invalid for bad config"
        assert len(report.errors) > 0, "Should have errors"

        print("✓ ValidationEngine basic tests passed")
        return True
    except Exception as e:
        print(f"✗ ValidationEngine test failed: {e}")
        return False

def test_type_validation():
    """Test type validation specifically."""
    try:
        from app.utils.validation import ValidationEngine

        engine = ValidationEngine()

        # Test type errors with auto-corrections
        config = {
            'detection_confidence_threshold': "0.7",  # String -> float
            'camera_width': 1280.5,  # Float -> int
            'enable_roi': "true"  # String -> bool
        }

        report = engine.validate_config_changes(config)

        # Should have auto-corrections
        assert 'detection_confidence_threshold' in report.auto_corrections
        assert report.auto_corrections['detection_confidence_threshold'] == 0.7

        assert 'camera_width' in report.auto_corrections
        assert report.auto_corrections['camera_width'] == 1280

        print("✓ Type validation tests passed")
        return True
    except Exception as e:
        print(f"✗ Type validation test failed: {e}")
        return False

def test_range_validation():
    """Test range validation specifically."""
    try:
        from app.utils.validation import ValidationEngine

        engine = ValidationEngine()

        # Test out-of-range values
        config = {
            'detection_confidence_threshold': 1.5,  # > 1.0
            'camera_fps': 200,  # > 120
            'gemini_temperature': -0.1  # < 0.0
        }

        report = engine.validate_config_changes(config)

        # Should have errors and auto-corrections
        assert not report.is_valid
        assert 'detection_confidence_threshold' in report.errors
        assert 'camera_fps' in report.errors
        assert 'gemini_temperature' in report.errors

        # Should suggest clamped values
        assert report.auto_corrections['detection_confidence_threshold'] == 1.0
        assert report.auto_corrections['camera_fps'] == 120
        assert report.auto_corrections['gemini_temperature'] == 0.0

        print("✓ Range validation tests passed")
        return True
    except Exception as e:
        print(f"✗ Range validation test failed: {e}")
        return False

def test_dependency_validation():
    """Test dependency validation specifically."""
    try:
        from app.utils.validation import ValidationEngine

        engine = ValidationEngine()

        # Test ROI dependency
        config = {
            'enable_roi': True,
            'roi_x': 100,
            'roi_width': 1500,  # Too wide for camera
            'roi_height': 500,
            'camera_width': 1280,
            'camera_height': 720
        }

        report = engine.validate_config_changes(config)

        # Should detect ROI extending beyond camera bounds
        assert not report.is_valid
        assert 'roi_x' in report.errors

        print("✓ Dependency validation tests passed")
        return True
    except Exception as e:
        print(f"✗ Dependency validation test failed: {e}")
        return False

def test_performance():
    """Test validation performance."""
    try:
        from app.utils.validation import ValidationEngine
        import time

        engine = ValidationEngine()

        # Large config for performance test
        config = {
            'detection_confidence_threshold': 0.6,
            'detection_iou_threshold': 0.4,
            'camera_width': 1920,
            'camera_height': 1080,
            'camera_fps': 30,
            'gemini_timeout': 45,
            'gemini_temperature': 0.8,
            'use_gpu': True,
            'max_memory_usage_mb': 4096,
            'app_theme': 'Dark',
            'enable_roi': False
        }

        start_time = time.time()
        report = engine.validate_config_changes(config)
        elapsed_time = (time.time() - start_time) * 1000

        # Should complete within 500ms requirement
        assert elapsed_time < 500, f"Validation took {elapsed_time:.2f}ms (> 500ms)"
        assert report.validation_time_ms < 500, f"Reported time {report.validation_time_ms:.2f}ms (> 500ms)"

        print(f"✓ Performance test passed ({elapsed_time:.2f}ms)")
        return True
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running ValidationEngine Tests")
    print("=" * 40)

    tests = [
        test_basic_import,
        test_validation_report,
        test_validation_engine,
        test_type_validation,
        test_range_validation,
        test_dependency_validation,
        test_performance
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1

    print("\n" + "=" * 40)
    print(f"Tests completed: {passed} passed, {failed} failed")

    if failed == 0:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())