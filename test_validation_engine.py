#!/usr/bin/env python3
"""Comprehensive test suite for the Settings Validation Engine.

Tests all five validation stages:
1. Type validation
2. Range validation
3. Dependency validation
4. Resource validation
5. Service compatibility

Also tests caching, performance, and edge cases.
"""

import pytest
import asyncio
import time
import tempfile
import os
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# Import the validation engine components
from app.utils.validation import (
    ValidationEngine, ValidationReport, ValidationError,
    EnvironmentValidator, InputValidator
)


class TestValidationReport:
    """Test ValidationReport dataclass functionality."""

    def test_initialization(self):
        """Test ValidationReport initialization."""
        report = ValidationReport(is_valid=True)

        assert report.is_valid is True
        assert report.errors == {}
        assert report.warnings == {}
        assert report.auto_corrections == {}
        assert report.validation_time_ms == 0.0
        assert report.stages_completed == []
        assert report.cache_hits == 0

    def test_add_error(self):
        """Test adding errors to report."""
        report = ValidationReport(is_valid=True)

        report.add_error("test_field", "Test error message")

        assert report.is_valid is False
        assert "test_field" in report.errors
        assert "Test error message" in report.errors["test_field"]

    def test_add_warning(self):
        """Test adding warnings to report."""
        report = ValidationReport(is_valid=True)

        report.add_warning("test_field", "Test warning message")

        assert report.is_valid is True  # Warnings don't affect validity
        assert "test_field" in report.warnings
        assert "Test warning message" in report.warnings["test_field"]

    def test_suggest_correction(self):
        """Test auto-correction suggestions."""
        report = ValidationReport(is_valid=True)

        report.suggest_correction("test_field", "corrected_value")

        assert report.auto_corrections["test_field"] == "corrected_value"

    def test_field_status(self):
        """Test field status checking."""
        report = ValidationReport(is_valid=True)

        # Valid field
        assert report.get_field_status("valid_field") == "valid"

        # Warning field
        report.add_warning("warning_field", "Warning")
        assert report.get_field_status("warning_field") == "warning"

        # Error field
        report.add_error("error_field", "Error")
        assert report.get_field_status("error_field") == "error"


class TestValidationEngine:
    """Test ValidationEngine functionality."""

    @pytest.fixture
    def engine(self):
        """Create ValidationEngine instance for testing."""
        return ValidationEngine()

    @pytest.fixture
    def valid_config(self):
        """Valid configuration for testing."""
        return {
            'detection_confidence_threshold': 0.5,
            'detection_iou_threshold': 0.45,
            'camera_width': 1280,
            'camera_height': 720,
            'camera_fps': 30,
            'gemini_api_key': 'AIza1234567890123456789012345678901234567',
            'gemini_model': 'gemini-1.5-flash',
            'gemini_timeout': 30,
            'gemini_temperature': 0.7,
            'use_gpu': True,
            'max_memory_usage_mb': 2048,
            'app_theme': 'Dark',
            'enable_roi': False,
            'enable_ai_analysis': True,
            'data_dir': 'data',
            'models_dir': 'data/models'
        }

    def test_type_validation_success(self, engine, valid_config):
        """Test successful type validation."""
        report = engine.validate_config_changes(valid_config)

        # Should have no type errors for valid config
        type_errors = [field for field in report.errors.keys()
                      if any('Invalid type' in msg for msg in report.errors[field])]
        assert len(type_errors) == 0

    def test_type_validation_failures(self, engine):
        """Test type validation failures and auto-corrections."""
        invalid_config = {
            'detection_confidence_threshold': "0.5",  # String instead of float
            'camera_width': 1280.5,  # Float instead of int
            'enable_roi': "true",  # String instead of bool
            'use_gpu': 1  # Int instead of bool (but convertible)
        }

        report = engine.validate_config_changes(invalid_config)

        # Should have errors
        assert not report.is_valid
        assert 'detection_confidence_threshold' in report.errors
        assert 'camera_width' in report.errors
        assert 'enable_roi' in report.errors

        # Should have auto-corrections
        assert 'detection_confidence_threshold' in report.auto_corrections
        assert report.auto_corrections['detection_confidence_threshold'] == 0.5
        assert 'camera_width' in report.auto_corrections
        assert report.auto_corrections['camera_width'] == 1280

    def test_range_validation(self, engine):
        """Test range validation."""
        out_of_range_config = {
            'detection_confidence_threshold': 1.5,  # > 1.0
            'camera_fps': 200,  # > 120
            'gemini_temperature': -0.1,  # < 0.0
            'camera_brightness': 150  # > 100
        }

        report = engine.validate_config_changes(out_of_range_config)

        assert not report.is_valid
        assert 'detection_confidence_threshold' in report.errors
        assert 'camera_fps' in report.errors
        assert 'gemini_temperature' in report.errors
        assert 'camera_brightness' in report.errors

        # Should suggest clamped values
        assert report.auto_corrections['detection_confidence_threshold'] == 1.0
        assert report.auto_corrections['camera_fps'] == 120
        assert report.auto_corrections['gemini_temperature'] == 0.0
        assert report.auto_corrections['camera_brightness'] == 100

    def test_dependency_validation_roi(self, engine):
        """Test ROI dependency validation."""
        roi_config = {
            'enable_roi': True,
            'roi_x': 100,
            'roi_y': 100,
            'roi_width': 1500,  # Extends beyond camera width
            'roi_height': 500,
            'camera_width': 1280,
            'camera_height': 720
        }

        report = engine.validate_config_changes(roi_config)

        assert not report.is_valid
        assert 'roi_x' in report.errors
        assert 'ROI extends beyond camera width' in report.errors['roi_x'][0]

    def test_dependency_validation_ai(self, engine):
        """Test AI dependency validation."""
        ai_config = {
            'enable_ai_analysis': True,
            'gemini_api_key': ''  # Empty API key
        }

        report = engine.validate_config_changes(ai_config)

        assert not report.is_valid
        assert 'gemini_api_key' in report.errors

    @patch('cv2.VideoCapture')
    def test_resource_validation_camera(self, mock_cv2, engine):
        """Test camera resource validation."""
        # Mock successful camera
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cv2.return_value = mock_cap

        config = {'last_webcam_index': 0}
        report = engine.validate_config_changes(config)

        # Should not have camera errors
        camera_errors = [field for field in report.errors.keys()
                        if 'camera' in field.lower()]
        assert len(camera_errors) == 0

    @patch('cv2.VideoCapture')
    def test_resource_validation_camera_failure(self, mock_cv2, engine):
        """Test camera resource validation failure."""
        # Mock failed camera
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.return_value = mock_cap

        config = {'last_webcam_index': 5}
        report = engine.validate_config_changes(config)

        assert not report.is_valid
        assert 'last_webcam_index' in report.errors

    def test_resource_validation_directory(self, engine):
        """Test directory resource validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'data_dir': temp_dir,
                'models_dir': os.path.join(temp_dir, 'models')
            }

            report = engine.validate_config_changes(config)

            # Should create directories and validate successfully
            assert os.path.exists(config['models_dir'])

    def test_service_compatibility_gemini(self, engine):
        """Test Gemini service compatibility."""
        config = {
            'enable_ai_analysis': True,
            'gemini_model': 'invalid-model-name'
        }

        report = engine.validate_config_changes(config)

        assert not report.is_valid
        assert 'gemini_model' in report.errors
        assert report.auto_corrections['gemini_model'] == 'gemini-1.5-flash'

    def test_caching_mechanism(self, engine):
        """Test validation caching."""
        config = {'last_webcam_index': 0}

        # First validation
        with patch('cv2.VideoCapture') as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cv2.return_value = mock_cap

            report1 = engine.validate_config_changes(config)
            first_call_count = mock_cv2.call_count

        # Second validation (should use cache)
        with patch('cv2.VideoCapture') as mock_cv2:
            report2 = engine.validate_config_changes(config)

            # Should not call CV2 again due to caching
            assert mock_cv2.call_count == 0
            assert report2.cache_hits > 0

    def test_performance_requirement(self, engine, valid_config):
        """Test that validation completes within 500ms requirement."""
        start_time = time.time()
        report = engine.validate_config_changes(valid_config)
        elapsed_time = (time.time() - start_time) * 1000

        # Should complete within 500ms
        assert elapsed_time < 500
        assert report.validation_time_ms < 500

    def test_all_stages_completed(self, engine, valid_config):
        """Test that all validation stages are completed."""
        report = engine.validate_config_changes(valid_config)

        expected_stages = [
            "type_validation",
            "range_validation",
            "dependency_validation",
            "resource_validation",
            "service_compatibility"
        ]

        assert all(stage in report.stages_completed for stage in expected_stages)

    def test_error_isolation(self, engine):
        """Test that errors in one stage don't prevent other stages."""
        config = {
            'detection_confidence_threshold': "invalid",  # Type error
            'camera_fps': 200,  # Range error
            'enable_roi': True,  # Will cause dependency error
            'roi_width': 0,
            'roi_height': 0
        }

        report = engine.validate_config_changes(config)

        # Should complete all stages despite errors
        assert len(report.stages_completed) == 5
        assert not report.is_valid

    def test_empty_config(self, engine):
        """Test validation with empty configuration."""
        report = engine.validate_config_changes({})

        # Should complete successfully with no errors
        assert report.is_valid
        assert len(report.stages_completed) == 5

    @pytest.mark.asyncio
    async def test_async_validation(self, engine):
        """Test async validation functionality."""
        config = {
            'enable_ai_analysis': True,
            'gemini_api_key': 'AIza1234567890123456789012345678901234567'
        }

        report = await engine.validate_config_changes_async(config)

        assert isinstance(report, ValidationReport)
        assert len(report.stages_completed) == 5

    def test_cache_ttl(self, engine):
        """Test cache TTL functionality."""
        # Set very short TTL for testing
        engine._cache_ttl = 0.1

        config = {'preferred_model': 'yolo12n'}

        # First validation
        report1 = engine.validate_config_changes(config)

        # Wait for cache to expire
        time.sleep(0.2)

        # Second validation (cache should be expired)
        report2 = engine.validate_config_changes(config)

        # Cache should not be used
        assert report2.cache_hits == 0

    def test_clear_cache(self, engine):
        """Test cache clearing."""
        config = {'preferred_model': 'yolo12n'}

        # Populate cache
        engine.validate_config_changes(config)
        assert len(engine._validation_cache) > 0

        # Clear cache
        engine.clear_cache()
        assert len(engine._validation_cache) == 0
        assert len(engine._cache_timestamps) == 0


class TestValidationIntegration:
    """Integration tests for the validation engine."""

    def test_real_world_config_changes(self):
        """Test validation with realistic configuration changes."""
        engine = ValidationEngine()

        # Simulate changing camera resolution
        config_changes = {
            'camera_width': 1920,
            'camera_height': 1080,
            'camera_fps': 60,
            'detection_confidence_threshold': 0.7,
            'enable_roi': True,
            'roi_x': 100,
            'roi_y': 100,
            'roi_width': 1720,  # Within new camera bounds
            'roi_height': 880
        }

        report = engine.validate_config_changes(config_changes)

        # Should be valid (assuming resources are available)
        roi_valid = 'roi_x' not in report.errors and 'roi_y' not in report.errors
        assert roi_valid  # ROI should fit in new camera resolution

    def test_batch_validation_performance(self):
        """Test performance with multiple field changes."""
        engine = ValidationEngine()

        # Large configuration change
        large_config = {
            'detection_confidence_threshold': 0.6,
            'detection_iou_threshold': 0.4,
            'camera_width': 1920,
            'camera_height': 1080,
            'camera_fps': 30,
            'camera_brightness': 10,
            'camera_contrast': 5,
            'camera_saturation': 0,
            'gemini_timeout': 45,
            'gemini_temperature': 0.8,
            'gemini_max_tokens': 4096,
            'use_gpu': True,
            'max_memory_usage_mb': 4096,
            'batch_size': 16,
            'target_fps': 60,
            'app_theme': 'Light',
            'enable_logging': True,
            'enable_roi': False,
            'enable_ai_analysis': False
        }

        start_time = time.time()
        report = engine.validate_config_changes(large_config)
        elapsed_time = (time.time() - start_time) * 1000

        # Should handle large configs efficiently
        assert elapsed_time < 500
        assert len(report.stages_completed) == 5


class TestValidationEdgeCases:
    """Test edge cases and error conditions."""

    def test_none_values(self):
        """Test handling of None values."""
        engine = ValidationEngine()

        config = {
            'detection_confidence_threshold': None,
            'camera_width': None,
            'gemini_api_key': None
        }

        report = engine.validate_config_changes(config)

        # Should handle None values gracefully
        assert not report.is_valid

    def test_extreme_values(self):
        """Test handling of extreme values."""
        engine = ValidationEngine()

        config = {
            'detection_confidence_threshold': float('inf'),
            'camera_width': -1000,
            'gemini_timeout': 0,
            'max_memory_usage_mb': 999999
        }

        report = engine.validate_config_changes(config)

        # Should reject extreme values
        assert not report.is_valid
        assert len(report.errors) > 0

    def test_unicode_strings(self):
        """Test handling of Unicode strings."""
        engine = ValidationEngine()

        config = {
            'app_theme': '🌙 Dark Mode',
            'language': 'español',
            'camera_device_name': 'Camera™ 📹'
        }

        report = engine.validate_config_changes(config)

        # Should handle Unicode gracefully
        assert isinstance(report, ValidationReport)

    def test_concurrent_validation(self):
        """Test thread safety with concurrent validations."""
        import threading

        engine = ValidationEngine()
        results = []

        def validate_config():
            config = {'detection_confidence_threshold': 0.5}
            report = engine.validate_config_changes(config)
            results.append(report.is_valid)

        # Run multiple validations concurrently
        threads = [threading.Thread(target=validate_config) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All validations should succeed
        assert all(results)
        assert len(results) == 10


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])