#!/usr/bin/env python3
"""Example usage of the Settings Validation Engine.

This script demonstrates how to integrate the ValidationEngine with the
Webcam Master Checker application to validate configuration changes
before applying them to the running system.
"""

import sys
import os
import json
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.utils.validation import ValidationEngine, ValidationReport
from app.config.settings import Config, load_config, save_config


class SettingsValidator:
    """Integration wrapper for the ValidationEngine with the application."""

    def __init__(self):
        self.engine = ValidationEngine()
        self.current_config = None

    def load_current_config(self, config_path: str = "config.json") -> Config:
        """Load the current configuration."""
        self.current_config = load_config(config_path)
        return self.current_config

    def validate_settings_changes(self, new_settings: Dict[str, Any]) -> ValidationReport:
        """Validate settings changes against current configuration.

        Args:
            new_settings: Dictionary of setting changes to validate

        Returns:
            ValidationReport: Comprehensive validation results
        """
        # Get current config as dict for comparison
        current_dict = self.current_config.to_dict() if self.current_config else {}

        # Merge new settings with current for full validation context
        merged_settings = {**current_dict, **new_settings}

        # Run validation
        return self.engine.validate_config_changes(merged_settings, current_dict)

    def apply_validated_changes(self, new_settings: Dict[str, Any],
                              config_path: str = "config.json") -> tuple[bool, ValidationReport]:
        """Validate and apply settings changes if valid.

        Args:
            new_settings: Settings changes to apply
            config_path: Path to config file

        Returns:
            tuple: (success, validation_report)
        """
        # Validate changes
        report = self.validate_settings_changes(new_settings)

        if not report.is_valid:
            print("❌ Validation failed - changes not applied")
            return False, report

        # Apply changes to current config
        if self.current_config:
            for key, value in new_settings.items():
                if hasattr(self.current_config, key):
                    setattr(self.current_config, key, value)
                else:
                    # Add to extra fields
                    self.current_config.extra[key] = value

            # Save updated config
            save_config(self.current_config, config_path)
            print("✅ Settings applied successfully")
            return True, report
        else:
            print("❌ No current config loaded")
            return False, report

    def get_auto_corrected_settings(self, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Get auto-corrected version of settings.

        Args:
            new_settings: Original settings

        Returns:
            dict: Settings with auto-corrections applied
        """
        report = self.validate_settings_changes(new_settings)

        # Apply auto-corrections
        corrected = new_settings.copy()
        for field, correction in report.auto_corrections.items():
            corrected[field] = correction

        return corrected

    def print_validation_report(self, report: ValidationReport) -> None:
        """Print a user-friendly validation report."""
        print(f"\n📊 Validation Report")
        print(f"Valid: {'✅ Yes' if report.is_valid else '❌ No'}")
        print(f"Validation Time: {report.validation_time_ms:.2f}ms")
        print(f"Stages Completed: {len(report.stages_completed)}/5")
        print(f"Cache Hits: {report.cache_hits}")

        if report.errors:
            print(f"\n🚫 Errors ({len(report.errors)} fields):")
            for field, messages in report.errors.items():
                print(f"  • {field}:")
                for msg in messages:
                    print(f"    - {msg}")

        if report.warnings:
            print(f"\n⚠️  Warnings ({len(report.warnings)} fields):")
            for field, messages in report.warnings.items():
                print(f"  • {field}:")
                for msg in messages:
                    print(f"    - {msg}")

        if report.auto_corrections:
            print(f"\n🔧 Auto-Corrections ({len(report.auto_corrections)} fields):")
            for field, value in report.auto_corrections.items():
                print(f"  • {field}: {value}")


def example_camera_settings_change():
    """Example: Changing camera settings."""
    print("🎥 Example: Camera Settings Change")
    print("=" * 50)

    validator = SettingsValidator()
    validator.load_current_config()

    # Simulate user changing camera settings
    new_camera_settings = {
        'camera_width': 1920,
        'camera_height': 1080,
        'camera_fps': 60,
        'camera_brightness': 10
    }

    print("Proposed camera settings:")
    for key, value in new_camera_settings.items():
        print(f"  {key}: {value}")

    # Validate changes
    report = validator.validate_settings_changes(new_camera_settings)
    validator.print_validation_report(report)

    return report.is_valid


def example_detection_settings_change():
    """Example: Changing detection settings."""
    print("\n🔍 Example: Detection Settings Change")
    print("=" * 50)

    validator = SettingsValidator()
    validator.load_current_config()

    # Simulate user changing detection settings
    new_detection_settings = {
        'detection_confidence_threshold': 0.8,
        'detection_iou_threshold': 0.4,
        'enable_roi': True,
        'roi_x': 100,
        'roi_y': 100,
        'roi_width': 1720,  # This might be too wide
        'roi_height': 880
    }

    print("Proposed detection settings:")
    for key, value in new_detection_settings.items():
        print(f"  {key}: {value}")

    # Validate changes
    report = validator.validate_settings_changes(new_detection_settings)
    validator.print_validation_report(report)

    if not report.is_valid and report.auto_corrections:
        print("\n🔧 Applying auto-corrections...")
        corrected_settings = validator.get_auto_corrected_settings(new_detection_settings)
        print("Corrected settings:")
        for key, value in corrected_settings.items():
            if key in report.auto_corrections:
                print(f"  {key}: {value} (corrected)")
            else:
                print(f"  {key}: {value}")

    return report.is_valid


def example_ai_settings_change():
    """Example: Changing AI settings."""
    print("\n🤖 Example: AI Settings Change")
    print("=" * 50)

    validator = SettingsValidator()
    validator.load_current_config()

    # Simulate user enabling AI with invalid settings
    new_ai_settings = {
        'enable_ai_analysis': True,
        'gemini_api_key': 'invalid_key',  # Invalid format
        'gemini_model': 'gpt-4',  # Wrong model
        'gemini_temperature': 1.5,  # Out of range
        'gemini_timeout': 300
    }

    print("Proposed AI settings:")
    for key, value in new_ai_settings.items():
        print(f"  {key}: {value}")

    # Validate changes
    report = validator.validate_settings_changes(new_ai_settings)
    validator.print_validation_report(report)

    return report.is_valid


def example_invalid_types():
    """Example: Handling invalid data types."""
    print("\n🔄 Example: Invalid Data Types")
    print("=" * 50)

    validator = SettingsValidator()
    validator.load_current_config()

    # Simulate settings with wrong types
    invalid_settings = {
        'detection_confidence_threshold': "0.7",  # String instead of float
        'camera_width': 1280.5,  # Float instead of int
        'enable_roi': "true",  # String instead of bool
        'camera_fps': "60"  # String instead of int
    }

    print("Settings with wrong types:")
    for key, value in invalid_settings.items():
        print(f"  {key}: {value} ({type(value).__name__})")

    # Validate changes
    report = validator.validate_settings_changes(invalid_settings)
    validator.print_validation_report(report)

    if report.auto_corrections:
        print("\n✨ Auto-corrected values:")
        for key, value in report.auto_corrections.items():
            print(f"  {key}: {value} ({type(value).__name__})")

    return report.is_valid


def example_performance_test():
    """Example: Performance test with large configuration."""
    print("\n⚡ Example: Performance Test")
    print("=" * 50)

    validator = SettingsValidator()
    validator.load_current_config()

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
        'enable_roi': True,
        'roi_x': 100,
        'roi_y': 100,
        'roi_width': 1720,
        'roi_height': 880,
        'gemini_timeout': 45,
        'gemini_temperature': 0.8,
        'gemini_max_tokens': 4096,
        'enable_ai_analysis': False,
        'use_gpu': True,
        'max_memory_usage_mb': 4096,
        'batch_size': 16,
        'target_fps': 60,
        'app_theme': 'Light',
        'enable_logging': True
    }

    print(f"Validating {len(large_config)} settings...")

    import time
    start_time = time.time()
    report = validator.validate_settings_changes(large_config)
    elapsed_time = (time.time() - start_time) * 1000

    print(f"Manual timing: {elapsed_time:.2f}ms")
    validator.print_validation_report(report)

    # Performance requirement: < 500ms
    performance_ok = elapsed_time < 500
    print(f"\n⚡ Performance: {'✅ PASS' if performance_ok else '❌ FAIL'} "
          f"({elapsed_time:.2f}ms < 500ms)")

    return performance_ok


def main():
    """Run all examples."""
    print("🔧 Settings Validation Engine Examples")
    print("=" * 60)

    examples = [
        example_camera_settings_change,
        example_detection_settings_change,
        example_ai_settings_change,
        example_invalid_types,
        example_performance_test
    ]

    results = []
    for example in examples:
        try:
            result = example()
            results.append(result)
        except Exception as e:
            print(f"❌ Example {example.__name__} failed: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print("📊 Example Results Summary:")
    for i, (example, result) in enumerate(zip(examples, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {i+1}. {example.__name__}: {status}")

    total_passed = sum(results)
    print(f"\n🎯 Overall: {total_passed}/{len(examples)} examples passed")

    return 0 if total_passed == len(examples) else 1


if __name__ == "__main__":
    sys.exit(main())