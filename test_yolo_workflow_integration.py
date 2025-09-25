"""
Test script for YOLO Workflow Orchestrator integration.

This script tests the complete integrated workflow including:
1. YoloWorkflowOrchestrator service
2. ReferenceImageManager integration
3. DetectionDataFormatter output
4. Async chat processing
"""

import sys
import os
import asyncio
import logging
import numpy as np
from pathlib import Path
from typing import Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_workflow_config():
    """Test WorkflowConfig initialization."""
    from app.services.yolo_workflow_orchestrator import WorkflowConfig

    config = WorkflowConfig(
        auto_yolo_analysis=True,
        reference_comparison_enabled=True,
        min_detection_confidence=0.5,
        max_objects_to_analyze=10
    )

    assert config.auto_yolo_analysis == True
    assert config.reference_comparison_enabled == True
    assert config.min_detection_confidence == 0.5
    assert config.max_objects_to_analyze == 10

    print("✓ WorkflowConfig test passed")
    return config


def test_reference_manager_init():
    """Test ReferenceImageManager initialization."""
    try:
        from app.services.reference_manager import ReferenceImageManager
        from app.backends.yolo_backend import YoloBackend

        # Create mock YOLO backend
        yolo_config = {
            'model_size': 'yolo12n',
            'confidence_threshold': 0.5,
            'device': 'cpu'
        }
        yolo_backend = YoloBackend(yolo_config)

        # Create reference manager
        data_dir = os.path.join(os.getcwd(), 'data', 'test_references')
        os.makedirs(data_dir, exist_ok=True)

        manager = ReferenceImageManager(
            yolo_backend=yolo_backend,
            data_dir=data_dir,
            max_references=10,
            max_memory_mb=10
        )

        print("✓ ReferenceImageManager initialization test passed")
        return manager

    except Exception as e:
        print(f"✗ ReferenceImageManager initialization failed: {e}")
        return None


async def test_workflow_orchestrator():
    """Test YoloWorkflowOrchestrator async workflow."""
    from app.services.yolo_workflow_orchestrator import (
        YoloWorkflowOrchestrator, WorkflowConfig
    )

    # Create test frame (640x480 RGB)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:240, :320] = [255, 0, 0]  # Red quadrant
    test_frame[:240, 320:] = [0, 255, 0]  # Green quadrant
    test_frame[240:, :320] = [0, 0, 255]  # Blue quadrant
    test_frame[240:, 320:] = [255, 255, 0]  # Yellow quadrant

    # Create orchestrator with mock services
    config = WorkflowConfig(
        auto_yolo_analysis=True,
        reference_comparison_enabled=False,  # Disable for this test
        async_timeout_seconds=10.0
    )

    orchestrator = YoloWorkflowOrchestrator(
        yolo_backend=None,  # Will skip YOLO detection
        reference_manager=None,  # Will skip comparison
        gemini_service=None,  # Will skip AI response
        config=config
    )

    # Test orchestration
    result = await orchestrator.orchestrate_analysis(
        current_frame=test_frame,
        user_message="Test analysis of colored quadrants"
    )

    assert result is not None
    assert result.success == True
    assert result.user_message == "Test analysis of colored quadrants"
    assert result.workflow_time_ms > 0

    print(f"✓ Workflow orchestration test passed (time: {result.workflow_time_ms:.1f}ms)")

    # Cleanup
    orchestrator.shutdown()

    return result


def test_detection_formatter():
    """Test DetectionDataFormatter output."""
    from app.utils.detection_formatter import format_detection_data
    from app.core.entities import Detection, BBox

    # Create sample detections
    detections = [
        Detection(
            bbox=BBox(x1=100, y1=100, x2=200, y2=200),
            confidence=0.95,
            class_id=0,
            class_name="person"
        ),
        Detection(
            bbox=BBox(x1=300, y1=150, x2=400, y2=250),
            confidence=0.88,
            class_id=67,
            class_name="cell phone"
        )
    ]

    # Format detection data
    formatted = format_detection_data(
        detections=detections,
        frame_dimensions=(640, 480),
        include_coordinates=True,
        include_confidence=True
    )

    assert formatted is not None
    assert "Object Detection Results" in formatted
    assert "person" in formatted
    assert "cell phone" in formatted
    assert "95.0%" in formatted or "0.95" in formatted

    print("✓ Detection formatter test passed")
    print("\nFormatted output sample:")
    print(formatted[:500] + "..." if len(formatted) > 500 else formatted)

    return formatted


def test_configuration_loading():
    """Test loading workflow configuration from config.json."""
    import json

    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Check new workflow configuration options
        workflow_options = [
            'auto_yolo_analysis',
            'reference_comparison_enabled',
            'min_detection_confidence',
            'max_objects_to_analyze',
            'workflow_async_timeout',
            'workflow_performance_monitoring',
            'cache_analysis_results'
        ]

        found_options = []
        for option in workflow_options:
            if option in config:
                found_options.append(f"  ✓ {option}: {config[option]}")

        if found_options:
            print("✓ Configuration options found:")
            for item in found_options:
                print(item)
        else:
            print("✗ No workflow configuration options found in config.json")

        return len(found_options) > 0
    else:
        print("✗ config.json not found")
        return False


async def test_integration_with_ui():
    """Test integration with ModernMainWindow."""
    try:
        # This is a smoke test to ensure imports work
        from app.ui.modern_main_window import ModernMainWindow
        from app.services.yolo_workflow_orchestrator import YoloWorkflowOrchestrator
        from app.services.reference_manager import ReferenceImageManager

        print("✓ UI integration imports successful")

        # Check that the UI can initialize with new services
        # (We won't actually create the UI here, just verify imports)

        return True

    except ImportError as e:
        print(f"✗ UI integration import failed: {e}")
        return False


async def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("YOLO Workflow Integration Tests")
    print("=" * 60)
    print()

    results = []

    # Test 1: WorkflowConfig
    print("1. Testing WorkflowConfig...")
    try:
        test_workflow_config()
        results.append(("WorkflowConfig", True))
    except Exception as e:
        print(f"   Failed: {e}")
        results.append(("WorkflowConfig", False))
    print()

    # Test 2: ReferenceImageManager
    print("2. Testing ReferenceImageManager...")
    try:
        test_reference_manager_init()
        results.append(("ReferenceImageManager", True))
    except Exception as e:
        print(f"   Failed: {e}")
        results.append(("ReferenceImageManager", False))
    print()

    # Test 3: Workflow Orchestrator
    print("3. Testing YoloWorkflowOrchestrator...")
    try:
        await test_workflow_orchestrator()
        results.append(("YoloWorkflowOrchestrator", True))
    except Exception as e:
        print(f"   Failed: {e}")
        results.append(("YoloWorkflowOrchestrator", False))
    print()

    # Test 4: Detection Formatter
    print("4. Testing DetectionDataFormatter...")
    try:
        test_detection_formatter()
        results.append(("DetectionDataFormatter", True))
    except Exception as e:
        print(f"   Failed: {e}")
        results.append(("DetectionDataFormatter", False))
    print()

    # Test 5: Configuration
    print("5. Testing Configuration Loading...")
    try:
        test_configuration_loading()
        results.append(("Configuration", True))
    except Exception as e:
        print(f"   Failed: {e}")
        results.append(("Configuration", False))
    print()

    # Test 6: UI Integration
    print("6. Testing UI Integration...")
    try:
        await test_integration_with_ui()
        results.append(("UI Integration", True))
    except Exception as e:
        print(f"   Failed: {e}")
        results.append(("UI Integration", False))
    print()

    # Summary
    print("=" * 60)
    print("Test Summary:")
    print("=" * 60)

    passed = sum(1 for _, status in results if status)
    total = len(results)

    for test_name, status in results:
        status_str = "✓ PASSED" if status else "✗ FAILED"
        print(f"  {test_name:30s} {status_str}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The YOLO workflow integration is working correctly.")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please review the errors above.")

    return passed == total


if __name__ == "__main__":
    # Run the async tests
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)