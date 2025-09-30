"""
Test script to verify reference_manager integration fix.

This tests that capture_reference_sync properly adds references
so that get_all_references() returns them correctly.
"""
import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add the current directory to path for app imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from app.services.reference_manager import ReferenceImageManager
from app.backends.yolo_backend import YoloBackend
from app.config.settings import load_config


def test_reference_manager_sync_capture():
    """Test that capture_reference_sync adds references to the registry."""

    print("=" * 70)
    print("Testing ReferenceImageManager Integration Fix")
    print("=" * 70)

    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config()
    print(f"   ✓ Config loaded: {config.data_dir}")

    # Initialize YOLO backend
    print("\n2. Initializing YOLO backend...")
    yolo_config = {
        'model_size': getattr(config, 'model_size', 'yolo12n'),
        'confidence_threshold': getattr(config, 'min_detection_confidence', 0.5),
        'device': 'cpu'
    }
    yolo_backend = YoloBackend(yolo_config)

    # Load the model
    if not yolo_backend.load_priority_model(config.models_dir, yolo_config['model_size']):
        print("   ✗ YOLO backend failed to load, cannot continue test")
        return False

    print(f"   ✓ YOLO backend loaded: {yolo_backend.is_loaded}")

    # Initialize reference manager
    print("\n3. Initializing ReferenceImageManager...")
    reference_data_dir = os.path.join(config.data_dir, "reference_data")
    os.makedirs(reference_data_dir, exist_ok=True)

    reference_manager = ReferenceImageManager(
        yolo_backend=yolo_backend,
        data_dir=reference_data_dir,
        max_references=100
    )
    print(f"   ✓ ReferenceImageManager initialized")

    # Check initial state
    print("\n4. Checking initial state...")
    initial_refs = reference_manager.get_all_references()
    print(f"   Initial reference count: {len(initial_refs)}")

    # Create a test image (640x480 blue image with white circle)
    print("\n5. Creating test reference image...")
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:, :] = (255, 100, 50)  # Blue background
    cv2.circle(test_image, (320, 240), 100, (255, 255, 255), -1)  # White circle
    print(f"   ✓ Test image created: {test_image.shape}")

    # Capture reference using synchronous method (THE FIX)
    print("\n6. Capturing reference using capture_reference_sync()...")
    try:
        reference_id = reference_manager.capture_reference_sync(
            test_image,
            confidence_threshold=0.5
        )
        print(f"   ✓ Reference captured: {reference_id}")
    except Exception as e:
        print(f"   ✗ Failed to capture reference: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify reference was added to registry
    print("\n7. Verifying reference in registry...")
    all_refs = reference_manager.get_all_references()
    print(f"   Total references: {len(all_refs)}")

    if len(all_refs) == 0:
        print("   ✗ FAILED: get_all_references() returned empty!")
        return False

    # Find our reference
    found = False
    for ref in all_refs:
        if ref['reference_id'] == reference_id:
            found = True
            print(f"   ✓ Reference found in registry:")
            print(f"     - ID: {ref['reference_id']}")
            print(f"     - Timestamp: {ref['timestamp']}")
            print(f"     - Detection count: {ref['detection_count']}")
            print(f"     - File size: {ref['file_size_kb']:.1f} KB")
            break

    if not found:
        print(f"   ✗ FAILED: Reference {reference_id} not found in registry!")
        return False

    # Retrieve reference data
    print("\n8. Retrieving reference data...")
    try:
        ref_data = reference_manager.get_reference(reference_id)
        print(f"   ✓ Reference retrieved successfully")
        print(f"     - Detections: {ref_data['detection_count']}")
        print(f"     - Thumbnail shape: {ref_data['thumbnail'].shape}")
    except Exception as e:
        print(f"   ✗ Failed to retrieve reference: {e}")
        return False

    # Cleanup
    print("\n9. Cleaning up test reference...")
    reference_manager._delete_reference(reference_id)
    print(f"   ✓ Test reference deleted")

    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED - Integration fix is working correctly!")
    print("=" * 70)

    return True


if __name__ == "__main__":
    try:
        success = test_reference_manager_sync_capture()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)