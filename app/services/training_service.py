"""Training service for YOLO model training with custom objects."""

import logging
import json
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
import numpy as np
import cv2
from datetime import datetime
import random

logger = logging.getLogger(__name__)


def get_best_device() -> tuple[str, str]:
    """Detect the best available device for YOLO training.

    Returns:
        Tuple of (device_string, device_name):
        - device_string: 'cuda', 'mps', or 'cpu' for PyTorch/YOLO
        - device_name: Human-readable device name
    """
    try:
        import torch

        # Check for NVIDIA CUDA GPU
        if torch.cuda.is_available():
            try:
                device_name = torch.cuda.get_device_name(0)
                return 'cuda', device_name
            except Exception as e:
                logger.warning(f"CUDA available but failed to get device name: {e}")
                return 'cuda', 'CUDA GPU'

        # Check for Apple Metal Performance Shaders (MPS)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps', 'Apple Metal (MPS)'

        # Fallback to CPU
        return 'cpu', 'CPU'

    except ImportError:
        logger.warning("PyTorch not available for device detection, defaulting to CPU")
        return 'cpu', 'CPU'
    except Exception as e:
        logger.error(f"Error detecting device: {e}")
        return 'cpu', 'CPU'


def get_device_memory_info(device: str) -> Optional[Dict[str, Any]]:
    """Get memory information for the specified device.

    Args:
        device: Device string ('cuda', 'mps', or 'cpu')

    Returns:
        Dictionary with memory info or None if unavailable
    """
    try:
        import torch

        if device == 'cuda' and torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)

            return {
                'total_mb': total_memory / (1024 ** 2),
                'allocated_mb': allocated / (1024 ** 2),
                'reserved_mb': reserved / (1024 ** 2),
                'free_mb': (total_memory - reserved) / (1024 ** 2)
            }

        return None

    except Exception as e:
        logger.warning(f"Could not get memory info for {device}: {e}")
        return None


class ImageAugmentor:
    """Image augmentation utilities for training data enhancement.

    Augmentation methods are categorized into two types:

    1. DETERMINISTIC methods - produce the same result every time, applied ONCE:
       - horizontal_flip, vertical_flip, edge_enhance

    2. STOCHASTIC methods - use randomization, applied multiple times (augmentation_factor):
       - rotation, scaling, translation, brightness, contrast, saturation, hue,
         gaussian_blur, motion_blur, gaussian_noise, salt_pepper_noise, sharpness
    """

    # Deterministic augmentation methods (same output every time)
    DETERMINISTIC_METHODS = {'horizontal_flip', 'vertical_flip', 'edge_enhance'}

    # Stochastic augmentation methods (random parameters, different each time)
    STOCHASTIC_METHODS = {
        'rotation', 'scaling', 'translation', 'brightness', 'contrast',
        'saturation', 'hue', 'gaussian_blur', 'motion_blur', 'gaussian_noise',
        'salt_pepper_noise', 'sharpness'
    }

    # Spatial augmentation methods that change object position/size and require bbox recalculation
    SPATIAL_AUGMENTATIONS = {'rotation', 'scaling', 'translation'}

    @staticmethod
    def augment_horizontal_flip(image: np.ndarray) -> np.ndarray:
        """Apply horizontal flip (mirror image).

        Args:
            image: Input image

        Returns:
            Horizontally flipped image
        """
        return cv2.flip(image, 1)

    @staticmethod
    def augment_vertical_flip(image: np.ndarray) -> np.ndarray:
        """Apply vertical flip.

        Args:
            image: Input image

        Returns:
            Vertically flipped image
        """
        return cv2.flip(image, 0)

    @staticmethod
    def augment_rotation(image: np.ndarray, angle: Optional[float] = None,
                        return_bbox: bool = False) -> tuple[np.ndarray, Optional[tuple]]:
        """Apply rotation with random or specified angle.

        Args:
            image: Input image
            angle: Rotation angle in degrees (if None, randomly chosen from -30 to +30)
            return_bbox: If True, returns (rotated_image, bbox), else returns rotated_image

        Returns:
            If return_bbox=False: Rotated image
            If return_bbox=True: Tuple of (rotated_image, bbox) where bbox is (cx, cy, w, h) normalized
        """
        if angle is None:
            angle = random.uniform(-30, 30)

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new dimensions to avoid cropping
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust transformation matrix
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(image, matrix, (new_w, new_h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))

        if return_bbox:
            # Calculate bbox for rotated image
            # Original object filled the frame (w x h), now canvas expanded to (new_w x new_h)
            # Object is still centered but smaller relative to new canvas
            bbox_width = w / new_w
            bbox_height = h / new_h
            bbox = (0.5, 0.5, bbox_width, bbox_height)  # Still centered
            return rotated, bbox
        else:
            return rotated

    @staticmethod
    def augment_scaling(image: np.ndarray, scale_factor: Optional[float] = None,
                       return_bbox: bool = False) -> tuple[np.ndarray, Optional[tuple]]:
        """Apply scaling (zoom in/out).

        Args:
            image: Input image
            scale_factor: Scale factor (if None, randomly chosen from 0.8 to 1.2)
            return_bbox: If True, returns (scaled_image, bbox), else returns scaled_image

        Returns:
            If return_bbox=False: Scaled image
            If return_bbox=True: Tuple of (scaled_image, bbox) where bbox is (cx, cy, w, h) normalized
        """
        if scale_factor is None:
            scale_factor = random.uniform(0.8, 1.2)

        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)

        # Resize image
        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Calculate bbox based on scaling operation
        if scale_factor < 1.0:
            # Scaled down and padded - object is smaller
            top = (h - new_h) // 2
            bottom = h - new_h - top
            left = (w - new_w) // 2
            right = w - new_w - left
            scaled = cv2.copyMakeBorder(scaled, top, bottom, left, right,
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # Object occupies only the center portion
            bbox_width = scale_factor
            bbox_height = scale_factor
            bbox = (0.5, 0.5, bbox_width, bbox_height)
        else:
            # Scaled up and cropped - object fills frame but some parts are outside
            top = (new_h - h) // 2
            left = (new_w - w) // 2
            scaled = scaled[top:top+h, left:left+w]
            # Object still fills the frame after crop
            bbox = (0.5, 0.5, 1.0, 1.0)

        if return_bbox:
            return scaled, bbox
        else:
            return scaled

    @staticmethod
    def augment_translation(image: np.ndarray, tx: Optional[int] = None,
                           ty: Optional[int] = None, return_bbox: bool = False) -> tuple[np.ndarray, Optional[tuple]]:
        """Apply translation (shift).

        Args:
            image: Input image
            tx: Translation in x direction (if None, random within ±10% of width)
            ty: Translation in y direction (if None, random within ±10% of height)
            return_bbox: If True, returns (translated_image, bbox), else returns translated_image

        Returns:
            If return_bbox=False: Translated image
            If return_bbox=True: Tuple of (translated_image, bbox) where bbox is (cx, cy, w, h) normalized
        """
        h, w = image.shape[:2]

        if tx is None:
            tx = int(random.uniform(-0.1, 0.1) * w)
        if ty is None:
            ty = int(random.uniform(-0.1, 0.1) * h)

        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(image, matrix, (w, h),
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0))

        if return_bbox:
            # Calculate bbox after translation with proper clipping
            # When image is translated, parts move outside frame and get cut off

            # Normalized translation offsets
            tx_norm = tx / w
            ty_norm = ty / h

            # Calculate visible region after translation and clipping
            # Original object spans [0, 1] in both dimensions
            # After translation, it spans [tx_norm, 1+tx_norm] and [ty_norm, 1+ty_norm]
            # But the visible frame is [0, 1] x [0, 1], so we need to clip

            left = max(0.0, tx_norm)
            right = min(1.0, 1.0 + tx_norm)
            top = max(0.0, ty_norm)
            bottom = min(1.0, 1.0 + ty_norm)

            # Calculate final bbox (center and size of visible portion)
            bbox_width = right - left
            bbox_height = bottom - top
            bbox_cx = (left + right) / 2.0
            bbox_cy = (top + bottom) / 2.0

            bbox = (bbox_cx, bbox_cy, bbox_width, bbox_height)
            return translated, bbox
        else:
            return translated

    @staticmethod
    def augment_brightness(image: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
        """Apply brightness adjustment.

        Args:
            image: Input image
            factor: Brightness factor (if None, randomly chosen from 0.7 to 1.3)
                   1.0 = no change, <1.0 = darker, >1.0 = brighter

        Returns:
            Brightness-adjusted image
        """
        if factor is None:
            factor = random.uniform(0.7, 1.3)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def augment_contrast(image: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
        """Apply contrast adjustment.

        Args:
            image: Input image
            factor: Contrast factor (if None, randomly chosen from 0.7 to 1.3)
                   1.0 = no change, <1.0 = less contrast, >1.0 = more contrast

        Returns:
            Contrast-adjusted image
        """
        if factor is None:
            factor = random.uniform(0.7, 1.3)

        # Convert to LAB color space for better contrast adjustment
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        mean_l = lab[:, :, 0].mean()
        lab[:, :, 0] = np.clip((lab[:, :, 0] - mean_l) * factor + mean_l, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    @staticmethod
    def augment_saturation(image: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
        """Apply saturation adjustment.

        Args:
            image: Input image
            factor: Saturation factor (if None, randomly chosen from 0.7 to 1.3)
                   1.0 = no change, <1.0 = less saturated, >1.0 = more saturated

        Returns:
            Saturation-adjusted image
        """
        if factor is None:
            factor = random.uniform(0.7, 1.3)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def augment_hue(image: np.ndarray, shift: Optional[int] = None) -> np.ndarray:
        """Apply hue shift.

        Args:
            image: Input image
            shift: Hue shift in degrees (if None, randomly chosen from -20 to +20)

        Returns:
            Hue-shifted image
        """
        if shift is None:
            shift = random.randint(-20, 20)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def augment_gaussian_blur(image: np.ndarray, kernel_size: Optional[int] = None) -> np.ndarray:
        """Apply Gaussian blur.

        Args:
            image: Input image
            kernel_size: Blur kernel size (if None, randomly chosen from [3, 5, 7])

        Returns:
            Blurred image
        """
        if kernel_size is None:
            kernel_size = random.choice([3, 5, 7])

        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    @staticmethod
    def augment_motion_blur(image: np.ndarray, kernel_size: Optional[int] = None) -> np.ndarray:
        """Apply motion blur.

        Args:
            image: Input image
            kernel_size: Motion blur kernel size (if None, randomly chosen from [5, 7, 9])

        Returns:
            Motion-blurred image
        """
        if kernel_size is None:
            kernel_size = random.choice([5, 7, 9])

        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size

        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def augment_gaussian_noise(image: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        """Apply Gaussian noise.

        Args:
            image: Input image
            sigma: Standard deviation of noise (if None, randomly chosen from 5 to 15)

        Returns:
            Noisy image
        """
        if sigma is None:
            sigma = random.uniform(5, 15)

        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)

    @staticmethod
    def augment_salt_pepper_noise(image: np.ndarray, amount: Optional[float] = None) -> np.ndarray:
        """Apply salt and pepper noise.

        Args:
            image: Input image
            amount: Proportion of pixels to be noisy (if None, randomly chosen from 0.01 to 0.05)

        Returns:
            Noisy image
        """
        if amount is None:
            amount = random.uniform(0.01, 0.05)

        noisy = image.copy()
        h, w = image.shape[:2]

        # Salt (white) noise
        num_salt = int(amount * h * w / 2)
        coords = [np.random.randint(0, i - 1, num_salt) for i in (h, w)]
        noisy[coords[0], coords[1]] = 255

        # Pepper (black) noise
        num_pepper = int(amount * h * w / 2)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in (h, w)]
        noisy[coords[0], coords[1]] = 0

        return noisy

    @staticmethod
    def augment_sharpness(image: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
        """Apply sharpness enhancement.

        Args:
            image: Input image
            factor: Sharpness factor (if None, randomly chosen from 0.5 to 1.5)
                   Higher values = sharper image

        Returns:
            Sharpened image
        """
        if factor is None:
            factor = random.uniform(0.5, 1.5)

        # Sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]]) * factor

        sharpened = cv2.filter2D(image, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def augment_edge_enhance(image: np.ndarray) -> np.ndarray:
        """Apply edge enhancement.

        Args:
            image: Input image

        Returns:
            Edge-enhanced image
        """
        # Edge detection kernel
        kernel = np.array([[-1, -1, -1],
                          [-1, 8, -1],
                          [-1, -1, -1]])

        edges = cv2.filter2D(image, -1, kernel)
        enhanced = cv2.addWeighted(image, 1.0, edges, 0.3, 0)
        return np.clip(enhanced, 0, 255).astype(np.uint8)

    @staticmethod
    def get_augmentation_function(aug_type: str) -> Optional[Callable]:
        """Get augmentation function by name.

        Args:
            aug_type: Augmentation type name

        Returns:
            Augmentation function or None if not found
        """
        augmentation_map = {
            'horizontal_flip': ImageAugmentor.augment_horizontal_flip,
            'vertical_flip': ImageAugmentor.augment_vertical_flip,
            'rotation': ImageAugmentor.augment_rotation,
            'scaling': ImageAugmentor.augment_scaling,
            'translation': ImageAugmentor.augment_translation,
            'brightness': ImageAugmentor.augment_brightness,
            'contrast': ImageAugmentor.augment_contrast,
            'saturation': ImageAugmentor.augment_saturation,
            'hue': ImageAugmentor.augment_hue,
            'gaussian_blur': ImageAugmentor.augment_gaussian_blur,
            'motion_blur': ImageAugmentor.augment_motion_blur,
            'gaussian_noise': ImageAugmentor.augment_gaussian_noise,
            'salt_pepper_noise': ImageAugmentor.augment_salt_pepper_noise,
            'sharpness': ImageAugmentor.augment_sharpness,
            'edge_enhance': ImageAugmentor.augment_edge_enhance,
        }
        return augmentation_map.get(aug_type)


class TrainingObject:
    """Represents a training object with image and metadata."""

    def __init__(self, image: np.ndarray, label: str, bbox: Optional[tuple] = None,
                 object_id: Optional[str] = None):
        """Initialize training object.

        Args:
            image: Cropped object image
            label: Object class label
            bbox: Optional bounding box (x1, y1, x2, y2)
            object_id: Unique identifier
        """
        self.image = image
        self.label = label
        self.bbox = bbox
        self.object_id = object_id or self._generate_id()
        self.timestamp = datetime.now()

    def _generate_id(self) -> str:
        """Generate unique ID for object."""
        return f"obj_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'object_id': self.object_id,
            'label': self.label,
            'bbox': self.bbox,
            'timestamp': self.timestamp.isoformat()
        }


class TrainingService:
    """Service for managing object training dataset and model training."""

    def __init__(self, data_dir: str = "data/training", config: Optional[Dict[str, Any]] = None):
        """Initialize training service.

        Args:
            data_dir: Base directory for training data
            config: Optional configuration dictionary
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}

        self.objects: List[TrainingObject] = []
        self._load_objects()

    def add_object(self, image: np.ndarray, label: str, bbox: Optional[tuple] = None) -> TrainingObject:
        """Add object to training dataset.

        Args:
            image: Cropped object image
            label: Object class label
            bbox: Optional bounding box

        Returns:
            Created TrainingObject
        """
        obj = TrainingObject(image, label, bbox)
        self.objects.append(obj)

        # Save object image
        self._save_object_image(obj)
        self._save_metadata()

        logger.info(f"Added training object: {obj.label} (ID: {obj.object_id})")
        return obj

    def get_object(self, object_id: str) -> Optional[TrainingObject]:
        """Get object by ID.

        Args:
            object_id: Object identifier

        Returns:
            TrainingObject or None if not found
        """
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None

    def update_object(self, object_id: str, label: Optional[str] = None) -> bool:
        """Update object properties.

        Args:
            object_id: Object identifier
            label: New label (optional)

        Returns:
            True if updated successfully, False if object not found
        """
        obj = self.get_object(object_id)
        if not obj:
            return False

        if label is not None:
            old_label = obj.label
            obj.label = label
            logger.info(f"Updated object {object_id} label: {old_label} -> {label}")

        self._save_metadata()
        return True

    def delete_object(self, object_id: str) -> bool:
        """Delete object from dataset.

        Args:
            object_id: Object identifier

        Returns:
            True if deleted successfully, False if object not found
        """
        obj = self.get_object(object_id)
        if not obj:
            return False

        # Remove from list
        self.objects.remove(obj)

        # Delete image file
        img_path = self.data_dir / f"{object_id}.png"
        if img_path.exists():
            img_path.unlink()

        self._save_metadata()
        logger.info(f"Deleted training object: {object_id}")
        return True

    def get_all_objects(self) -> List[TrainingObject]:
        """Get all training objects.

        Returns:
            List of all TrainingObjects
        """
        return self.objects.copy()

    def get_object_count(self) -> Dict[str, int]:
        """Get count of objects.

        Returns:
            Dictionary with 'total' count
        """
        return {
            'total': len(self.objects)
        }

    def export_dataset(self, export_dir: str, format: str = 'yolo') -> bool:
        """Export all objects as training dataset.

        Args:
            export_dir: Directory to export dataset
            format: Dataset format ('yolo' or 'coco')

        Returns:
            True if exported successfully, False otherwise
        """
        try:
            export_path = Path(export_dir)
            export_path.mkdir(parents=True, exist_ok=True)

            all_objects = self.get_all_objects()
            if not all_objects:
                logger.warning("No objects to export")
                return False

            if format == 'yolo':
                return self._export_yolo_format(export_path, all_objects)
            else:
                logger.error(f"Unsupported format: {format}")
                return False

        except Exception as e:
            logger.error(f"Error exporting dataset: {e}")
            return False

    def _export_all_objects(self, export_dir: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Export all objects as training dataset in YOLO format.

        Args:
            export_dir: Directory to export dataset
            config: Optional configuration dictionary with augmentation settings

        Returns:
            True if exported successfully, False otherwise
        """
        try:
            export_path = Path(export_dir)
            export_path.mkdir(parents=True, exist_ok=True)

            all_objects = self.get_all_objects()
            if not all_objects:
                logger.warning("No objects to export")
                return False

            # Get augmentation settings from config
            enable_augmentation = False
            augmentation_factor = 3
            augmentation_types = ['horizontal_flip', 'rotation', 'brightness']

            if config:
                enable_augmentation = config.get('enable_augmentation', False)
                augmentation_factor = config.get('augmentation_factor', 3)
                augmentation_types = config.get('augmentation_types', augmentation_types)

            return self._export_yolo_format(
                export_path,
                all_objects,
                enable_augmentation=enable_augmentation,
                augmentation_factor=augmentation_factor,
                augmentation_types=augmentation_types
            )

        except Exception as e:
            logger.error(f"Error exporting all objects: {e}")
            return False

    def _apply_augmentations(self, image: np.ndarray, augmentation_types: List[str],
                            augmentation_factor: int) -> List[tuple[np.ndarray, str, Optional[tuple]]]:
        """Apply augmentations to an image with intelligent deterministic vs stochastic handling.

        This implements a hybrid augmentation approach:

        - DETERMINISTIC methods (horizontal_flip, vertical_flip, edge_enhance):
          Applied exactly ONCE, regardless of augmentation_factor, since they always
          produce the same result.

        - STOCHASTIC methods (rotation, brightness, noise, etc.):
          Applied augmentation_factor times, with each application using different
          random parameters to create diverse variations.

        - SPATIAL methods (rotation, scaling, translation):
          These change the object's position/size, so proper bbox calculation is performed.

        Args:
            image: Input image
            augmentation_types: List of augmentation type names to apply
            augmentation_factor: Number of times to apply STOCHASTIC methods
                                (Deterministic methods are always applied once)
                                Example: factor=3 means stochastic methods applied 3 times

        Returns:
            List of tuples (augmented_image, augmentation_identifier, bbox)
            where:
            - augmentation_identifier is formatted as "method_name_N" (e.g., "rotation_1")
              or "method_name" for deterministic methods
            - bbox is (cx, cy, w, h) in normalized coordinates, or None if unchanged
        """
        augmented_images = []

        # Apply each selected augmentation method
        for aug_type in augmentation_types:
            aug_func = ImageAugmentor.get_augmentation_function(aug_type)
            if aug_func:
                # Check if this is a spatial augmentation that needs bbox calculation
                is_spatial = aug_type in ImageAugmentor.SPATIAL_AUGMENTATIONS

                # Determine iteration count based on method type
                if aug_type in ImageAugmentor.DETERMINISTIC_METHODS:
                    # Deterministic: apply exactly once
                    iterations = 1
                    iteration_note = "deterministic"
                elif aug_type in ImageAugmentor.STOCHASTIC_METHODS:
                    # Stochastic: apply augmentation_factor times
                    iterations = augmentation_factor
                    iteration_note = "stochastic"
                else:
                    # Unknown method: treat as stochastic (safe default)
                    logger.warning(f"Unknown augmentation type '{aug_type}', treating as stochastic")
                    iterations = augmentation_factor
                    iteration_note = "unknown"

                # Apply augmentation 'iterations' times
                for i in range(iterations):
                    try:
                        # Apply augmentation to original image
                        # Stochastic methods generate new random parameters each time
                        if is_spatial:
                            # Spatial augmentations support bbox calculation
                            result = aug_func(image.copy(), return_bbox=True)
                            if isinstance(result, tuple) and len(result) == 2:
                                aug_image, bbox = result
                            else:
                                # Fallback if bbox not returned
                                aug_image = result
                                bbox = None
                        else:
                            # Non-spatial augmentations don't change bbox
                            aug_image = aug_func(image.copy())
                            bbox = None

                        # Create unique identifier
                        if iterations == 1:
                            # Deterministic: simple identifier without number
                            aug_identifier = aug_type
                        else:
                            # Stochastic: include iteration number
                            aug_identifier = f"{aug_type}_{i+1}"

                        augmented_images.append((aug_image, aug_identifier, bbox))
                    except Exception as e:
                        logger.warning(f"Failed to apply augmentation '{aug_type}' "
                                     f"(iteration {i+1}/{iterations}, {iteration_note}): {e}")

        return augmented_images

    def _export_yolo_format(self, export_dir: Path, objects: List[TrainingObject],
                           enable_augmentation: bool = False,
                           augmentation_factor: int = 3,
                           augmentation_types: Optional[List[str]] = None) -> bool:
        """Export dataset in YOLO format with optional data augmentation.

        The augmentation factor is applied per selected method, not per original image.
        This means each augmentation method is applied multiple times to generate diverse
        variations with different random parameters.

        Args:
            export_dir: Export directory
            objects: List of objects to export
            enable_augmentation: Whether to apply data augmentation
            augmentation_factor: How many times to apply EACH augmentation method
                                Example: 3 images × 5 methods × 3 factor = 3 + 45 = 48 total
            augmentation_types: List of augmentation types to apply

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory structure
            images_dir = export_dir / "images" / "train"
            labels_dir = export_dir / "labels" / "train"
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            # Get unique class labels
            class_labels = sorted(list(set(obj.label for obj in objects)))
            class_map = {label: idx for idx, label in enumerate(class_labels)}

            # Counter for images
            img_counter = 0

            # Calculate total images with hybrid deterministic/stochastic approach
            if enable_augmentation and augmentation_types:
                # Separate deterministic and stochastic methods
                deterministic_methods = [m for m in augmentation_types if m in ImageAugmentor.DETERMINISTIC_METHODS]
                stochastic_methods = [m for m in augmentation_types if m in ImageAugmentor.STOCHASTIC_METHODS]

                # Deterministic: 1 image per method
                deterministic_count = len(deterministic_methods) * 1

                # Stochastic: augmentation_factor images per method
                stochastic_count = len(stochastic_methods) * augmentation_factor

                # Total augmented images per original image
                augmented_per_image = deterministic_count + stochastic_count
                total_augmented = len(objects) * augmented_per_image
                total_images = len(objects) + total_augmented
            else:
                total_images = len(objects)
                deterministic_methods = []
                stochastic_methods = []

            logger.info(f"Exporting dataset with augmentation: {enable_augmentation}")
            if enable_augmentation:
                logger.info(f"Augmentation approach: Hybrid (deterministic 1×, stochastic {augmentation_factor}×)")
                logger.info(f"Selected augmentation methods ({len(augmentation_types)} total):")
                if deterministic_methods:
                    logger.info(f"  • Deterministic (1× each): {', '.join(deterministic_methods)}")
                if stochastic_methods:
                    logger.info(f"  • Stochastic ({augmentation_factor}× each): {', '.join(stochastic_methods)}")

                # Check for spatial augmentations with bbox tracking
                spatial_methods = [m for m in augmentation_types if m in ImageAugmentor.SPATIAL_AUGMENTATIONS]
                if spatial_methods:
                    logger.info(f"  • Spatial augmentations with bbox tracking: {', '.join(spatial_methods)}")
                    logger.info(f"    (These augmentations change object position/size, bbox will be calculated)")

                logger.info(f"Total images to export: {total_images} = {len(objects)} original + "
                          f"{total_augmented} augmented ({len(deterministic_methods)}×1 + {len(stochastic_methods)}×{augmentation_factor} per image)")

            # Export images and labels
            for obj_idx, obj in enumerate(objects):
                # Save original image
                img_name = f"{img_counter:04d}.jpg"
                img_path = images_dir / img_name
                cv2.imwrite(str(img_path), obj.image)

                # Create label file (center_x, center_y, width, height - normalized)
                h, w = obj.image.shape[:2]
                class_idx = class_map[obj.label]

                # For cropped objects, bbox is the full image
                label_path = labels_dir / f"{img_counter:04d}.txt"
                with open(label_path, 'w') as f:
                    # Full image bbox in normalized YOLO format
                    f.write(f"{class_idx} 0.5 0.5 1.0 1.0\n")

                img_counter += 1

                # Apply augmentations if enabled
                if enable_augmentation and augmentation_types:
                    # Get augmented images with their identifiers and bboxes
                    # Each method is applied augmentation_factor times
                    augmented_images = self._apply_augmentations(
                        obj.image, augmentation_types, augmentation_factor
                    )

                    # Save each augmented image with descriptive filename
                    for aug_image, aug_identifier, bbox in augmented_images:
                        # Create filename with augmentation identifier
                        # Example: 0001_rotation_1.jpg, 0001_rotation_2.jpg, 0001_brightness_1.jpg, etc.
                        aug_img_name = f"{img_counter:04d}_{aug_identifier}.jpg"
                        aug_img_path = images_dir / aug_img_name
                        cv2.imwrite(str(aug_img_path), aug_image)

                        # Create label file for augmented image
                        aug_h, aug_w = aug_image.shape[:2]
                        aug_label_path = labels_dir / f"{img_counter:04d}_{aug_identifier}.txt"

                        # Use calculated bbox if available, otherwise use default full-image bbox
                        if bbox is not None:
                            # Spatial augmentation with calculated bbox
                            cx, cy, bw, bh = bbox
                            # Ensure bbox values are within valid range [0, 1]
                            cx = max(0.0, min(1.0, cx))
                            cy = max(0.0, min(1.0, cy))
                            bw = max(0.0, min(1.0, bw))
                            bh = max(0.0, min(1.0, bh))
                        else:
                            # Non-spatial augmentation or no bbox returned - object still fills frame
                            cx, cy, bw, bh = 0.5, 0.5, 1.0, 1.0

                        with open(aug_label_path, 'w') as f:
                            # YOLO format: class_idx center_x center_y width height (normalized)
                            f.write(f"{class_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                        img_counter += 1

                # Log progress every 10 objects
                if (obj_idx + 1) % 10 == 0:
                    logger.info(f"Exported {obj_idx + 1}/{len(objects)} objects "
                              f"({img_counter} total images so far)")

            # Create data.yaml
            yaml_content = {
                'path': str(export_dir.absolute()),
                'train': 'images/train',
                'val': 'images/train',  # FIX: Changed from 'labels/train' to 'images/train' for validation
                'names': {idx: label for label, idx in class_map.items()}
            }

            yaml_path = export_dir / "data.yaml"
            with open(yaml_path, 'w') as f:
                import yaml
                yaml.dump(yaml_content, f)

            logger.info(f"Export complete: {len(objects)} objects exported as {img_counter} total images")
            if enable_augmentation:
                # Calculate actual multiplier with hybrid approach
                deterministic_methods = [m for m in augmentation_types if m in ImageAugmentor.DETERMINISTIC_METHODS]
                stochastic_methods = [m for m in augmentation_types if m in ImageAugmentor.STOCHASTIC_METHODS]
                augmented_per_image = len(deterministic_methods) + (len(stochastic_methods) * augmentation_factor)
                multiplier = augmented_per_image + 1  # +1 for original

                logger.info(f"Augmentation: Applied {len(deterministic_methods)} deterministic (1×) + "
                          f"{len(stochastic_methods)} stochastic ({augmentation_factor}×) methods")
                logger.info(f"Dataset increased by {multiplier}× = 1 original + {augmented_per_image} augmented per image")
                logger.info(f"Each original image generated {augmented_per_image} augmented variations "
                          f"({len(deterministic_methods)} deterministic + {len(stochastic_methods)}×{augmentation_factor} stochastic)")

            return True

        except Exception as e:
            logger.error(f"Error in YOLO export: {e}")
            return False

    def train_model(self, epochs: int = 50, batch_size: int = 8, img_size: int = 640,
                   progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                   cancellation_check: Optional[Callable[[], bool]] = None,
                   device: Optional[str] = None,
                   model_architecture: str = "yolo11n.pt") -> bool:
        """Train YOLO model with transfer learning using pretrained weights.

        This method uses transfer learning by starting with pretrained YOLO weights
        and fine-tuning them on your custom dataset. This approach requires much less
        data and training time compared to training from scratch.

        Args:
            epochs: Number of training epochs (default 50 for transfer learning).
                   Transfer learning typically needs 30-50 epochs vs 100+ for training from scratch.
            batch_size: Training batch size
            img_size: Image size for training
            progress_callback: Optional callback for progress updates (receives dict with metrics)
            cancellation_check: Optional callback that returns True if training should be cancelled
            device: Device to use for training ('auto', 'cuda', 'mps', 'cpu').
                   If None or 'auto', automatically detects best available device.
            model_architecture: YOLO pretrained model file (default 'yolo11n.pt').
                              Available models: yolo11n.pt (nano), yolo11s.pt (small),
                              yolo11m.pt (medium), yolo11l.pt (large), yolo11x.pt (extra large).
                              The .pt file contains pretrained weights for transfer learning.
                              Use .yaml files only for training from scratch (requires 100+ epochs).

        Returns:
            True if training completed successfully, False otherwise (includes cancellation)

        Note:
            Transfer learning advantages:
            - Much less training data required (can work with 5-10 images per class)
            - Faster training time (30-50 epochs typical)
            - Better results with limited data
            - Leverages knowledge from pretrained models
        """
        try:
            from ultralytics import YOLO
            import shutil
            import time
            import gc

            # Force garbage collection to free memory before training
            gc.collect()
            logger.info("Memory cleanup: Garbage collection completed")

            # Optional: Check available memory if psutil is available
            try:
                import psutil
                memory = psutil.virtual_memory()
                available_gb = memory.available / (1024**3)
                used_percent = memory.percent

                logger.info(f"System memory status: {available_gb:.2f} GB available ({100-used_percent:.1f}% free)")

                if available_gb < 2.0:
                    logger.warning(f"LOW MEMORY WARNING: Only {available_gb:.2f} GB available")
                    logger.warning("Training may be slow or fail. Consider closing other applications.")

                    if progress_callback:
                        progress_callback({
                            'status': 'warning',
                            'message': f'Low memory: {available_gb:.2f} GB available. Training may be slower.'
                        })
                elif available_gb < 4.0:
                    logger.info(f"Memory notice: {available_gb:.2f} GB available - using memory-safe settings")
            except ImportError:
                # psutil not available - that's fine, continue without memory check
                logger.debug("psutil not available - skipping memory check")

            # Determine device to use
            if device is None or device == 'auto' or device == '':
                training_device, device_name = get_best_device()
                logger.info(f"Auto-detected training device: {device_name} ({training_device})")
            else:
                # Use user-specified device
                training_device = device.lower()
                device_name = device

                # Validate device availability
                if training_device == 'cuda':
                    try:
                        import torch
                        if not torch.cuda.is_available():
                            logger.warning("CUDA requested but not available, falling back to CPU")
                            training_device = 'cpu'
                            device_name = 'CPU (fallback)'
                    except ImportError:
                        logger.warning("PyTorch not available, falling back to CPU")
                        training_device = 'cpu'
                        device_name = 'CPU (fallback)'
                elif training_device == 'mps':
                    try:
                        import torch
                        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                            logger.warning("MPS requested but not available, falling back to CPU")
                            training_device = 'cpu'
                            device_name = 'CPU (fallback)'
                    except ImportError:
                        logger.warning("PyTorch not available, falling back to CPU")
                        training_device = 'cpu'
                        device_name = 'CPU (fallback)'

                logger.info(f"Using user-specified training device: {device_name} ({training_device})")

            # Get memory info if GPU
            memory_info = get_device_memory_info(training_device)
            if memory_info:
                logger.info(f"GPU Memory: {memory_info['free_mb']:.0f}MB free / {memory_info['total_mb']:.0f}MB total")

            # Export dataset first (uses all objects with augmentation if configured)
            dataset_dir = self.data_dir / "exported_dataset"
            if not self._export_all_objects(str(dataset_dir), config=self.config):
                logger.error("Failed to export dataset for training")
                return False

            # Check for cancellation before starting
            if cancellation_check and cancellation_check():
                logger.info("Training cancelled before starting")
                return False

            # Determine training mode based on model file extension
            is_pretrained = model_architecture.endswith('.pt')
            is_from_scratch = model_architecture.endswith('.yaml')

            if is_pretrained:
                logger.info(f"Initializing model with transfer learning: {model_architecture}")
                logger.info("Using pretrained weights for better performance with limited data")
                logger.info(f"Transfer learning approach: fine-tuning existing knowledge on your custom dataset")
            elif is_from_scratch:
                logger.info(f"Initializing model for training from scratch: {model_architecture}")
                logger.info("NOTE: Training from scratch with random initialization (no pretrained weights)")
                logger.info(f"This requires more data and epochs but creates a fully custom model")
            else:
                logger.warning(f"Unknown model file type: {model_architecture}")
                logger.info("Expected .pt (pretrained) or .yaml (from scratch)")

            try:
                model = YOLO(model_architecture)
                logger.info(f"Model loaded successfully: {model_architecture}")
            except Exception as e:
                logger.error(f"Failed to load model '{model_architecture}': {e}")

                # Smart fallback based on original intention
                if is_pretrained or not is_from_scratch:
                    # Try pretrained fallback first
                    logger.info("Available pretrained models: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt")
                    logger.info("Falling back to yolo11n.pt (nano pretrained model with transfer learning)")
                    try:
                        model = YOLO('yolo11n.pt')
                        logger.info("Fallback successful: using yolo11n.pt with transfer learning")
                    except Exception as e2:
                        logger.error(f"Pretrained fallback failed: {e2}")
                        return False
                else:
                    # Fallback to from-scratch for .yaml files
                    logger.info("Available architectures: yolo11n.yaml, yolo11s.yaml, yolo11m.yaml, yolo11l.yaml, yolo11x.yaml")
                    logger.info("Falling back to yolo11n.yaml (nano architecture, training from scratch)")
                    try:
                        model = YOLO('yolo11n.yaml')
                        logger.info("Fallback successful: using yolo11n.yaml")
                    except Exception as e2:
                        logger.error(f"Fallback failed: {e2}")
                        return False

            # Prepare training arguments
            data_yaml = dataset_dir / "data.yaml"

            # Track epoch timing for accurate ETA calculation
            # Key variables for proper per-epoch time tracking
            epoch_start_time = time.time()  # Start time of current epoch
            epoch_durations = []  # List of individual epoch durations (not cumulative)
            training_overall_start = time.time()  # Overall training start (for total elapsed)

            # Helper function to format ETA in human-readable format
            def format_eta(seconds: int) -> str:
                """Format seconds as human-readable time string.

                Args:
                    seconds: Number of seconds

                Returns:
                    Formatted string like "2h 15m 30s" or "5m 30s" or "45s"
                """
                if seconds < 0:
                    return "Almost done"

                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                secs = seconds % 60

                if hours > 0:
                    return f"{hours}h {minutes}m {secs}s"
                elif minutes > 0:
                    return f"{minutes}m {secs}s"
                else:
                    return f"{secs}s"

            # Define callback for training progress updates
            def on_train_epoch_end(trainer):
                """Called at the end of each training epoch with accurate ETA calculation."""
                nonlocal epoch_start_time  # Need to update this for next epoch

                try:
                    # Check for cancellation
                    if cancellation_check and cancellation_check():
                        logger.info("Training cancellation requested")
                        trainer.stop = True  # Signal YOLO to stop training
                        return

                    # Extract metrics from trainer
                    current_epoch = trainer.epoch + 1  # YOLO uses 0-based indexing
                    total_epochs = trainer.epochs

                    # Calculate this epoch's duration (not cumulative time!)
                    epoch_end_time = time.time()
                    epoch_duration = epoch_end_time - epoch_start_time
                    epoch_durations.append(epoch_duration)

                    # Calculate progress percentage
                    progress_percent = round((current_epoch / total_epochs) * 100, 1)

                    # Calculate ETA with intelligent moving average
                    # Strategy: Use moving window average, excluding first epochs if they're outliers
                    min_epochs_for_avg = 2  # Minimum epochs before we start averaging

                    if len(epoch_durations) == 0:
                        # No epochs completed yet (shouldn't happen in this callback)
                        avg_epoch_time = 0
                        eta_seconds = 0
                        eta_formatted = "Calculating..."
                    elif len(epoch_durations) == 1:
                        # First epoch just completed - use it as rough estimate
                        avg_epoch_time = epoch_duration
                        remaining_epochs = total_epochs - current_epoch
                        eta_seconds = int(avg_epoch_time * remaining_epochs)
                        eta_formatted = format_eta(eta_seconds) + " (initial estimate)"
                    else:
                        # Multiple epochs completed - use moving average
                        # Exclude first epoch if we have enough data (first epoch often has initialization overhead)
                        if len(epoch_durations) > min_epochs_for_avg:
                            # Use moving window of last 5 epochs (or all except first if fewer)
                            window_size = min(5, len(epoch_durations) - 1)
                            recent_durations = epoch_durations[-window_size:]
                        else:
                            # Use all epochs if we don't have enough data yet
                            recent_durations = epoch_durations

                        avg_epoch_time = sum(recent_durations) / len(recent_durations)
                        remaining_epochs = total_epochs - current_epoch

                        if remaining_epochs > 0:
                            eta_seconds = int(avg_epoch_time * remaining_epochs)
                            eta_formatted = format_eta(eta_seconds)
                        else:
                            eta_seconds = 0
                            eta_formatted = "Almost done"

                    # Calculate training speed (epochs per minute)
                    epochs_per_minute = round(60 / avg_epoch_time, 2) if avg_epoch_time > 0 else 0

                    # Calculate estimated completion time
                    from datetime import timedelta
                    if eta_seconds > 0:
                        completion_time = datetime.now() + timedelta(seconds=eta_seconds)
                        completion_time_str = completion_time.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        completion_time_str = "N/A"

                    # Extract loss metrics (from trainer.metrics if available)
                    metrics = {
                        # Epoch progress
                        'epoch': current_epoch,
                        'total_epochs': total_epochs,
                        'progress_percent': progress_percent,

                        # Timing metrics
                        'epoch_duration': round(epoch_duration, 1),
                        'avg_epoch_time': round(avg_epoch_time, 1),
                        'eta_seconds': eta_seconds,
                        'eta_formatted': eta_formatted,
                        'completion_time': completion_time_str,
                        'epochs_per_minute': epochs_per_minute,

                        # Total elapsed time
                        'total_elapsed': round(epoch_end_time - training_overall_start, 1)
                    }

                    # Try to get loss values from trainer
                    if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                        # Sum all loss components for total loss
                        total_loss = sum(trainer.loss_items)
                        metrics['loss'] = float(total_loss)
                    elif hasattr(trainer, 'tloss'):
                        metrics['loss'] = float(trainer.tloss)

                    # Get additional metrics if available
                    if hasattr(trainer, 'metrics') and trainer.metrics is not None:
                        metrics_dict = trainer.metrics
                        if hasattr(metrics_dict, 'results_dict'):
                            results = metrics_dict.results_dict
                            metrics['precision'] = results.get('metrics/precision(B)', 0.0)
                            metrics['recall'] = results.get('metrics/recall(B)', 0.0)
                            metrics['mAP50'] = results.get('metrics/mAP50(B)', 0.0)
                            metrics['mAP50-95'] = results.get('metrics/mAP50-95(B)', 0.0)

                    # Call progress callback with metrics
                    if progress_callback:
                        progress_callback(metrics)

                    # Enhanced logging with detailed progress information
                    loss_str = f"{metrics.get('loss', 0.0):.4f}" if 'loss' in metrics else "N/A"
                    logger.info(
                        f"Epoch {current_epoch}/{total_epochs} ({progress_percent}%) - "
                        f"Duration: {epoch_duration:.1f}s - "
                        f"ETA: {eta_formatted} - "
                        f"Loss: {loss_str} - "
                        f"Speed: {epochs_per_minute} epochs/min"
                    )

                    # Log completion time on first few epochs for user reference
                    if current_epoch <= 3 and eta_seconds > 0:
                        logger.info(f"  Estimated completion: {completion_time_str}")

                    # Reset epoch start time for next epoch
                    epoch_start_time = time.time()

                except Exception as e:
                    logger.error(f"Error in training callback: {e}")
                    # Still try to reset epoch start time even on error
                    epoch_start_time = time.time()

            # Add callback to model
            model.add_callback('on_train_epoch_end', on_train_epoch_end)

            # Start training
            training_mode = 'transfer_learning' if is_pretrained else 'from_scratch'
            mode_description = 'with transfer learning' if is_pretrained else 'FROM SCRATCH'

            logger.info(f"Starting model training {mode_description} on {device_name}...")
            logger.info(f"Training with {epochs} epochs on custom dataset")
            if is_pretrained:
                logger.info(f"Model initialized with pretrained weights (transfer learning)")
            else:
                logger.info(f"Model initialized with random weights (training from scratch)")

            # Notify start of training with device info
            if progress_callback:
                progress_callback({
                    'status': 'training_started',
                    'epoch': 0,
                    'total_epochs': epochs,
                    'device': device_name,
                    'device_type': training_device,
                    'training_mode': training_mode
                })

            # Train with device specification and memory-safe parameters
            logger.info("Training configuration: Memory-safe mode enabled")
            logger.info(f"  - workers=0 (single-threaded data loading to prevent memory fragmentation)")
            logger.info(f"  - cache=False (no dataset caching in RAM)")
            logger.info(f"  - batch_size={batch_size} (images loaded simultaneously)")

            try:
                results = model.train(
                    data=str(data_yaml),
                    epochs=epochs,
                    batch=batch_size,
                    imgsz=img_size,
                    device=training_device,  # Specify device for training
                    workers=0,               # MEMORY FIX: Single-threaded data loading
                    cache=False,             # MEMORY FIX: Don't cache images in RAM
                    verbose=True,
                    project='runs/detect',
                    name='train'
                )
            except RuntimeError as e:
                # Handle GPU out-of-memory or other runtime errors
                if 'out of memory' in str(e).lower() or 'cuda' in str(e).lower():
                    logger.error(f"GPU training failed: {e}")
                    logger.info("Retrying training on CPU...")

                    # Retry on CPU
                    training_device = 'cpu'
                    device_name = 'CPU (fallback after GPU error)'

                    if progress_callback:
                        progress_callback({
                            'status': 'retrying_on_cpu',
                            'message': 'GPU training failed, retrying on CPU...'
                        })

                    results = model.train(
                        data=str(data_yaml),
                        epochs=epochs,
                        batch=batch_size,
                        imgsz=img_size,
                        device='cpu',
                        workers=0,               # MEMORY FIX: Single-threaded data loading
                        cache=False,             # MEMORY FIX: Don't cache images in RAM
                        verbose=True,
                        project='runs/detect',
                        name='train'
                    )
                else:
                    raise

            # Check if training was cancelled
            if cancellation_check and cancellation_check():
                logger.info("Training was cancelled by user")
                return False

            logger.info("Training completed successfully")

            # Get the actual save directory from training results
            # YOLO creates runs/detect/train, train2, train3, etc. incrementally
            save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else Path('runs/detect/train')
            trained_model_path = save_dir / "weights" / "best.pt"

            # Use absolute paths for reliability
            trained_model_path = trained_model_path.resolve()
            target_model_dir = Path("data/models").resolve()
            target_model_path = target_model_dir / "model.pt"

            logger.info(f"Looking for trained model at: {trained_model_path}")

            if not trained_model_path.exists():
                # Try to find the model in the most recent train directory
                logger.warning(f"Model not found at {trained_model_path}, searching for latest training run...")
                runs_dir = Path("runs/detect").resolve()
                if runs_dir.exists():
                    # Find all train directories (train, train2, train3, etc.)
                    train_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('train')],
                                      key=lambda x: x.stat().st_mtime, reverse=True)

                    if train_dirs:
                        latest_train_dir = train_dirs[0]
                        trained_model_path = latest_train_dir / "weights" / "best.pt"
                        logger.info(f"Found latest training directory: {latest_train_dir}")
                        logger.info(f"Checking for model at: {trained_model_path}")

            if trained_model_path.exists():
                try:
                    # Verify the model file has content
                    model_size = trained_model_path.stat().st_size
                    if model_size == 0:
                        logger.error(f"Trained model file is empty: {trained_model_path}")
                        return False

                    logger.info(f"Found trained model ({model_size / 1024 / 1024:.2f} MB) at: {trained_model_path}")

                    # Create models directory if it doesn't exist
                    target_model_dir.mkdir(parents=True, exist_ok=True)

                    # Backup existing model.pt if it exists
                    if target_model_path.exists():
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        backup_path = target_model_dir / f"model_backup_{timestamp}.pt"
                        logger.info(f"Backing up existing model to: {backup_path}")
                        shutil.copy2(str(target_model_path), str(backup_path))

                    # Copy the trained model to the expected location
                    logger.info(f"Copying trained model from {trained_model_path} to {target_model_path}")
                    shutil.copy2(str(trained_model_path), str(target_model_path))

                    # Verify the copy was successful
                    if target_model_path.exists():
                        copied_size = target_model_path.stat().st_size
                        if copied_size == model_size:
                            logger.info(f"✓ Trained model successfully saved to {target_model_path} ({copied_size / 1024 / 1024:.2f} MB)")

                            # Also save a timestamped backup of this training
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            backup_path = target_model_dir / f"trained_model_{timestamp}.pt"
                            shutil.copy2(str(trained_model_path), str(backup_path))
                            logger.info(f"✓ Backup model saved to {backup_path}")

                            # Try to verify the model can be loaded
                            try:
                                test_model = YOLO(str(target_model_path))
                                logger.info(f"✓ Model validation successful - model can be loaded by YOLO")
                            except Exception as e:
                                logger.warning(f"Model copied but validation failed: {e}")

                            return True
                        else:
                            logger.error(f"File copy size mismatch! Source: {model_size}, Target: {copied_size}")
                            return False
                    else:
                        logger.error(f"Failed to copy model - target file does not exist after copy operation")
                        return False

                except Exception as e:
                    logger.error(f"Error copying trained model: {e}", exc_info=True)
                    return False
            else:
                logger.error(f"Trained model not found at expected path: {trained_model_path}")
                logger.error("Training may have failed or been cancelled before model was saved")

                # List what's actually in the runs/detect directory for debugging
                runs_dir = Path("runs/detect").resolve()
                if runs_dir.exists():
                    logger.info(f"Contents of {runs_dir}:")
                    for item in runs_dir.iterdir():
                        logger.info(f"  - {item.name}")

                return False

            return True

        except ImportError:
            logger.error("Ultralytics package not installed")
            return False
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return False

    def _save_object_image(self, obj: TrainingObject):
        """Save object image to disk.

        Args:
            obj: TrainingObject to save
        """
        img_path = self.data_dir / f"{obj.object_id}.png"
        cv2.imwrite(str(img_path), obj.image)

    def _save_metadata(self):
        """Save objects metadata to JSON file."""
        metadata = {
            'objects': [obj.to_dict() for obj in self.objects]
        }

        metadata_path = self.data_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _load_objects(self):
        """Load objects from disk on initialization."""
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            return

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            for obj_data in metadata.get('objects', []):
                img_path = self.data_dir / f"{obj_data['object_id']}.png"
                if img_path.exists():
                    image = cv2.imread(str(img_path))
                    if image is not None:
                        obj = TrainingObject(
                            image=image,
                            label=obj_data['label'],
                            bbox=obj_data.get('bbox'),
                            object_id=obj_data['object_id']
                        )
                        self.objects.append(obj)

            logger.info(f"Loaded {len(self.objects)} training objects")

        except Exception as e:
            logger.error(f"Error loading objects: {e}")