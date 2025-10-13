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
    def _get_mean_border_color(image: np.ndarray, border_width: int = 5) -> tuple:
        """Calculate mean color from image border pixels.

        This is used to fill empty regions created by spatial transformations
        (rotation, translation, scaling) with a natural color instead of black.

        Args:
            image: Input image (BGR or grayscale)
            border_width: Width of border to sample (default 5 pixels)

        Returns:
            Tuple of mean color (B, G, R) for BGR images or single value for grayscale
        """
        h, w = image.shape[:2]

        # Ensure border width doesn't exceed image dimensions
        border_width = min(border_width, h // 2, w // 2)

        if border_width < 1:
            # Image too small, return black
            if len(image.shape) == 3:
                return (0, 0, 0)
            else:
                return 0

        try:
            # Extract border regions
            top_border = image[0:border_width, :]
            bottom_border = image[-border_width:, :]
            left_border = image[:, 0:border_width]
            right_border = image[:, -border_width:]

            # Combine all border pixels
            if len(image.shape) == 3:
                # Color image (BGR)
                all_borders = np.vstack([
                    top_border.reshape(-1, 3),
                    bottom_border.reshape(-1, 3),
                    left_border.reshape(-1, 3),
                    right_border.reshape(-1, 3)
                ])

                # Calculate mean for each channel
                mean_color = np.mean(all_borders, axis=0)

                # Check for NaN or invalid values
                if np.any(np.isnan(mean_color)):
                    return (0, 0, 0)

                # Convert to integer tuple
                return tuple(map(int, mean_color))
            else:
                # Grayscale image
                all_borders = np.concatenate([
                    top_border.flatten(),
                    bottom_border.flatten(),
                    left_border.flatten(),
                    right_border.flatten()
                ])

                mean_value = np.mean(all_borders)

                # Check for NaN
                if np.isnan(mean_value):
                    return 0

                return int(mean_value)
        except Exception as e:
            logger.warning(f"Error calculating border color: {e}, using black")
            return (0, 0, 0) if len(image.shape) == 3 else 0

    @staticmethod
    def _fill_background_with_random_colors(rotated: np.ndarray, original: np.ndarray,
                                            background_region: Optional[tuple] = None) -> np.ndarray:
        """Fill empty regions in rotated image with random colors from background region.

        This creates more realistic augmented images by sampling colors from a specified
        background region and using random values within the min/max range for each channel.

        Args:
            rotated: Rotated image with black/solid color gaps
            original: Original image before rotation
            background_region: Optional (x1, y1, x2, y2) region to sample background colors from

        Returns:
            Rotated image with gaps filled with realistic random colors
        """
        if background_region is None:
            # No background region specified, return as-is
            return rotated

        try:
            # Extract background region from original image
            bx1, by1, bx2, by2 = map(int, background_region)
            h, w = original.shape[:2]

            # Validate and clamp background region
            bx1 = max(0, min(bx1, w - 1))
            bx2 = max(bx1 + 1, min(bx2, w))
            by1 = max(0, min(by1, h - 1))
            by2 = max(by1 + 1, min(by2, h))

            # Extract background pixels
            bg_pixels = original[by1:by2, bx1:bx2]

            if bg_pixels.size == 0:
                logger.warning("Empty background region, skipping intelligent filling")
                return rotated

            # Calculate min/max for each channel (B, G, R)
            min_vals = bg_pixels.min(axis=(0, 1))  # Shape: (3,)
            max_vals = bg_pixels.max(axis=(0, 1))  # Shape: (3,)

            # Find empty regions (pixels that are black or very dark, indicating gaps from rotation)
            # We detect gaps by looking for pixels that are uniformly black/dark
            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            empty_mask = gray < 5  # Threshold for "empty" pixels

            # Create random colors for empty pixels
            result = rotated.copy()
            num_empty = np.sum(empty_mask)

            if num_empty > 0:
                # Generate random values for each channel
                for c in range(3):  # B, G, R
                    random_values = np.random.randint(
                        int(min_vals[c]),
                        int(max_vals[c]) + 1,
                        size=num_empty,
                        dtype=np.uint8
                    )
                    result[empty_mask, c] = random_values

                logger.debug(f"Filled {num_empty} pixels with random background colors "
                           f"(B: [{min_vals[0]}, {max_vals[0]}], "
                           f"G: [{min_vals[1]}, {max_vals[1]}], "
                           f"R: [{min_vals[2]}, {max_vals[2]}])")

            return result

        except Exception as e:
            logger.warning(f"Error filling background with random colors: {e}, returning original")
            return rotated

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

        Note:
            Empty regions created by rotation are filled with the mean color
            from the original image's 5-pixel border (not black).
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

        # Calculate mean border color from original image
        border_color = ImageAugmentor._get_mean_border_color(image, border_width=5)

        rotated = cv2.warpAffine(image, matrix, (new_w, new_h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=border_color)

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

        Note:
            When scaling down, padding uses the mean color from the original
            image's 5-pixel border (not black).
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

            # Calculate mean border color from original image
            border_color = ImageAugmentor._get_mean_border_color(image, border_width=5)

            scaled = cv2.copyMakeBorder(scaled, top, bottom, left, right,
                                       cv2.BORDER_CONSTANT, value=border_color)
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

        Note:
            Empty regions created by translation are filled with the mean color
            from the original image's 5-pixel border (not black).
        """
        h, w = image.shape[:2]

        if tx is None:
            tx = int(random.uniform(-0.1, 0.1) * w)
        if ty is None:
            ty = int(random.uniform(-0.1, 0.1) * h)

        # Calculate mean border color from original image
        border_color = ImageAugmentor._get_mean_border_color(image, border_width=5)

        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(image, matrix, (w, h),
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=border_color)

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
    """Represents a training object with full frame and bbox annotation.

    IMPORTANT: This class now stores FULL FRAMES with bounding box annotations,
    not cropped objects. This ensures proper bbox localization during training.
    """

    def __init__(self, image: np.ndarray, label: str, bbox: tuple,
                 background_region: Optional[tuple] = None,
                 segmentation: Optional[List[float]] = None,
                 threshold: Optional[int] = None,
                 object_id: Optional[str] = None,
                 image_id: Optional[str] = None):
        """Initialize training object.

        Args:
            image: FULL FRAME image (NOT cropped!)
            label: Object class label
            bbox: Bounding box coordinates (x1, y1, x2, y2) in image pixel coordinates
                  REQUIRED - must specify where the object is in the full frame
            background_region: Optional background region (x1, y1, x2, y2) for sampling
                             background colors during augmentation. If None, uses image
                             border mean color as fallback.
            segmentation: Optional YOLO segmentation points [x1, y1, x2, y2, ...] normalized to [0, 1]
            threshold: Optional threshold value used for segmentation extraction
            object_id: Unique identifier for this specific object
            image_id: Identifier to group objects from the same source image
        """
        if bbox is None:
            raise ValueError("bbox is required! Must provide object location in full frame.")

        self.image = image  # FULL FRAME (not cropped)
        self.label = label
        self.bbox = bbox  # (x1, y1, x2, y2) in pixel coordinates
        self.background_region = background_region  # Optional (x1, y1, x2, y2) or None
        self.segmentation = segmentation or []  # YOLO segmentation format
        self.threshold = threshold  # Threshold value used
        self.object_id = object_id or self._generate_id()
        self.image_id = image_id or self._generate_id()  # NEW: Track source image
        self.timestamp = datetime.now()

        # Validate bbox
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
            logger.warning(f"Invalid bbox {bbox} for image {w}x{h}, will clamp to valid range")

        # Validate background_region if provided
        if background_region is not None:
            bx1, by1, bx2, by2 = background_region
            if bx1 < 0 or by1 < 0 or bx2 > w or by2 > h or bx1 >= bx2 or by1 >= by2:
                logger.warning(f"Invalid background_region {background_region} for image {w}x{h}, will be ignored")
                self.background_region = None

    def _generate_id(self) -> str:
        """Generate unique ID for object."""
        return f"obj_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def get_cropped_object(self) -> np.ndarray:
        """Extract cropped object from full frame using bbox.

        Returns:
            Cropped object image
        """
        x1, y1, x2, y2 = map(int, self.bbox)
        h, w = self.image.shape[:2]

        # Clamp coordinates
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        return self.image[y1:y2, x1:x2].copy()

    def get_bbox_normalized(self) -> tuple:
        """Get normalized bbox in YOLO format (cx, cy, w, h).

        Returns:
            Tuple of (center_x, center_y, width, height) normalized to [0, 1]
        """
        h, w = self.image.shape[:2]
        x1, y1, x2, y2 = self.bbox

        # Calculate center and size
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bbox_w = (x2 - x1) / w
        bbox_h = (y2 - y1) / h

        # Clamp to valid range [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bbox_w = max(0.0, min(1.0, bbox_w))
        bbox_h = max(0.0, min(1.0, bbox_h))

        return (cx, cy, bbox_w, bbox_h)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        h, w = self.image.shape[:2]
        return {
            'object_id': self.object_id,
            'image_id': self.image_id,  # NEW: Store image_id
            'label': self.label,
            'bbox': self.bbox,
            'background_region': self.background_region,  # Store background region
            'segmentation': self.segmentation,  # Store segmentation points
            'threshold': self.threshold,  # Store threshold value
            'image_size': (w, h),  # Store for validation
            'timestamp': self.timestamp.isoformat()
        }


class TrainingService:
    """Service for managing object training dataset and model training.

    IMAGE_ID-BASED GROUPING SYSTEM:
    ================================
    This service uses an image_id-based architecture to efficiently manage objects
    that come from the same source frame. Key characteristics:

    1. STORAGE:
       - Images are saved by image_id (not object_id): e.g., "abc123.png"
       - Multiple objects from the same frame share ONE image file
       - No duplicate images = significant storage savings

    2. GROUPING:
       - Objects with the same image_id are grouped together
       - During export, objects are grouped by image_id
       - ONE label file per image contains ALL objects from that frame

    3. LABEL EXPORT:
       - For image_id "abc123" with 3 objects (person, car, dog):
         - Creates: "0000.jpg" (full frame)
         - Creates: "0000.txt" with 3 lines:
           0 0.5 0.3 0.2 0.1  # person
           1 0.7 0.6 0.15 0.2  # car
           2 0.2 0.4 0.1 0.15  # dog

    4. BENEFITS:
       - Reduced storage (no duplicate images)
       - Consistent annotations across objects from same frame
       - Proper YOLO format (multiple objects per image)
       - Simplified data management

    Example workflow:
        # Frame 1 has 2 objects -> image_id="frame1_timestamp"
        service.add_object(frame1, "person", bbox1, image_id="frame1_timestamp")
        service.add_object(frame1, "car", bbox2, image_id="frame1_timestamp")

        # Frame 2 has 1 object -> image_id="frame2_timestamp"
        service.add_object(frame2, "dog", bbox3, image_id="frame2_timestamp")

        # Storage: 2 image files (frame1_timestamp.png, frame2_timestamp.png)
        # Export: 2 label files with proper multi-object annotations
    """

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

    def add_object(self, image: np.ndarray, label: str, bbox: tuple,
                   background_region: Optional[tuple] = None,
                   segmentation: Optional[List[float]] = None,
                   threshold: Optional[int] = None,
                   image_id: Optional[str] = None) -> TrainingObject:
        """Add object to training dataset.

        Args:
            image: FULL FRAME image (NOT cropped!)
            label: Object class label
            bbox: Bounding box (x1, y1, x2, y2) in pixel coordinates - REQUIRED!
            background_region: Optional background region (x1, y1, x2, y2) for augmentation
            segmentation: Optional YOLO segmentation points [x1, y1, x2, y2, ...] normalized
            threshold: Optional threshold value used for segmentation
            image_id: Optional identifier to group objects from the same source image

        Returns:
            Created TrainingObject
        """
        if bbox is None:
            raise ValueError("bbox is required! Must provide object location in full frame.")

        obj = TrainingObject(
            image, label, bbox,
            background_region=background_region,
            segmentation=segmentation,
            threshold=threshold,
            image_id=image_id
        )
        self.objects.append(obj)

        # Save object image
        self._save_object_image(obj)
        self._save_metadata()

        bg_info = f" with background region {background_region}" if background_region else ""
        seg_info = f", {len(segmentation) // 2} segmentation points" if segmentation else ""
        img_id_info = f", image_id={image_id}" if image_id else ""
        logger.info(f"Added training object: {obj.label} at bbox {bbox}{bg_info}{seg_info}{img_id_info} (ID: {obj.object_id})")
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

        Only deletes the shared image file if this is the last object using that image_id.
        This prevents accidentally deleting images that are still referenced by other objects.

        Args:
            object_id: Object identifier

        Returns:
            True if deleted successfully, False if object not found
        """
        obj = self.get_object(object_id)
        if not obj:
            return False

        # Store image_id before removing object
        image_id = obj.image_id

        # Remove from list
        self.objects.remove(obj)

        # Check if any other objects still reference this image_id
        other_objects_with_same_image = [o for o in self.objects if o.image_id == image_id]

        if not other_objects_with_same_image:
            # This was the last object using this image - safe to delete
            img_path = self.data_dir / f"{image_id}.png"
            if img_path.exists():
                img_path.unlink()
                logger.debug(f"Deleted shared image {image_id}.png (no other objects reference it)")
        else:
            logger.debug(f"Image {image_id}.png kept (still used by {len(other_objects_with_same_image)} other object(s))")

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

    def _clear_train_folder(self, train_dir: Path):
        """Clear all images and labels from train folder before augmentation.

        This ensures that old augmented images don't mix with new ones,
        preventing training on outdated or inconsistent data.

        Args:
            train_dir: Path to train directory (e.g., export_dir/images/train or export_dir/labels/train)
        """
        try:
            if not train_dir.exists():
                logger.debug(f"Train directory does not exist yet: {train_dir}")
                return

            # Count files before deletion
            deleted_count = 0

            # Clear image files
            for file_path in train_dir.iterdir():
                if file_path.is_file():
                    # Only delete image and label files (safety check)
                    if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.txt']:
                        try:
                            file_path.unlink()
                            deleted_count += 1
                            logger.debug(f"Deleted: {file_path.name}")
                        except Exception as e:
                            logger.warning(f"Failed to delete {file_path.name}: {e}")
                    else:
                        logger.warning(f"Skipping non-image/label file: {file_path.name}")

            if deleted_count > 0:
                logger.info(f"Cleared {deleted_count} files from {train_dir}")
            else:
                logger.debug(f"No files to clear in {train_dir}")

        except Exception as e:
            logger.error(f"Error clearing train folder {train_dir}: {e}")
            # Don't fail the entire export if cleanup fails
            # Just log the error and continue

    def _add_random_padding(self, image: np.ndarray,
                           min_pad_percent: float = 0.15,
                           max_pad_percent: float = 0.60) -> tuple[np.ndarray, tuple]:
        """Add random padding around an image to create position and scale diversity.

        This solves the "cropped object training" problem where all training images
        have objects centered and filling the frame (bbox always 0.5, 0.5, 1.0, 1.0).

        By adding random padding:
        - Objects appear at different positions (not always centered)
        - Objects appear at different scales (not always full-frame)
        - Model learns proper object localization

        Args:
            image: Input image (cropped object)
            min_pad_percent: Minimum padding as percentage of image dimensions (default 15%)
            max_pad_percent: Maximum padding as percentage of image dimensions (default 60%)

        Returns:
            Tuple of (padded_image, bbox) where bbox is (cx, cy, w, h) in normalized coordinates
        """
        h, w = image.shape[:2]

        # Generate random padding for each side independently
        # This creates diverse object positions (not always centered)
        pad_top = random.randint(int(h * min_pad_percent), int(h * max_pad_percent))
        pad_bottom = random.randint(int(h * min_pad_percent), int(h * max_pad_percent))
        pad_left = random.randint(int(w * min_pad_percent), int(w * max_pad_percent))
        pad_right = random.randint(int(w * min_pad_percent), int(w * max_pad_percent))

        # Calculate mean border color from original image (before padding)
        # This ensures padding matches the image's natural edge color, not black
        border_color = ImageAugmentor._get_mean_border_color(image, border_width=5)

        # Add padding with mean border color (NOT black!)
        padded_img = cv2.copyMakeBorder(
            image,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=border_color
        )

        # Calculate bbox for the original object within the padded image
        new_h, new_w = padded_img.shape[:2]

        # Object center in padded image (absolute pixels)
        obj_center_x = pad_left + (w / 2)
        obj_center_y = pad_top + (h / 2)

        # Normalize to [0, 1] for YOLO format
        bbox_cx = obj_center_x / new_w
        bbox_cy = obj_center_y / new_h
        bbox_w = w / new_w
        bbox_h = h / new_h

        # Ensure bbox values are valid [0, 1]
        bbox_cx = max(0.0, min(1.0, bbox_cx))
        bbox_cy = max(0.0, min(1.0, bbox_cy))
        bbox_w = max(0.0, min(1.0, bbox_w))
        bbox_h = max(0.0, min(1.0, bbox_h))

        bbox = (bbox_cx, bbox_cy, bbox_w, bbox_h)

        return padded_img, bbox

    def _export_yolo_format(self, export_dir: Path, objects: List[TrainingObject],
                           enable_augmentation: bool = False,
                           augmentation_factor: int = 3,
                           augmentation_types: Optional[List[str]] = None) -> bool:
        """Export dataset in YOLO format with optional data augmentation.

        IMPORTANT: Objects are now grouped by image_id to create one label file per image.
        Multiple objects from the same source image will share one image file and one label
        file with multiple lines (one per object).

        Args:
            export_dir: Export directory
            objects: List of objects to export
            enable_augmentation: Whether to apply data augmentation
            augmentation_factor: How many times to apply augmentation
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

            # Clear existing files from train folder to prevent old augmented images from mixing
            logger.info("Clearing existing files from train folders...")
            self._clear_train_folder(images_dir)
            self._clear_train_folder(labels_dir)
            logger.info("Train folders cleared - ready for fresh export")

            # Get unique class labels
            class_labels = sorted(list(set(obj.label for obj in objects)))
            class_map = {label: idx for idx, label in enumerate(class_labels)}

            # Group objects by image_id - THIS IS THE KEY FIX
            from collections import defaultdict
            image_groups = defaultdict(list)
            for obj in objects:
                image_groups[obj.image_id].append(obj)

            logger.info(f"Grouped {len(objects)} objects into {len(image_groups)} unique images")

            # Counter for images
            img_counter = 0

            # Statistics tracking for bbox diversity
            bbox_stats = {
                'positions_x': [],  # Center X positions
                'positions_y': [],  # Center Y positions
                'widths': [],       # Bbox widths
                'heights': []       # Bbox heights
            }

            # Calculate total images with rotation-only approach
            if enable_augmentation and 'rotation' in augmentation_types:
                num_rotations = augmentation_factor - 1 if augmentation_factor > 1 else 0
                total_augmented = len(image_groups) * num_rotations
                total_images = len(image_groups) + total_augmented
            else:
                total_images = len(image_groups)
                num_rotations = 0

            logger.info(f"Exporting dataset with rotation augmentation: {enable_augmentation}")
            if enable_augmentation and 'rotation' in augmentation_types:
                logger.info(f"Augmentation approach: Bbox-only rotation (preserves background)")
                logger.info(f"  - Only objects within their bboxes are rotated")
                logger.info(f"  - Background remains intact, objects rotate within their regions")
                logger.info(f"Augmentation factor: {augmentation_factor} (1 original + {num_rotations} rotated versions)")
                logger.info(f"Rotation angles: Random between 0° and 360° (full rotation range)")

                # Check if background regions are provided
                objects_with_bg = sum(1 for obj in objects if obj.background_region is not None)
                if objects_with_bg > 0:
                    logger.info(f"Background regions: {objects_with_bg}/{len(objects)} objects have background regions")
                    logger.info(f"  → Using min/max random color filling for empty pixels in rotated bbox")
                else:
                    logger.info(f"Background regions: None provided")
                    logger.info(f"  → Using mean border color for empty pixels in rotated bbox")

                logger.info(f"Total images to export: {total_images} = {len(image_groups)} original + "
                          f"{total_augmented} rotated ({num_rotations} rotations per image)")

            # Export images and labels - GROUP BY IMAGE_ID
            for img_idx, (image_id, objects_on_image) in enumerate(image_groups.items()):
                # Use the first object's image (all objects from same image share the same frame)
                source_image = objects_on_image[0].image

                # Save FULL FRAME once for all objects on this image
                img_name = f"{img_counter:04d}.jpg"
                img_path = images_dir / img_name
                cv2.imwrite(str(img_path), source_image)

                # Create label file with ALL objects from this image
                label_path = labels_dir / f"{img_counter:04d}.txt"
                with open(label_path, 'w') as f:
                    for obj in objects_on_image:
                        # Get class index
                        class_idx = class_map[obj.label]

                        # Get normalized bbox
                        cx, cy, bw, bh = obj.get_bbox_normalized()

                        # Write one line per object
                        if obj.segmentation and len(obj.segmentation) > 0:
                            # YOLO segmentation format: class_idx x1 y1 x2 y2 x3 y3 ... (normalized)
                            seg_str = ' '.join(f"{coord:.6f}" for coord in obj.segmentation)
                            f.write(f"{class_idx} {seg_str}\n")
                        else:
                            # Fallback to YOLO bbox format: class_idx center_x center_y width height (normalized)
                            f.write(f"{class_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                        # Track bbox statistics
                        bbox_stats['positions_x'].append(cx)
                        bbox_stats['positions_y'].append(cy)
                        bbox_stats['widths'].append(bw)
                        bbox_stats['heights'].append(bh)

                img_counter += 1

                # Apply rotation augmentation if enabled (bbox-only rotation approach)
                # Now handles multiple objects per image
                if enable_augmentation and 'rotation' in augmentation_types:
                    num_rotations = augmentation_factor - 1 if augmentation_factor > 1 else 0

                    for rotation_idx in range(num_rotations):
                        # Start with original image
                        augmented_image = source_image.copy()
                        h, w = source_image.shape[:2]

                        # Rotate EACH object independently and composite onto same image
                        for obj in objects_on_image:
                            # Generate random rotation angle for this object
                            angle = random.uniform(0, 360)

                            # Extract bbox coordinates
                            x1, y1, x2, y2 = map(int, obj.bbox)

                            # Add padding to bbox for rotation space
                            padding = 50
                            padded_x1 = max(0, x1 - padding)
                            padded_y1 = max(0, y1 - padding)
                            padded_x2 = min(w, x2 + padding)
                            padded_y2 = min(h, y2 + padding)

                            # Extract bbox region with padding
                            bbox_region = source_image[padded_y1:padded_y2, padded_x1:padded_x2].copy()
                            bbox_h, bbox_w = bbox_region.shape[:2]

                            if bbox_h == 0 or bbox_w == 0:
                                logger.warning(f"Invalid bbox region size: {bbox_w}x{bbox_h}, skipping object rotation")
                                continue

                            # Rotate bbox region only
                            bbox_center = (bbox_w // 2, bbox_h // 2)
                            rotation_matrix = cv2.getRotationMatrix2D(bbox_center, angle, 1.0)

                            # Rotate with black border initially
                            rotated_bbox = cv2.warpAffine(
                                bbox_region,
                                rotation_matrix,
                                (bbox_w, bbox_h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0)
                            )

                            # Fill empty pixels in rotated bbox
                            if obj.background_region is not None:
                                rotated_bbox = ImageAugmentor._fill_background_with_random_colors(
                                    rotated_bbox, source_image, obj.background_region
                                )
                            else:
                                border_color = ImageAugmentor._get_mean_border_color(bbox_region, border_width=5)
                                gray = cv2.cvtColor(rotated_bbox, cv2.COLOR_BGR2GRAY)
                                empty_mask = gray < 5
                                if np.any(empty_mask):
                                    rotated_bbox[empty_mask] = border_color

                            # Paste rotated bbox back onto augmented image
                            augmented_image[padded_y1:padded_y2, padded_x1:padded_x2] = rotated_bbox

                        # Save augmented image (with all rotated objects)
                        aug_img_name = f"{img_counter:04d}_rotation_{rotation_idx + 1}.jpg"
                        aug_img_path = images_dir / aug_img_name
                        cv2.imwrite(str(aug_img_path), augmented_image)

                        # Create label file with ALL objects (same bbox positions)
                        aug_label_path = labels_dir / f"{img_counter:04d}_rotation_{rotation_idx + 1}.txt"
                        with open(aug_label_path, 'w') as f:
                            for obj in objects_on_image:
                                class_idx = class_map[obj.label]
                                cx, cy, bw, bh = obj.get_bbox_normalized()

                                if obj.segmentation and len(obj.segmentation) > 0:
                                    seg_str = ' '.join(f"{coord:.6f}" for coord in obj.segmentation)
                                    f.write(f"{class_idx} {seg_str}\n")
                                else:
                                    f.write(f"{class_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                                # Track bbox statistics
                                bbox_stats['positions_x'].append(cx)
                                bbox_stats['positions_y'].append(cy)
                                bbox_stats['widths'].append(bw)
                                bbox_stats['heights'].append(bh)

                        img_counter += 1

                        logger.debug(f"Created rotation {rotation_idx + 1}/{num_rotations} with {len(objects_on_image)} objects")

                # Log progress every 10 images
                if (img_idx + 1) % 10 == 0:
                    logger.info(f"Exported {img_idx + 1}/{len(image_groups)} unique images "
                              f"({img_counter} total images with augmentation)")

            # Create data.yaml
            yaml_content = {
                'path': str(export_dir.absolute()),
                'train': 'images/train',
                'val': 'images/train',
                'names': {idx: label for label, idx in class_map.items()}
            }

            yaml_path = export_dir / "data.yaml"
            with open(yaml_path, 'w') as f:
                import yaml
                yaml.dump(yaml_content, f)

            logger.info(f"Export complete: {len(objects)} objects grouped into {len(image_groups)} unique images")
            logger.info(f"Total exported images: {img_counter} (including augmentation)")
            if enable_augmentation and 'rotation' in augmentation_types:
                multiplier = augmentation_factor

                logger.info(f"Augmentation: Bbox-only rotation applied (background preserved)")
                logger.info(f"Dataset increased by {multiplier}× = 1 original + {num_rotations} rotated per image")
                logger.info(f"Each unique image generated {num_rotations} rotated variations")

            # Log bbox diversity statistics
            if bbox_stats['positions_x']:
                logger.info("")
                logger.info("=" * 80)
                logger.info("BOUNDING BOX DIVERSITY STATISTICS")
                logger.info("=" * 80)
                logger.info("This shows the diversity in object positions and scales.")
                logger.info("Good diversity prevents the 'all objects centered and full-frame' problem.")
                logger.info("")

                # Calculate statistics
                pos_x = bbox_stats['positions_x']
                pos_y = bbox_stats['positions_y']
                widths = bbox_stats['widths']
                heights = bbox_stats['heights']

                # Position diversity (center points)
                avg_x = sum(pos_x) / len(pos_x)
                avg_y = sum(pos_y) / len(pos_y)
                min_x, max_x = min(pos_x), max(pos_x)
                min_y, max_y = min(pos_y), max(pos_y)
                std_x = (sum((x - avg_x) ** 2 for x in pos_x) / len(pos_x)) ** 0.5
                std_y = (sum((y - avg_y) ** 2 for y in pos_y) / len(pos_y)) ** 0.5

                # Scale diversity (bbox sizes)
                avg_w = sum(widths) / len(widths)
                avg_h = sum(heights) / len(heights)
                min_w, max_w = min(widths), max(widths)
                min_h, max_h = min(heights), max(heights)
                std_w = (sum((w - avg_w) ** 2 for w in widths) / len(widths)) ** 0.5
                std_h = (sum((h - avg_h) ** 2 for h in heights) / len(heights)) ** 0.5

                logger.info("POSITION DIVERSITY (Center Points):")
                logger.info(f"  Center X: avg={avg_x:.3f}, range=[{min_x:.3f}, {max_x:.3f}], std={std_x:.3f}")
                logger.info(f"  Center Y: avg={avg_y:.3f}, range=[{min_y:.3f}, {max_y:.3f}], std={std_y:.3f}")
                logger.info(f"  Analysis: Objects spread across {(max_x - min_x) * 100:.1f}% of width, "
                          f"{(max_y - min_y) * 100:.1f}% of height")

                # Interpret position diversity
                if std_x < 0.05 and std_y < 0.05:
                    logger.warning("  WARNING: Low position diversity - objects too centered!")
                elif std_x > 0.15 and std_y > 0.15:
                    logger.info("  EXCELLENT: High position diversity - objects well distributed!")
                else:
                    logger.info("  GOOD: Moderate position diversity")

                logger.info("")
                logger.info("SCALE DIVERSITY (Object Sizes):")
                logger.info(f"  Width: avg={avg_w:.3f}, range=[{min_w:.3f}, {max_w:.3f}], std={std_w:.3f}")
                logger.info(f"  Height: avg={avg_h:.3f}, range=[{min_h:.3f}, {max_h:.3f}], std={std_h:.3f}")
                logger.info(f"  Analysis: Object sizes vary from {min_w * 100:.1f}% to {max_w * 100:.1f}% of image width")

                # Interpret scale diversity
                if avg_w > 0.95 and avg_h > 0.95:
                    logger.warning("  WARNING: Objects fill entire frame - no scale diversity!")
                elif avg_w < 0.5 and avg_h < 0.5:
                    logger.info("  EXCELLENT: Objects are small relative to frame - good scale diversity!")
                else:
                    logger.info("  GOOD: Objects have moderate size relative to frame")

                logger.info("")
                logger.info("EXPECTED MODEL BEHAVIOR:")
                if std_x > 0.10 and std_y > 0.10 and avg_w < 0.7 and avg_h < 0.7:
                    logger.info("  The model should learn to detect individual objects with accurate bboxes.")
                    logger.info("  Bboxes should tightly fit around objects, not encompass entire regions.")
                elif avg_w > 0.9 or avg_h > 0.9:
                    logger.warning("  The model may still struggle with accurate localization.")
                    logger.warning("  Consider retraining with more aggressive padding (60-80%).")
                else:
                    logger.info("  The model should perform better than before, but may need fine-tuning.")

                logger.info("")
                logger.info(f"Total bounding boxes: {len(pos_x)} (across {img_counter} training images)")
                logger.info("=" * 80)
                logger.info("")

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

            # Validate training objects exist before starting
            if not self.objects:
                logger.error("No training objects available for training")
                if progress_callback:
                    progress_callback({
                        'status': 'error',
                        'message': 'No training objects found. Please add objects before training.'
                    })
                return False

            logger.info(f"Training with {len(self.objects)} objects")

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
                    # Check for cancellation - IMMEDIATE STOP via exception
                    if cancellation_check and cancellation_check():
                        logger.info("=" * 60)
                        logger.info("TRAINING CANCELLATION REQUESTED")
                        logger.info(f"Current epoch: {trainer.epoch + 1}/{trainer.epochs}")
                        logger.info("Stopping immediately via KeyboardInterrupt...")
                        logger.info("=" * 60)

                        # Raise KeyboardInterrupt to immediately stop training (even mid-epoch)
                        # This is more immediate than trainer.stop = True which waits for epoch end
                        raise KeyboardInterrupt("Training cancelled by user")

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

                except KeyboardInterrupt:
                    # Re-raise to propagate cancellation to main try-except
                    logger.info("Cancellation exception raised - propagating to stop training")
                    raise
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

            # OPTIMIZED TRAINING CONFIGURATION
            # Get optimization settings from config with smart defaults
            workers = self.config.get('training_workers', 2)  # Default: 2 workers
            cache_mode = self.config.get('training_cache', 'ram')  # Default: RAM caching

            # Determine cache setting based on dataset size
            # For small datasets (<500 images), RAM caching is highly beneficial
            dataset_image_count = len(self.objects) if hasattr(self, 'objects') and self.objects else 0
            if cache_mode == 'auto':
                # Auto-detect: use RAM cache for datasets < 500 images
                cache_setting = 'ram' if dataset_image_count < 500 else False
                logger.info(f"Auto-cache: Using {'RAM' if cache_setting == 'ram' else 'no'} cache for {dataset_image_count} images")
            elif cache_mode in ['ram', 'disk', True]:
                cache_setting = cache_mode
            else:
                cache_setting = False

            # Log optimized training configuration
            logger.info("OPTIMIZED Training Configuration:")
            logger.info(f"  - Device: {training_device}")
            logger.info(f"  - Batch size: {batch_size} images per batch")
            logger.info(f"  - Workers: {workers} (parallel data loading threads)")
            logger.info(f"  - Cache: {cache_setting} ({'RAM caching enabled - faster epochs!' if cache_setting == 'ram' else 'No caching'})")
            logger.info(f"  - Image size: {img_size}x{img_size}")
            logger.info(f"  - Epochs: {epochs}")

            # Performance estimation
            if cache_setting == 'ram':
                logger.info("💡 Performance boost: RAM caching will speed up epochs significantly (30-50% faster)")
            if workers > 0:
                logger.info(f"💡 Performance boost: {workers} workers will prevent GPU starvation (20-30% faster)")

            # Training results variable (will be set in try block)
            results = None
            training_cancelled = False

            try:
                results = model.train(
                    data=str(data_yaml),
                    epochs=epochs,
                    batch=batch_size,
                    imgsz=img_size,
                    device=training_device,  # Specify device for training
                    workers=workers,         # OPTIMIZED: Parallel data loading
                    cache=cache_setting,     # OPTIMIZED: RAM caching for small datasets
                    verbose=True,
                    project='runs/detect',
                    name='train'
                )

            except KeyboardInterrupt:
                # IMMEDIATE CANCELLATION - User requested to stop training
                training_cancelled = True
                logger.info("=" * 60)
                logger.info("TRAINING CANCELLED BY USER")
                logger.info("Training stopped immediately (mid-epoch interruption)")
                logger.info("=" * 60)

                # Notify via progress callback
                if progress_callback:
                    progress_callback({
                        'status': 'cancelled',
                        'message': 'Training cancelled by user'
                    })

                # Return False to indicate training did not complete
                return False

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

                    try:
                        results = model.train(
                            data=str(data_yaml),
                            epochs=epochs,
                            batch=batch_size,
                            imgsz=img_size,
                            device='cpu',
                            workers=workers,         # OPTIMIZED: Parallel data loading
                            cache=cache_setting,     # OPTIMIZED: RAM caching for small datasets
                            verbose=True,
                            project='runs/detect',
                            name='train'
                        )
                    except KeyboardInterrupt:
                        # Cancellation during CPU retry
                        training_cancelled = True
                        logger.info("Training cancelled during CPU retry")
                        if progress_callback:
                            progress_callback({
                                'status': 'cancelled',
                                'message': 'Training cancelled by user'
                            })
                        return False
                else:
                    raise

            finally:
                # RESOURCE CLEANUP - Always execute, even on cancellation
                logger.info("Performing resource cleanup...")

                try:
                    # Clear GPU cache if using CUDA
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("GPU cache cleared")

                    # Force garbage collection to free memory
                    import gc
                    gc.collect()
                    logger.info("Garbage collection completed")

                except Exception as cleanup_error:
                    logger.warning(f"Error during resource cleanup: {cleanup_error}")

                # Log cleanup completion
                if training_cancelled:
                    logger.info("Resource cleanup completed after cancellation")
                else:
                    logger.info("Resource cleanup completed after training")

            # Check if training was cancelled (redundant check for safety)
            if cancellation_check and cancellation_check():
                logger.info("Training was cancelled by user (post-training check)")
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

        Images are saved by image_id to avoid duplicates when multiple objects
        come from the same source image. Only saves if the image doesn't already exist.

        This ensures that when multiple objects are extracted from the same source
        frame, they share ONE image file instead of creating duplicate copies.

        Args:
            obj: TrainingObject to save
        """
        # Save by image_id instead of object_id to consolidate duplicates
        img_path = self.data_dir / f"{obj.image_id}.png"

        # Only save if image doesn't already exist (avoid duplicate writes)
        if not img_path.exists():
            cv2.imwrite(str(img_path), obj.image)
            logger.debug(f"Saved new image for image_id: {obj.image_id} (object: {obj.object_id})")
        else:
            logger.debug(f"Image already exists for image_id: {obj.image_id}, skipping duplicate save (object: {obj.object_id})")

    def _save_metadata(self):
        """Save objects metadata to JSON file."""
        metadata = {
            'objects': [obj.to_dict() for obj in self.objects]
        }

        metadata_path = self.data_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _load_objects(self):
        """Load objects from disk on initialization.

        Images are loaded by image_id, which allows multiple objects from the same
        source frame to share a single image file. This approach:
        - Reduces storage space (no duplicate images)
        - Maintains consistency across objects from the same frame
        - Enables efficient label grouping during export
        """
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            return

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            loaded_count = 0
            skipped_count = 0

            for obj_data in metadata.get('objects', []):
                # Load by image_id (all objects from same source share one image file)
                image_id = obj_data['image_id']
                img_path = self.data_dir / f"{image_id}.png"

                if img_path.exists():
                    image = cv2.imread(str(img_path))
                    if image is not None:
                        obj = TrainingObject(
                            image=image,
                            label=obj_data['label'],
                            bbox=obj_data.get('bbox'),
                            background_region=obj_data.get('background_region'),  # Load background region
                            segmentation=obj_data.get('segmentation', []),  # Load segmentation points
                            threshold=obj_data.get('threshold'),  # Load threshold value
                            object_id=obj_data['object_id'],
                            image_id=image_id  # Use loaded image_id
                        )
                        self.objects.append(obj)
                        loaded_count += 1
                    else:
                        logger.warning(f"Failed to read image file: {img_path}")
                        skipped_count += 1
                else:
                    logger.warning(f"Image file not found for object {obj_data['object_id']}: {img_path}")
                    skipped_count += 1

            logger.info(f"Loaded {loaded_count} training objects")
            if skipped_count > 0:
                logger.warning(f"Skipped {skipped_count} objects due to missing or unreadable images")

        except Exception as e:
            logger.error(f"Error loading objects: {e}")