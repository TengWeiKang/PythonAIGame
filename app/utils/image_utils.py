"""Image processing utilities."""

import cv2
import numpy as np
from typing import Tuple, Optional

def resize_image(image: np.ndarray, max_width: int = 960, max_height: int = 720) -> np.ndarray:
    """Resize image while maintaining aspect ratio."""
    h, w = image.shape[:2]
    
    if w <= max_width and h <= max_height:
        return image
    
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def convert_color_space(image: np.ndarray, conversion: int) -> np.ndarray:
    """Convert image color space."""
    return cv2.cvtColor(image, conversion)

def crop_image(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop image using bounding box coordinates."""
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(x1, min(x2, w))
    y2 = max(y1, min(y2, h))
    
    return image[y1:y2, x1:x2]

def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur to image."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)