"""Reference image management service."""

import logging
from typing import Optional
from pathlib import Path
import numpy as np
import cv2
from datetime import datetime

logger = logging.getLogger(__name__)


class ReferenceManager:
    """Service for managing reference images."""

    def __init__(self, data_dir: str = "data/reference"):
        """Initialize reference manager.

        Args:
            data_dir: Directory for storing reference images
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._reference_image: Optional[np.ndarray] = None
        self._reference_path: Optional[Path] = None
        self._reference_timestamp: Optional[datetime] = None

    def set_reference_from_array(self, image: np.ndarray, save: bool = True) -> bool:
        """Set reference image from numpy array.

        Args:
            image: Reference image as numpy array
            save: Whether to save image to disk

        Returns:
            True if successful, False otherwise
        """
        try:
            if image is None or image.size == 0:
                logger.error("Invalid image array")
                return False

            self._reference_image = image.copy()
            self._reference_timestamp = datetime.now()

            if save:
                # Generate filename with timestamp
                filename = f"reference_{self._reference_timestamp.strftime('%Y%m%d_%H%M%S')}.png"
                filepath = self.data_dir / filename

                # Save image
                cv2.imwrite(str(filepath), image)
                self._reference_path = filepath
                logger.info(f"Reference image saved: {filepath}")

            return True

        except Exception as e:
            logger.error(f"Error setting reference image: {e}")
            return False

    def load_reference_from_file(self, filepath: str) -> bool:
        """Load reference image from file.

        Args:
            filepath: Path to image file

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            path = Path(filepath)

            if not path.exists():
                logger.error(f"Reference image not found: {filepath}")
                return False

            image = cv2.imread(str(path))

            if image is None:
                logger.error(f"Failed to load image: {filepath}")
                return False

            self._reference_image = image
            self._reference_path = path
            self._reference_timestamp = datetime.now()

            logger.info(f"Reference image loaded: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error loading reference image: {e}")
            return False

    def get_reference(self) -> Optional[np.ndarray]:
        """Get current reference image.

        Returns:
            Reference image as numpy array, or None if not set
        """
        return self._reference_image.copy() if self._reference_image is not None else None

    def has_reference(self) -> bool:
        """Check if reference image is set.

        Returns:
            True if reference image exists, False otherwise
        """
        return self._reference_image is not None

    def clear_reference(self):
        """Clear current reference image."""
        self._reference_image = None
        self._reference_path = None
        self._reference_timestamp = None
        logger.info("Reference image cleared")

    def get_reference_info(self) -> Optional[dict]:
        """Get reference image metadata.

        Returns:
            Dictionary with image info, or None if no reference set
        """
        if not self.has_reference():
            return None

        h, w = self._reference_image.shape[:2]

        return {
            'width': w,
            'height': h,
            'channels': self._reference_image.shape[2] if len(self._reference_image.shape) > 2 else 1,
            'path': str(self._reference_path) if self._reference_path else None,
            'timestamp': self._reference_timestamp.isoformat() if self._reference_timestamp else None
        }

    def save_current_reference(self, filename: Optional[str] = None) -> Optional[str]:
        """Save current reference image to disk.

        Args:
            filename: Optional custom filename (without extension)

        Returns:
            Path to saved file, or None on failure
        """
        if not self.has_reference():
            logger.warning("No reference image to save")
            return None

        try:
            if filename:
                filepath = self.data_dir / f"{filename}.png"
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filepath = self.data_dir / f"reference_{timestamp}.png"

            cv2.imwrite(str(filepath), self._reference_image)
            self._reference_path = filepath
            logger.info(f"Reference image saved: {filepath}")

            return str(filepath)

        except Exception as e:
            logger.error(f"Error saving reference image: {e}")
            return None