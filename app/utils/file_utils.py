"""File system utilities."""

import os
import cv2
import numpy as np
import logging
from typing import List, Optional, Tuple
from datetime import datetime

def read_yolo_labels(path: str) -> List:
    """
    Read YOLO labels file, return list of [class_idx, x_center, y_center, w, h] (float)
    """
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                vals = [float(x) if i>0 else int(x) for i,x in enumerate(parts)]
                out.append(vals)
    except FileNotFoundError:
        pass
    return out

def save_yolo_labels(path: str, labels: List) -> None:
    """Save YOLO format labels to file."""
    try:
        # Ensure directory exists
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        with open(path, "w", encoding="utf-8") as f:
            for label in labels:
                if len(label) >= 5:
                    class_id = int(label[0])
                    coords = " ".join([f"{float(x):.6f}" for x in label[1:5]])
                    f.write(f"{class_id} {coords}\n")
        logging.debug(f"Successfully saved YOLO labels to {path}")
    except (OSError, IOError, PermissionError) as e:
        logging.error(f"Failed to save YOLO labels to {path}: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error saving YOLO labels to {path}: {e}")
        raise

def get_file_extension(filepath: str) -> str:
    """Get file extension in lowercase."""
    return os.path.splitext(filepath)[1].lower()

def ensure_file_exists(filepath: str) -> bool:
    """Check if file exists and is readable."""
    return os.path.isfile(filepath) and os.access(filepath, os.R_OK)


def ensure_directory_exists(directory_path: str) -> bool:
    """Ensure directory exists, create if it doesn't."""
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except (OSError, PermissionError) as e:
        logging.error(f"Failed to create directory {directory_path}: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error creating directory {directory_path}: {e}")
        return False


def save_reference_image(image: np.ndarray, data_dir: str, source: str = "unknown") -> Optional[str]:
    """
    Save reference image to persistent storage.
    
    Args:
        image: OpenCV image array (BGR format)
        data_dir: Base data directory path
        source: Source description (e.g., "file", "camera")
        
    Returns:
        Path to saved image file or None if failed
    """
    try:
        # Create reference directory
        reference_dir = os.path.join(data_dir, "reference")
        if not ensure_directory_exists(reference_dir):
            return None
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reference_{source}_{timestamp}.png"
        file_path = os.path.join(reference_dir, filename)
        
        # Save image in high quality PNG format
        success = cv2.imwrite(file_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        if success:
            logging.info(f"Reference image saved: {file_path}")
            return file_path
        else:
            logging.error(f"Failed to save reference image to {file_path}")
            return None
            
    except (OSError, IOError, PermissionError) as e:
        logging.error(f"File system error saving reference image: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error saving reference image: {e}")
        return None


def load_reference_image(file_path: str) -> Optional[np.ndarray]:
    """
    Load reference image from file.
    
    Args:
        file_path: Path to image file
        
    Returns:
        OpenCV image array or None if failed
    """
    try:
        if not ensure_file_exists(file_path):
            return None
            
        image = cv2.imread(file_path)
        if image is not None:
            logging.debug(f"Reference image loaded: {file_path}")
            return image
        else:
            logging.warning(f"Failed to load image from {file_path} - file may be corrupted")
            return None
            
    except (OSError, IOError, PermissionError) as e:
        logging.error(f"File system error loading reference image: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error loading reference image: {e}")
        return None


def get_latest_reference_image(data_dir: str) -> Optional[str]:
    """
    Get path to the most recently saved reference image.
    
    Args:
        data_dir: Base data directory path
        
    Returns:
        Path to latest reference image or None if none found
    """
    try:
        reference_dir = os.path.join(data_dir, "reference")
        if not os.path.exists(reference_dir):
            return None
        
        # Get all reference image files
        image_files = []
        for filename in os.listdir(reference_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and filename.startswith('reference_'):
                file_path = os.path.join(reference_dir, filename)
                if ensure_file_exists(file_path):
                    image_files.append((file_path, os.path.getmtime(file_path)))
        
        if not image_files:
            return None
        
        # Sort by modification time (newest first)
        image_files.sort(key=lambda x: x[1], reverse=True)
        latest_file = image_files[0][0]
        
        logging.debug(f"Latest reference image found: {latest_file}")
        return latest_file
        
    except (OSError, PermissionError) as e:
        logging.error(f"File system error finding latest reference image: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error finding latest reference image: {e}")
        return None


def cleanup_old_reference_images(data_dir: str, keep_count: int = 5) -> int:
    """
    Clean up old reference images, keeping only the most recent ones.
    
    Args:
        data_dir: Base data directory path
        keep_count: Number of recent images to keep
        
    Returns:
        Number of files cleaned up
    """
    try:
        reference_dir = os.path.join(data_dir, "reference")
        if not os.path.exists(reference_dir):
            return 0
        
        # Get all reference image files with timestamps
        image_files = []
        for filename in os.listdir(reference_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and filename.startswith('reference_'):
                file_path = os.path.join(reference_dir, filename)
                if ensure_file_exists(file_path):
                    image_files.append((file_path, os.path.getmtime(file_path)))
        
        if len(image_files) <= keep_count:
            return 0
        
        # Sort by modification time (oldest first for deletion)
        image_files.sort(key=lambda x: x[1])
        
        # Delete oldest files
        files_to_delete = image_files[:-keep_count]
        deleted_count = 0
        
        for file_path, _ in files_to_delete:
            try:
                os.remove(file_path)
                deleted_count += 1
                logging.debug(f"Deleted old reference image: {file_path}")
            except (OSError, PermissionError) as e:
                logging.error(f"Failed to delete {file_path}: {e}")
            except Exception as e:
                logging.error(f"Unexpected error deleting {file_path}: {e}")
        
        if deleted_count > 0:
            logging.info(f"Cleaned up {deleted_count} old reference images")
        return deleted_count
        
    except (OSError, PermissionError) as e:
        logging.error(f"File system error during reference image cleanup: {e}")
        return 0
    except Exception as e:
        logging.error(f"Unexpected error during reference image cleanup: {e}")
        return 0


def get_image_info(image: np.ndarray) -> dict:
    """Get basic information about an image."""
    if image is None:
        return {}
    
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    
    return {
        'width': width,
        'height': height,
        'channels': channels,
        'dtype': str(image.dtype),
        'size_bytes': image.nbytes
    }