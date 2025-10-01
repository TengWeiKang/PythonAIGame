"""Training service for YOLO model training with custom objects."""

import logging
import json
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
import numpy as np
import cv2
from datetime import datetime

logger = logging.getLogger(__name__)


class TrainingObject:
    """Represents a training object with image and metadata."""

    def __init__(self, image: np.ndarray, label: str, bbox: Optional[tuple] = None,
                 confirmed: bool = False, object_id: Optional[str] = None):
        """Initialize training object.

        Args:
            image: Cropped object image
            label: Object class label
            bbox: Optional bounding box (x1, y1, x2, y2)
            confirmed: Whether object is confirmed for training
            object_id: Unique identifier
        """
        self.image = image
        self.label = label
        self.bbox = bbox
        self.confirmed = confirmed
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
            'confirmed': self.confirmed,
            'timestamp': self.timestamp.isoformat()
        }


class TrainingService:
    """Service for managing object training dataset and model training."""

    def __init__(self, data_dir: str = "data/training"):
        """Initialize training service.

        Args:
            data_dir: Base directory for training data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.objects: List[TrainingObject] = []
        self._load_objects()

    def add_object(self, image: np.ndarray, label: str, bbox: Optional[tuple] = None,
                   auto_confirm: bool = False) -> TrainingObject:
        """Add object to training dataset.

        Args:
            image: Cropped object image
            label: Object class label
            bbox: Optional bounding box
            auto_confirm: Whether to auto-confirm the object

        Returns:
            Created TrainingObject
        """
        obj = TrainingObject(image, label, bbox, confirmed=auto_confirm)
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

    def update_object(self, object_id: str, label: Optional[str] = None,
                     confirmed: Optional[bool] = None) -> bool:
        """Update object properties.

        Args:
            object_id: Object identifier
            label: New label (optional)
            confirmed: New confirmed status (optional)

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

        if confirmed is not None:
            obj.confirmed = confirmed
            logger.info(f"Updated object {object_id} confirmed status: {confirmed}")

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

    def get_confirmed_objects(self) -> List[TrainingObject]:
        """Get only confirmed training objects.

        Returns:
            List of confirmed TrainingObjects
        """
        return [obj for obj in self.objects if obj.confirmed]

    def get_object_count(self) -> Dict[str, int]:
        """Get count of objects by status.

        Returns:
            Dictionary with 'total', 'confirmed', 'unconfirmed' counts
        """
        confirmed = sum(1 for obj in self.objects if obj.confirmed)
        return {
            'total': len(self.objects),
            'confirmed': confirmed,
            'unconfirmed': len(self.objects) - confirmed
        }

    def export_dataset(self, export_dir: str, format: str = 'yolo') -> bool:
        """Export confirmed objects as training dataset.

        Args:
            export_dir: Directory to export dataset
            format: Dataset format ('yolo' or 'coco')

        Returns:
            True if exported successfully, False otherwise
        """
        try:
            export_path = Path(export_dir)
            export_path.mkdir(parents=True, exist_ok=True)

            confirmed = self.get_confirmed_objects()
            if not confirmed:
                logger.warning("No confirmed objects to export")
                return False

            if format == 'yolo':
                return self._export_yolo_format(export_path, confirmed)
            else:
                logger.error(f"Unsupported format: {format}")
                return False

        except Exception as e:
            logger.error(f"Error exporting dataset: {e}")
            return False

    def _export_yolo_format(self, export_dir: Path, objects: List[TrainingObject]) -> bool:
        """Export dataset in YOLO format.

        Args:
            export_dir: Export directory
            objects: List of objects to export

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

            # Export images and labels
            for idx, obj in enumerate(objects):
                # Save image
                img_name = f"{idx:04d}.jpg"
                img_path = images_dir / img_name
                cv2.imwrite(str(img_path), obj.image)

                # Create label file (center_x, center_y, width, height - normalized)
                h, w = obj.image.shape[:2]
                class_idx = class_map[obj.label]

                # For cropped objects, bbox is the full image
                label_path = labels_dir / f"{idx:04d}.txt"
                with open(label_path, 'w') as f:
                    # Full image bbox in normalized YOLO format
                    f.write(f"{class_idx} 0.5 0.5 1.0 1.0\n")

            # Create data.yaml
            yaml_content = {
                'path': str(export_dir.absolute()),
                'train': 'images/train',
                'val': 'images/train',  # Use same for validation
                'names': {idx: label for label, idx in class_map.items()}
            }

            yaml_path = export_dir / "data.yaml"
            with open(yaml_path, 'w') as f:
                import yaml
                yaml.dump(yaml_content, f)

            logger.info(f"Exported {len(objects)} objects to {export_dir}")
            return True

        except Exception as e:
            logger.error(f"Error in YOLO export: {e}")
            return False

    def train_model(self, base_model: str = "yolo12n.pt", epochs: int = 10,
                   batch_size: int = 8, img_size: int = 640,
                   progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> bool:
        """Train YOLO model with confirmed objects.

        Args:
            base_model: Base YOLO model to fine-tune
            epochs: Number of training epochs
            batch_size: Training batch size
            img_size: Image size for training
            progress_callback: Optional callback for progress updates

        Returns:
            True if training completed successfully, False otherwise
        """
        try:
            from ultralytics import YOLO
            import shutil

            # Export dataset first
            dataset_dir = self.data_dir / "exported_dataset"
            if not self.export_dataset(str(dataset_dir)):
                logger.error("Failed to export dataset for training")
                return False

            # Load base model
            logger.info(f"Loading base model: {base_model}")
            model = YOLO(base_model)

            # Prepare training arguments
            data_yaml = dataset_dir / "data.yaml"

            # Start training
            logger.info("Starting model training...")
            results = model.train(
                data=str(data_yaml),
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                verbose=True,
                project='runs/detect',
                name='train'
            )

            logger.info("Training completed successfully")

            # Copy trained model to the expected location for inference
            trained_model_path = Path("runs/detect/train/weights/best.pt")
            target_model_dir = Path("data/models")
            target_model_path = target_model_dir / "model.pt"

            if trained_model_path.exists():
                # Create models directory if it doesn't exist
                target_model_dir.mkdir(parents=True, exist_ok=True)

                # Copy the trained model to the expected location
                logger.info(f"Copying trained model from {trained_model_path} to {target_model_path}")
                shutil.copy2(str(trained_model_path), str(target_model_path))
                logger.info(f"Trained model successfully saved to {target_model_path}")

                # Also save a timestamped backup
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = target_model_dir / f"trained_model_{timestamp}.pt"
                shutil.copy2(str(trained_model_path), str(backup_path))
                logger.info(f"Backup model saved to {backup_path}")
            else:
                logger.error(f"Trained model not found at expected path: {trained_model_path}")
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
                            confirmed=obj_data.get('confirmed', False),
                            object_id=obj_data['object_id']
                        )
                        self.objects.append(obj)

            logger.info(f"Loaded {len(self.objects)} training objects")

        except Exception as e:
            logger.error(f"Error loading objects: {e}")