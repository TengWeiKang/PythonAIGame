"""Training service for YOLO model training with custom objects."""

import logging
import json
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
import numpy as np
import cv2
from datetime import datetime

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

    def __init__(self, data_dir: str = "data/training"):
        """Initialize training service.

        Args:
            data_dir: Base directory for training data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

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

    def _export_all_objects(self, export_dir: str) -> bool:
        """Export all objects as training dataset in YOLO format.

        Args:
            export_dir: Directory to export dataset

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

            return self._export_yolo_format(export_path, all_objects)

        except Exception as e:
            logger.error(f"Error exporting all objects: {e}")
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

    def train_model(self, epochs: int = 100, batch_size: int = 8, img_size: int = 640,
                   progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                   cancellation_check: Optional[Callable[[], bool]] = None,
                   device: Optional[str] = None,
                   model_architecture: str = "yolo11n.yaml") -> bool:
        """Train YOLO model from scratch with all objects.

        This method trains a completely custom YOLO model with random initialization
        (no pretrained weights). The model learns only from your custom dataset,
        making it fully specialized for your specific objects.

        Args:
            epochs: Number of training epochs (default 100 for training from scratch).
                   Training from scratch typically requires more epochs than fine-tuning.
            batch_size: Training batch size
            img_size: Image size for training
            progress_callback: Optional callback for progress updates (receives dict with metrics)
            cancellation_check: Optional callback that returns True if training should be cancelled
            device: Device to use for training ('auto', 'cuda', 'mps', 'cpu').
                   If None or 'auto', automatically detects best available device.
            model_architecture: YOLO architecture YAML file (default 'yolo11n.yaml').
                              Available architectures: yolo11n.yaml (nano), yolo11s.yaml (small),
                              yolo11m.yaml (medium), yolo11l.yaml (large), yolo11x.yaml (extra large).
                              The YAML file defines the network structure without pretrained weights.

        Returns:
            True if training completed successfully, False otherwise (includes cancellation)

        Note:
            Training from scratch (random weights) typically requires:
            - More training epochs (100+ recommended vs 10-50 for fine-tuning)
            - More training data for good results
            - Longer training time
            - The model will be completely custom to your dataset
        """
        try:
            from ultralytics import YOLO
            import shutil
            import time

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

            # Export dataset first (uses all objects)
            dataset_dir = self.data_dir / "exported_dataset"
            if not self._export_all_objects(str(dataset_dir)):
                logger.error("Failed to export dataset for training")
                return False

            # Check for cancellation before starting
            if cancellation_check and cancellation_check():
                logger.info("Training cancelled before starting")
                return False

            # Initialize model from architecture (training from scratch with random weights)
            logger.info(f"Initializing model architecture from scratch: {model_architecture}")
            logger.info("NOTE: Training from scratch with random initialization (no pretrained weights)")
            logger.info(f"This will take longer than fine-tuning but creates a fully custom model for your dataset")

            try:
                model = YOLO(model_architecture)
                logger.info(f"Model architecture loaded successfully: {model_architecture}")
            except Exception as e:
                logger.error(f"Failed to load model architecture '{model_architecture}': {e}")
                logger.info("Available architectures: yolo11n.yaml, yolo11s.yaml, yolo11m.yaml, yolo11l.yaml, yolo11x.yaml")
                logger.info("Falling back to yolo11n.yaml (nano architecture)")
                try:
                    model = YOLO('yolo11n.yaml')
                    logger.info("Fallback successful: using yolo11n.yaml")
                except Exception as e2:
                    logger.error(f"Fallback failed: {e2}")
                    return False

            # Prepare training arguments
            data_yaml = dataset_dir / "data.yaml"

            # Track training start time for ETA calculation
            training_start_time = time.time()
            epoch_times = []

            # Define callback for training progress updates
            def on_train_epoch_end(trainer):
                """Called at the end of each training epoch."""
                try:
                    # Check for cancellation
                    if cancellation_check and cancellation_check():
                        logger.info("Training cancellation requested")
                        trainer.stop = True  # Signal YOLO to stop training
                        return

                    # Extract metrics from trainer
                    current_epoch = trainer.epoch + 1  # YOLO uses 0-based indexing
                    total_epochs = trainer.epochs

                    # Calculate ETA
                    epoch_time = time.time() - training_start_time
                    epoch_times.append(epoch_time)

                    if len(epoch_times) > 1:
                        avg_epoch_time = sum(epoch_times) / len(epoch_times)
                        remaining_epochs = total_epochs - current_epoch
                        eta_seconds = int(avg_epoch_time * remaining_epochs)
                    else:
                        eta_seconds = 0

                    # Extract loss metrics (from trainer.metrics if available)
                    metrics = {
                        'epoch': current_epoch,
                        'total_epochs': total_epochs,
                        'eta_seconds': eta_seconds
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

                    logger.info(f"Epoch {current_epoch}/{total_epochs} completed")

                except Exception as e:
                    logger.error(f"Error in training callback: {e}")

            # Add callback to model
            model.add_callback('on_train_epoch_end', on_train_epoch_end)

            # Start training
            logger.info(f"Starting model training FROM SCRATCH on {device_name}...")
            logger.info(f"Training with {epochs} epochs on custom dataset")
            logger.info(f"Model will be initialized with random weights (no pretrained base)")

            # Notify start of training with device info
            if progress_callback:
                progress_callback({
                    'status': 'training_started',
                    'epoch': 0,
                    'total_epochs': epochs,
                    'device': device_name,
                    'device_type': training_device,
                    'training_mode': 'from_scratch'
                })

            # Train with device specification
            try:
                results = model.train(
                    data=str(data_yaml),
                    epochs=epochs,
                    batch=batch_size,
                    imgsz=img_size,
                    device=training_device,  # Specify device for training
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