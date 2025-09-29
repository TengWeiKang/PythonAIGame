"""Training service for model training operations."""
from __future__ import annotations
import os
import yaml
import shutil
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from ..core.exceptions import ModelError
from ..config.settings import Config
from ..core.device_utils import DeviceDetector, detect_and_validate_device

# Try to import ultralytics if present
HAS_ULTRALYTICS = False
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

class TrainingService:
    """Service for handling model training operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.training_in_progress = False

    def setup_training_config(self, dataset_path: str, class_names: list) -> str:
        """Create YAML configuration file for training."""
        config_path = os.path.join(self.config.data_dir, "training_config.yaml")
        
        training_config = {
            'path': os.path.abspath(dataset_path),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {i: name for i, name in enumerate(class_names)}
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(training_config, f, default_flow_style=False)
        
        return config_path

    def train_model_with_confirmed_objects(self,
                                          object_training_service,
                                          base_model: str = None,
                                          epochs: Optional[int] = None,
                                          batch_size: Optional[int] = None,
                                          save_path: Optional[str] = None,
                                          progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
        """
        Train a YOLO model using only confirmed objects.
        
        Args:
            object_training_service: ObjectTrainingService instance
            base_model: Base model path/name (e.g., 'yolo11n.pt')
            epochs: Number of training epochs
            batch_size: Training batch size
            save_path: Path to save the final trained model
            progress_callback: Callback for progress updates (message, progress)
        
        Returns:
            Dict containing training results and model path
        """
        if not HAS_ULTRALYTICS:
            raise ModelError("Ultralytics not available for training")
        
        if self.training_in_progress:
            raise ModelError("Training already in progress")
        
        try:
            self.training_in_progress = True
            
            if progress_callback:
                progress_callback("Checking confirmed objects...", 0.0)
            
            # Check if we have confirmed objects
            confirmed_count = object_training_service.get_confirmed_count()
            if confirmed_count == 0:
                raise ModelError("No confirmed objects found. Please confirm some objects first.")
            
            # Export confirmed objects to YOLO format
            if progress_callback:
                progress_callback("Exporting confirmed objects...", 0.1)
            
            dataset_path = object_training_service.export_confirmed_dataset("yolo")
            dataset_config_path = os.path.join(dataset_path, "dataset.yaml")
            
            if progress_callback:
                progress_callback("Setting up training configuration...", 0.2)
            
            # Use config values or defaults
            epochs = epochs or getattr(self.config, 'train_epochs', 50)
            batch_size = batch_size or getattr(self.config, 'batch_size', 16)
            base_model = base_model or getattr(self.config, 'model_size', 'yolo11n.pt')
            
            # Ensure models directory exists
            models_dir = Path(getattr(self.config, 'models_dir', 'models'))
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Create unique training run name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            training_name = f"training_run_{timestamp}"
            
            # Load base model
            if progress_callback:
                progress_callback("Loading base model...", 0.3)
            
            self.model = YOLO(base_model)
            
            # Detect and validate device
            prefer_gpu = getattr(self.config, 'use_gpu', False)
            device_info = DeviceDetector.detect_device(prefer_gpu)
            selected_device = device_info.device
            
            # Log device selection information
            device_info_str = DeviceDetector.get_device_info_string(device_info)
            print(f"Training Device Detection: {device_info_str}")
            
            if progress_callback:
                progress_callback(f"Using device: {device_info.device_name or selected_device}", 0.35)
            
            # Setup training parameters
            train_args = {
                'data': dataset_config_path,
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': getattr(self.config, 'img_size', 640),
                'device': selected_device,
                'project': str(models_dir),
                'name': training_name,
                'save': True,
                'save_period': max(1, epochs // 10),  # Save every 10% of epochs
                'patience': 50,  # Early stopping patience
                'verbose': True
            }
            
            if progress_callback:
                progress_callback("Starting model training...", 0.4)
            
            # Start training with progress tracking
            results = self._train_with_progress_tracking(train_args, progress_callback)
            
            # Get the best model path
            training_run_dir = models_dir / training_name
            best_model_path = training_run_dir / "weights" / "best.pt"
            last_model_path = training_run_dir / "weights" / "last.pt"
            
            if progress_callback:
                progress_callback("Saving trained model...", 0.95)
            
            # Save model to specified location or as "model.pt" in models directory
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                final_model_path = save_path
            else:
                # Always save as "model.pt" to create focused model persistence
                final_model_path = models_dir / "model.pt"
            
            # Copy the best model to the final location
            if best_model_path.exists():
                shutil.copy2(best_model_path, final_model_path)
                model_used = "best"
            elif last_model_path.exists():
                shutil.copy2(last_model_path, final_model_path)
                model_used = "last"
            else:
                raise ModelError("No trained model weights found")
            
            if progress_callback:
                progress_callback("Training completed successfully!", 1.0)
            
            return {
                'success': True,
                'model_path': str(final_model_path),
                'training_dir': str(training_run_dir),
                'dataset_path': dataset_path,
                'epochs_completed': epochs,
                'confirmed_objects_count': confirmed_count,
                'model_used': model_used,
                'results': results
            }
            
        except Exception as e:
            error_msg = f"Training failed: {e}"
            if progress_callback:
                progress_callback(error_msg, -1.0)
            raise ModelError(error_msg)
        
        finally:
            self.training_in_progress = False
    
    def _train_with_progress_tracking(self, train_args: Dict[str, Any], 
                                    progress_callback: Optional[Callable[[str, float], None]] = None):
        """Train model with progress tracking."""
        
        # Custom callback class to track training progress
        class ProgressTracker:
            def __init__(self, epochs, callback):
                self.epochs = epochs
                self.callback = callback
                self.current_epoch = 0
                
            def on_train_epoch_end(self, trainer):
                self.current_epoch += 1
                if self.callback:
                    progress = 0.4 + (self.current_epoch / self.epochs) * 0.5  # 40% to 90%
                    self.callback(f"Training epoch {self.current_epoch}/{self.epochs}...", progress)
        
        # Start training
        results = self.model.train(**train_args)
        return results

    def train_model(self, 
                   dataset_config_path: str,
                   base_model: str = None,
                   epochs: Optional[int] = None,
                   batch_size: Optional[int] = None,
                   progress_callback: Optional[Callable[[str, float], None]] = None) -> bool:
        """Train a new model (legacy method for backward compatibility)."""
        if not HAS_ULTRALYTICS:
            raise ModelError("Ultralytics not available for training")
        
        if self.training_in_progress:
            raise ModelError("Training already in progress")
        
        try:
            self.training_in_progress = True
            
            # Use config values or defaults
            epochs = epochs or getattr(self.config, 'train_epochs', 50)
            batch_size = batch_size or getattr(self.config, 'batch_size', 16)
            base_model = base_model or getattr(self.config, 'model_size', 'yolo11n.pt')
            
            # Load base model
            self.model = YOLO(base_model)
            
            # Detect and validate device
            prefer_gpu = getattr(self.config, 'use_gpu', False)
            device_info = DeviceDetector.detect_device(prefer_gpu)
            selected_device = device_info.device
            
            # Log device selection information
            device_info_str = DeviceDetector.get_device_info_string(device_info)
            print(f"Training Device Detection: {device_info_str}")
            
            # Setup training parameters
            train_args = {
                'data': dataset_config_path,
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': getattr(self.config, 'img_size', 640),
                'device': selected_device,
                'project': getattr(self.config, 'models_dir', 'models'),
                'name': 'training_run',
                'save': True,
                'save_period': max(1, epochs // 10)  # Save every 10% of epochs
            }
            
            if progress_callback:
                progress_callback("Starting training...", 0.0)
            
            # Start training
            results = self.model.train(**train_args)
            
            if progress_callback:
                progress_callback("Training completed!", 1.0)
            
            return True
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"Training failed: {e}", -1.0)
            raise ModelError(f"Training failed: {e}")
        
        finally:
            self.training_in_progress = False

    def validate_model(self, model_path: str, dataset_config_path: str) -> Dict[str, Any]:
        """Validate a trained model."""
        if not HAS_ULTRALYTICS:
            raise ModelError("Ultralytics not available for validation")
        
        try:
            model = YOLO(model_path)
            results = model.val(data=dataset_config_path)
            
            # Extract key metrics
            metrics = {
                'map50': float(results.box.map50),
                'map': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
            }
            
            return metrics
            
        except Exception as e:
            raise ModelError(f"Validation failed: {e}")

    def export_model(self, model_path: str, format: str = 'onnx') -> str:
        """Export model to different format."""
        if not HAS_ULTRALYTICS:
            raise ModelError("Ultralytics not available for export")
        
        try:
            model = YOLO(model_path)
            exported_path = model.export(format=format)
            return exported_path
            
        except Exception as e:
            raise ModelError(f"Export failed: {e}")

    def is_training(self) -> bool:
        """Check if training is currently in progress."""
        return self.training_in_progress

    def stop_training(self) -> None:
        """Stop current training (if possible)."""
        # Note: YOLO doesn't provide a direct way to stop training
        # This is a placeholder for future implementation
        self.training_in_progress = False

    def get_training_history(self) -> Dict[str, Any]:
        """Get training history from the last run."""
        results_path = os.path.join(self.config.models_dir, 'training_run', 'results.csv')
        
        if not os.path.exists(results_path):
            return {}
        
        try:
            import pandas as pd
            df = pd.read_csv(results_path)
            return df.to_dict('records')
        except ImportError:
            # If pandas not available, return basic info
            return {'message': 'Training history available in CSV format'}
        except Exception as e:
            return {'error': str(e)}