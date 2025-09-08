"""Object Training Service for managing training datasets."""

import os
import json
import shutil
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import uuid
import logging

from ..config.settings import Config
from ..core.exceptions import ServiceError


class ObjectTrainingService:
    """Service for managing object training datasets."""
    
    def __init__(self, config: Config, data_dir: str = "data/training_objects"):
        self.config = config
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Dataset structure
        self.images_dir = self.data_dir / "images"
        self.metadata_dir = self.data_dir / "metadata"
        self.exports_dir = self.data_dir / "exports"
        
        # Create directories
        for directory in [self.images_dir, self.metadata_dir, self.exports_dir]:
            directory.mkdir(exist_ok=True)
        
        self._initialize_dataset_index()
    
    def _initialize_dataset_index(self) -> None:
        """Initialize or load the dataset index."""
        self.index_file = self.data_dir / "dataset_index.json"
        
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.dataset_index = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load dataset index: {e}")
                self.dataset_index = {"objects": {}, "classes": [], "version": "1.0"}
        else:
            self.dataset_index = {"objects": {}, "classes": [], "version": "1.0"}
            self._save_dataset_index()
    
    def _save_dataset_index(self) -> None:
        """Save the dataset index to file."""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.dataset_index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save dataset index: {e}")
            raise ServiceError(f"Failed to save dataset index: {e}")
    
    def save_object(self, name: str, image: np.ndarray, coordinates: Tuple[float, float, float, float],
                   source_image: Optional[np.ndarray] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a training object with metadata.
        
        Args:
            name: Object class name
            image: Cropped object image
            coordinates: Bounding box coordinates (x1, y1, x2, y2)
            source_image: Original source image (optional)
            metadata: Additional metadata (optional)
        
        Returns:
            str: Object ID
        """
        try:
            # Generate unique object ID
            object_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename
            filename = f"{name}_{timestamp}_{object_id[:8]}"
            
            # Save cropped object image
            object_image_path = self.images_dir / f"{filename}_object.jpg"
            cv2.imwrite(str(object_image_path), image)
            
            # Save source image if provided
            source_image_path = None
            if source_image is not None:
                source_image_path = self.images_dir / f"{filename}_source.jpg"
                cv2.imwrite(str(source_image_path), source_image)
            
            # Prepare object metadata
            object_metadata = {
                'id': object_id,
                'name': name,
                'timestamp': timestamp,
                'created_at': datetime.now().isoformat(),
                'confirmed': False,  # New field to track confirmation status
                'coordinates': {
                    'x1': float(coordinates[0]),
                    'y1': float(coordinates[1]),
                    'x2': float(coordinates[2]),
                    'y2': float(coordinates[3])
                },
                'image_path': str(object_image_path.relative_to(self.data_dir)),
                'source_image_path': str(source_image_path.relative_to(self.data_dir)) if source_image_path else None,
                'image_shape': image.shape,
                'source_image_shape': source_image.shape if source_image is not None else None
            }
            
            # Add custom metadata
            if metadata:
                object_metadata['custom'] = metadata
            
            # Save metadata file
            metadata_path = self.metadata_dir / f"{filename}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(object_metadata, f, indent=2, ensure_ascii=False)
            
            # Update dataset index
            self.dataset_index["objects"][object_id] = object_metadata
            
            # Update classes list
            if name not in self.dataset_index["classes"]:
                self.dataset_index["classes"].append(name)
                self.dataset_index["classes"].sort()
            
            # Save updated index
            self._save_dataset_index()
            
            self.logger.info(f"Saved object '{name}' with ID '{object_id}'")
            return object_id
            
        except Exception as e:
            self.logger.error(f"Failed to save object: {e}")
            raise ServiceError(f"Failed to save object: {e}")
    
    def load_objects(self, class_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load all training objects or objects of a specific class.
        
        Args:
            class_name: Optional class name filter
        
        Returns:
            List of object metadata dictionaries
        """
        try:
            objects = []
            
            for object_id, metadata in self.dataset_index["objects"].items():
                if class_name is None or metadata["name"] == class_name:
                    # Load the actual images
                    object_data = metadata.copy()
                    
                    # Load object image
                    object_image_path = self.data_dir / metadata["image_path"]
                    if object_image_path.exists():
                        object_data["image"] = cv2.imread(str(object_image_path))
                    else:
                        self.logger.warning(f"Object image not found: {object_image_path}")
                        object_data["image"] = None
                    
                    # Load source image if available
                    if metadata.get("source_image_path"):
                        source_image_path = self.data_dir / metadata["source_image_path"]
                        if source_image_path.exists():
                            object_data["source_image"] = cv2.imread(str(source_image_path))
                        else:
                            object_data["source_image"] = None
                    else:
                        object_data["source_image"] = None
                    
                    objects.append(object_data)
            
            # Sort by creation time (newest first)
            objects.sort(key=lambda x: x["created_at"], reverse=True)
            
            return objects
            
        except Exception as e:
            self.logger.error(f"Failed to load objects: {e}")
            raise ServiceError(f"Failed to load objects: {e}")
    
    def get_object(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific object by ID."""
        try:
            if object_id not in self.dataset_index["objects"]:
                return None
            
            metadata = self.dataset_index["objects"][object_id]
            object_data = metadata.copy()
            
            # Load images
            object_image_path = self.data_dir / metadata["image_path"]
            if object_image_path.exists():
                object_data["image"] = cv2.imread(str(object_image_path))
            else:
                object_data["image"] = None
            
            if metadata.get("source_image_path"):
                source_image_path = self.data_dir / metadata["source_image_path"]
                if source_image_path.exists():
                    object_data["source_image"] = cv2.imread(str(source_image_path))
                else:
                    object_data["source_image"] = None
            else:
                object_data["source_image"] = None
            
            return object_data
            
        except Exception as e:
            self.logger.error(f"Failed to get object {object_id}: {e}")
            raise ServiceError(f"Failed to get object: {e}")
    
    def update_object(self, object_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update object metadata.
        
        Args:
            object_id: Object ID
            updates: Dictionary of updates
        
        Returns:
            bool: Success status
        """
        try:
            if object_id not in self.dataset_index["objects"]:
                return False
            
            old_metadata = self.dataset_index["objects"][object_id]
            
            # Update allowed fields
            allowed_updates = ["name", "custom", "confirmed"]
            for key, value in updates.items():
                if key in allowed_updates:
                    old_metadata[key] = value
            
            # Update modified timestamp
            old_metadata["modified_at"] = datetime.now().isoformat()
            
            # If name changed, update classes list
            if "name" in updates:
                new_name = updates["name"]
                if new_name not in self.dataset_index["classes"]:
                    self.dataset_index["classes"].append(new_name)
                    self.dataset_index["classes"].sort()
                
                # Remove old class if no longer used
                old_name = old_metadata.get("name")
                if old_name and old_name != new_name:
                    if not any(obj["name"] == old_name for obj in self.dataset_index["objects"].values() if obj != old_metadata):
                        self.dataset_index["classes"].remove(old_name)
            
            # Update metadata file
            timestamp = old_metadata["timestamp"]
            filename = f"{old_metadata['name']}_{timestamp}_{object_id[:8]}"
            metadata_path = self.metadata_dir / f"{filename}.json"
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(old_metadata, f, indent=2, ensure_ascii=False)
            
            # Save updated index
            self._save_dataset_index()
            
            self.logger.info(f"Updated object {object_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update object {object_id}: {e}")
            raise ServiceError(f"Failed to update object: {e}")
    
    def delete_object(self, object_id: str) -> bool:
        """
        Delete a training object.
        
        Args:
            object_id: Object ID to delete
        
        Returns:
            bool: Success status
        """
        try:
            if object_id not in self.dataset_index["objects"]:
                return False
            
            metadata = self.dataset_index["objects"][object_id]
            
            # Delete image files
            try:
                object_image_path = self.data_dir / metadata["image_path"]
                if object_image_path.exists():
                    object_image_path.unlink()
                
                if metadata.get("source_image_path"):
                    source_image_path = self.data_dir / metadata["source_image_path"]
                    if source_image_path.exists():
                        source_image_path.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to delete image files for {object_id}: {e}")
            
            # Delete metadata file
            try:
                timestamp = metadata["timestamp"]
                filename = f"{metadata['name']}_{timestamp}_{object_id[:8]}"
                metadata_path = self.metadata_dir / f"{filename}.json"
                if metadata_path.exists():
                    metadata_path.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to delete metadata file for {object_id}: {e}")
            
            # Remove from index
            old_name = metadata["name"]
            del self.dataset_index["objects"][object_id]
            
            # Clean up classes list if needed
            if not any(obj["name"] == old_name for obj in self.dataset_index["objects"].values()):
                if old_name in self.dataset_index["classes"]:
                    self.dataset_index["classes"].remove(old_name)
            
            # Save updated index
            self._save_dataset_index()
            
            self.logger.info(f"Deleted object {object_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete object {object_id}: {e}")
            raise ServiceError(f"Failed to delete object: {e}")
    
    def get_classes(self) -> List[str]:
        """Get list of all object classes."""
        return self.dataset_index["classes"].copy()
    
    def get_object_count(self, class_name: Optional[str] = None) -> int:
        """Get count of objects, optionally filtered by class."""
        if class_name is None:
            return len(self.dataset_index["objects"])
        else:
            return sum(1 for obj in self.dataset_index["objects"].values() if obj["name"] == class_name)
    
    def export_dataset(self, export_format: str = "yolo", output_path: Optional[str] = None) -> str:
        """
        Export dataset in specified format.
        
        Args:
            export_format: Export format ('yolo', 'coco', 'pascal_voc')
            output_path: Optional output path
        
        Returns:
            str: Export path
        """
        try:
            if export_format.lower() == "yolo":
                return self._export_yolo_format(output_path)
            elif export_format.lower() == "coco":
                return self._export_coco_format(output_path)
            elif export_format.lower() == "pascal_voc":
                return self._export_pascal_voc_format(output_path)
            else:
                raise ServiceError(f"Unsupported export format: {export_format}")
                
        except Exception as e:
            self.logger.error(f"Failed to export dataset: {e}")
            raise ServiceError(f"Failed to export dataset: {e}")
    
    def _export_yolo_format(self, output_path: Optional[str] = None) -> str:
        """Export dataset in YOLO format."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.exports_dir / f"yolo_export_{timestamp}"
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create YOLO directory structure
        images_dir = output_path / "images"
        labels_dir = output_path / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        # Create classes.txt file
        classes_file = output_path / "classes.txt"
        with open(classes_file, 'w') as f:
            for class_name in self.dataset_index["classes"]:
                f.write(f"{class_name}\n")
        
        # Export objects
        for object_id, metadata in self.dataset_index["objects"].items():
            if metadata.get("source_image_path") and metadata.get("coordinates"):
                # Copy source image
                source_image_path = self.data_dir / metadata["source_image_path"]
                if source_image_path.exists():
                    timestamp = metadata["timestamp"]
                    filename = f"{metadata['name']}_{timestamp}_{object_id[:8]}"
                    
                    # Copy image
                    dest_image_path = images_dir / f"{filename}.jpg"
                    shutil.copy2(source_image_path, dest_image_path)
                    
                    # Create YOLO label
                    coords = metadata["coordinates"]
                    source_shape = metadata.get("source_image_shape", [480, 640, 3])
                    img_height, img_width = source_shape[:2]
                    
                    # Convert to YOLO format
                    class_id = self.dataset_index["classes"].index(metadata["name"])
                    center_x = (coords["x1"] + coords["x2"]) / 2 / img_width
                    center_y = (coords["y1"] + coords["y2"]) / 2 / img_height
                    width = (coords["x2"] - coords["x1"]) / img_width
                    height = (coords["y2"] - coords["y1"]) / img_height
                    
                    # Write label file
                    label_path = labels_dir / f"{filename}.txt"
                    with open(label_path, 'w') as f:
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        # Create dataset.yaml file
        yaml_content = f"""
# YOLO Dataset Configuration
path: {output_path.absolute()}
train: images
val: images

nc: {len(self.dataset_index["classes"])}
names: {self.dataset_index["classes"]}
"""
        
        yaml_file = output_path / "dataset.yaml"
        with open(yaml_file, 'w') as f:
            f.write(yaml_content.strip())
        
        self.logger.info(f"Exported YOLO dataset to {output_path}")
        return str(output_path)
    
    def _export_coco_format(self, output_path: Optional[str] = None) -> str:
        """Export dataset in COCO format."""
        # Implementation for COCO format export
        raise NotImplementedError("COCO format export not yet implemented")
    
    def _export_pascal_voc_format(self, output_path: Optional[str] = None) -> str:
        """Export dataset in Pascal VOC format."""
        # Implementation for Pascal VOC format export
        raise NotImplementedError("Pascal VOC format export not yet implemented")
    
    def validate_dataset(self) -> Dict[str, Any]:
        """Validate dataset integrity."""
        try:
            validation_results = {
                "valid": True,
                "issues": [],
                "statistics": {
                    "total_objects": len(self.dataset_index["objects"]),
                    "total_classes": len(self.dataset_index["classes"]),
                    "class_distribution": {}
                }
            }
            
            # Check class distribution
            for class_name in self.dataset_index["classes"]:
                count = self.get_object_count(class_name)
                validation_results["statistics"]["class_distribution"][class_name] = count
            
            # Validate file existence
            missing_files = []
            for object_id, metadata in self.dataset_index["objects"].items():
                # Check object image
                object_image_path = self.data_dir / metadata["image_path"]
                if not object_image_path.exists():
                    missing_files.append(f"Object image missing for {object_id}: {object_image_path}")
                
                # Check source image if specified
                if metadata.get("source_image_path"):
                    source_image_path = self.data_dir / metadata["source_image_path"]
                    if not source_image_path.exists():
                        missing_files.append(f"Source image missing for {object_id}: {source_image_path}")
            
            if missing_files:
                validation_results["valid"] = False
                validation_results["issues"].extend(missing_files)
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Failed to validate dataset: {e}")
            return {
                "valid": False,
                "issues": [f"Validation error: {e}"],
                "statistics": {}
            }
    
    def confirm_object(self, object_id: str) -> bool:
        """
        Confirm an object as ready for training.
        
        Args:
            object_id: Object ID to confirm
        
        Returns:
            bool: Success status
        """
        return self.update_object(object_id, {"confirmed": True})
    
    def unconfirm_object(self, object_id: str) -> bool:
        """
        Unconfirm an object (mark as not ready for training).
        
        Args:
            object_id: Object ID to unconfirm
        
        Returns:
            bool: Success status
        """
        return self.update_object(object_id, {"confirmed": False})
    
    def load_confirmed_objects(self, class_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load only confirmed training objects.
        
        Args:
            class_name: Optional class name filter
        
        Returns:
            List of confirmed object metadata dictionaries
        """
        try:
            objects = []
            
            for object_id, metadata in self.dataset_index["objects"].items():
                # Only include confirmed objects
                if not metadata.get("confirmed", False):
                    continue
                    
                if class_name is None or metadata["name"] == class_name:
                    # Load the actual images
                    object_data = metadata.copy()
                    
                    # Load object image
                    object_image_path = self.data_dir / metadata["image_path"]
                    if object_image_path.exists():
                        object_data["image"] = cv2.imread(str(object_image_path))
                    else:
                        self.logger.warning(f"Object image not found: {object_image_path}")
                        object_data["image"] = None
                    
                    # Load source image if available
                    if metadata.get("source_image_path"):
                        source_image_path = self.data_dir / metadata["source_image_path"]
                        if source_image_path.exists():
                            object_data["source_image"] = cv2.imread(str(source_image_path))
                        else:
                            object_data["source_image"] = None
                    else:
                        object_data["source_image"] = None
                    
                    objects.append(object_data)
            
            # Sort by creation time (newest first)
            objects.sort(key=lambda x: x["created_at"], reverse=True)
            
            return objects
            
        except Exception as e:
            self.logger.error(f"Failed to load confirmed objects: {e}")
            raise ServiceError(f"Failed to load confirmed objects: {e}")
    
    def get_confirmed_count(self, class_name: Optional[str] = None) -> int:
        """Get count of confirmed objects, optionally filtered by class."""
        if class_name is None:
            return sum(1 for obj in self.dataset_index["objects"].values() 
                      if obj.get("confirmed", False))
        else:
            return sum(1 for obj in self.dataset_index["objects"].values() 
                      if obj["name"] == class_name and obj.get("confirmed", False))
    
    def get_confirmation_status(self, object_id: str) -> Optional[bool]:
        """Get confirmation status of an object."""
        if object_id not in self.dataset_index["objects"]:
            return None
        return self.dataset_index["objects"][object_id].get("confirmed", False)
    
    def export_confirmed_dataset(self, export_format: str = "yolo", output_path: Optional[str] = None) -> str:
        """
        Export only confirmed objects in specified format.
        
        Args:
            export_format: Export format ('yolo', 'coco', 'pascal_voc')
            output_path: Optional output path
        
        Returns:
            str: Export path
        """
        try:
            if export_format.lower() == "yolo":
                return self._export_confirmed_yolo_format(output_path)
            else:
                raise ServiceError(f"Confirmed export for format '{export_format}' not yet implemented")
                
        except Exception as e:
            self.logger.error(f"Failed to export confirmed dataset: {e}")
            raise ServiceError(f"Failed to export confirmed dataset: {e}")
    
    def _export_confirmed_yolo_format(self, output_path: Optional[str] = None) -> str:
        """Export only confirmed objects in YOLO format."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.exports_dir / f"yolo_confirmed_export_{timestamp}"
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create YOLO directory structure
        images_dir = output_path / "images"
        labels_dir = output_path / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        # Get confirmed classes only
        confirmed_classes = []
        for object_id, metadata in self.dataset_index["objects"].items():
            if metadata.get("confirmed", False) and metadata["name"] not in confirmed_classes:
                confirmed_classes.append(metadata["name"])
        
        confirmed_classes.sort()
        
        # Create classes.txt file
        classes_file = output_path / "classes.txt"
        with open(classes_file, 'w') as f:
            for class_name in confirmed_classes:
                f.write(f"{class_name}\n")
        
        # Export confirmed objects only
        exported_count = 0
        for object_id, metadata in self.dataset_index["objects"].items():
            if not metadata.get("confirmed", False):
                continue
                
            if metadata.get("source_image_path") and metadata.get("coordinates"):
                # Copy source image
                source_image_path = self.data_dir / metadata["source_image_path"]
                if source_image_path.exists():
                    timestamp = metadata["timestamp"]
                    filename = f"{metadata['name']}_{timestamp}_{object_id[:8]}"
                    
                    # Copy image
                    dest_image_path = images_dir / f"{filename}.jpg"
                    shutil.copy2(source_image_path, dest_image_path)
                    
                    # Create YOLO label
                    coords = metadata["coordinates"]
                    source_shape = metadata.get("source_image_shape", [480, 640, 3])
                    img_height, img_width = source_shape[:2]
                    
                    # Convert to YOLO format
                    class_id = confirmed_classes.index(metadata["name"])
                    center_x = (coords["x1"] + coords["x2"]) / 2 / img_width
                    center_y = (coords["y1"] + coords["y2"]) / 2 / img_height
                    width = (coords["x2"] - coords["x1"]) / img_width
                    height = (coords["y2"] - coords["y1"]) / img_height
                    
                    # Write label file
                    label_path = labels_dir / f"{filename}.txt"
                    with open(label_path, 'w') as f:
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                    
                    exported_count += 1
        
        # Create dataset.yaml file
        yaml_content = f"""
# YOLO Dataset Configuration (Confirmed Objects Only)
path: {output_path.absolute()}
train: images
val: images

nc: {len(confirmed_classes)}
names: {confirmed_classes}
"""
        
        yaml_file = output_path / "dataset.yaml"
        with open(yaml_file, 'w') as f:
            f.write(yaml_content.strip())
        
        self.logger.info(f"Exported {exported_count} confirmed objects to YOLO dataset at {output_path}")
        return str(output_path)

    def cleanup_orphaned_files(self) -> int:
        """Clean up orphaned files not referenced in the index."""
        try:
            cleaned_count = 0
            
            # Get all referenced files
            referenced_files = set()
            for metadata in self.dataset_index["objects"].values():
                referenced_files.add(self.data_dir / metadata["image_path"])
                if metadata.get("source_image_path"):
                    referenced_files.add(self.data_dir / metadata["source_image_path"])
            
            # Check images directory
            for image_file in self.images_dir.iterdir():
                if image_file.is_file() and image_file not in referenced_files:
                    image_file.unlink()
                    cleaned_count += 1
            
            # Check metadata directory
            referenced_metadata_files = set()
            for object_id, metadata in self.dataset_index["objects"].items():
                timestamp = metadata["timestamp"]
                filename = f"{metadata['name']}_{timestamp}_{object_id[:8]}"
                referenced_metadata_files.add(self.metadata_dir / f"{filename}.json")
            
            for metadata_file in self.metadata_dir.iterdir():
                if metadata_file.is_file() and metadata_file not in referenced_metadata_files:
                    metadata_file.unlink()
                    cleaned_count += 1
            
            self.logger.info(f"Cleaned up {cleaned_count} orphaned files")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup orphaned files: {e}")
            return 0