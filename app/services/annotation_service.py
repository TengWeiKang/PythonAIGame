"""Annotation service for image labeling and annotation management."""
from __future__ import annotations
import os
import json
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image, ImageDraw
import cv2
import numpy as np
from ..core.entities import Detection, BBox
from ..core.exceptions import ValidationError
from ..utils.geometry import xyxy_to_xywh_norm, xywh_to_xyxy
from ..utils.file_utils import save_yolo_labels, read_yolo_labels
from ..config.settings import Config

class AnnotationService:
    """Service for handling image annotation operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.class_names = {}  # class_id -> name mapping

    def load_class_names(self, class_file_path: str) -> None:
        """Load class names from file."""
        try:
            if class_file_path.endswith('.yaml') or class_file_path.endswith('.yml'):
                import yaml
                with open(class_file_path, 'r') as f:
                    data = yaml.safe_load(f)
                    self.class_names = data.get('names', {})
            elif class_file_path.endswith('.json'):
                with open(class_file_path, 'r') as f:
                    self.class_names = json.load(f)
            else:
                # Assume it's a text file with class names per line
                with open(class_file_path, 'r') as f:
                    names = [line.strip() for line in f.readlines()]
                    self.class_names = {i: name for i, name in enumerate(names)}
        except Exception as e:
            print(f"Failed to load class names: {e}")

    def set_class_names(self, names: Dict[int, str]) -> None:
        """Set class names dictionary."""
        self.class_names = names

    def annotate_image(self, 
                      image: np.ndarray,
                      detections: List[Detection],
                      draw_labels: bool = True,
                      draw_confidence: bool = True,
                      draw_boxes: bool = True) -> np.ndarray:
        """Draw annotations on image."""
        annotated_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            class_name = self.class_names.get(detection.class_id, f"Class {detection.class_id}")
            
            # Draw bounding box
            if draw_boxes:
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label and confidence
            if draw_labels or draw_confidence:
                label_parts = []
                if draw_labels:
                    label_parts.append(class_name)
                if draw_confidence:
                    label_parts.append(f"{detection.score:.2f}")
                
                label = " ".join(label_parts)
                
                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Draw background rectangle
                cv2.rectangle(
                    annotated_image,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    (0, 255, 0),
                    -1
                )
                
                # Draw text
                cv2.putText(
                    annotated_image,
                    label,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
        
        return annotated_image

    def save_annotation(self, 
                       image_path: str,
                       detections: List[Detection],
                       output_dir: Optional[str] = None) -> str:
        """Save annotation in YOLO format."""
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
        
        # Get image dimensions
        try:
            image = Image.open(image_path)
            img_width, img_height = image.size
        except Exception as e:
            raise ValidationError(f"Failed to read image {image_path}: {e}")
        
        # Convert detections to YOLO format
        yolo_annotations = []
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            cx, cy, w, h = xyxy_to_xywh_norm((x1, y1, x2, y2), img_width, img_height)
            yolo_annotations.append([detection.class_id, cx, cy, w, h])
        
        # Save to file
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        annotation_path = os.path.join(output_dir, f"{base_name}.txt")
        
        save_yolo_labels(annotation_path, yolo_annotations)
        return annotation_path

    def load_annotations(self, annotation_path: str, image_width: int, image_height: int) -> List[Detection]:
        """Load annotations from YOLO format file."""
        yolo_labels = read_yolo_labels(annotation_path)
        detections = []
        
        for label in yolo_labels:
            if len(label) >= 5:
                class_id = int(label[0])
                cx, cy, w, h = label[1:5]
                
                # Convert from normalized to pixel coordinates
                x1, y1, x2, y2 = xywh_to_xyxy((cx, cy, w, h), image_width, image_height)
                
                detection = Detection(
                    class_id=class_id,
                    score=1.0,  # Assume perfect confidence for ground truth
                    bbox=(x1, y1, x2, y2)
                )
                detections.append(detection)
        
        return detections

    def create_annotation_project(self, 
                                 images_dir: str,
                                 project_name: str,
                                 class_names: List[str]) -> str:
        """Create a new annotation project structure."""
        project_dir = os.path.join(self.config.data_dir, "annotations", project_name)
        
        # Create directory structure
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(os.path.join(project_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "labels"), exist_ok=True)
        
        # Save class names
        class_mapping = {i: name for i, name in enumerate(class_names)}
        with open(os.path.join(project_dir, "classes.json"), 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        # Create project metadata
        metadata = {
            'name': project_name,
            'images_dir': images_dir,
            'num_classes': len(class_names),
            'classes': class_mapping,
            'created': os.path.getmtime(project_dir)
        }
        
        with open(os.path.join(project_dir, "project.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return project_dir

    def get_annotation_stats(self, annotations_dir: str) -> Dict[str, Any]:
        """Get statistics about annotations in a directory."""
        stats = {
            'total_files': 0,
            'total_annotations': 0,
            'class_distribution': {},
            'average_annotations_per_image': 0.0
        }
        
        annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]
        stats['total_files'] = len(annotation_files)
        
        if not annotation_files:
            return stats
        
        total_annotations = 0
        class_counts = {}
        
        for annotation_file in annotation_files:
            annotation_path = os.path.join(annotations_dir, annotation_file)
            labels = read_yolo_labels(annotation_path)
            
            total_annotations += len(labels)
            
            for label in labels:
                if len(label) >= 1:
                    class_id = int(label[0])
                    class_name = self.class_names.get(class_id, f"Class_{class_id}")
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        stats['total_annotations'] = total_annotations
        stats['class_distribution'] = class_counts
        stats['average_annotations_per_image'] = total_annotations / len(annotation_files)
        
        return stats