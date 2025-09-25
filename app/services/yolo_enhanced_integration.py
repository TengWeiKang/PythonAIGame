"""
Integration module for YOLO backend with Enhanced Detection Service.

This module shows how to seamlessly integrate the enhanced detection extraction
with the existing YOLO backend for comprehensive object analysis.
"""

import time
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import logging

from ..core.entities import Detection
from ..backends.yolo_backend import YoloBackend
from .enhanced_detection_service import EnhancedDetectionService

logger = logging.getLogger(__name__)


class YoloEnhancedAnalyzer:
    """
    Analyzer that combines YOLO detection with comprehensive metadata extraction.

    This class provides a unified interface for running YOLO inference and
    extracting rich analytical data for AI-powered analysis.
    """

    def __init__(self, yolo_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced YOLO analyzer.

        Args:
            yolo_config: Configuration for YOLO backend
        """
        self.yolo_backend = YoloBackend(yolo_config or {})
        self.detection_service = EnhancedDetectionService(
            clustering_eps=50.0,  # Optimized for typical webcam scenarios
            clustering_min_samples=2,
            cache_size=100
        )
        self.class_names = None
        self._last_analysis = None

    def load_model(self, model_path: str) -> bool:
        """
        Load YOLO model and extract class names.

        Args:
            model_path: Path to YOLO model file

        Returns:
            True if successful
        """
        success = self.yolo_backend.load_model(model_path)

        if success:
            # Extract class names from model
            model_info = self.yolo_backend.get_model_info()
            self.class_names = model_info.get('class_names', None)
            logger.info(f"Loaded YOLO model with {len(self.class_names) if self.class_names else 0} classes")

        return success

    def analyze_frame(self, image: np.ndarray,
                      conf_threshold: float = 0.5,
                      iou_threshold: float = 0.45) -> Dict[str, Any]:
        """
        Run YOLO detection and extract comprehensive analysis data.

        Args:
            image: Input image as numpy array
            conf_threshold: Confidence threshold for YOLO
            iou_threshold: IOU threshold for NMS

        Returns:
            Dictionary with comprehensive detection analysis
        """
        start_time = time.perf_counter()

        # Run YOLO inference
        detections = self.yolo_backend.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        # Add class names to detections if available
        if self.class_names:
            for detection in detections:
                if detection.class_id < len(self.class_names):
                    detection.class_name = self.class_names[detection.class_id]

        # Extract comprehensive analysis
        analysis = self.detection_service.extract_comprehensive_detection_data(
            detections,
            image.shape,
            self.class_names
        )

        # Add YOLO-specific metadata
        analysis['yolo_metadata'] = {
            'model_info': self.yolo_backend.get_model_info(),
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'inference_time_ms': (time.perf_counter() - start_time) * 1000
        }

        # Store for reference
        self._last_analysis = analysis

        return analysis

    def compare_with_reference(self, current_analysis: Dict[str, Any],
                              reference_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare current frame analysis with a reference frame.

        Args:
            current_analysis: Analysis of current frame
            reference_analysis: Analysis of reference frame

        Returns:
            Comparison results with changes highlighted
        """
        comparison = {
            'timestamp': current_analysis['timestamp'],
            'reference_timestamp': reference_analysis['timestamp'],
            'changes': {},
            'similarity_metrics': {},
            'object_changes': {}
        }

        # Compare object counts
        curr_summary = current_analysis['summary']
        ref_summary = reference_analysis['summary']

        comparison['changes']['object_count'] = {
            'current': curr_summary['total_objects'],
            'reference': ref_summary['total_objects'],
            'difference': curr_summary['total_objects'] - ref_summary['total_objects']
        }

        # Compare class distributions
        curr_classes = set(curr_summary['class_distribution'].keys())
        ref_classes = set(ref_summary['class_distribution'].keys())

        comparison['changes']['classes'] = {
            'added': list(curr_classes - ref_classes),
            'removed': list(ref_classes - curr_classes),
            'common': list(curr_classes & ref_classes)
        }

        # Compare spatial patterns
        curr_patterns = set(current_analysis['spatial_analysis']['spatial_patterns'])
        ref_patterns = set(reference_analysis['spatial_analysis']['spatial_patterns'])

        comparison['changes']['spatial_patterns'] = {
            'current': list(curr_patterns),
            'reference': list(ref_patterns),
            'new_patterns': list(curr_patterns - ref_patterns),
            'lost_patterns': list(ref_patterns - curr_patterns)
        }

        # Calculate similarity score
        similarity_score = self._calculate_similarity(current_analysis, reference_analysis)
        comparison['similarity_metrics']['overall_score'] = similarity_score

        # Determine significance of changes
        change_significance = self._assess_change_significance(comparison)
        comparison['change_significance'] = change_significance

        return comparison

    def generate_ai_prompt_data(self, analysis: Dict[str, Any],
                               context: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate structured data optimized for AI prompt generation.

        Args:
            analysis: Detection analysis data
            context: Optional context for the AI analysis

        Returns:
            Structured data for AI prompts
        """
        summary = analysis['summary']
        spatial = analysis['spatial_analysis']

        # Create natural language descriptions
        scene_description = self._generate_scene_description(analysis)

        # Prepare structured prompt data
        prompt_data = {
            'scene_description': scene_description,
            'object_inventory': self._format_object_inventory(analysis['objects']),
            'spatial_relationships': self._format_spatial_relationships(spatial),
            'key_metrics': {
                'object_count': summary['total_objects'],
                'unique_classes': summary['unique_classes'],
                'average_confidence': summary['confidence_stats']['mean'],
                'scene_coverage': summary['coverage_stats']['total_coverage_percent'],
                'clustering_level': self._assess_clustering_level(spatial['clusters'])
            },
            'notable_features': self._identify_notable_features(analysis),
            'context': context
        }

        return prompt_data

    def _generate_scene_description(self, analysis: Dict[str, Any]) -> str:
        """Generate natural language scene description."""
        summary = analysis['summary']
        objects = analysis['objects']

        if not objects:
            return "The scene appears to be empty with no detected objects."

        # Start with object count
        description = f"The scene contains {summary['total_objects']} object"
        if summary['total_objects'] > 1:
            description += "s"

        # Add class distribution
        class_dist = summary['class_distribution']
        if class_dist:
            class_list = []
            for cls, count in class_dist.items():
                if count == 1:
                    class_list.append(f"1 {cls}")
                else:
                    class_list.append(f"{count} {cls}s")

            description += f": {', '.join(class_list)}"

        # Add spatial distribution
        position_dist = summary.get('position_distribution', {})
        if position_dist:
            dominant_position = max(position_dist.items(), key=lambda x: x[1])
            if dominant_position[1] > len(objects) / 2:
                description += f". Most objects are located in the {dominant_position[0]} region"

        # Add coverage information
        coverage = summary['coverage_stats']['total_coverage_percent']
        if coverage > 50:
            description += f". Objects cover a significant portion ({coverage:.1f}%) of the frame"
        elif coverage > 20:
            description += f". Objects moderately fill the frame ({coverage:.1f}% coverage)"
        else:
            description += f". Objects occupy a small portion of the frame ({coverage:.1f}% coverage)"

        return description

    def _format_object_inventory(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format object inventory for AI consumption."""
        inventory = []

        for i, obj in enumerate(objects):
            item = {
                'index': i,
                'class': obj['class_name'],
                'confidence': f"{obj['confidence']:.1%}",
                'position': obj['position'],
                'size': f"{obj['dimensions']['width']:.0f}x{obj['dimensions']['height']:.0f}",
                'orientation': obj['orientation']['type']
            }
            inventory.append(item)

        return inventory

    def _format_spatial_relationships(self, spatial: Dict[str, Any]) -> Dict[str, Any]:
        """Format spatial relationships for AI understanding."""
        relationships = {
            'clusters': [],
            'overlaps': [],
            'proximities': []
        }

        # Format clusters
        for cluster in spatial['clusters']:
            relationships['clusters'].append({
                'size': cluster['num_objects'],
                'cohesion': f"{cluster['cohesion_score']:.2f}",
                'dominant_class': cluster.get('dominant_class', 'mixed')
            })

        # Format significant overlaps
        for overlap in spatial['overlaps'][:5]:  # Top 5 overlaps
            if overlap['iou'] > 0.1:  # Only significant overlaps
                relationships['overlaps'].append({
                    'objects': f"{overlap['object1_idx']} and {overlap['object2_idx']}",
                    'overlap_level': overlap['overlap_type']
                })

        # Format close proximities
        for prox in spatial['proximities'][:5]:  # Top 5 proximities
            relationships['proximities'].append({
                'objects': f"{prox['object1_idx']} and {prox['object2_idx']}",
                'distance': f"{prox['distance']:.0f}px",
                'direction': prox['direction']
            })

        return relationships

    def _identify_notable_features(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify notable features in the scene."""
        features = []
        summary = analysis['summary']
        spatial = analysis['spatial_analysis']

        # Check for high confidence detections
        if summary['confidence_stats']['mean'] > 0.85:
            features.append("High confidence detections")

        # Check for clustering
        if len(spatial['clusters']) > 0:
            features.append(f"{len(spatial['clusters'])} object clusters detected")

        # Check for overlaps
        significant_overlaps = [o for o in spatial['overlaps'] if o['iou'] > 0.3]
        if significant_overlaps:
            features.append(f"{len(significant_overlaps)} significant object overlaps")

        # Check for spatial patterns
        patterns = spatial['spatial_patterns']
        if 'grid_formation' in patterns:
            features.append("Grid formation pattern detected")
        if 'horizontal_alignment' in patterns or 'vertical_alignment' in patterns:
            features.append("Object alignment detected")

        # Check coverage
        coverage = summary['coverage_stats']['total_coverage_percent']
        if coverage > 70:
            features.append("High scene density")
        elif coverage < 10:
            features.append("Sparse object distribution")

        return features

    def _calculate_similarity(self, current: Dict[str, Any],
                            reference: Dict[str, Any]) -> float:
        """Calculate similarity score between two analyses."""
        score = 0.0
        weights = {
            'object_count': 0.3,
            'class_match': 0.3,
            'spatial_patterns': 0.2,
            'coverage': 0.2
        }

        # Object count similarity
        curr_count = current['summary']['total_objects']
        ref_count = reference['summary']['total_objects']
        if ref_count > 0:
            count_sim = 1.0 - abs(curr_count - ref_count) / max(curr_count, ref_count)
            score += weights['object_count'] * count_sim

        # Class distribution similarity
        curr_classes = set(current['summary']['class_distribution'].keys())
        ref_classes = set(reference['summary']['class_distribution'].keys())
        if ref_classes:
            class_sim = len(curr_classes & ref_classes) / len(curr_classes | ref_classes)
            score += weights['class_match'] * class_sim

        # Spatial pattern similarity
        curr_patterns = set(current['spatial_analysis']['spatial_patterns'])
        ref_patterns = set(reference['spatial_analysis']['spatial_patterns'])
        if ref_patterns:
            pattern_sim = len(curr_patterns & ref_patterns) / len(curr_patterns | ref_patterns)
            score += weights['spatial_patterns'] * pattern_sim

        # Coverage similarity
        curr_coverage = current['summary']['coverage_stats']['total_coverage_percent']
        ref_coverage = reference['summary']['coverage_stats']['total_coverage_percent']
        if ref_coverage > 0:
            coverage_sim = 1.0 - abs(curr_coverage - ref_coverage) / max(curr_coverage, ref_coverage)
            score += weights['coverage'] * coverage_sim

        return min(score, 1.0)

    def _assess_change_significance(self, comparison: Dict[str, Any]) -> str:
        """Assess the significance of changes between frames."""
        object_diff = abs(comparison['changes']['object_count']['difference'])
        classes_added = len(comparison['changes']['classes']['added'])
        classes_removed = len(comparison['changes']['classes']['removed'])
        similarity = comparison['similarity_metrics']['overall_score']

        if similarity > 0.85 and object_diff <= 1:
            return "minimal"
        elif similarity > 0.6 and object_diff <= 3:
            return "minor"
        elif similarity > 0.4:
            return "moderate"
        else:
            return "major"

    def _assess_clustering_level(self, clusters: List[Dict[str, Any]]) -> str:
        """Assess the level of object clustering."""
        if not clusters:
            return "none"
        elif len(clusters) == 1:
            return "single_cluster"
        elif len(clusters) <= 3:
            return "moderate"
        else:
            return "high"

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.detection_service.get_performance_stats()

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.detection_service.reset_stats()


# Example usage function
def example_usage():
    """Example of how to use the enhanced YOLO analyzer."""
    import cv2

    # Create analyzer
    analyzer = YoloEnhancedAnalyzer()

    # Load YOLO model
    analyzer.load_model("yolo11n.pt")  # or any YOLO model

    # Capture frame from webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        # Analyze frame
        analysis = analyzer.analyze_frame(frame)

        # Generate AI prompt data
        prompt_data = analyzer.generate_ai_prompt_data(
            analysis,
            context="Monitoring classroom activity"
        )

        # Print scene description
        print(f"Scene: {prompt_data['scene_description']}")
        print(f"Notable features: {', '.join(prompt_data['notable_features'])}")

        # Save analysis to file
        with open("frame_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)

        print("Analysis saved to frame_analysis.json")


if __name__ == "__main__":
    example_usage()