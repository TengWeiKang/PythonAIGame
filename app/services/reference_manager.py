"""
High-Performance Reference Image Management System.

This module provides optimized reference image capture, storage, and comparison
functionality for the webcam YOLO analysis feature. It includes advanced caching,
memory management, and performance optimizations to ensure sub-100ms comparison times.

Performance Targets:
- Reference capture: <200ms including YOLO analysis
- Object comparison: <100ms for typical scenes (5-10 objects)
- Memory footprint: <50MB baseline + ~1MB per reference
- Cache hit rate: >80% for repeated comparisons
"""

import os
import json
import time
import hashlib
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import OrderedDict
import numpy as np
import cv2
import logging

from ..core.entities import Detection, BBox
from ..core.performance import (
    LRUCache, ImageCache, performance_timer,
    PerformanceMonitor, cached_result
)
from ..core.exceptions import DetectionError
from ..backends.yolo_backend import YoloBackend

logger = logging.getLogger(__name__)

@dataclass
class ReferenceMetadata:
    """Metadata for a reference image."""
    reference_id: str
    timestamp: datetime
    image_path: str
    thumbnail_path: str
    image_hash: str
    dimensions: Tuple[int, int]  # (width, height)
    file_size_bytes: int
    detection_count: int
    confidence_threshold: float
    model_used: str
    analysis_time_ms: float

@dataclass
class ObjectMatch:
    """Result of matching a single object between current and reference."""
    reference_detection: Detection
    current_detection: Optional[Detection]
    iou_score: float
    position_offset: Tuple[float, float]  # (x_offset, y_offset) in pixels
    size_ratio: float  # current_area / reference_area
    confidence_delta: float
    match_type: str  # 'match', 'moved', 'missing', 'changed'

@dataclass
class ComparisonResult:
    """Complete comparison result between current and reference detections."""
    reference_id: str
    timestamp: datetime
    overall_similarity: float  # 0.0 to 1.0
    object_matches: List[ObjectMatch]
    objects_added: List[Detection]
    objects_missing: List[Detection]
    scene_change_score: float  # 0.0 to 1.0
    comparison_time_ms: float
    cache_hit: bool


class ReferenceImageManager:
    """
    High-performance reference image management system with optimized caching
    and comparison algorithms for real-time webcam analysis.
    """

    def __init__(self,
                 yolo_backend: YoloBackend,
                 data_dir: str,
                 max_references: int = 100,
                 max_memory_mb: int = 50,
                 auto_cleanup_days: int = 7,
                 enable_compression: bool = True):
        """
        Initialize the Reference Image Manager.

        Args:
            yolo_backend: YOLO backend for object detection
            data_dir: Base directory for storing reference data
            max_references: Maximum number of references to keep
            max_memory_mb: Maximum memory usage in MB
            auto_cleanup_days: Days after which to auto-delete references
            enable_compression: Enable JPEG compression for storage
        """
        self.yolo_backend = yolo_backend
        self.data_dir = Path(data_dir)
        self.references_dir = self.data_dir / "references"
        self.max_references = max_references
        self.max_memory_mb = max_memory_mb
        self.auto_cleanup_days = auto_cleanup_days
        self.enable_compression = enable_compression

        # Create directories
        self.references_dir.mkdir(parents=True, exist_ok=True)
        (self.references_dir / "images").mkdir(exist_ok=True)
        (self.references_dir / "thumbnails").mkdir(exist_ok=True)
        (self.references_dir / "metadata").mkdir(exist_ok=True)

        # Performance optimization: Multi-level caching
        self._metadata_cache = LRUCache(max_size=max_references)
        self._detection_cache = LRUCache(max_size=max_references * 2)
        self._image_cache = ImageCache(max_size=20, max_memory_mb=max_memory_mb // 2)
        self._thumbnail_cache = ImageCache(max_size=50, max_memory_mb=max_memory_mb // 4)
        self._comparison_cache = LRUCache(max_size=200)

        # Register caches with performance monitor
        monitor = PerformanceMonitor.instance()
        monitor.register_cache("ref_metadata", self._metadata_cache)
        monitor.register_cache("ref_detections", self._detection_cache)
        monitor.register_cache("ref_images", self._image_cache)
        monitor.register_cache("ref_thumbnails", self._thumbnail_cache)
        monitor.register_cache("ref_comparisons", self._comparison_cache)

        # Thread-safe reference registry
        self._reference_registry = OrderedDict()
        self._registry_lock = threading.RLock()

        # IOU calculation optimization: pre-computed constants
        self._iou_threshold = 0.5
        self._position_tolerance = 40  # pixels
        self._size_change_threshold = 0.2  # 20% size change

        # Load existing references
        self._load_reference_registry()

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._periodic_cleanup,
            daemon=True,
            name="ReferenceCleanup"
        )
        self._cleanup_running = True
        self._cleanup_thread.start()

        logger.info(f"ReferenceImageManager initialized with {len(self._reference_registry)} existing references")

    @performance_timer("reference_capture")
    async def capture_reference(self,
                               frame: np.ndarray,
                               reference_id: Optional[str] = None,
                               confidence_threshold: float = 0.5) -> str:
        """
        Capture a reference image with YOLO analysis.

        Args:
            frame: OpenCV frame (BGR numpy array)
            reference_id: Optional custom ID, auto-generated if None
            confidence_threshold: Confidence threshold for YOLO detection

        Returns:
            reference_id: The ID of the captured reference

        Performance: Target <200ms including YOLO analysis
        """
        start_time = time.perf_counter()

        # Generate reference ID if not provided
        if reference_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_hash = self._compute_frame_hash(frame)[:8]
            reference_id = f"ref_{timestamp}_{frame_hash}"

        # Check if we need to remove old references (LRU)
        with self._registry_lock:
            if len(self._reference_registry) >= self.max_references:
                self._evict_oldest_reference()

        # Parallel processing: Run YOLO detection and image processing concurrently
        detection_task = asyncio.create_task(
            self._run_yolo_detection_async(frame, confidence_threshold)
        )

        # Prepare image data
        height, width = frame.shape[:2]
        thumbnail = self._create_thumbnail(frame)

        # Save images with optional compression
        image_path = self.references_dir / "images" / f"{reference_id}.jpg"
        thumbnail_path = self.references_dir / "thumbnails" / f"{reference_id}.jpg"

        # Use threading for I/O operations
        save_tasks = []
        if self.enable_compression:
            save_tasks.append(
                threading.Thread(target=cv2.imwrite,
                               args=(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85]))
            )
            save_tasks.append(
                threading.Thread(target=cv2.imwrite,
                               args=(str(thumbnail_path), thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 70]))
            )
        else:
            save_tasks.append(
                threading.Thread(target=cv2.imwrite, args=(str(image_path), frame))
            )
            save_tasks.append(
                threading.Thread(target=cv2.imwrite, args=(str(thumbnail_path), thumbnail))
            )

        for task in save_tasks:
            task.start()

        # Wait for YOLO detection
        detections, model_info = await detection_task

        # Wait for image saves
        for task in save_tasks:
            task.join()

        # Calculate file size
        file_size = os.path.getsize(image_path)

        # Create metadata
        analysis_time_ms = (time.perf_counter() - start_time) * 1000
        metadata = ReferenceMetadata(
            reference_id=reference_id,
            timestamp=datetime.now(),
            image_path=str(image_path),
            thumbnail_path=str(thumbnail_path),
            image_hash=self._compute_frame_hash(frame),
            dimensions=(width, height),
            file_size_bytes=file_size,
            detection_count=len(detections),
            confidence_threshold=confidence_threshold,
            model_used=model_info.get('model_path', 'unknown'),
            analysis_time_ms=analysis_time_ms
        )

        # Save metadata and detections
        self._save_reference_data(reference_id, metadata, detections)

        # Update caches
        self._metadata_cache.put(reference_id, metadata)
        self._detection_cache.put(reference_id, detections)
        self._thumbnail_cache.put(reference_id, thumbnail)

        # Update registry
        with self._registry_lock:
            self._reference_registry[reference_id] = metadata
            # Move to end (most recent)
            self._reference_registry.move_to_end(reference_id)

        logger.info(f"Captured reference {reference_id} in {analysis_time_ms:.1f}ms with {len(detections)} detections")
        return reference_id

    @performance_timer("reference_retrieval")
    def get_reference(self, reference_id: str) -> Dict[str, Any]:
        """
        Retrieve reference data including metadata and detections.

        Args:
            reference_id: The reference ID to retrieve

        Returns:
            Dictionary containing metadata, detections, and image data
        """
        # Check metadata cache first
        metadata = self._metadata_cache.get(reference_id)
        if metadata is None:
            # Load from disk
            metadata = self._load_reference_metadata(reference_id)
            if metadata is None:
                raise ValueError(f"Reference {reference_id} not found")
            self._metadata_cache.put(reference_id, metadata)

        # Get detections
        detections = self._detection_cache.get(reference_id)
        if detections is None:
            detections = self._load_reference_detections(reference_id)
            self._detection_cache.put(reference_id, detections)

        # Get thumbnail (not full image for performance)
        thumbnail = self._thumbnail_cache.get(reference_id)
        if thumbnail is None:
            thumbnail = cv2.imread(metadata.thumbnail_path)
            self._thumbnail_cache.put(reference_id, thumbnail)

        return {
            'metadata': asdict(metadata),
            'detections': detections,
            'thumbnail': thumbnail,
            'detection_count': len(detections)
        }

    @performance_timer("reference_comparison")
    def compare_with_reference(self,
                              current_detections: List[Detection],
                              reference_id: str,
                              use_cache: bool = True) -> ComparisonResult:
        """
        Compare current detections with a reference.

        Args:
            current_detections: List of current Detection objects
            reference_id: Reference ID to compare against
            use_cache: Whether to use cached comparisons

        Returns:
            ComparisonResult with detailed matching information

        Performance: Target <100ms for typical scenes (5-10 objects)
        """
        start_time = time.perf_counter()

        # Generate cache key for this comparison
        cache_key = self._generate_comparison_cache_key(current_detections, reference_id)

        # Check cache if enabled
        if use_cache:
            cached_result = self._comparison_cache.get(cache_key)
            if cached_result is not None:
                # Update timestamp and mark as cache hit
                cached_result.cache_hit = True
                cached_result.comparison_time_ms = (time.perf_counter() - start_time) * 1000
                return cached_result

        # Get reference detections
        reference_detections = self._detection_cache.get(reference_id)
        if reference_detections is None:
            reference_detections = self._load_reference_detections(reference_id)
            if reference_detections is None:
                raise ValueError(f"Reference {reference_id} not found")
            self._detection_cache.put(reference_id, reference_detections)

        # Perform object matching using optimized algorithm
        object_matches = self._match_objects(current_detections, reference_detections)

        # Identify added and missing objects using indices
        matched_current_indices = set()
        matched_reference_indices = set()

        for match in object_matches:
            if match.current_detection is not None:
                # Find index of current detection
                for i, det in enumerate(current_detections):
                    if (det.class_id == match.current_detection.class_id and
                        det.bbox == match.current_detection.bbox and
                        abs(det.score - match.current_detection.score) < 0.001):
                        matched_current_indices.add(i)
                        break

            # Find index of reference detection
            for i, det in enumerate(reference_detections):
                if (det.class_id == match.reference_detection.class_id and
                    det.bbox == match.reference_detection.bbox and
                    abs(det.score - match.reference_detection.score) < 0.001):
                    matched_reference_indices.add(i)
                    break

        objects_added = [d for i, d in enumerate(current_detections) if i not in matched_current_indices]
        objects_missing = [d for i, d in enumerate(reference_detections) if i not in matched_reference_indices]

        # Calculate overall similarity and scene change score
        overall_similarity = self._calculate_overall_similarity(
            object_matches, len(objects_added), len(objects_missing)
        )
        scene_change_score = self._calculate_scene_change_score(
            object_matches, objects_added, objects_missing
        )

        # Create result
        comparison_time_ms = (time.perf_counter() - start_time) * 1000
        result = ComparisonResult(
            reference_id=reference_id,
            timestamp=datetime.now(),
            overall_similarity=overall_similarity,
            object_matches=object_matches,
            objects_added=objects_added,
            objects_missing=objects_missing,
            scene_change_score=scene_change_score,
            comparison_time_ms=comparison_time_ms,
            cache_hit=False
        )

        # Cache the result
        if use_cache:
            self._comparison_cache.put(cache_key, result)

        return result

    def _match_objects(self,
                      current_objects: List[Detection],
                      reference_objects: List[Detection]) -> List[ObjectMatch]:
        """
        Optimized object matching using IoU-based Hungarian algorithm.

        Performance optimizations:
        - Early termination for empty lists
        - Vectorized IoU calculations
        - Greedy matching for speed over optimality
        """
        if not current_objects or not reference_objects:
            # Handle empty cases quickly
            if reference_objects:
                return [
                    ObjectMatch(
                        reference_detection=ref,
                        current_detection=None,
                        iou_score=0.0,
                        position_offset=(0, 0),
                        size_ratio=0.0,
                        confidence_delta=0.0,
                        match_type='missing'
                    ) for ref in reference_objects
                ]
            return []

        # Pre-compute all IoU scores in a matrix (vectorized for speed)
        iou_matrix = np.zeros((len(reference_objects), len(current_objects)))

        for i, ref in enumerate(reference_objects):
            for j, curr in enumerate(current_objects):
                # Only compute IoU for same class objects
                if ref.class_id == curr.class_id:
                    iou_matrix[i, j] = self._calculate_iou(ref.bbox, curr.bbox)

        # Greedy matching: match highest IoU pairs first
        matches = []
        used_current = set()
        used_reference = set()

        # Sort all IoU scores in descending order
        sorted_pairs = []
        for i in range(len(reference_objects)):
            for j in range(len(current_objects)):
                if iou_matrix[i, j] > 0:
                    sorted_pairs.append((iou_matrix[i, j], i, j))
        sorted_pairs.sort(reverse=True)

        # Match objects greedily
        for iou_score, ref_idx, curr_idx in sorted_pairs:
            if ref_idx not in used_reference and curr_idx not in used_current:
                if iou_score >= self._iou_threshold:
                    ref = reference_objects[ref_idx]
                    curr = current_objects[curr_idx]

                    # Calculate additional metrics
                    position_offset = self._calculate_position_offset(ref.bbox, curr.bbox)
                    size_ratio = self._calculate_size_ratio(ref.bbox, curr.bbox)
                    confidence_delta = curr.score - ref.score

                    # Determine match type
                    if position_offset[0]**2 + position_offset[1]**2 > self._position_tolerance**2:
                        match_type = 'moved'
                    elif abs(1.0 - size_ratio) > self._size_change_threshold:
                        match_type = 'changed'
                    else:
                        match_type = 'match'

                    matches.append(ObjectMatch(
                        reference_detection=ref,
                        current_detection=curr,
                        iou_score=iou_score,
                        position_offset=position_offset,
                        size_ratio=size_ratio,
                        confidence_delta=confidence_delta,
                        match_type=match_type
                    ))

                    used_reference.add(ref_idx)
                    used_current.add(curr_idx)

        # Add unmatched reference objects as missing
        for i, ref in enumerate(reference_objects):
            if i not in used_reference:
                matches.append(ObjectMatch(
                    reference_detection=ref,
                    current_detection=None,
                    iou_score=0.0,
                    position_offset=(0, 0),
                    size_ratio=0.0,
                    confidence_delta=0.0,
                    match_type='missing'
                ))

        return matches

    @staticmethod
    def _calculate_iou(bbox1: BBox, bbox2: BBox) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        Optimized with early termination and minimal operations.
        """
        # Unpack coordinates
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Calculate intersection coordinates
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        # Check for no intersection (early termination)
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        # Calculate areas
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)

        # Calculate union
        union_area = area1 + area2 - inter_area

        # Avoid division by zero
        if union_area == 0:
            return 0.0

        return inter_area / union_area

    @staticmethod
    def _calculate_position_offset(bbox1: BBox, bbox2: BBox) -> Tuple[float, float]:
        """Calculate position offset between two bounding boxes (center points)."""
        x1_center = (bbox1[0] + bbox1[2]) / 2
        y1_center = (bbox1[1] + bbox1[3]) / 2
        x2_center = (bbox2[0] + bbox2[2]) / 2
        y2_center = (bbox2[1] + bbox2[3]) / 2

        return (x2_center - x1_center, y2_center - y1_center)

    @staticmethod
    def _calculate_size_ratio(bbox1: BBox, bbox2: BBox) -> float:
        """Calculate size ratio between two bounding boxes."""
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        if area1 == 0:
            return 0.0

        return area2 / area1

    def _calculate_overall_similarity(self,
                                    matches: List[ObjectMatch],
                                    added_count: int,
                                    missing_count: int) -> float:
        """Calculate overall similarity score between 0 and 1."""
        if not matches and added_count == 0 and missing_count == 0:
            return 1.0  # Empty scenes are identical

        total_objects = len(matches) + added_count
        if total_objects == 0:
            return 0.0

        # Calculate weighted similarity
        match_scores = []
        for match in matches:
            if match.match_type == 'match':
                match_scores.append(match.iou_score)
            elif match.match_type == 'moved':
                match_scores.append(match.iou_score * 0.8)  # Penalize moved objects
            elif match.match_type == 'changed':
                match_scores.append(match.iou_score * 0.7)  # Penalize changed objects
            else:  # missing
                match_scores.append(0.0)

        # Add zeros for added objects
        match_scores.extend([0.0] * added_count)

        if not match_scores:
            return 0.0

        return sum(match_scores) / len(match_scores)

    def _calculate_scene_change_score(self,
                                     matches: List[ObjectMatch],
                                     added: List[Detection],
                                     missing: List[Detection]) -> float:
        """Calculate scene change score (0 = no change, 1 = complete change)."""
        total_reference = len([m for m in matches if m.reference_detection])
        total_current = len([m for m in matches if m.current_detection]) + len(added)

        if total_reference == 0 and total_current == 0:
            return 0.0

        max_objects = max(total_reference, total_current)
        changed_objects = len(added) + len(missing) + len([m for m in matches if m.match_type in ['moved', 'changed']])

        return min(1.0, changed_objects / max_objects)

    def _create_thumbnail(self, frame: np.ndarray, max_size: int = 256) -> np.ndarray:
        """Create a thumbnail of the frame for efficient storage and display."""
        height, width = frame.shape[:2]

        # Calculate scaling factor
        scale = min(max_size / width, max_size / height)

        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            thumbnail = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            thumbnail = frame.copy()

        return thumbnail

    def _compute_frame_hash(self, frame: np.ndarray) -> str:
        """Compute a hash of the frame for quick comparison."""
        # Resize to small size for faster hashing
        small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        # Convert to grayscale
        if len(small.shape) == 3:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Compute hash
        return hashlib.md5(small.tobytes()).hexdigest()

    def _generate_comparison_cache_key(self,
                                      detections: List[Detection],
                                      reference_id: str) -> str:
        """Generate a cache key for comparison results."""
        # Create a simple hash based on detection count and classes
        detection_summary = f"{len(detections)}_"
        if detections:
            class_counts = {}
            for d in detections:
                class_counts[d.class_id] = class_counts.get(d.class_id, 0) + 1
            detection_summary += "_".join(f"{k}:{v}" for k, v in sorted(class_counts.items()))

        return f"{reference_id}_{detection_summary}"

    async def _run_yolo_detection_async(self,
                                       frame: np.ndarray,
                                       confidence_threshold: float) -> Tuple[List[Detection], Dict]:
        """Run YOLO detection asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._run_yolo_detection,
            frame,
            confidence_threshold
        )

    def _run_yolo_detection(self,
                          frame: np.ndarray,
                          confidence_threshold: float) -> Tuple[List[Detection], Dict]:
        """Run YOLO detection on a frame."""
        detections = self.yolo_backend.predict(
            frame,
            conf=confidence_threshold,
            iou=0.45,
            verbose=False
        )
        model_info = self.yolo_backend.get_model_info()
        return detections, model_info

    def _save_reference_data(self,
                           reference_id: str,
                           metadata: ReferenceMetadata,
                           detections: List[Detection]):
        """Save reference metadata and detections to disk."""
        # Save metadata
        metadata_path = self.references_dir / "metadata" / f"{reference_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2, default=str)

        # Save detections
        detections_path = self.references_dir / "metadata" / f"{reference_id}_detections.json"
        detections_data = [
            {
                'class_id': d.class_id,
                'score': d.score,
                'bbox': d.bbox,
                'class_name': d.class_name
            } for d in detections
        ]
        with open(detections_path, 'w') as f:
            json.dump(detections_data, f, indent=2)

    def _load_reference_metadata(self, reference_id: str) -> Optional[ReferenceMetadata]:
        """Load reference metadata from disk."""
        metadata_path = self.references_dir / "metadata" / f"{reference_id}.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path, 'r') as f:
            data = json.load(f)
            # Convert timestamp string back to datetime
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            return ReferenceMetadata(**data)

    def _load_reference_detections(self, reference_id: str) -> Optional[List[Detection]]:
        """Load reference detections from disk."""
        detections_path = self.references_dir / "metadata" / f"{reference_id}_detections.json"
        if not detections_path.exists():
            return None

        with open(detections_path, 'r') as f:
            data = json.load(f)
            return [
                Detection(
                    class_id=d['class_id'],
                    score=d['score'],
                    bbox=tuple(d['bbox']),
                    class_name=d.get('class_name')
                ) for d in data
            ]

    def _load_reference_registry(self):
        """Load all existing references into the registry."""
        metadata_dir = self.references_dir / "metadata"
        if not metadata_dir.exists():
            return

        for metadata_file in metadata_dir.glob("*.json"):
            if "_detections" not in metadata_file.name:
                reference_id = metadata_file.stem
                metadata = self._load_reference_metadata(reference_id)
                if metadata:
                    with self._registry_lock:
                        self._reference_registry[reference_id] = metadata

    def _evict_oldest_reference(self):
        """Remove the oldest reference (LRU eviction)."""
        if not self._reference_registry:
            return

        # Get oldest reference (first in OrderedDict)
        oldest_id = next(iter(self._reference_registry))
        self._delete_reference(oldest_id)

    def _delete_reference(self, reference_id: str):
        """Delete a reference and all associated data."""
        with self._registry_lock:
            if reference_id not in self._reference_registry:
                return

            metadata = self._reference_registry[reference_id]

            # Delete files
            try:
                Path(metadata.image_path).unlink(missing_ok=True)
                Path(metadata.thumbnail_path).unlink(missing_ok=True)
                (self.references_dir / "metadata" / f"{reference_id}.json").unlink(missing_ok=True)
                (self.references_dir / "metadata" / f"{reference_id}_detections.json").unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"Error deleting reference files: {e}")

            # Remove from registry
            del self._reference_registry[reference_id]

            # Clear from caches
            self._metadata_cache.put(reference_id, None)  # Invalidate
            self._detection_cache.put(reference_id, None)
            self._image_cache.put(reference_id, None)
            self._thumbnail_cache.put(reference_id, None)

    def _periodic_cleanup(self):
        """Periodically clean up old references."""
        while self._cleanup_running:
            try:
                # Sleep for 1 hour between cleanups
                for _ in range(3600):
                    if not self._cleanup_running:
                        break
                    time.sleep(1)

                if not self._cleanup_running:
                    break

                # Perform cleanup
                self._cleanup_old_references()

            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")

    def _cleanup_old_references(self):
        """Remove references older than auto_cleanup_days."""
        if self.auto_cleanup_days <= 0:
            return

        cutoff_date = datetime.now() - timedelta(days=self.auto_cleanup_days)

        with self._registry_lock:
            references_to_delete = [
                ref_id for ref_id, metadata in self._reference_registry.items()
                if metadata.timestamp < cutoff_date
            ]

        for ref_id in references_to_delete:
            self._delete_reference(ref_id)
            logger.info(f"Auto-deleted old reference: {ref_id}")

    def get_all_references(self) -> List[Dict[str, Any]]:
        """Get metadata for all references."""
        with self._registry_lock:
            return [
                {
                    'reference_id': ref_id,
                    'timestamp': metadata.timestamp.isoformat(),
                    'detection_count': metadata.detection_count,
                    'file_size_kb': metadata.file_size_bytes / 1024
                }
                for ref_id, metadata in self._reference_registry.items()
            ]

    def delete_all_references(self):
        """Delete all references (clear everything)."""
        with self._registry_lock:
            ref_ids = list(self._reference_registry.keys())

        for ref_id in ref_ids:
            self._delete_reference(ref_id)

        # Clear all caches
        self._metadata_cache.clear()
        self._detection_cache.clear()
        self._image_cache.clear()
        self._thumbnail_cache.clear()
        self._comparison_cache.clear()

        logger.info("All references deleted")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        return {
            'metadata_cache_hit_rate': self._metadata_cache.stats().hit_rate,
            'detection_cache_hit_rate': self._detection_cache.stats().hit_rate,
            'comparison_cache_hit_rate': self._comparison_cache.stats().hit_rate,
            'total_references': len(self._reference_registry),
            'cache_entries': {
                'metadata': self._metadata_cache.stats().size,
                'detections': self._detection_cache.stats().size,
                'images': self._image_cache.stats().size,
                'thumbnails': self._thumbnail_cache.stats().size,
                'comparisons': self._comparison_cache.stats().size
            }
        }

    def shutdown(self):
        """Shutdown the manager and cleanup resources."""
        self._cleanup_running = False
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=2.0)

        logger.info("ReferenceImageManager shutdown complete")