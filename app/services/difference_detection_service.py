"""Real-time difference detection and highlighting service."""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
import threading
import time
from dataclasses import dataclass

from ..core.exceptions import WebcamError


@dataclass
class DifferenceRegion:
    """Represents a detected difference region."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    description: str = ""
    

@dataclass
class DifferenceDetectionResult:
    """Result of difference detection analysis."""
    regions: List[DifferenceRegion]
    overall_similarity: float
    processing_time_ms: float
    method_used: str


class DifferenceDetectionService:
    """Service for detecting and highlighting differences between images."""
    
    def __init__(self):
        """Initialize the difference detection service."""
        self._reference_image = None
        self._current_image = None
        self._detection_threshold = 30
        self._min_contour_area = 500
        self._blur_kernel_size = 5
        self._dilation_iterations = 2
        
        # For real-time processing
        self._is_processing = False
        self._processing_thread = None
        self._last_result = None
        self._callbacks = []
    
    def set_reference_image(self, image: np.ndarray) -> None:
        """Set the reference image for comparison."""
        self._reference_image = image.copy() if image is not None else None
    
    def set_detection_parameters(self, threshold: int = 30, min_area: int = 500, 
                                blur_size: int = 5, dilation_iter: int = 2):
        """Set parameters for difference detection."""
        self._detection_threshold = threshold
        self._min_contour_area = min_area
        self._blur_kernel_size = blur_size
        self._dilation_iterations = dilation_iter
    
    def add_callback(self, callback: Callable[[DifferenceDetectionResult], None]):
        """Add callback for real-time detection results."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[DifferenceDetectionResult], None]):
        """Remove callback from the list."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def detect_differences_opencv(self, reference: np.ndarray, 
                                 current: np.ndarray) -> DifferenceDetectionResult:
        """Detect differences using OpenCV methods."""
        start_time = time.time()
        
        if reference is None or current is None:
            return DifferenceDetectionResult([], 0.0, 0.0, "opencv_basic")
        
        try:
            # Resize images to same dimensions if needed
            ref_processed, curr_processed = self._prepare_images(reference, current)
            
            # Convert to grayscale
            ref_gray = cv2.cvtColor(ref_processed, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_processed, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(ref_gray, curr_gray)
            
            # Apply blur to reduce noise
            if self._blur_kernel_size > 0:
                diff = cv2.GaussianBlur(diff, (self._blur_kernel_size, self._blur_kernel_size), 0)
            
            # Threshold the difference
            _, thresh = cv2.threshold(diff, self._detection_threshold, 255, cv2.THRESH_BINARY)
            
            # Morphological operations to connect nearby differences
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            if self._dilation_iterations > 0:
                thresh = cv2.dilate(thresh, kernel, iterations=self._dilation_iterations)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours into regions
            regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= self._min_contour_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate confidence based on area and intensity
                    roi_diff = diff[y:y+h, x:x+w]
                    avg_intensity = np.mean(roi_diff)
                    confidence = min(1.0, (avg_intensity / 255.0) * (area / 1000.0))
                    
                    regions.append(DifferenceRegion(
                        x=x, y=y, width=w, height=h,
                        confidence=confidence,
                        description=f"Change detected (area: {int(area)}px)"
                    ))
            
            # Calculate overall similarity
            total_diff_pixels = np.count_nonzero(thresh)
            total_pixels = thresh.shape[0] * thresh.shape[1]
            similarity = 1.0 - (total_diff_pixels / total_pixels)
            
            processing_time = (time.time() - start_time) * 1000
            
            return DifferenceDetectionResult(
                regions=regions,
                overall_similarity=similarity,
                processing_time_ms=processing_time,
                method_used="opencv_basic"
            )
            
        except Exception as e:
            print(f"Error in difference detection: {e}")
            return DifferenceDetectionResult([], 0.0, 0.0, "opencv_basic_error")
    
    def detect_differences_advanced(self, reference: np.ndarray, 
                                   current: np.ndarray) -> DifferenceDetectionResult:
        """Advanced difference detection with feature matching."""
        start_time = time.time()
        
        if reference is None or current is None:
            return DifferenceDetectionResult([], 0.0, 0.0, "advanced")
        
        try:
            # Prepare images
            ref_processed, curr_processed = self._prepare_images(reference, current)
            
            # Multi-scale analysis
            regions = []
            
            # Scale 1: Full resolution
            result_full = self.detect_differences_opencv(ref_processed, curr_processed)
            regions.extend(result_full.regions)
            
            # Scale 2: Half resolution for broader changes
            ref_half = cv2.resize(ref_processed, None, fx=0.5, fy=0.5)
            curr_half = cv2.resize(curr_processed, None, fx=0.5, fy=0.5)
            result_half = self.detect_differences_opencv(ref_half, curr_half)
            
            # Scale back up the coordinates
            for region in result_half.regions:
                scaled_region = DifferenceRegion(
                    x=region.x * 2,
                    y=region.y * 2,
                    width=region.width * 2,
                    height=region.height * 2,
                    confidence=region.confidence * 0.8,  # Reduce confidence for scaled detection
                    description=f"Broad change detected (scaled)"
                )
                regions.append(scaled_region)
            
            # Remove duplicate/overlapping regions
            regions = self._merge_overlapping_regions(regions)
            
            # Calculate overall similarity (average of both scales)
            similarity = (result_full.overall_similarity + result_half.overall_similarity) / 2.0
            
            processing_time = (time.time() - start_time) * 1000
            
            return DifferenceDetectionResult(
                regions=regions,
                overall_similarity=similarity,
                processing_time_ms=processing_time,
                method_used="advanced_multiscale"
            )
            
        except Exception as e:
            print(f"Error in advanced difference detection: {e}")
            return DifferenceDetectionResult([], 0.0, 0.0, "advanced_error")
    
    def start_realtime_detection(self, update_interval_ms: int = 500):
        """Start real-time difference detection."""
        if self._is_processing:
            return
        
        self._is_processing = True
        self._processing_thread = threading.Thread(
            target=self._realtime_worker,
            args=(update_interval_ms / 1000.0,),
            daemon=True
        )
        self._processing_thread.start()
    
    def stop_realtime_detection(self):
        """Stop real-time difference detection."""
        self._is_processing = False
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=1.0)
    
    def update_current_image(self, image: np.ndarray):
        """Update the current image for comparison."""
        self._current_image = image.copy() if image is not None else None
    
    def get_last_result(self) -> Optional[DifferenceDetectionResult]:
        """Get the last detection result."""
        return self._last_result
    
    def create_highlighted_image(self, base_image: np.ndarray, 
                                regions: List[DifferenceRegion],
                                highlight_color: Tuple[int, int, int] = (0, 255, 0),
                                transparency: float = 0.3) -> np.ndarray:
        """Create image with highlighted difference regions."""
        if base_image is None or not regions:
            return base_image
        
        result = base_image.copy()
        overlay = base_image.copy()
        
        for region in regions:
            # Draw filled rectangle on overlay
            cv2.rectangle(
                overlay,
                (region.x, region.y),
                (region.x + region.width, region.y + region.height),
                highlight_color,
                -1
            )
            
            # Draw border
            cv2.rectangle(
                result,
                (region.x, region.y),
                (region.x + region.width, region.y + region.height),
                highlight_color,
                2
            )
            
            # Add confidence text
            if region.confidence > 0.5:
                text = f"{region.confidence:.2f}"
                cv2.putText(
                    result,
                    text,
                    (region.x, region.y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    highlight_color,
                    1
                )
        
        # Blend overlay with original
        cv2.addWeighted(overlay, transparency, result, 1 - transparency, 0, result)
        
        return result
    
    def _prepare_images(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare images for comparison (resize, align, etc.)."""
        if img1.shape != img2.shape:
            # Resize img2 to match img1
            height, width = img1.shape[:2]
            img2 = cv2.resize(img2, (width, height))
        
        return img1, img2
    
    def _merge_overlapping_regions(self, regions: List[DifferenceRegion]) -> List[DifferenceRegion]:
        """Merge overlapping or nearby difference regions."""
        if not regions:
            return regions
        
        # Sort by area (largest first)
        regions.sort(key=lambda r: r.width * r.height, reverse=True)
        
        merged = []
        for region in regions:
            # Check if this region overlaps significantly with any merged region
            merged_with_existing = False
            
            for merged_region in merged:
                overlap = self._calculate_overlap(region, merged_region)
                if overlap > 0.3:  # 30% overlap threshold
                    # Merge regions by expanding the existing one
                    merged_region.x = min(merged_region.x, region.x)
                    merged_region.y = min(merged_region.y, region.y)
                    merged_region.width = max(
                        merged_region.x + merged_region.width,
                        region.x + region.width
                    ) - merged_region.x
                    merged_region.height = max(
                        merged_region.y + merged_region.height,
                        region.y + region.height
                    ) - merged_region.y
                    merged_region.confidence = max(merged_region.confidence, region.confidence)
                    merged_with_existing = True
                    break
            
            if not merged_with_existing:
                merged.append(region)
        
        return merged
    
    def _calculate_overlap(self, region1: DifferenceRegion, region2: DifferenceRegion) -> float:
        """Calculate overlap ratio between two regions."""
        # Calculate intersection rectangle
        x1 = max(region1.x, region2.x)
        y1 = max(region1.y, region2.y)
        x2 = min(region1.x + region1.width, region2.x + region2.width)
        y2 = min(region1.y + region1.height, region2.y + region2.height)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0  # No intersection
        
        # Calculate areas
        intersection_area = (x2 - x1) * (y2 - y1)
        region1_area = region1.width * region1.height
        region2_area = region2.width * region2.height
        union_area = region1_area + region2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _realtime_worker(self, interval_seconds: float):
        """Background worker for real-time detection."""
        while self._is_processing:
            try:
                if self._reference_image is not None and self._current_image is not None:
                    # Perform detection
                    result = self.detect_differences_advanced(
                        self._reference_image, 
                        self._current_image
                    )
                    
                    self._last_result = result
                    
                    # Notify callbacks
                    for callback in self._callbacks:
                        try:
                            callback(result)
                        except Exception as e:
                            print(f"Error in difference detection callback: {e}")
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                print(f"Error in realtime detection worker: {e}")
                time.sleep(interval_seconds)


class DifferenceVisualizer:
    """Helper class for visualizing difference detection results."""
    
    @staticmethod
    def create_difference_overlay(reference: np.ndarray, current: np.ndarray,
                                regions: List[DifferenceRegion]) -> np.ndarray:
        """Create a side-by-side comparison with difference overlay."""
        if reference is None or current is None:
            return None
        
        # Prepare images
        ref_resized = reference.copy()
        curr_resized = current.copy()
        
        # Ensure same dimensions
        if ref_resized.shape != curr_resized.shape:
            height, width = ref_resized.shape[:2]
            curr_resized = cv2.resize(curr_resized, (width, height))
        
        # Create side-by-side comparison
        combined = np.hstack([ref_resized, curr_resized])
        
        # Draw difference regions on both sides
        height, width = ref_resized.shape[:2]
        
        for region in regions:
            color = (0, 255, 0) if region.confidence > 0.7 else (0, 255, 255)
            thickness = 2 if region.confidence > 0.5 else 1
            
            # Left side (reference)
            cv2.rectangle(
                combined,
                (region.x, region.y),
                (region.x + region.width, region.y + region.height),
                color,
                thickness
            )
            
            # Right side (current) - offset by width
            cv2.rectangle(
                combined,
                (region.x + width, region.y),
                (region.x + region.width + width, region.y + region.height),
                color,
                thickness
            )
            
            # Add labels
            if region.confidence > 0.5:
                cv2.putText(
                    combined,
                    f"{region.confidence:.2f}",
                    (region.x, region.y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
        
        return combined
    
    @staticmethod
    def create_heatmap(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Create a difference heatmap."""
        if reference is None or current is None:
            return None
        
        # Prepare images
        if reference.shape != current.shape:
            height, width = reference.shape[:2]
            current = cv2.resize(current, (width, height))
        
        # Convert to grayscale
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(ref_gray, curr_gray)
        
        # Apply colormap for heatmap effect
        heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        
        return heatmap