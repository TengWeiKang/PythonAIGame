"""Comprehensive unit tests for core entities.

Tests cover data validation, serialization, and business logic
for Detection, MasterObject, and other core entities.
"""
import pytest
import json
from dataclasses import FrozenInstanceError
from app.core.entities import Detection, MasterObject, DetectionResult, AnalysisResult
from app.core.exceptions import ValidationError


class TestDetection:
    """Test suite for Detection entity."""

    def test_valid_detection_creation(self):
        """Test creating a valid detection."""
        detection = Detection(
            class_id=0,
            score=0.85,
            bbox=(100, 100, 200, 200)
        )

        assert detection.class_id == 0
        assert detection.score == 0.85
        assert detection.bbox == (100, 100, 200, 200)

    def test_detection_with_optional_fields(self):
        """Test detection with all optional fields."""
        detection = Detection(
            class_id=1,
            score=0.92,
            bbox=(50, 50, 150, 150),
            class_name="person",
            confidence_level="high"
        )

        assert detection.class_name == "person"
        assert detection.confidence_level == "high"

    def test_detection_immutability(self):
        """Test that detection objects are immutable."""
        detection = Detection(
            class_id=0,
            score=0.85,
            bbox=(100, 100, 200, 200)
        )

        with pytest.raises(FrozenInstanceError):
            detection.score = 0.95

    def test_detection_bbox_validation(self):
        """Test bounding box validation."""
        # Valid bbox
        Detection(class_id=0, score=0.5, bbox=(10, 10, 50, 50))

        # Invalid bbox - wrong number of coordinates
        with pytest.raises(ValueError):
            Detection(class_id=0, score=0.5, bbox=(10, 10, 50))

        # Invalid bbox - x2 <= x1
        with pytest.raises(ValueError):
            Detection(class_id=0, score=0.5, bbox=(50, 10, 40, 50))

        # Invalid bbox - y2 <= y1
        with pytest.raises(ValueError):
            Detection(class_id=0, score=0.5, bbox=(10, 50, 50, 40))

        # Invalid bbox - negative coordinates
        with pytest.raises(ValueError):
            Detection(class_id=0, score=0.5, bbox=(-10, 10, 50, 50))

    def test_detection_score_validation(self):
        """Test score validation."""
        # Valid scores
        Detection(class_id=0, score=0.0, bbox=(10, 10, 50, 50))
        Detection(class_id=0, score=1.0, bbox=(10, 10, 50, 50))
        Detection(class_id=0, score=0.5, bbox=(10, 10, 50, 50))

        # Invalid scores
        with pytest.raises(ValueError):
            Detection(class_id=0, score=-0.1, bbox=(10, 10, 50, 50))

        with pytest.raises(ValueError):
            Detection(class_id=0, score=1.1, bbox=(10, 10, 50, 50))

    def test_detection_class_id_validation(self):
        """Test class ID validation."""
        # Valid class IDs
        Detection(class_id=0, score=0.5, bbox=(10, 10, 50, 50))
        Detection(class_id=999, score=0.5, bbox=(10, 10, 50, 50))

        # Invalid class IDs
        with pytest.raises(ValueError):
            Detection(class_id=-1, score=0.5, bbox=(10, 10, 50, 50))

    def test_detection_area_calculation(self):
        """Test bounding box area calculation."""
        detection = Detection(
            class_id=0,
            score=0.85,
            bbox=(10, 10, 60, 40)
        )

        area = detection.area()
        expected_area = (60 - 10) * (40 - 10)  # 50 * 30 = 1500
        assert area == expected_area

    def test_detection_center_calculation(self):
        """Test center point calculation."""
        detection = Detection(
            class_id=0,
            score=0.85,
            bbox=(10, 10, 60, 40)
        )

        center_x, center_y = detection.center()
        expected_x = (10 + 60) / 2  # 35
        expected_y = (10 + 40) / 2  # 25

        assert center_x == expected_x
        assert center_y == expected_y

    def test_detection_width_height(self):
        """Test width and height properties."""
        detection = Detection(
            class_id=0,
            score=0.85,
            bbox=(10, 10, 60, 40)
        )

        assert detection.width() == 50
        assert detection.height() == 30

    def test_detection_iou_calculation(self):
        """Test Intersection over Union calculation."""
        detection1 = Detection(
            class_id=0,
            score=0.85,
            bbox=(10, 10, 50, 50)
        )
        detection2 = Detection(
            class_id=0,
            score=0.90,
            bbox=(30, 30, 70, 70)
        )

        iou = detection1.iou(detection2)

        # Calculate expected IoU
        # Intersection: (30,30) to (50,50) = 20*20 = 400
        # Union: 40*40 + 40*40 - 400 = 1600 + 1600 - 400 = 2800
        # IoU = 400/2800 = 1/7 ≈ 0.143
        expected_iou = 400 / 2800
        assert abs(iou - expected_iou) < 0.001

    def test_detection_no_overlap_iou(self):
        """Test IoU calculation for non-overlapping boxes."""
        detection1 = Detection(
            class_id=0,
            score=0.85,
            bbox=(10, 10, 30, 30)
        )
        detection2 = Detection(
            class_id=0,
            score=0.90,
            bbox=(50, 50, 70, 70)
        )

        iou = detection1.iou(detection2)
        assert iou == 0.0

    def test_detection_identical_iou(self):
        """Test IoU calculation for identical boxes."""
        detection1 = Detection(
            class_id=0,
            score=0.85,
            bbox=(10, 10, 50, 50)
        )
        detection2 = Detection(
            class_id=1,
            score=0.90,
            bbox=(10, 10, 50, 50)
        )

        iou = detection1.iou(detection2)
        assert iou == 1.0

    def test_detection_serialization(self):
        """Test detection serialization to dict."""
        detection = Detection(
            class_id=1,
            score=0.85,
            bbox=(10, 20, 100, 200),
            class_name="person"
        )

        data = detection.to_dict()

        expected = {
            'class_id': 1,
            'score': 0.85,
            'bbox': (10, 20, 100, 200),
            'class_name': 'person',
            'confidence_level': None
        }

        assert data == expected

    def test_detection_deserialization(self):
        """Test detection deserialization from dict."""
        data = {
            'class_id': 1,
            'score': 0.85,
            'bbox': (10, 20, 100, 200),
            'class_name': 'person'
        }

        detection = Detection.from_dict(data)

        assert detection.class_id == 1
        assert detection.score == 0.85
        assert detection.bbox == (10, 20, 100, 200)
        assert detection.class_name == 'person'

    def test_detection_json_serialization(self):
        """Test JSON serialization round-trip."""
        detection = Detection(
            class_id=2,
            score=0.75,
            bbox=(5, 15, 95, 185),
            class_name="car"
        )

        # Serialize to JSON
        json_str = json.dumps(detection.to_dict())

        # Deserialize from JSON
        data = json.loads(json_str)
        restored_detection = Detection.from_dict(data)

        assert restored_detection == detection


class TestMasterObject:
    """Test suite for MasterObject entity."""

    def test_valid_master_object_creation(self):
        """Test creating a valid master object."""
        master_obj = MasterObject(
            class_id=0,
            name="test_object",
            bbox_norm=(0.1, 0.1, 0.2, 0.2)
        )

        assert master_obj.class_id == 0
        assert master_obj.name == "test_object"
        assert master_obj.bbox_norm == (0.1, 0.1, 0.2, 0.2)

    def test_master_object_immutability(self):
        """Test that master objects are immutable."""
        master_obj = MasterObject(
            class_id=0,
            name="test_object",
            bbox_norm=(0.1, 0.1, 0.2, 0.2)
        )

        with pytest.raises(FrozenInstanceError):
            master_obj.name = "new_name"

    def test_master_object_normalized_bbox_validation(self):
        """Test normalized bounding box validation."""
        # Valid normalized bbox
        MasterObject(
            class_id=0,
            name="valid",
            bbox_norm=(0.1, 0.1, 0.9, 0.9)
        )

        # Invalid - coordinates outside [0,1]
        with pytest.raises(ValueError):
            MasterObject(
                class_id=0,
                name="invalid",
                bbox_norm=(-0.1, 0.1, 0.9, 0.9)
            )

        with pytest.raises(ValueError):
            MasterObject(
                class_id=0,
                name="invalid",
                bbox_norm=(0.1, 0.1, 1.1, 0.9)
            )

        # Invalid - x2 <= x1
        with pytest.raises(ValueError):
            MasterObject(
                class_id=0,
                name="invalid",
                bbox_norm=(0.5, 0.1, 0.4, 0.9)
            )

    def test_master_object_to_absolute_coordinates(self):
        """Test conversion to absolute coordinates."""
        master_obj = MasterObject(
            class_id=0,
            name="test_object",
            bbox_norm=(0.25, 0.25, 0.75, 0.75)
        )

        image_width, image_height = 640, 480
        x1, y1, x2, y2 = master_obj.to_absolute(image_width, image_height)

        assert x1 == 160  # 0.25 * 640
        assert y1 == 120  # 0.25 * 480
        assert x2 == 480  # 0.75 * 640
        assert y2 == 360  # 0.75 * 480

    def test_master_object_area_calculation(self):
        """Test normalized area calculation."""
        master_obj = MasterObject(
            class_id=0,
            name="test_object",
            bbox_norm=(0.1, 0.2, 0.6, 0.8)
        )

        area = master_obj.area()
        expected_area = (0.6 - 0.1) * (0.8 - 0.2)  # 0.5 * 0.6 = 0.3
        assert abs(area - expected_area) < 0.001

    def test_master_object_center_calculation(self):
        """Test normalized center calculation."""
        master_obj = MasterObject(
            class_id=0,
            name="test_object",
            bbox_norm=(0.2, 0.3, 0.8, 0.7)
        )

        center_x, center_y = master_obj.center()
        expected_x = (0.2 + 0.8) / 2  # 0.5
        expected_y = (0.3 + 0.7) / 2  # 0.5

        assert center_x == expected_x
        assert center_y == expected_y

    def test_master_object_serialization(self):
        """Test master object serialization."""
        master_obj = MasterObject(
            class_id=1,
            name="car",
            bbox_norm=(0.1, 0.2, 0.5, 0.8),
            description="A red car"
        )

        data = master_obj.to_dict()

        expected = {
            'class_id': 1,
            'name': 'car',
            'bbox_norm': (0.1, 0.2, 0.5, 0.8),
            'description': 'A red car'
        }

        assert data == expected

    def test_master_object_deserialization(self):
        """Test master object deserialization."""
        data = {
            'class_id': 1,
            'name': 'car',
            'bbox_norm': (0.1, 0.2, 0.5, 0.8),
            'description': 'A red car'
        }

        master_obj = MasterObject.from_dict(data)

        assert master_obj.class_id == 1
        assert master_obj.name == 'car'
        assert master_obj.bbox_norm == (0.1, 0.2, 0.5, 0.8)
        assert master_obj.description == 'A red car'


class TestDetectionResult:
    """Test suite for DetectionResult entity."""

    def test_detection_result_creation(self, sample_detections):
        """Test creating a detection result."""
        result = DetectionResult(
            frame_id=123,
            timestamp=1634567890.5,
            detections=sample_detections,
            processing_time=0.15
        )

        assert result.frame_id == 123
        assert result.timestamp == 1634567890.5
        assert len(result.detections) == 3
        assert result.processing_time == 0.15

    def test_detection_result_immutability(self, sample_detections):
        """Test detection result immutability."""
        result = DetectionResult(
            frame_id=123,
            timestamp=1634567890.5,
            detections=sample_detections
        )

        with pytest.raises((FrozenInstanceError, AttributeError)):
            result.frame_id = 456

    def test_detection_result_empty_detections(self):
        """Test detection result with no detections."""
        result = DetectionResult(
            frame_id=123,
            timestamp=1634567890.5,
            detections=[]
        )

        assert len(result.detections) == 0
        assert result.has_detections() is False

    def test_detection_result_has_detections(self, sample_detections):
        """Test has_detections method."""
        result = DetectionResult(
            frame_id=123,
            timestamp=1634567890.5,
            detections=sample_detections
        )

        assert result.has_detections() is True

    def test_detection_result_get_detections_by_class(self, sample_detections):
        """Test filtering detections by class."""
        result = DetectionResult(
            frame_id=123,
            timestamp=1634567890.5,
            detections=sample_detections
        )

        class_0_detections = result.get_detections_by_class(0)
        assert len(class_0_detections) == 2

        class_1_detections = result.get_detections_by_class(1)
        assert len(class_1_detections) == 1

        class_2_detections = result.get_detections_by_class(2)
        assert len(class_2_detections) == 0

    def test_detection_result_get_high_confidence_detections(self, sample_detections):
        """Test filtering high-confidence detections."""
        result = DetectionResult(
            frame_id=123,
            timestamp=1634567890.5,
            detections=sample_detections
        )

        high_conf = result.get_high_confidence_detections(threshold=0.9)
        assert len(high_conf) == 1  # Only one detection >= 0.9

        medium_conf = result.get_high_confidence_detections(threshold=0.8)
        assert len(medium_conf) == 2  # Two detections >= 0.8

    def test_detection_result_statistics(self, sample_detections):
        """Test detection result statistics."""
        result = DetectionResult(
            frame_id=123,
            timestamp=1634567890.5,
            detections=sample_detections
        )

        stats = result.get_statistics()

        assert stats['total_detections'] == 3
        assert stats['unique_classes'] == 2
        assert stats['avg_confidence'] > 0
        assert stats['max_confidence'] == 0.92
        assert stats['min_confidence'] == 0.78

    def test_detection_result_serialization(self, sample_detections):
        """Test detection result serialization."""
        result = DetectionResult(
            frame_id=123,
            timestamp=1634567890.5,
            detections=sample_detections,
            processing_time=0.15
        )

        data = result.to_dict()

        assert data['frame_id'] == 123
        assert data['timestamp'] == 1634567890.5
        assert len(data['detections']) == 3
        assert data['processing_time'] == 0.15


class TestAnalysisResult:
    """Test suite for AnalysisResult entity."""

    def test_analysis_result_creation(self):
        """Test creating an analysis result."""
        result = AnalysisResult(
            request_id="req_123",
            analysis_type="image_comparison",
            result_text="Images are similar with minor differences",
            confidence=0.87,
            processing_time=2.3
        )

        assert result.request_id == "req_123"
        assert result.analysis_type == "image_comparison"
        assert "similar" in result.result_text
        assert result.confidence == 0.87
        assert result.processing_time == 2.3

    def test_analysis_result_validation(self):
        """Test analysis result validation."""
        # Valid confidence
        AnalysisResult(
            request_id="req_123",
            analysis_type="test",
            result_text="test result",
            confidence=0.5
        )

        # Invalid confidence - too high
        with pytest.raises(ValueError):
            AnalysisResult(
                request_id="req_123",
                analysis_type="test",
                result_text="test result",
                confidence=1.1
            )

        # Invalid confidence - negative
        with pytest.raises(ValueError):
            AnalysisResult(
                request_id="req_123",
                analysis_type="test",
                result_text="test result",
                confidence=-0.1
            )

    def test_analysis_result_with_metadata(self):
        """Test analysis result with additional metadata."""
        metadata = {
            "model_version": "1.5",
            "language": "en",
            "tokens_used": 150
        }

        result = AnalysisResult(
            request_id="req_123",
            analysis_type="image_analysis",
            result_text="Analysis complete",
            confidence=0.95,
            metadata=metadata
        )

        assert result.metadata == metadata
        assert result.metadata["model_version"] == "1.5"

    def test_analysis_result_serialization(self):
        """Test analysis result serialization."""
        result = AnalysisResult(
            request_id="req_123",
            analysis_type="image_analysis",
            result_text="Analysis complete",
            confidence=0.95,
            processing_time=1.5,
            metadata={"model": "gemini-1.5"}
        )

        data = result.to_dict()

        assert data['request_id'] == "req_123"
        assert data['analysis_type'] == "image_analysis"
        assert data['result_text'] == "Analysis complete"
        assert data['confidence'] == 0.95
        assert data['processing_time'] == 1.5
        assert data['metadata'] == {"model": "gemini-1.5"}


# Edge cases and error conditions
class TestEntityEdgeCases:
    """Test edge cases and error conditions for all entities."""

    def test_detection_with_zero_area_bbox(self):
        """Test detection with zero-area bounding box."""
        with pytest.raises(ValueError):
            Detection(
                class_id=0,
                score=0.5,
                bbox=(10, 10, 10, 10)  # Zero area
            )

    def test_master_object_with_zero_area_bbox(self):
        """Test master object with zero-area normalized bbox."""
        with pytest.raises(ValueError):
            MasterObject(
                class_id=0,
                name="zero_area",
                bbox_norm=(0.5, 0.5, 0.5, 0.5)  # Zero area
            )

    def test_detection_with_extremely_large_bbox(self):
        """Test detection with very large coordinates."""
        # Should work with large but valid coordinates
        detection = Detection(
            class_id=0,
            score=0.5,
            bbox=(0, 0, 9999, 9999)
        )

        assert detection.area() == 9999 * 9999

    def test_detection_precision_handling(self):
        """Test handling of floating-point precision in calculations."""
        detection = Detection(
            class_id=0,
            score=0.333333333,  # Repeating decimal
            bbox=(1, 1, 4, 4)
        )

        # Should handle precision correctly
        assert abs(detection.score - 0.333333333) < 1e-9

    def test_empty_string_handling(self):
        """Test handling of empty strings in optional fields."""
        # Empty class name should be allowed
        detection = Detection(
            class_id=0,
            score=0.5,
            bbox=(10, 10, 50, 50),
            class_name=""
        )

        assert detection.class_name == ""

        # Empty master object name should not be allowed
        with pytest.raises(ValueError):
            MasterObject(
                class_id=0,
                name="",  # Empty name
                bbox_norm=(0.1, 0.1, 0.9, 0.9)
            )

    def test_unicode_string_handling(self):
        """Test handling of Unicode strings."""
        # Unicode class names should work
        detection = Detection(
            class_id=0,
            score=0.5,
            bbox=(10, 10, 50, 50),
            class_name="人物"  # Chinese for "person"
        )

        assert detection.class_name == "人物"

        # Unicode master object names
        master_obj = MasterObject(
            class_id=0,
            name="объект",  # Russian for "object"
            bbox_norm=(0.1, 0.1, 0.9, 0.9)
        )

        assert master_obj.name == "объект"

    def test_very_long_string_handling(self):
        """Test handling of very long strings."""
        long_name = "a" * 1000

        # Should handle long names appropriately
        master_obj = MasterObject(
            class_id=0,
            name=long_name,
            bbox_norm=(0.1, 0.1, 0.9, 0.9)
        )

        assert len(master_obj.name) == 1000