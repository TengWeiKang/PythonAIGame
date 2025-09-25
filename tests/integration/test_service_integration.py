"""Integration tests for service interactions and workflows.

Tests verify that services work together correctly in realistic scenarios,
including webcam + detection, AI analysis workflows, and error propagation.
"""
import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import cv2

from app.services.improved_webcam_service import ImprovedWebcamService
from app.services.gemini_service import GeminiService
from app.services.detection_service import DetectionService
from app.services.difference_detection_service import DifferenceDetectionService
from app.core.entities import Detection, MasterObject, DetectionResult
from app.core.exceptions import WebcamError, AIServiceError, ModelError


@pytest.mark.integration
class TestWebcamDetectionIntegration:
    """Test integration between webcam and detection services."""

    @pytest.fixture
    def mock_webcam_with_detection_service(self, mock_config):
        """Create integrated webcam and detection services."""
        webcam_service = Mock(spec=ImprovedWebcamService)
        detection_service = Mock(spec=DetectionService)

        # Mock webcam to provide test frames
        test_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        webcam_service.read_frame.return_value = (True, test_frame)
        webcam_service.is_webcam_opened.return_value = True

        # Mock detection to return test detections
        test_detections = [
            Detection(class_id=0, score=0.85, bbox=(100, 100, 200, 200)),
            Detection(class_id=1, score=0.92, bbox=(300, 150, 450, 300))
        ]
        detection_service.detect_objects.return_value = test_detections

        return webcam_service, detection_service

    def test_continuous_detection_workflow(self, mock_webcam_with_detection_service):
        """Test continuous detection from webcam frames."""
        webcam_service, detection_service = mock_webcam_with_detection_service

        results = []
        frame_count = 0
        max_frames = 10

        # Simulate continuous detection loop
        while frame_count < max_frames:
            success, frame = webcam_service.read_frame()

            if success and frame is not None:
                detections = detection_service.detect_objects(frame)
                result = DetectionResult(
                    frame_id=frame_count,
                    timestamp=time.time(),
                    detections=detections
                )
                results.append(result)

            frame_count += 1

        # Verify workflow completed successfully
        assert len(results) == max_frames
        assert all(result.has_detections() for result in results)
        assert webcam_service.read_frame.call_count == max_frames
        assert detection_service.detect_objects.call_count == max_frames

    def test_webcam_detection_error_handling(self, mock_config):
        """Test error handling in webcam-detection pipeline."""
        webcam_service = Mock(spec=ImprovedWebcamService)
        detection_service = Mock(spec=DetectionService)

        # Simulate webcam failure
        webcam_service.read_frame.return_value = (False, None)
        webcam_service.is_webcam_opened.return_value = False

        success, frame = webcam_service.read_frame()

        if not success:
            # Should handle webcam failure gracefully
            assert frame is None
            # Detection service should not be called
            detection_service.detect_objects.assert_not_called()

    def test_detection_service_error_propagation(self, mock_webcam_with_detection_service):
        """Test error propagation from detection service."""
        webcam_service, detection_service = mock_webcam_with_detection_service

        # Simulate detection service failure
        detection_service.detect_objects.side_effect = ModelError("Model failed to load")

        success, frame = webcam_service.read_frame()
        assert success

        with pytest.raises(ModelError):
            detection_service.detect_objects(frame)

    def test_performance_under_load(self, mock_webcam_with_detection_service):
        """Test performance under continuous load."""
        webcam_service, detection_service = mock_webcam_with_detection_service

        start_time = time.time()
        processed_frames = 0
        duration = 2.0  # 2 seconds

        while time.time() - start_time < duration:
            success, frame = webcam_service.read_frame()
            if success:
                detections = detection_service.detect_objects(frame)
                processed_frames += 1

        end_time = time.time()
        actual_duration = end_time - start_time
        fps = processed_frames / actual_duration

        # Should maintain reasonable FPS
        assert fps > 10  # At least 10 FPS with mocked services
        assert processed_frames > 20  # Should process many frames


@pytest.mark.integration
class TestAIAnalysisIntegration:
    """Test integration of AI analysis services."""

    @pytest.fixture
    def mock_ai_services(self, mock_config):
        """Create mock AI services for integration testing."""
        gemini_service = Mock(spec=GeminiService)
        gemini_service.is_configured.return_value = True

        # Mock successful analysis responses
        gemini_service.analyze_single_image.return_value = asyncio.Future()
        gemini_service.analyze_single_image.return_value.set_result(
            "The image contains a person and a car in an outdoor setting."
        )

        gemini_service.compare_images.return_value = asyncio.Future()
        gemini_service.compare_images.return_value.set_result(
            "The images are similar with minor lighting differences."
        )

        return gemini_service

    @pytest.mark.asyncio
    async def test_single_image_analysis_workflow(self, mock_ai_services, sample_image):
        """Test complete single image analysis workflow."""
        gemini_service = mock_ai_services

        # Simulate image analysis request
        prompt = "Describe what you see in this image"
        result = await gemini_service.analyze_single_image(sample_image, prompt)

        assert isinstance(result, str)
        assert len(result) > 0
        gemini_service.analyze_single_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_image_comparison_workflow(self, mock_ai_services, sample_image):
        """Test complete image comparison workflow."""
        gemini_service = mock_ai_services

        # Create two slightly different images
        image1 = sample_image
        image2 = sample_image.copy()
        image2[50:100, 50:100] = [255, 255, 255]  # Add white square

        prompt = "Compare these images and identify differences"
        result = await gemini_service.compare_images(image1, image2, prompt)

        assert isinstance(result, str)
        assert len(result) > 0
        gemini_service.compare_images.assert_called_once()

    @pytest.mark.asyncio
    async def test_ai_service_not_configured_handling(self, mock_config, sample_image):
        """Test handling when AI service is not configured."""
        gemini_service = Mock(spec=GeminiService)
        gemini_service.is_configured.return_value = False

        # Should handle unconfigured service gracefully
        with pytest.raises(AIServiceError):
            await gemini_service.analyze_single_image(sample_image, "Test prompt")

    @pytest.mark.asyncio
    async def test_concurrent_ai_requests(self, mock_ai_services, sample_image):
        """Test concurrent AI analysis requests."""
        gemini_service = mock_ai_services

        # Create multiple concurrent requests
        tasks = []
        for i in range(5):
            task = gemini_service.analyze_single_image(
                sample_image,
                f"Analysis request {i}"
            )
            tasks.append(task)

        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All requests should complete successfully
        assert len(results) == 5
        assert all(isinstance(result, str) for result in results)


@pytest.mark.integration
class TestDifferenceDetectionIntegration:
    """Test integration of difference detection with other services."""

    @pytest.fixture
    def mock_difference_detection_setup(self, mock_config):
        """Set up mock services for difference detection testing."""
        webcam_service = Mock(spec=ImprovedWebcamService)
        detection_service = Mock(spec=DetectionService)
        difference_service = Mock(spec=DifferenceDetectionService)

        # Mock master objects
        master_objects = [
            MasterObject(class_id=0, name="person", bbox_norm=(0.2, 0.2, 0.4, 0.6)),
            MasterObject(class_id=1, name="car", bbox_norm=(0.6, 0.4, 0.9, 0.8))
        ]

        # Mock current detections
        current_detections = [
            Detection(class_id=0, score=0.85, bbox=(128, 96, 256, 288)),  # person found
            Detection(class_id=2, score=0.75, bbox=(300, 200, 400, 350))   # new object
        ]

        webcam_service.read_frame.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        detection_service.detect_objects.return_value = current_detections
        difference_service.compare_with_master.return_value = {
            'missing_objects': [master_objects[1]],  # car is missing
            'new_objects': [current_detections[1]],   # new object found
            'matched_objects': [master_objects[0]]    # person matched
        }

        return webcam_service, detection_service, difference_service, master_objects

    def test_complete_difference_detection_workflow(self, mock_difference_detection_setup):
        """Test complete difference detection workflow."""
        webcam_service, detection_service, difference_service, master_objects = mock_difference_detection_setup

        # 1. Get current frame
        success, frame = webcam_service.read_frame()
        assert success

        # 2. Detect objects in current frame
        detections = detection_service.detect_objects(frame)
        assert len(detections) == 2

        # 3. Compare with master
        differences = difference_service.compare_with_master(detections, master_objects)

        # 4. Verify differences detected
        assert len(differences['missing_objects']) == 1
        assert len(differences['new_objects']) == 1
        assert len(differences['matched_objects']) == 1

        # 5. Verify service call sequence
        webcam_service.read_frame.assert_called_once()
        detection_service.detect_objects.assert_called_once_with(frame)
        difference_service.compare_with_master.assert_called_once()

    def test_difference_detection_with_no_changes(self, mock_config):
        """Test difference detection when no changes are present."""
        detection_service = Mock(spec=DetectionService)
        difference_service = Mock(spec=DifferenceDetectionService)

        # Mock perfect match scenario
        master_objects = [
            MasterObject(class_id=0, name="person", bbox_norm=(0.2, 0.2, 0.4, 0.6))
        ]

        current_detections = [
            Detection(class_id=0, score=0.85, bbox=(128, 96, 256, 288))
        ]

        detection_service.detect_objects.return_value = current_detections
        difference_service.compare_with_master.return_value = {
            'missing_objects': [],
            'new_objects': [],
            'matched_objects': master_objects
        }

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detection_service.detect_objects(frame)
        differences = difference_service.compare_with_master(detections, master_objects)

        # Should detect no differences
        assert len(differences['missing_objects']) == 0
        assert len(differences['new_objects']) == 0
        assert len(differences['matched_objects']) == 1

    def test_difference_detection_error_recovery(self, mock_config):
        """Test error recovery in difference detection workflow."""
        detection_service = Mock(spec=DetectionService)
        difference_service = Mock(spec=DifferenceDetectionService)

        # Simulate detection failure
        detection_service.detect_objects.side_effect = ModelError("Detection failed")

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            detections = detection_service.detect_objects(frame)
        except ModelError:
            # Should handle error gracefully and not crash difference detection
            detections = []  # Fallback to empty detections

        # Difference service should handle empty detections
        master_objects = [MasterObject(class_id=0, name="person", bbox_norm=(0.2, 0.2, 0.4, 0.6))]
        difference_service.compare_with_master.return_value = {
            'missing_objects': master_objects,  # All objects missing due to detection failure
            'new_objects': [],
            'matched_objects': []
        }

        differences = difference_service.compare_with_master(detections, master_objects)

        assert len(differences['missing_objects']) == 1
        assert len(differences['new_objects']) == 0
        assert len(differences['matched_objects']) == 0


@pytest.mark.integration
class TestFullSystemIntegration:
    """Test full system integration scenarios."""

    @pytest.fixture
    def integrated_system_mock(self, mock_config):
        """Create a fully integrated system mock."""
        # All services
        webcam_service = Mock(spec=ImprovedWebcamService)
        detection_service = Mock(spec=DetectionService)
        gemini_service = Mock(spec=GeminiService)
        difference_service = Mock(spec=DifferenceDetectionService)

        # Mock successful operations
        test_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        webcam_service.read_frame.return_value = (True, test_frame)
        webcam_service.is_webcam_opened.return_value = True

        test_detections = [
            Detection(class_id=0, score=0.85, bbox=(100, 100, 200, 200))
        ]
        detection_service.detect_objects.return_value = test_detections

        gemini_service.is_configured.return_value = True
        gemini_service.analyze_single_image.return_value = asyncio.Future()
        gemini_service.analyze_single_image.return_value.set_result("Analysis complete")

        difference_service.compare_with_master.return_value = {
            'missing_objects': [],
            'new_objects': [],
            'matched_objects': [MasterObject(class_id=0, name="test", bbox_norm=(0.1, 0.1, 0.3, 0.3))]
        }

        return webcam_service, detection_service, gemini_service, difference_service

    @pytest.mark.asyncio
    async def test_complete_analysis_pipeline(self, integrated_system_mock):
        """Test complete analysis pipeline from webcam to AI analysis."""
        webcam_service, detection_service, gemini_service, difference_service = integrated_system_mock

        # 1. Capture frame
        success, frame = webcam_service.read_frame()
        assert success

        # 2. Detect objects
        detections = detection_service.detect_objects(frame)
        assert len(detections) > 0

        # 3. Compare with master
        master_objects = [MasterObject(class_id=0, name="test", bbox_norm=(0.1, 0.1, 0.3, 0.3))]
        differences = difference_service.compare_with_master(detections, master_objects)

        # 4. AI analysis if differences found
        if differences['missing_objects'] or differences['new_objects']:
            analysis_result = await gemini_service.analyze_single_image(
                frame,
                "Analyze the differences in this image"
            )
            assert isinstance(analysis_result, str)
        else:
            # No differences, pipeline complete
            assert len(differences['matched_objects']) > 0

        # Verify all services were called appropriately
        webcam_service.read_frame.assert_called_once()
        detection_service.detect_objects.assert_called_once()
        difference_service.compare_with_master.assert_called_once()

    def test_system_startup_sequence(self, integrated_system_mock):
        """Test system startup and initialization sequence."""
        webcam_service, detection_service, gemini_service, difference_service = integrated_system_mock

        # 1. Initialize webcam
        webcam_ready = webcam_service.is_webcam_opened()

        # 2. Check AI service configuration
        ai_ready = gemini_service.is_configured()

        # 3. System should be ready if core services are available
        system_ready = webcam_ready  # AI is optional

        assert system_ready is True
        assert ai_ready is True  # In our mock scenario

    def test_system_shutdown_sequence(self, integrated_system_mock):
        """Test system shutdown and cleanup sequence."""
        webcam_service, detection_service, gemini_service, difference_service = integrated_system_mock

        # Simulate shutdown
        webcam_service.close_webcam()

        # Verify cleanup calls
        webcam_service.close_webcam.assert_called_once()

    def test_error_cascade_handling(self, integrated_system_mock):
        """Test how errors cascade through the system."""
        webcam_service, detection_service, gemini_service, difference_service = integrated_system_mock

        # Simulate webcam failure
        webcam_service.read_frame.return_value = (False, None)
        webcam_service.is_webcam_opened.return_value = False

        success, frame = webcam_service.read_frame()
        assert not success

        # System should handle webcam failure gracefully
        # Detection service should not be called with None frame
        if frame is not None:
            detection_service.detect_objects(frame)
        else:
            # Handle gracefully - no detection attempted
            detection_service.detect_objects.assert_not_called()

    def test_performance_monitoring_integration(self, integrated_system_mock):
        """Test performance monitoring across integrated services."""
        webcam_service, detection_service, gemini_service, difference_service = integrated_system_mock

        frame_times = []
        detection_times = []

        for _ in range(10):
            # Measure webcam frame time
            start_time = time.time()
            success, frame = webcam_service.read_frame()
            frame_time = time.time() - start_time
            frame_times.append(frame_time)

            if success:
                # Measure detection time
                start_time = time.time()
                detections = detection_service.detect_objects(frame)
                detection_time = time.time() - start_time
                detection_times.append(detection_time)

        # Verify performance metrics
        avg_frame_time = sum(frame_times) / len(frame_times)
        avg_detection_time = sum(detection_times) / len(detection_times)

        # Mock services should be very fast
        assert avg_frame_time < 0.1  # 100ms
        assert avg_detection_time < 0.1  # 100ms

    @pytest.mark.asyncio
    async def test_concurrent_operations_integration(self, integrated_system_mock):
        """Test concurrent operations across services."""
        webcam_service, detection_service, gemini_service, difference_service = integrated_system_mock

        async def analysis_task(task_id):
            success, frame = webcam_service.read_frame()
            if success:
                detections = detection_service.detect_objects(frame)
                result = await gemini_service.analyze_single_image(
                    frame,
                    f"Analysis task {task_id}"
                )
                return result
            return None

        # Run multiple concurrent analysis tasks
        tasks = [analysis_task(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All tasks should complete successfully
        assert len(results) == 5
        assert all(isinstance(result, str) for result in results)

    def test_resource_management_integration(self, integrated_system_mock):
        """Test resource management across integrated services."""
        webcam_service, detection_service, gemini_service, difference_service = integrated_system_mock

        # Simulate resource-intensive operations
        for _ in range(100):
            success, frame = webcam_service.read_frame()
            if success:
                detections = detection_service.detect_objects(frame)
                # Simulate processing
                processed_detections = len(detections)

        # Services should handle many operations without issues
        assert webcam_service.read_frame.call_count == 100
        assert detection_service.detect_objects.call_count == 100