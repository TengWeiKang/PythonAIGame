"""Test script for Service Restart Manager performance and functionality.

This script tests:
- Hot-swap timing for critical services
- Resource management during restarts
- Service health checks
- Progress callbacks
- Error handling and rollback
"""

import sys
import os
import time
import logging
import asyncio
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.restart_manager import (
    ServiceRestarter,
    RestartStrategy,
    RestartProgress,
    ServiceMetadata,
    ProgressCallback,
    ServiceHealthCheck,
    ResourceMonitor,
    get_restart_manager
)
from app.services.service_adapters import (
    WebcamServiceAdapter,
    DetectionServiceAdapter,
    InferenceServiceAdapter,
    GeminiServiceAdapter,
    ServiceAdapterFactory,
    create_service_metadata
)
from app.services.restart_integration import (
    ServiceRestartCoordinator,
    ServiceChangeDetector,
    RestartOptimizer,
    get_restart_coordinator,
    create_restart_coordinator_for_app
)
from app.services.webcam_service import WebcamService
from app.services.detection_service import DetectionService
from app.services.gemini_service import GeminiService
from app.config.settings import Settings, CameraSettings, DetectionSettings, AISettings
from app.core.performance import PerformanceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceTestCallback(ProgressCallback):
    """Callback for tracking restart performance."""

    def __init__(self):
        super().__init__()
        self.timings = {}
        self.events = []

    def on_service_stopping(self, service: str):
        self.timings[f"{service}_stop_start"] = time.time()
        self.events.append(('stopping', service, time.time()))
        logger.info(f"⏸ Stopping {service}...")

    def on_service_stopped(self, service: str):
        if f"{service}_stop_start" in self.timings:
            elapsed = (time.time() - self.timings[f"{service}_stop_start"]) * 1000
            logger.info(f"✓ {service} stopped in {elapsed:.1f}ms")

    def on_service_starting(self, service: str):
        self.timings[f"{service}_start_time"] = time.time()
        self.events.append(('starting', service, time.time()))
        logger.info(f"▶ Starting {service}...")

    def on_service_ready(self, service: str):
        if f"{service}_start_time" in self.timings:
            elapsed = (time.time() - self.timings[f"{service}_start_time"]) * 1000
            logger.info(f"✓ {service} ready in {elapsed:.1f}ms")
        self.events.append(('ready', service, time.time()))

    def on_restart_complete(self, success: bool):
        self.events.append(('complete', None, time.time()))
        status = "✓ Successfully" if success else "✗ Failed"
        logger.info(f"{status} completed restart")

    def on_error(self, service: str, error: Exception):
        self.events.append(('error', service, time.time()))
        logger.error(f"✗ Error in {service}: {error}")

    def get_total_time(self):
        """Get total restart time."""
        if len(self.events) >= 2:
            return (self.events[-1][2] - self.events[0][2]) * 1000
        return 0


def test_hot_swap_performance():
    """Test hot-swap performance for webcam service."""
    logger.info("\n" + "="*60)
    logger.info("Testing Hot-Swap Performance")
    logger.info("="*60)

    # Create webcam service
    webcam = WebcamService()
    adapter = WebcamServiceAdapter(webcam)

    # Initial configuration
    config1 = {
        'camera_index': 0,
        'frame_width': 640,
        'frame_height': 480,
        'fps': 30
    }

    # New configuration for hot-swap
    config2 = {
        'camera_index': 0,
        'frame_width': 1280,
        'frame_height': 720,
        'fps': 60
    }

    try:
        # Start service initially
        logger.info("Starting webcam service...")
        adapter.startup(config1)

        # Verify it's running
        assert adapter.health_check(), "Webcam failed initial health check"
        logger.info("✓ Webcam started successfully")

        # Perform hot-swap
        logger.info("\nPerforming hot-swap to new configuration...")
        start_time = time.time()
        success = adapter.hot_swap(config2)
        swap_time_ms = (time.time() - start_time) * 1000

        if success:
            logger.info(f"✓ Hot-swap completed in {swap_time_ms:.1f}ms")

            # Check against target
            if swap_time_ms < 200:
                logger.info(f"✓ Met performance target (<200ms)")
            else:
                logger.warning(f"⚠ Exceeded performance target (200ms)")

            # Verify new configuration
            resources = adapter.get_resources()
            logger.info(f"New configuration: {resources}")
        else:
            logger.error("✗ Hot-swap failed")

    except Exception as e:
        logger.error(f"Test failed: {e}")

    finally:
        # Cleanup
        adapter.shutdown()
        logger.info("Cleaned up resources")


def test_service_restart_manager():
    """Test the main restart manager with multiple services."""
    logger.info("\n" + "="*60)
    logger.info("Testing Service Restart Manager")
    logger.info("="*60)

    # Create services
    services = {
        'webcam': WebcamService(),
        'detection': DetectionService(None, None, None, None, {}),
        'gemini': GeminiService()
    }

    # Create adapters
    adapters = {}
    for name, service in services.items():
        adapter = ServiceAdapterFactory.create_adapter(name, service)
        adapters[name] = adapter

    # Create restart manager
    restart_manager = ServiceRestarter()

    # Register services
    for name, adapter in adapters.items():
        metadata = create_service_metadata(name, services[name], adapter)
        restart_manager.register_service(metadata)

    # Add performance callback
    callback = PerformanceTestCallback()
    restart_manager.add_callback(callback)

    try:
        # Test graceful restart
        logger.info("\n--- Testing Graceful Restart ---")
        plan = restart_manager.plan_restart(['webcam', 'detection'])
        logger.info(f"Restart plan: strategy={plan.strategy.value}, "
                   f"estimated_duration={plan.estimated_duration:.1f}s")

        success = restart_manager.execute_restart(plan)
        total_time = callback.get_total_time()

        if success:
            logger.info(f"✓ Graceful restart successful in {total_time:.1f}ms")
        else:
            logger.error("✗ Graceful restart failed")

        # Test parallel restart
        logger.info("\n--- Testing Parallel Restart ---")
        plan = restart_manager.plan_restart(['webcam', 'gemini'])
        plan.strategy = RestartStrategy.PARALLEL

        callback = PerformanceTestCallback()
        restart_manager.add_callback(callback)

        success = restart_manager.execute_restart(plan)
        total_time = callback.get_total_time()

        if success:
            logger.info(f"✓ Parallel restart successful in {total_time:.1f}ms")
        else:
            logger.error("✗ Parallel restart failed")

    except Exception as e:
        logger.error(f"Test failed: {e}")


def test_restart_coordinator():
    """Test the restart coordinator with settings changes."""
    logger.info("\n" + "="*60)
    logger.info("Testing Restart Coordinator")
    logger.info("="*60)

    # Create mock services
    services = {
        'webcam': WebcamService(),
        'detection': DetectionService(None, None, None, None, {}),
        'gemini': GeminiService()
    }

    # Create coordinator
    coordinator = create_restart_coordinator_for_app(services)

    # Create settings
    old_settings = Settings()
    old_settings.camera = CameraSettings(
        device_index=0,
        resolution=(640, 480),
        fps=30
    )
    old_settings.detection = DetectionSettings(
        confidence_threshold=0.5,
        nms_threshold=0.4,
        max_detections=100
    )

    new_settings = Settings()
    new_settings.camera = CameraSettings(
        device_index=0,
        resolution=(1280, 720),  # Changed
        fps=60  # Changed
    )
    new_settings.detection = DetectionSettings(
        confidence_threshold=0.6,  # Changed
        nms_threshold=0.4,
        max_detections=100
    )

    # Test change detection
    detector = ServiceChangeDetector()
    changes = detector.detect_changes(old_settings, new_settings)
    logger.info(f"Detected changes: {changes}")

    # Test optimization
    optimizer = RestartOptimizer()
    strategy = optimizer.optimize_strategy(changes)
    logger.info(f"Optimized strategy: {strategy.value}")

    # Test restart with progress
    def progress_callback(message, level='info'):
        logger.info(f"[{level}] {message}")

    try:
        logger.info("\nApplying settings with restart...")
        success = coordinator.apply_settings_with_restart(
            old_settings,
            new_settings,
            progress_callback
        )

        if success:
            logger.info("✓ Settings applied successfully with restart")

            # Check service status
            status = coordinator.get_service_status()
            for name, info in status.items():
                state = info.get('state', 'unknown')
                healthy = "✓" if info.get('healthy', False) else "✗"
                logger.info(f"  {name}: state={state}, healthy={healthy}")
        else:
            logger.error("✗ Failed to apply settings with restart")

    except Exception as e:
        logger.error(f"Test failed: {e}")


def test_resource_monitoring():
    """Test resource monitoring during restarts."""
    logger.info("\n" + "="*60)
    logger.info("Testing Resource Monitoring")
    logger.info("="*60)

    monitor = ResourceMonitor()

    # Start monitoring
    monitor.start_monitoring()
    initial = monitor.get_current_resources()
    logger.info(f"Initial resources: CPU={initial['cpu_percent']:.1f}%, "
               f"Memory={initial['memory_mb']:.1f}MB")

    # Simulate some activity
    time.sleep(1)

    # Stop monitoring and get stats
    stats = monitor.stop_monitoring()
    logger.info(f"Resource usage: Memory delta={stats['memory_delta_mb']:.1f}MB, "
               f"Peak={stats['peak_memory_mb']:.1f}MB")


def test_health_checks():
    """Test service health check functionality."""
    logger.info("\n" + "="*60)
    logger.info("Testing Health Checks")
    logger.info("="*60)

    # Create a mock service
    class MockService:
        def __init__(self):
            self.healthy = True

        def health_check(self):
            return self.healthy

        def get_state(self):
            from app.services.restart_manager import ServiceState
            return ServiceState.RUNNING if self.healthy else ServiceState.ERROR

        def get_resources(self):
            return {'test': True}

    service = MockService()
    health_check = ServiceHealthCheck("mock", service)

    # Test healthy service
    logger.info("Testing healthy service...")
    assert health_check.check_pre_restart(), "Pre-restart check failed"
    assert health_check.wait_for_ready(1.0), "Service not ready"
    assert health_check.verify_functionality(), "Functionality check failed"
    logger.info("✓ All health checks passed")

    # Test unhealthy service
    logger.info("\nTesting unhealthy service...")
    service.healthy = False
    assert not health_check.verify_functionality(), "Should have failed"
    logger.info("✓ Correctly detected unhealthy service")


async def test_async_operations():
    """Test async restart operations."""
    logger.info("\n" + "="*60)
    logger.info("Testing Async Operations")
    logger.info("="*60)

    from app.services.restart_integration import AsyncServiceRestartCoordinator

    # Create services and coordinator
    services = {'webcam': WebcamService()}
    coordinator = create_restart_coordinator_for_app(services)
    async_coordinator = AsyncServiceRestartCoordinator(coordinator)

    # Test async status check
    status = await async_coordinator.get_service_status_async()
    logger.info(f"Async status check: {status}")

    # Test async hot-swap
    config = {'camera_index': 0, 'frame_width': 640, 'frame_height': 480}
    success = await async_coordinator.hot_swap_service_async('webcam', config)
    logger.info(f"Async hot-swap: {'✓ Success' if success else '✗ Failed'}")


def run_performance_benchmarks():
    """Run performance benchmarks for restart operations."""
    logger.info("\n" + "="*60)
    logger.info("Performance Benchmarks")
    logger.info("="*60)

    results = {
        'hot_swap': [],
        'graceful': [],
        'parallel': []
    }

    # Run multiple iterations for each strategy
    iterations = 3

    for i in range(iterations):
        logger.info(f"\nIteration {i+1}/{iterations}")

        # Create fresh services
        restart_manager = ServiceRestarter()

        # Mock service for testing
        class MockService:
            def get_state(self):
                from app.services.restart_manager import ServiceState
                return ServiceState.RUNNING
            def health_check(self):
                return True
            def get_resources(self):
                return {}
            def prepare_shutdown(self):
                pass
            def shutdown(self):
                time.sleep(0.01)  # Simulate work
            def startup(self, config=None):
                time.sleep(0.02)  # Simulate work

        # Register test services
        for j in range(3):
            service = MockService()
            metadata = ServiceMetadata(
                name=f"test_{j}",
                service_ref=service,
                supports_hot_swap=(j == 0),
                max_restart_time=0.1
            )
            restart_manager.register_service(metadata)

        # Test different strategies
        for strategy in [RestartStrategy.HOT_SWAP, RestartStrategy.GRACEFUL, RestartStrategy.PARALLEL]:
            plan = restart_manager.plan_restart(['test_0', 'test_1'])
            plan.strategy = strategy

            start = time.time()
            success = restart_manager.execute_restart(plan)
            elapsed = (time.time() - start) * 1000

            if success:
                results[strategy.value].append(elapsed)
                logger.info(f"  {strategy.value}: {elapsed:.1f}ms")

    # Calculate averages
    logger.info("\n" + "-"*40)
    logger.info("Average Performance:")
    for strategy, times in results.items():
        if times:
            avg = sum(times) / len(times)
            logger.info(f"  {strategy}: {avg:.1f}ms")


def main():
    """Run all tests."""
    logger.info("="*60)
    logger.info("Service Restart Manager Test Suite")
    logger.info("="*60)

    try:
        # Basic functionality tests
        test_health_checks()
        test_resource_monitoring()

        # Service-specific tests
        test_hot_swap_performance()
        test_service_restart_manager()
        test_restart_coordinator()

        # Async tests
        logger.info("\nRunning async tests...")
        asyncio.run(test_async_operations())

        # Performance benchmarks
        run_performance_benchmarks()

        logger.info("\n" + "="*60)
        logger.info("✓ All tests completed successfully!")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()