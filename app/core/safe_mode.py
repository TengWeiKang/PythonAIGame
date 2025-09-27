"""Safe Mode Manager for minimal configuration fallback system.

This module provides a safe mode system that:
- Provides minimal, stable configuration when errors occur
- Disables problematic features temporarily
- Allows gradual re-enablement of features
- Tracks feature stability over time
- Provides emergency recovery capabilities
- Maintains system usability during troubleshooting

Safe mode ensures the application remains functional even when
configuration errors or system issues occur.
"""
from __future__ import annotations

import logging
import time
import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class SafeModeReason(Enum):
    """Reasons for entering safe mode."""
    CONFIG_ERROR = "config_error"
    SERVICE_FAILURE = "service_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CRITICAL_ERROR = "critical_error"
    MANUAL_ACTIVATION = "manual_activation"
    STARTUP_FAILURE = "startup_failure"
    RECOVERY_MODE = "recovery_mode"


class FeatureState(Enum):
    """States of features in safe mode."""
    ENABLED = "enabled"          # Feature is working normally
    DISABLED = "disabled"        # Feature is temporarily disabled
    TESTING = "testing"          # Feature is being tested for stability
    FAILED = "failed"           # Feature has failed and is disabled
    RESTRICTED = "restricted"    # Feature has limited functionality


@dataclass(slots=True)
class FeatureConfig:
    """Configuration for a feature in safe mode."""
    name: str
    state: FeatureState
    priority: int  # 1 = critical, 10 = optional
    dependencies: List[str] = field(default_factory=list)
    test_function: Optional[str] = None
    safe_config: Dict[str, Any] = field(default_factory=dict)

    # Stability tracking
    last_test_time: Optional[datetime] = None
    test_success_count: int = 0
    test_failure_count: int = 0
    stability_score: float = 0.0  # 0.0 to 1.0

    # Timing
    disabled_since: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None

    def calculate_stability_score(self) -> float:
        """Calculate stability score based on test history."""
        total_tests = self.test_success_count + self.test_failure_count
        if total_tests == 0:
            return 0.5  # Unknown stability

        # Base score from success rate
        success_rate = self.test_success_count / total_tests

        # Penalty for recent failures
        penalty = 0.0
        if self.last_failure_time:
            hours_since_failure = (datetime.now() - self.last_failure_time).total_seconds() / 3600
            if hours_since_failure < 1:
                penalty = 0.3  # Recent failure
            elif hours_since_failure < 24:
                penalty = 0.1  # Failure within day

        self.stability_score = max(0.0, success_rate - penalty)
        return self.stability_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'state': self.state.value,
            'priority': self.priority,
            'dependencies': self.dependencies,
            'test_function': self.test_function,
            'safe_config': self.safe_config,
            'last_test_time': self.last_test_time.isoformat() if self.last_test_time else None,
            'test_success_count': self.test_success_count,
            'test_failure_count': self.test_failure_count,
            'stability_score': self.stability_score,
            'disabled_since': self.disabled_since.isoformat() if self.disabled_since else None,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None
        }


@dataclass(slots=True)
class SafeModeState:
    """Current state of safe mode system."""
    active: bool
    reason: Optional[SafeModeReason]
    activated_at: Optional[datetime]
    activated_by: str = "system"

    # Feature states
    features: Dict[str, FeatureConfig] = field(default_factory=dict)

    # Configuration
    safe_config: Dict[str, Any] = field(default_factory=dict)
    original_config: Optional[Dict[str, Any]] = None

    # Statistics
    total_activations: int = 0
    successful_exits: int = 0
    failed_exits: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            'active': self.active,
            'reason': self.reason.value if self.reason else None,
            'activated_at': self.activated_at.isoformat() if self.activated_at else None,
            'activated_by': self.activated_by,
            'features': {name: feat.to_dict() for name, feat in self.features.items()},
            'safe_config': self.safe_config,
            'original_config': self.original_config,
            'total_activations': self.total_activations,
            'successful_exits': self.successful_exits,
            'failed_exits': self.failed_exits
        }


class SafeModeManager:
    """Manages safe mode operations and feature stability."""

    def __init__(self, config_manager: Any = None, data_dir: Optional[Path] = None):
        self.config_manager = config_manager
        self.data_dir = data_dir or Path("data")
        self.safe_mode_file = self.data_dir / "safe_mode_state.json"

        # State
        self.state = SafeModeState(active=False, reason=None, activated_at=None)
        self._lock = threading.RLock()

        # Feature test functions
        self.test_functions: Dict[str, Callable[[], bool]] = {}

        # Default safe configuration
        self.default_safe_config = self._create_default_safe_config()

        # Initialize features
        self._initialize_features()

        # Load persisted state
        self._load_state()

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def enter_safe_mode(self, reason: SafeModeReason, triggered_by: str = "system",
                       preserve_features: Optional[List[str]] = None) -> bool:
        """Enter safe mode with specified reason."""
        with self._lock:
            try:
                if self.state.active:
                    logger.warning(f"Already in safe mode (reason: {self.state.reason})")
                    return True

                logger.warning(f"Entering safe mode: {reason.value} (triggered by: {triggered_by})")

                # Store original configuration
                if self.config_manager and hasattr(self.config_manager, 'get_current_config'):
                    try:
                        self.state.original_config = self.config_manager.get_current_config().to_dict()
                    except Exception as e:
                        logger.error(f"Failed to store original config: {e}")

                # Activate safe mode
                self.state.active = True
                self.state.reason = reason
                self.state.activated_at = datetime.now()
                self.state.activated_by = triggered_by
                self.state.total_activations += 1

                # Apply safe configuration
                safe_config = self.get_safe_config(preserve_features)
                self.state.safe_config = safe_config

                # Disable problematic features
                self._disable_problematic_features(preserve_features)

                # Apply safe configuration to system
                if self.config_manager and hasattr(self.config_manager, 'apply_safe_config'):
                    try:
                        self.config_manager.apply_safe_config(safe_config)
                    except Exception as e:
                        logger.error(f"Failed to apply safe config: {e}")

                # Save state
                self._save_state()

                logger.info(f"Safe mode activated successfully (reason: {reason.value})")
                return True

            except Exception as e:
                logger.error(f"Failed to enter safe mode: {e}")
                return False

    def exit_safe_mode(self, force: bool = False) -> bool:
        """Exit safe mode and restore normal operation."""
        with self._lock:
            try:
                if not self.state.active:
                    logger.info("Not currently in safe mode")
                    return True

                logger.info("Attempting to exit safe mode")

                # Check if it's safe to exit
                if not force and not self._is_safe_to_exit():
                    logger.warning("Not safe to exit safe mode - use force=True to override")
                    return False

                # Test critical features first
                if not force:
                    critical_features = [f for f in self.state.features.values() if f.priority <= 3]
                    for feature in critical_features:
                        if not self.test_feature(feature.name):
                            logger.warning(f"Critical feature {feature.name} failed test - cannot exit safe mode")
                            return False

                # Restore original configuration
                success = True
                if self.state.original_config and self.config_manager:
                    try:
                        # Apply original configuration
                        if hasattr(self.config_manager, 'apply_config_dict'):
                            self.config_manager.apply_config_dict(self.state.original_config)
                        else:
                            logger.warning("Config manager doesn't support applying config dict")
                    except Exception as e:
                        logger.error(f"Failed to restore original config: {e}")
                        if not force:
                            return False
                        success = False

                # Re-enable features gradually
                if not self._reenable_features():
                    if not force:
                        logger.error("Failed to re-enable features - staying in safe mode")
                        return False
                    success = False

                # Deactivate safe mode
                self.state.active = False
                self.state.reason = None
                self.state.activated_at = None

                if success:
                    self.state.successful_exits += 1
                    logger.info("Successfully exited safe mode")
                else:
                    self.state.failed_exits += 1
                    logger.warning("Exited safe mode with errors (forced)")

                # Save state
                self._save_state()

                return True

            except Exception as e:
                logger.error(f"Failed to exit safe mode: {e}")
                self.state.failed_exits += 1
                self._save_state()
                return False

    def test_feature(self, feature_name: str) -> bool:
        """Test if a feature is working correctly."""
        with self._lock:
            try:
                feature = self.state.features.get(feature_name)
                if not feature:
                    logger.warning(f"Unknown feature: {feature_name}")
                    return False

                # Get test function
                test_func = self.test_functions.get(feature.test_function or feature_name)
                if not test_func:
                    logger.debug(f"No test function for feature: {feature_name}")
                    return True  # Assume OK if no test

                # Run test
                logger.debug(f"Testing feature: {feature_name}")
                start_time = time.time()

                try:
                    result = test_func()
                    duration = time.time() - start_time

                    # Update test statistics
                    feature.last_test_time = datetime.now()

                    if result:
                        feature.test_success_count += 1
                        logger.debug(f"Feature test passed: {feature_name} ({duration:.2f}s)")
                    else:
                        feature.test_failure_count += 1
                        feature.last_failure_time = datetime.now()
                        logger.warning(f"Feature test failed: {feature_name} ({duration:.2f}s)")

                    # Update stability score
                    feature.calculate_stability_score()

                    return result

                except Exception as e:
                    feature.test_failure_count += 1
                    feature.last_failure_time = datetime.now()
                    feature.calculate_stability_score()
                    logger.error(f"Feature test error for {feature_name}: {e}")
                    return False

            except Exception as e:
                logger.error(f"Failed to test feature {feature_name}: {e}")
                return False

    def disable_feature(self, feature_name: str, reason: str = "") -> bool:
        """Disable a specific feature."""
        with self._lock:
            try:
                feature = self.state.features.get(feature_name)
                if not feature:
                    logger.warning(f"Unknown feature: {feature_name}")
                    return False

                if feature.state == FeatureState.DISABLED:
                    logger.debug(f"Feature already disabled: {feature_name}")
                    return True

                logger.info(f"Disabling feature: {feature_name} (reason: {reason})")

                feature.state = FeatureState.DISABLED
                feature.disabled_since = datetime.now()

                # Disable dependent features
                self._disable_dependent_features(feature_name)

                # Save state
                self._save_state()

                return True

            except Exception as e:
                logger.error(f"Failed to disable feature {feature_name}: {e}")
                return False

    def enable_feature(self, feature_name: str, test_first: bool = True) -> bool:
        """Enable a specific feature."""
        with self._lock:
            try:
                feature = self.state.features.get(feature_name)
                if not feature:
                    logger.warning(f"Unknown feature: {feature_name}")
                    return False

                if feature.state == FeatureState.ENABLED:
                    logger.debug(f"Feature already enabled: {feature_name}")
                    return True

                # Check dependencies first
                for dep in feature.dependencies:
                    dep_feature = self.state.features.get(dep)
                    if not dep_feature or dep_feature.state != FeatureState.ENABLED:
                        logger.warning(f"Cannot enable {feature_name}: dependency {dep} not enabled")
                        return False

                # Test feature if requested
                if test_first and not self.test_feature(feature_name):
                    logger.warning(f"Feature test failed, not enabling: {feature_name}")
                    feature.state = FeatureState.FAILED
                    return False

                logger.info(f"Enabling feature: {feature_name}")

                feature.state = FeatureState.ENABLED
                feature.disabled_since = None

                # Save state
                self._save_state()

                return True

            except Exception as e:
                logger.error(f"Failed to enable feature {feature_name}: {e}")
                return False

    def get_safe_config(self, preserve_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get safe configuration for the application."""
        try:
            safe_config = self.default_safe_config.copy()

            # Apply feature-specific safe configurations
            for feature_name, feature in self.state.features.items():
                if preserve_features and feature_name in preserve_features:
                    continue  # Skip preserved features

                if feature.state in [FeatureState.DISABLED, FeatureState.FAILED]:
                    # Apply safe config for disabled features
                    safe_config.update(feature.safe_config)

            return safe_config

        except Exception as e:
            logger.error(f"Failed to generate safe config: {e}")
            return self.default_safe_config.copy()

    def is_in_safe_mode(self) -> bool:
        """Check if currently in safe mode."""
        return self.state.active

    def get_safe_mode_info(self) -> Dict[str, Any]:
        """Get information about current safe mode state."""
        with self._lock:
            return {
                'active': self.state.active,
                'reason': self.state.reason.value if self.state.reason else None,
                'activated_at': self.state.activated_at.isoformat() if self.state.activated_at else None,
                'activated_by': self.state.activated_by,
                'duration_minutes': (
                    (datetime.now() - self.state.activated_at).total_seconds() / 60
                    if self.state.activated_at else 0
                ),
                'feature_summary': self._get_feature_summary(),
                'statistics': {
                    'total_activations': self.state.total_activations,
                    'successful_exits': self.state.successful_exits,
                    'failed_exits': self.state.failed_exits
                }
            }

    def register_test_function(self, feature_name: str, test_func: Callable[[], bool]) -> None:
        """Register a test function for a feature."""
        self.test_functions[feature_name] = test_func
        logger.debug(f"Registered test function for feature: {feature_name}")

    def get_feature_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all features."""
        with self._lock:
            return {
                name: {
                    'state': feature.state.value,
                    'priority': feature.priority,
                    'stability_score': feature.stability_score,
                    'test_success_count': feature.test_success_count,
                    'test_failure_count': feature.test_failure_count,
                    'last_test_time': feature.last_test_time.isoformat() if feature.last_test_time else None,
                    'disabled_since': feature.disabled_since.isoformat() if feature.disabled_since else None
                }
                for name, feature in self.state.features.items()
            }

    def _create_default_safe_config(self) -> Dict[str, Any]:
        """Create default safe configuration."""
        return {
            # Application settings
            'debug': True,
            'enable_logging': True,
            'log_level': 'INFO',
            'performance_mode': 'Power_Saving',
            'max_memory_usage_mb': 1024,

            # Camera settings (minimal)
            'camera_width': 640,
            'camera_height': 480,
            'camera_fps': 15,
            'camera_buffer_size': 1,

            # Detection settings (conservative)
            'detection_confidence_threshold': 0.5,
            'detection_iou_threshold': 0.5,

            # Gemini settings (basic)
            'gemini_model': 'gemini-1.5-flash',  # Fastest model
            'gemini_timeout': 30,
            'gemini_temperature': 0.0,  # Deterministic
            'gemini_max_tokens': 1000,

            # UI settings (minimal)
            'startup_fullscreen': False,
            'remember_window_state': False,
            'window_width': 800,
            'window_height': 600,
        }

    def _initialize_features(self) -> None:
        """Initialize feature configurations."""
        features = [
            # Core features (priority 1-3)
            FeatureConfig(
                name="webcam",
                state=FeatureState.ENABLED,
                priority=1,
                test_function="test_webcam",
                safe_config={
                    'camera_width': 640,
                    'camera_height': 480,
                    'camera_fps': 15,
                    'use_gpu': False
                }
            ),
            FeatureConfig(
                name="detection",
                state=FeatureState.ENABLED,
                priority=2,
                dependencies=["webcam"],
                test_function="test_detection",
                safe_config={
                    'detection_confidence_threshold': 0.5,
                    'use_gpu': False
                }
            ),
            FeatureConfig(
                name="main_ui",
                state=FeatureState.ENABLED,
                priority=3,
                test_function="test_main_ui",
                safe_config={
                    'startup_fullscreen': False,
                    'window_width': 800,
                    'window_height': 600
                }
            ),

            # Optional features (priority 4-7)
            FeatureConfig(
                name="gemini_ai",
                state=FeatureState.ENABLED,
                priority=4,
                test_function="test_gemini",
                safe_config={
                    'gemini_model': 'gemini-1.5-flash',
                    'requests_per_minute': 5,
                    'enable_rate_limiting': True
                }
            ),
            FeatureConfig(
                name="gpu_acceleration",
                state=FeatureState.ENABLED,
                priority=5,
                test_function="test_gpu",
                safe_config={'use_gpu': False}
            ),
            FeatureConfig(
                name="advanced_ui",
                state=FeatureState.ENABLED,
                priority=6,
                dependencies=["main_ui"],
                safe_config={'advanced_features_enabled': False}
            ),
            FeatureConfig(
                name="performance_mode",
                state=FeatureState.ENABLED,
                priority=7,
                safe_config={'performance_mode': 'Power_Saving'}
            ),

            # Enhancement features (priority 8-10)
            FeatureConfig(
                name="auto_updates",
                state=FeatureState.ENABLED,
                priority=8,
                safe_config={'auto_updates_enabled': False}
            ),
            FeatureConfig(
                name="telemetry",
                state=FeatureState.ENABLED,
                priority=9,
                safe_config={'telemetry_enabled': False}
            ),
            FeatureConfig(
                name="experimental",
                state=FeatureState.ENABLED,
                priority=10,
                safe_config={'experimental_features_enabled': False}
            )
        ]

        # Convert to dictionary
        self.state.features = {feature.name: feature for feature in features}

    def _disable_problematic_features(self, preserve_features: Optional[List[str]] = None) -> None:
        """Disable features that are likely to cause problems."""
        preserve_features = preserve_features or []

        # Disable non-essential features in safe mode
        non_essential = ['gpu_acceleration', 'advanced_ui', 'performance_mode',
                        'auto_updates', 'telemetry', 'experimental']

        for feature_name in non_essential:
            if feature_name not in preserve_features:
                feature = self.state.features.get(feature_name)
                if feature:
                    feature.state = FeatureState.DISABLED
                    feature.disabled_since = datetime.now()

    def _disable_dependent_features(self, disabled_feature: str) -> None:
        """Disable features that depend on the disabled feature."""
        for feature in self.state.features.values():
            if disabled_feature in feature.dependencies:
                if feature.state != FeatureState.DISABLED:
                    logger.info(f"Disabling dependent feature: {feature.name}")
                    feature.state = FeatureState.DISABLED
                    feature.disabled_since = datetime.now()
                    # Recursively disable dependents
                    self._disable_dependent_features(feature.name)

    def _reenable_features(self) -> bool:
        """Re-enable features when exiting safe mode."""
        try:
            # Enable features in order of priority
            features_by_priority = sorted(
                self.state.features.values(),
                key=lambda f: f.priority
            )

            success_count = 0
            total_count = 0

            for feature in features_by_priority:
                if feature.state == FeatureState.DISABLED:
                    total_count += 1

                    # Test feature before enabling
                    if self.test_feature(feature.name):
                        feature.state = FeatureState.ENABLED
                        feature.disabled_since = None
                        success_count += 1
                        logger.info(f"Re-enabled feature: {feature.name}")
                    else:
                        logger.warning(f"Failed to re-enable feature: {feature.name}")
                        feature.state = FeatureState.FAILED

            logger.info(f"Re-enabled {success_count}/{total_count} features")
            return success_count == total_count

        except Exception as e:
            logger.error(f"Failed to re-enable features: {e}")
            return False

    def _is_safe_to_exit(self) -> bool:
        """Check if it's safe to exit safe mode."""
        try:
            # Check if we've been in safe mode long enough
            if self.state.activated_at:
                duration = datetime.now() - self.state.activated_at
                if duration < timedelta(minutes=2):
                    logger.debug("Haven't been in safe mode long enough")
                    return False

            # Check critical features
            critical_features = [f for f in self.state.features.values() if f.priority <= 3]
            for feature in critical_features:
                if feature.state == FeatureState.FAILED:
                    logger.warning(f"Critical feature {feature.name} is failed")
                    return False

                if feature.stability_score < 0.7:
                    logger.warning(f"Critical feature {feature.name} has low stability score: {feature.stability_score}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Failed to check if safe to exit: {e}")
            return False

    def _get_feature_summary(self) -> Dict[str, int]:
        """Get summary of feature states."""
        summary = {state.value: 0 for state in FeatureState}

        for feature in self.state.features.values():
            summary[feature.state.value] += 1

        return summary

    def _save_state(self) -> None:
        """Save safe mode state to disk."""
        try:
            with open(self.safe_mode_file, 'w', encoding='utf-8') as f:
                json.dump(self.state.to_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save safe mode state: {e}")

    def _load_state(self) -> None:
        """Load safe mode state from disk."""
        try:
            if self.safe_mode_file.exists():
                with open(self.safe_mode_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Restore basic state
                self.state.active = data.get('active', False)
                self.state.total_activations = data.get('total_activations', 0)
                self.state.successful_exits = data.get('successful_exits', 0)
                self.state.failed_exits = data.get('failed_exits', 0)

                # If we were in safe mode when the application closed,
                # we might want to stay in safe mode
                if self.state.active:
                    logger.warning("Application was previously in safe mode - remaining in safe mode")
                    self.state.reason = SafeModeReason.STARTUP_FAILURE
                    self.state.activated_at = datetime.now()

                logger.debug("Loaded safe mode state from disk")

        except Exception as e:
            logger.error(f"Failed to load safe mode state: {e}")


__all__ = [
    'SafeModeReason', 'FeatureState', 'FeatureConfig',
    'SafeModeState', 'SafeModeManager'
]