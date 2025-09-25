"""Dialog components package."""

from .settings_dialog import SettingsDialog
from .webcam_dialog import WebcamDialog
from .master_dialog import MasterDialog
from .object_classification_dialog import ObjectClassificationDialog
from .progress_dialog import SettingsProgressDialog, ProgressState, progress_manager
from .error_recovery_dialog import ErrorRecoveryDialog, ErrorDetails, ErrorSeverity, ActionType, show_error_dialog
from .success_dialog import SuccessDialog, ChangesSummary, PerformanceMetric, NextStep, show_success_dialog

__all__ = [
    "SettingsDialog",
    "WebcamDialog",
    "MasterDialog",
    "ObjectClassificationDialog",
    "SettingsProgressDialog",
    "ProgressState",
    "progress_manager",
    "ErrorRecoveryDialog",
    "ErrorDetails",
    "ErrorSeverity",
    "ActionType",
    "show_error_dialog",
    "SuccessDialog",
    "ChangesSummary",
    "PerformanceMetric",
    "NextStep",
    "show_success_dialog"
]