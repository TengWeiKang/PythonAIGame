"""
Main application package for Python Game Detection System.
"""

__version__ = "2.0.0"
__author__ = "Python Game Dev Team"

from .config.settings import Config, load_config, save_config
from .core.entities import Detection, MasterObject, MatchResult, PipelineState

__all__ = [
    "Config", "load_config", "save_config",
    "Detection", "MasterObject", "MatchResult", "PipelineState"
]