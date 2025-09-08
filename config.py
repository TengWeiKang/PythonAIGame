"""Configuration dataclass and loading utilities.

Provides a strongly-typed configuration object that can be injected into
services instead of relying on a global module-level dictionary.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict
import json, os

_DEFAULTS: Dict[str, Any] = {
    "python_version": "3.13.7",
    "img_size": 640,
    "target_fps": 30,
    "iou_match_threshold": 0.5,
    "master_tolerance_px": 40,
    "angle_tolerance_deg": 20,
    "use_gpu": True,
    "data_dir": "data",
    "models_dir": "data/models",
    "master_dir": "data/master",
    "locales_dir": "locales",
    "default_locale": "en",
    "results_export_dir": "data/results",
    "model_size": "yolo11n",
    "train_epochs": 10,
    "batch_size": 8,
    "debug": False,
    "last_webcam_index": 0,
    "preview_max_width": 960,
    "preview_max_height": 720,
    "camera_width": 1280,
    "camera_height": 720,
    "camera_fps": 30,
}

@dataclass(slots=True)
class Config:
    python_version: str = _DEFAULTS["python_version"]
    img_size: int = _DEFAULTS["img_size"]
    target_fps: int = _DEFAULTS["target_fps"]
    iou_match_threshold: float = _DEFAULTS["iou_match_threshold"]
    master_tolerance_px: int = _DEFAULTS["master_tolerance_px"]
    angle_tolerance_deg: int = _DEFAULTS["angle_tolerance_deg"]
    use_gpu: bool = _DEFAULTS["use_gpu"]
    data_dir: str = _DEFAULTS["data_dir"]
    models_dir: str = _DEFAULTS["models_dir"]
    master_dir: str = _DEFAULTS["master_dir"]
    locales_dir: str = _DEFAULTS["locales_dir"]
    default_locale: str = _DEFAULTS["default_locale"]
    results_export_dir: str = _DEFAULTS["results_export_dir"]
    model_size: str = _DEFAULTS["model_size"]
    train_epochs: int = _DEFAULTS["train_epochs"]
    batch_size: int = _DEFAULTS["batch_size"]
    debug: bool = _DEFAULTS["debug"]
    last_webcam_index: int = _DEFAULTS["last_webcam_index"]
    preview_max_width: int = _DEFAULTS["preview_max_width"]
    preview_max_height: int = _DEFAULTS["preview_max_height"]
    camera_width: int = _DEFAULTS["camera_width"]
    camera_height: int = _DEFAULTS["camera_height"]
    camera_fps: int = _DEFAULTS["camera_fps"]
    # Arbitrary extra values retained for forward compatibility
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # merge extra keys at top-level for saving
        extra = d.pop("extra", {})
        d.update(extra)
        return d

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, self.extra.get(key, default))


def load_config(path: str = "config.json") -> Config:
    data: Dict[str, Any] = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
        except Exception:
            data = {}
    merged = {**_DEFAULTS, **data}
    # capture unknown keys
    extra = {k: v for k, v in merged.items() if k not in Config.__annotations__}
    cfg = Config(**{k: merged[k] for k in Config.__annotations__ if k != 'extra'}, extra=extra)
    return cfg


def save_config(cfg: Config, path: str = "config.json") -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, indent=2)
    except Exception:
        pass

__all__ = ["Config", "load_config", "save_config"]
