"""Domain entities (data-only structures) used across services.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

BBox = Tuple[int,int,int,int]  # (x1,y1,x2,y2)

@dataclass(slots=True)
class Detection:
    class_id: int
    score: float
    bbox: BBox
    angle: Optional[float] = None

@dataclass(slots=True)
class MasterObject:
    class_id: int
    name: str
    bbox_norm: Tuple[float,float,float,float]  # (cx,cy,w,h) normalized
    angle: Optional[float] = None

@dataclass(slots=True)
class MatchResult:
    master: MasterObject
    detection: Optional[Detection]
    verdict: str  # match|near|misaligned|missing|extra
    iou: float
    offset_px: float
    angle_delta: Optional[float] = None

@dataclass(slots=True)
class PipelineState:
    frame: any  # numpy ndarray (BGR)
    detections: List[Detection]
    matches: List[MatchResult]
    feedback: List[str]
    latency_ms: int
    fps: float
