"""Domain entities (data-only structures) used across services."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any

BBox = Tuple[int,int,int,int]  # (x1,y1,x2,y2)

@dataclass(slots=True)
class Detection:
    class_id: int
    score: float
    bbox: BBox
    angle: Optional[float] = None
    class_name: Optional[str] = None  # Human-readable class name

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
    frame: Any  # numpy ndarray (BGR)
    detections: List[Detection]
    matches: List[MatchResult]
    feedback: List[str]
    latency_ms: int
    fps: float

@dataclass(slots=True)
class ChatbotContext:
    """Context information for chatbot integration."""
    user_message: str
    timestamp: str
    frame_dimensions: Tuple[int, int]  # (width, height)
    detection_results: List[Detection]
    comparison_summary: Optional[str] = None
    scene_description: Optional[str] = None

@dataclass(slots=True)
class ComparisonMetrics:
    """Metrics for image comparison analysis."""
    similarity_score: float  # 0.0 to 1.0
    objects_added: int
    objects_removed: int
    objects_moved: int
    objects_unchanged: int
    total_changes: int
    change_significance: str  # 'minor', 'moderate', 'major'