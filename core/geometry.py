"""Geometry / math helper functions (pure, easily unit tested)."""
from __future__ import annotations
from typing import Tuple
import math

def iou_xyxy(a, b) -> float:
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    denom = area_a + area_b - inter
    if denom <= 0: return 0.0
    return inter / denom

def centroid_xyxy(b):
    return ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0)

def centroid_distance(a, b) -> float:
    ax, ay = centroid_xyxy(a); bx, by = centroid_xyxy(b)
    return math.hypot(ax-bx, ay-by)

def angle_delta(a: float|None, b: float|None) -> float|None:
    if a is None or b is None: return None
    d = abs(a-b)
    # wrap (angles limited to [-90,90] typical) but keep simple
    return d
