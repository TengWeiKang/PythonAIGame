# tests/test_matching.py
import pytest
from utils import iou_xyxy, centroid_distance

def test_iou_no_overlap():
    a = [0,0,10,10]
    b = [20,20,30,30]
    assert iou_xyxy(a,b) == 0.0

def test_iou_full_overlap():
    a = [0,0,10,10]
    b = [0,0,10,10]
    assert pytest.approx(iou_xyxy(a,b)) == 1.0

def test_centroid_distance():
    a = [0,0,10,10]
    b = [10,10,20,20]
    assert centroid_distance(a,b) == pytest.approx(((5-15)**2 + (5-15)**2)**0.5)
