"""Geometry and bounding box utilities."""

import math
import cv2
import os

def xywh_to_xyxy(box, w=1, h=1, normalized=True):
    """Convert center-width-height format to x1,y1,x2,y2 format."""
    # box: (x_center, y_center, w, h) normalized 0..1 or absolute
    if normalized:
        cx, cy, bw, bh = box
        x1 = (cx - bw/2) * w
        y1 = (cy - bh/2) * h
        x2 = (cx + bw/2) * w
        y2 = (cy + bh/2) * h
    else:
        cx, cy, bw, bh = box
        x1 = cx - bw/2
        y1 = cy - bh/2
        x2 = cx + bw/2
        y2 = cy + bh/2
    return [int(x1), int(y1), int(x2), int(y2)]

def xyxy_to_xywh_norm(xyxy, img_w, img_h):
    """Convert x1,y1,x2,y2 format to normalized center-width-height format."""
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return [cx, cy, bw, bh]

def iou_xyxy(boxA, boxB):
    """Calculate Intersection over Union (IoU) for two boxes."""
    # boxA/B: [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    denom = float(boxAArea + boxBArea - interArea)
    if denom == 0:
        return 0.0
    return interArea / denom

def centroid(box):
    """Calculate centroid of bounding box."""
    x1,y1,x2,y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def centroid_distance(boxA, boxB):
    """Calculate distance between centroids of two boxes."""
    (ax,ay) = centroid(boxA)
    (bx,by) = centroid(boxB)
    return math.hypot(ax-bx, ay-by)

def estimate_orientation(img, bbox):
    """
    Estimate object orientation (degrees) within bbox in image.
    bbox: [x1,y1,x2,y2]
    Returns angle in degrees (-90..90). 0 means horizontal.
    """
    x1,y1,x2,y2 = bbox
    crop = img[max(0,y1):y2, max(0,x1):x2].copy()
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # use Canny and contours
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    # choose largest contour
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 10:
        return 0.0
    rect = cv2.minAreaRect(c)
    angle = rect[-1]
    # rect angle parsing: return normalized angle
    # OpenCV angle semantics: angle of the rotated rectangle.
    return float(angle)

def angle_delta(angle1, angle2):
    """Calculate the absolute difference between two angles."""
    if angle1 is None or angle2 is None:
        return None
    
    diff = abs(angle1 - angle2)
    # Normalize to 0-180 range
    if diff > 180:
        diff = 360 - diff
    return diff

def ensure_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)