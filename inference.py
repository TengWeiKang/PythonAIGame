# inference.py
"""
Model wrapper and master-image comparison logic.
Uses Ultralytics model if available, otherwise a simple stub/dummy.
"""

import os, time, random
from typing import List, Dict, Tuple
import numpy as np
from utils import xywh_to_xyxy, iou_xyxy, centroid_distance, estimate_orientation, xyxy_to_xywh_norm, read_yolo_labels, load_config

cfg = load_config()

# Try to import ultralytics if present
HAS_ULTRALYTICS = False
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception:
    HAS_ULTRALYTICS = False

class ModelWrapper:
    def __init__(self, weights_path=None):
        self.weights = weights_path or os.path.join(cfg.get("models_dir","models"), "best.pt")
        self.model = None
        self.loaded_source = None  # path or model name
        if HAS_ULTRALYTICS:
            self._attempt_load()
        else:
            print("[inference] Ultralytics not found; using dummy detections when debug enabled.")

    def _attempt_load(self):
        """Attempt layered loading: custom weights path (if exists) else configured model_size else fallbacks."""
        candidates: List[str] = []
        # 1. Explicit weights path if file exists
        if os.path.isfile(self.weights):
            candidates.append(self.weights)
        # 2. Configured model_size (e.g. yolo11n)
        ms = cfg.get('model_size')
        if ms:
            candidates.append(ms)
        # 3. Fallback list
        candidates.extend(['yolo11n', 'yolov8n'])
        tried = []
        for cand in candidates:
            if cand in tried:
                continue
            tried.append(cand)
            try:
                self.model = YOLO(cand)
                self.loaded_source = cand
                print(f"[inference] Loaded model: {cand}")
                return
            except Exception as e:
                # keep trying
                last_err = e
                continue
        print(f"[inference] Could not load any YOLO model (tried: {tried}). Using dummy detections.")
        self.model = None

    def predict(self, frame, img_size=640):
        """
        Returns list of detections: each is dict {class:int, score:float, bbox:[x1,y1,x2,y2]}
        """
        h, w = frame.shape[:2]
        if self.model:
            # Ultralytics returns preds; convert to our format
            try:
                results = self.model(frame, imgsz=img_size, verbose=False)
            except Exception as e:
                print(f"[inference] predict() error running model: {e}; falling back to dummy detections if debug")
                return self._dummy_detections(h, w) if cfg.get('debug') else []
            detections = []
            # results could be list-like for each input
            res = results[0]
            boxes = getattr(res, 'boxes', None)
            if boxes is None:
                return []
            for b in boxes:
                xyxy = b.xyxy[0].cpu().numpy() if hasattr(b.xyxy[0], "cpu") else np.array(b.xyxy[0])
                conf = float(b.conf[0]) if hasattr(b.conf[0], "cpu") else float(b.conf[0])
                cls = int(b.cls[0]) if hasattr(b.cls[0], "cpu") else int(b.cls[0])
                detections.append({"class": cls, "score": conf, "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]})
            return detections
        else:
            # Dummy detections (only if debug enabled) so UI shows boxes
            return self._dummy_detections(h, w) if cfg.get('debug') else []

    def _dummy_detections(self, h, w):
        """Generate deterministic-ish dummy detections for UI demonstration when no model loaded."""
        random.seed(int(time.time()) // 2)
        num = random.randint(0, 2)
        dets = []
        for _ in range(num):
            bw = random.randint(40, 120)
            bh = random.randint(40, 120)
            x1 = random.randint(0, max(1, w - bw))
            y1 = random.randint(0, max(1, h - bh))
            dets.append({
                'class': 0,
                'score': round(random.uniform(0.6, 0.95), 2),
                'bbox': [x1, y1, x1 + bw, y1 + bh]
            })
        return dets

def load_master_labels(master_dir=cfg.get("master_dir", "data/master")):
    master_img = os.path.join(master_dir, "master.jpg")
    master_labels = os.path.join(master_dir, "master_labels.txt")
    if not os.path.exists(master_img) or not os.path.exists(master_labels):
        return None
    labels = read_yolo_labels(master_labels)
    return {"image": master_img, "labels": labels}

def match_detections_to_master(detections: List[Dict], master_labels: List[List], image_shape: Tuple[int,int], 
                               iou_threshold=None, tolerance_px=None, angle_tolerance_deg=None):
    """
    detections: list of {'class', 'score', 'bbox':[x1,y1,x2,y2]}
    master_labels: list of [class_idx, cx,cy,w,h] normalized
    image_shape: (h,w)
    returns: list of matches for each master label with details
    """
    h, w = image_shape
    iou_threshold = iou_threshold or cfg.get("iou_match_threshold", 0.5)
    tolerance_px = tolerance_px or cfg.get("master_tolerance_px", 40)
    angle_tolerance_deg = angle_tolerance_deg or cfg.get("angle_tolerance_deg", 20)

    results = []

    # convert master labels to xyxy absolute
    master_abs = []
    for lab in master_labels:
        cls_idx = int(lab[0])
        xyxy = xywh_to_xyxy(lab[1:], w=w, h=h, normalized=True)
        master_abs.append({"class": cls_idx, "bbox": xyxy, "raw": lab})

    used_dets = set()
    for m in master_abs:
        best = None
        best_iou = 0.0
        best_det_idx = None
        for di, det in enumerate(detections):
            if det["class"] != m["class"]:
                continue
            i = iou_xyxy(m["bbox"], det["bbox"])
            if i > best_iou:
                best_iou = i
                best = det
                best_det_idx = di
        if best is None:
            results.append({
                "master_class": m["class"],
                "verdict": "missing",
                "master_bbox": m["bbox"],
                "detected": None,
                "offset_px": None,
                "angle_master": None,
                "angle_detected": None
            })
            continue
        # compute centroid distance & angle
        dist = centroid_distance(m["bbox"], best["bbox"])
        angle_master = estimate_orientation_from_bbox(m["bbox"], None, h, w)  # We'll compute angle on master if needed later
        # For detected, compute angle from frame when caller supplies frame. Here we leave None (caller can compute)
        angle_det = None

        # decide verdict
        if best_iou >= iou_threshold and dist <= tolerance_px:
            verdict = "match"
        elif dist <= tolerance_px * 2:
            verdict = "near"
        else:
            verdict = "misaligned"

        results.append({
            "master_class": m["class"],
            "verdict": verdict,
            "iou": best_iou,
            "master_bbox": m["bbox"],
            "detected": best,
            "offset_px": dist,
            "angle_master": angle_master,
            "angle_detected": angle_det
        })
        if best_det_idx is not None:
            used_dets.add(best_det_idx)

    # extras: detections that didn't match any master label
    extras = []
    for di, det in enumerate(detections):
        if di not in used_dets:
            extras.append(det)
    return results, extras

def estimate_orientation_from_bbox(norm_bbox, frame=None, h=None, w=None):
    """
    If you have the master image file, caller should open it and call estimate_orientation() from utils on absolute bbox.
    Here, we try to compute nothing; kept for scaffolding.
    """
    return 0.0
