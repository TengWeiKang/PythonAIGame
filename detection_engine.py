"""Deprecated in favor of services.pipeline.DetectionPipeline + services.webcam.WebcamManager.
Kept temporarily for backward compatibility; will be removed after migration.
"""
from __future__ import annotations

import cv2, time, threading
from typing import Callable, List, Dict, Any, Optional, Tuple


class DetectionEngine:  # pragma: no cover (legacy)
    """Manage camera capture, model inference, and (optional) master matching.

    Lifecycle:
      start(device_idx, master_provider) -> opens VideoCapture, schedules loop
      stop() -> cancels loop & releases camera

    Callback contract (on_update):
      on_update(frame_bgr, detections, matches, feedback_lines) where:
        frame_bgr: latest numpy ndarray frame (BGR)
        detections: list[dict] each with keys (class, score, bbox[x1,y1,x2,y2])
        matches: list[Any] (opaque to engine; produced by match function if provided)
        feedback_lines: list[str] textual feedback lines (already prepared)
    """

    def __init__(
        self,
        root,
        model,
        cfg: dict,
        match_fn: Optional[Callable[[List[Dict[str, Any]], Any, Tuple[int,int], float, int], Tuple[List[Any], List[Any]]]] = None,
        on_update: Optional[Callable[[Any, List[Dict[str, Any]], List[Any], List[str]], None]] = None,
    ) -> None:
        self.root = root
        self.model = model
        self.cfg = cfg
        self.match_fn = match_fn
        self.on_update = on_update
        self.running = False
        self.cap = None
        self._after_id = None
        self._master_provider: Optional[Callable[[], Optional[dict]]] = None

    # ---------------- Public API ----------------
    def start(self, device_idx: int, master_provider: Callable[[], Optional[dict]]):
        if self.running:
            return
        self._master_provider = master_provider
        self.cap = cv2.VideoCapture(device_idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            # Let caller surface UI error; just stop here
            self.cap.release(); self.cap = None
            return
        # Attempt to probe and select the maximum supported resolution before applying cfg overrides.
        try:
            self._select_max_resolution()
        except Exception:
            pass
        # Apply configured capture properties (best-effort)
        for prop, key in (
            (cv2.CAP_PROP_FRAME_WIDTH, 'camera_width'),
            (cv2.CAP_PROP_FRAME_HEIGHT, 'camera_height'),
            (cv2.CAP_PROP_FPS, 'camera_fps'),
        ):
            val = self.cfg.get(key)
            if val is not None:
                try: self.cap.set(prop, val)
                except Exception: pass
        self.running = True
        self._schedule_next(0)

    def stop(self):
        self.running = False
        if self.cap:
            try: self.cap.release()
            except Exception: pass
            self.cap = None
        if self._after_id is not None:
            try: self.root.after_cancel(self._after_id)
            except Exception: pass
            self._after_id = None

    def is_running(self) -> bool:
        return self.running

    # ---------------- Internal loop ----------------
    def _schedule_next(self, delay_ms: int):
        if not self.running:
            return
        self._after_id = self.root.after(max(1, delay_ms), self._loop)

    def _loop(self):
        if not (self.running and self.cap):
            return
        start_t = time.time()
        ok, frame = self.cap.read()
        if not ok:
            # Try again shortly
            self._schedule_next(50)
            return
        detections = []
        # Optional pre-resize before inference to reduce CPU/GPU load
        pre_resize_enabled = self.cfg.get('pre_resize', True)
        target_img_size = self.cfg.get('img_size', 640)
        infer_frame = frame
        scale = 1.0
        if pre_resize_enabled and target_img_size:
            h0, w0 = frame.shape[:2]
            max_side = max(h0, w0)
            if max_side > target_img_size:
                scale = target_img_size / float(max_side)
                new_w = int(w0 * scale)
                new_h = int(h0 * scale)
                try:
                    import cv2 as _cv2
                    infer_frame = _cv2.resize(frame, (new_w, new_h), interpolation=_cv2.INTER_AREA)
                except Exception:
                    infer_frame = frame
        try:
            detections = self.model.predict(infer_frame, img_size=target_img_size) or []
            # Scale detections back to original frame size if we resized pre-inference
            if scale != 1.0:
                inv = 1.0 / scale
                for det in detections:
                    x1, y1, x2, y2 = det.get('bbox', [0,0,0,0])
                    det['bbox'] = [int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv)]
        except Exception:
            # Fallback dummy detection (soft-fail) if debug
            if self.cfg.get('debug'):
                detections = [{"class": 0, "score": 0.9, "bbox": [50, 50, 150, 150]}]
        feedback_lines: List[str] = []
        matches: List[Any] = []
        master = self._master_provider() if self._master_provider else None
        if master and self.match_fn:
            try:
                matches, extras = self.match_fn(
                    detections,
                    master.get('labels', []),
                    (frame.shape[0], frame.shape[1]),
                    iou_threshold=self.cfg.get('iou_match_threshold', 0.5),
                    tolerance_px=self.cfg.get('master_tolerance_px', 40)
                )
                for m in matches:
                    det = getattr(m, 'detected', None) if not isinstance(m, dict) else m.get('detected')
                    verdict = getattr(m, 'verdict', None) if not isinstance(m, dict) else m.get('verdict')
                    mclass = getattr(m, 'master_class', None) if not isinstance(m, dict) else m.get('master_class')
                    offset_px = getattr(m, 'offset_px', 0) if not isinstance(m, dict) else m.get('offset_px', 0)
                    if det:
                        if verdict == 'match':
                            feedback_lines.append(f'Good! {mclass} in place.')
                        elif verdict == 'near':
                            feedback_lines.append(f'{mclass} slightly off ({int(offset_px)} px).')
                        elif verdict == 'misaligned':
                            feedback_lines.append(f'{mclass} misaligned ({int(offset_px)} px).')
                    else:
                        feedback_lines.append(f'{mclass} missing.')
            except Exception:
                # Matching failure shouldn't kill loop
                if self.cfg.get('debug'):
                    feedback_lines.append('[debug] matching failed')

        if self.on_update:
            try:
                self.on_update(frame, detections, matches, feedback_lines)
            except Exception:
                pass

        target_fps = self.cfg.get('target_fps', 30)
        frame_period_ms = int(1000 / max(1, target_fps))
        proc_ms = int((time.time() - start_t) * 1000)
        delay = max(1, frame_period_ms - proc_ms)
        self._schedule_next(delay)

    # ---------------- Resolution probing ----------------
    def _select_max_resolution(self):
        """Try a list of descending common resolutions and keep the first that the
        camera accepts. Stores result back into cfg as camera_width/height.
        This is a heuristic; DirectShow doesn't expose full mode list via OpenCV.
        """
        if not self.cap:
            return
        # If user already specified both width & height, respect them (skip auto)
        if self.cfg.get('camera_width') and self.cfg.get('camera_height'):
            return
        candidates = [
            (3840,2160),(2560,1440),(2560,1080),(2560,960),(1920,1200),(1920,1080),
            (1600,1200),(1600,900),(1536,864),(1440,900),(1366,768),(1280,1024),
            (1280,960),(1280,800),(1280,720),(1024,768),(800,600),(800,480),
            (720,480),(640,480),(640,360),(424,240),(320,240)
        ]
        original_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        original_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        chosen = None
        for w,h in candidates:
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                # Warm up a couple frames for some drivers to apply the mode
                for _ in range(2):
                    self.cap.read()
                aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if abs(aw - w) <= 32 and abs(ah - h) <= 32:
                    chosen = (aw, ah)
                    break
            except Exception:
                continue
        if chosen:
            self.cfg['camera_width'], self.cfg['camera_height'] = chosen
        else:
            # fallback restore original if we changed it and failed to find match
            if original_w and original_h:
                try:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_w)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_h)
                except Exception:
                    pass
