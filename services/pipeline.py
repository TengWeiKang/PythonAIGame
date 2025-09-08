"""High-level detection pipeline orchestration."""
from __future__ import annotations
import time
from typing import Callable, List, Optional
from app.core.entities import Detection, MasterObject, MatchResult, PipelineState

class DetectionPipeline:
    def __init__(self, frame_source, inference, matcher, feedback_builder, cfg):
        self.frame_source = frame_source
        self.inference = inference
        self.matcher = matcher
        self.feedback_builder = feedback_builder
        self.cfg = cfg
        self._listeners: List[Callable[[PipelineState], None]] = []
        self._running=False
        self._last_tick_time=None
        self._tk_root=None
        self._after_id=None
        self._master_provider: Optional[Callable[[], List[MasterObject]]] = None

    def set_master_provider(self, fn:Callable[[], List[MasterObject]]):
        self._master_provider=fn

    def add_listener(self, cb:Callable[[PipelineState], None]):
        self._listeners.append(cb)

    def start(self, tk_root, fps:int=None):
        if self._running: return
        self._tk_root=tk_root
        self._running=True
        self._schedule(0)

    def stop(self):
        self._running=False
        if self._after_id and self._tk_root:
            try: self._tk_root.after_cancel(self._after_id)
            except Exception: pass
            self._after_id=None

    def _schedule(self, delay_ms:int):
        if not self._running: return
        self._after_id = self._tk_root.after(max(1,delay_ms), self._tick)

    def _tick(self):
        if not self._running: return
        start=time.time()
        ok, frame = self.frame_source.read()
        if not ok or frame is None:
            self._schedule(50)
            return
        detections:List[Detection] = self.inference.predict(frame) or []
        masters = self._master_provider() if self._master_provider else []
        matches:List[MatchResult] = self.matcher.match(detections, masters, (frame.shape[0], frame.shape[1])) if masters else []
        feedback = self.feedback_builder(matches)
        end=time.time()
        latency_ms=int((end-start)*1000)
        fps = 1000.0/latency_ms if latency_ms>0 else 0.0
        state = PipelineState(frame=frame,detections=detections,matches=matches,feedback=feedback,latency_ms=latency_ms,fps=fps)
        for cb in self._listeners:
            try: cb(state)
            except Exception: pass
        target_fps = self.cfg.get('target_fps',30)
        frame_period = 1000//max(1,target_fps)
        delay = max(1, frame_period - latency_ms)
        self._schedule(delay)
