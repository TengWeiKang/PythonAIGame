"""Enhanced detection pipeline service."""
from __future__ import annotations
import time
from typing import Callable, List, Optional
from ..core.entities import Detection, MasterObject, MatchResult, PipelineState
from ..core.exceptions import DetectionError

class DetectionService:
    """High-level detection pipeline orchestration service."""
    
    def __init__(self, frame_source, inference_service, matcher, feedback_builder, config):
        self.frame_source = frame_source
        self.inference_service = inference_service
        self.matcher = matcher
        self.feedback_builder = feedback_builder
        self.config = config
        self._listeners: List[Callable[[PipelineState], None]] = []
        self._running = False
        self._last_tick_time = None
        self._tk_root = None
        self._after_id = None
        self._master_provider: Optional[Callable[[], List[MasterObject]]] = None

    def set_master_provider(self, fn: Callable[[], List[MasterObject]]) -> None:
        """Set the master objects provider function."""
        self._master_provider = fn

    def add_listener(self, callback: Callable[[PipelineState], None]) -> None:
        """Add a listener for pipeline state updates."""
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[PipelineState], None]) -> None:
        """Remove a pipeline state listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def start(self, tk_root, fps: Optional[int] = None) -> None:
        """Start the detection pipeline."""
        if self._running:
            return
        
        self._tk_root = tk_root
        self._running = True
        self._schedule(0)

    def stop(self) -> None:
        """Stop the detection pipeline."""
        self._running = False
        if self._after_id and self._tk_root:
            try:
                self._tk_root.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def is_running(self) -> bool:
        """Check if the pipeline is currently running."""
        return self._running

    def _schedule(self, delay_ms: int) -> None:
        """Schedule the next pipeline tick."""
        if not self._running:
            return
        self._after_id = self._tk_root.after(max(1, delay_ms), self._tick)

    def _tick(self) -> None:
        """Execute one pipeline iteration."""
        if not self._running:
            return
        
        try:
            start_time = time.time()
            
            # Read frame from source
            ok, frame = self.frame_source.read()
            if not ok or frame is None:
                self._schedule(50)  # Retry in 50ms
                return
            
            # Run inference
            detections: List[Detection] = self.inference_service.predict(frame) or []
            
            # Get master objects
            masters = self._master_provider() if self._master_provider else []
            
            # Match detections to masters
            matches: List[MatchResult] = []
            if masters and self.matcher:
                matches = self.matcher.match(detections, masters, (frame.shape[0], frame.shape[1]))
            
            # Generate feedback
            feedback = self.feedback_builder(matches) if self.feedback_builder else []
            
            # Calculate performance metrics
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
            
            # Create pipeline state
            state = PipelineState(
                frame=frame,
                detections=detections,
                matches=matches,
                feedback=feedback,
                latency_ms=latency_ms,
                fps=fps
            )
            
            # Notify all listeners
            for callback in self._listeners:
                try:
                    callback(state)
                except Exception as e:
                    print(f"Error in pipeline listener: {e}")
            
            # Calculate delay for next iteration
            target_fps = self.config.get('target_fps', 30)
            frame_period = 1000 // max(1, target_fps)
            delay = max(1, frame_period - latency_ms)
            
            self._schedule(delay)
            
        except Exception as e:
            print(f"Error in pipeline tick: {e}")
            self._schedule(100)  # Retry in 100ms

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            'running': self._running,
            'listeners': len(self._listeners),
            'has_master_provider': self._master_provider is not None
        }