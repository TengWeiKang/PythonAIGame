"""Optimized canvas components for high-performance image rendering."""

import tkinter as tk
from tkinter import Canvas
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
from typing import Optional, Tuple, Callable, Any
from collections import deque

from ..core.performance import performance_timer, LRUCache, PerformanceMonitor
from ..core.cache_manager import generate_image_hash

class OptimizedCanvas(Canvas):
    """High-performance canvas with optimized image rendering."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Performance optimizations
        self._image_cache = LRUCache(max_size=20)  # Cache rendered PhotoImages
        self._resize_cache = LRUCache(max_size=50)  # Cache resized images
        self._last_image_hash = None
        self._last_canvas_size = None
        
        # Rendering optimizations
        self._render_quality = 'medium'  # low, medium, high
        self._enable_smoothing = True
        self._max_render_size = (1920, 1080)  # Max size for performance
        
        # Async rendering
        self._render_thread = None
        self._render_queue = deque(maxlen=3)
        self._render_active = False
        self._render_lock = threading.Lock()
        
        # Bind resize event for cache invalidation
        self.bind('<Configure>', self._on_canvas_resize)
        
        # Register with performance monitor
        monitor = PerformanceMonitor.instance()
        monitor.register_cache("canvas_image", self._image_cache)
        monitor.register_cache("canvas_resize", self._resize_cache)
    
    def _on_canvas_resize(self, event):
        """Handle canvas resize events."""
        new_size = (event.width, event.height)
        if self._last_canvas_size != new_size:
            self._last_canvas_size = new_size
            # Clear resize cache on size change
            self._resize_cache.clear()
    
    def set_render_quality(self, quality: str):
        """Set rendering quality: 'low', 'medium', 'high'."""
        if quality in ['low', 'medium', 'high']:
            self._render_quality = quality
            # Clear caches when quality changes
            self._image_cache.clear()
            self._resize_cache.clear()
    
    @performance_timer("canvas_display_image")
    def display_image_optimized(self, image: np.ndarray, clear_canvas: bool = True) -> bool:
        """Display image with optimizations."""
        if image is None:
            return False
        
        try:
            # Generate image hash for caching
            image_hash = generate_image_hash(image)
            
            # Get canvas dimensions
            self.update_idletasks()
            canvas_width = self.winfo_width()
            canvas_height = self.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                return False
            
            canvas_size = (canvas_width, canvas_height)
            
            # Check if we can use cached version
            cache_key = f"{image_hash}:{canvas_size}:{self._render_quality}"
            cached_photo = self._image_cache.get(cache_key)
            
            if cached_photo is not None:
                # Use cached image
                if clear_canvas:
                    self.delete("all")
                self.create_image(
                    canvas_width // 2, 
                    canvas_height // 2,
                    anchor="center", 
                    image=cached_photo
                )
                # Keep reference to prevent garbage collection
                self.image_ref = cached_photo
                return True
            
            # Process and render new image
            processed_image = self._process_image_for_display(
                image, canvas_width, canvas_height
            )
            
            if processed_image is None:
                return False
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(processed_image)
            
            # Cache the PhotoImage
            self._image_cache.put(cache_key, photo)
            
            # Display on canvas
            if clear_canvas:
                self.delete("all")
            
            self.create_image(
                canvas_width // 2, 
                canvas_height // 2,
                anchor="center", 
                image=photo
            )
            
            # Keep reference to prevent garbage collection
            self.image_ref = photo
            self._last_image_hash = image_hash
            
            return True
            
        except Exception as e:
            print(f"Error displaying image: {e}")
            return False
    
    def display_image_async(self, image: np.ndarray, callback: Optional[Callable] = None):
        """Display image asynchronously to prevent UI blocking."""
        if not self._render_active:
            self._start_async_rendering()
        
        with self._render_lock:
            # Add to render queue (replace if queue is full)
            self._render_queue.append((image, callback))
    
    def _start_async_rendering(self):
        """Start async rendering thread."""
        if self._render_thread is None or not self._render_thread.is_alive():
            self._render_active = True
            self._render_thread = threading.Thread(
                target=self._async_render_worker,
                name="CanvasRenderer",
                daemon=True
            )
            self._render_thread.start()
    
    def _async_render_worker(self):
        """Background worker for async image rendering."""
        while self._render_active:
            try:
                render_task = None
                
                with self._render_lock:
                    if self._render_queue:
                        render_task = self._render_queue.popleft()
                
                if render_task:
                    image, callback = render_task
                    
                    # Render in background
                    success = False
                    try:
                        # Schedule UI update on main thread
                        self.after(0, lambda: self.display_image_optimized(image))
                        success = True
                    except Exception as e:
                        print(f"Async render error: {e}")
                    
                    # Call callback if provided
                    if callback:
                        self.after(0, lambda: callback(success))
                else:
                    time.sleep(0.01)  # Small delay when no tasks
                    
            except Exception as e:
                print(f"Async render worker error: {e}")
                time.sleep(0.1)
    
    def _process_image_for_display(self, image: np.ndarray, 
                                 canvas_width: int, canvas_height: int) -> Optional[Image.Image]:
        """Process image for optimal display."""
        try:
            # Create resize cache key
            img_shape = image.shape
            resize_key = f"{hash(image.tobytes()[:1000])}:{img_shape}:{canvas_width}x{canvas_height}:{self._render_quality}"
            
            # Check resize cache
            cached_processed = self._resize_cache.get(resize_key)
            if cached_processed is not None:
                return cached_processed
            
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Calculate optimal size
            img_height, img_width = image_rgb.shape[:2]
            
            # Limit maximum size for performance
            max_w, max_h = self._max_render_size
            if img_width > max_w or img_height > max_h:
                scale = min(max_w / img_width, max_h / img_height)
                img_width = int(img_width * scale)
                img_height = int(img_height * scale)
                image_rgb = cv2.resize(image_rgb, (img_width, img_height))
            
            # Calculate display scaling
            scale = min(canvas_width / img_width, canvas_height / img_height)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Choose interpolation method based on quality setting
            if self._render_quality == 'high':
                interpolation = cv2.INTER_LANCZOS4
            elif self._render_quality == 'medium':
                interpolation = cv2.INTER_LINEAR
            else:  # low quality
                interpolation = cv2.INTER_NEAREST
            
            # Resize image
            if (new_width, new_height) != (img_width, img_height):
                resized_image = cv2.resize(image_rgb, (new_width, new_height), 
                                         interpolation=interpolation)
            else:
                resized_image = image_rgb
            
            # Convert to PIL Image
            pil_image = Image.fromarray(resized_image.astype(np.uint8))
            
            # Apply additional optimizations based on quality
            if self._render_quality == 'low':
                # Reduce color depth for performance
                pil_image = pil_image.quantize(colors=64)
            elif self._enable_smoothing and self._render_quality == 'high':
                # Apply smoothing filter
                from PIL import ImageFilter
                pil_image = pil_image.filter(ImageFilter.SMOOTH)
            
            # Cache processed image
            self._resize_cache.put(resize_key, pil_image)
            
            return pil_image
            
        except Exception as e:
            print(f"Error processing image for display: {e}")
            return None
    
    def clear_caches(self):
        """Clear all image caches."""
        self._image_cache.clear()
        self._resize_cache.clear()
    
    def get_cache_stats(self) -> dict:
        """Get cache performance statistics."""
        return {
            'image_cache': {
                'hits': self._image_cache.stats().hits,
                'misses': self._image_cache.stats().misses,
                'hit_rate': self._image_cache.stats().hit_rate,
                'size': self._image_cache.stats().size
            },
            'resize_cache': {
                'hits': self._resize_cache.stats().hits,
                'misses': self._resize_cache.stats().misses,
                'hit_rate': self._resize_cache.stats().hit_rate,
                'size': self._resize_cache.stats().size
            }
        }
    
    def stop_async_rendering(self):
        """Stop async rendering thread."""
        self._render_active = False
        if self._render_thread and self._render_thread.is_alive():
            self._render_thread.join(timeout=1.0)

class VideoCanvas(OptimizedCanvas):
    """Specialized canvas for video display with frame rate optimization."""
    
    def __init__(self, parent, target_fps: int = 30, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.target_fps = target_fps
        self._frame_interval = 1.0 / target_fps
        self._last_frame_time = 0
        self._dropped_frames = 0
        self._total_frames = 0
        
        # Frame dropping strategy for performance
        self._enable_frame_dropping = True
        self._max_dropped_consecutive = 2
        self._consecutive_drops = 0
    
    @performance_timer("video_canvas_display")
    def display_frame(self, frame: np.ndarray) -> bool:
        """Display video frame with frame rate control."""
        current_time = time.time()
        self._total_frames += 1
        
        # Check if we should drop this frame for performance
        if self._enable_frame_dropping and self._should_drop_frame(current_time):
            self._dropped_frames += 1
            self._consecutive_drops += 1
            return False
        
        # Reset consecutive drops counter
        self._consecutive_drops = 0
        
        # Display frame
        success = self.display_image_optimized(frame)
        self._last_frame_time = current_time
        
        # Record performance metrics
        if self._total_frames % 30 == 0:  # Every 30 frames
            drop_rate = self._dropped_frames / self._total_frames
            monitor = PerformanceMonitor.instance()
            monitor.record_operation_time("video_frame_drop_rate", drop_rate)
        
        return success
    
    def _should_drop_frame(self, current_time: float) -> bool:
        """Determine if frame should be dropped for performance."""
        # Don't drop if enough time has passed
        time_since_last = current_time - self._last_frame_time
        if time_since_last >= self._frame_interval:
            return False
        
        # Don't drop too many consecutive frames
        if self._consecutive_drops >= self._max_dropped_consecutive:
            return False
        
        # Check system performance
        monitor = PerformanceMonitor.instance()
        current_metrics = monitor.get_current_metrics()
        
        if current_metrics:
            # Drop frame if CPU usage is high
            if current_metrics.cpu_percent > 80:
                return True
            
            # Drop frame if memory usage is high
            if current_metrics.memory_percent > 85:
                return True
        
        return False
    
    def get_performance_stats(self) -> dict:
        """Get video performance statistics."""
        stats = self.get_cache_stats()
        
        if self._total_frames > 0:
            drop_rate = self._dropped_frames / self._total_frames
            effective_fps = self.target_fps * (1 - drop_rate)
        else:
            drop_rate = 0
            effective_fps = 0
        
        stats['video'] = {
            'target_fps': self.target_fps,
            'effective_fps': effective_fps,
            'total_frames': self._total_frames,
            'dropped_frames': self._dropped_frames,
            'drop_rate': drop_rate
        }
        
        return stats

class ChatCanvas(Canvas):
    """Optimized canvas for chat message rendering."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Virtual scrolling for performance with many messages
        self._virtual_scrolling = True
        self._visible_message_buffer = 50  # Messages to keep rendered
        self._message_height_cache = LRUCache(max_size=200)
        
        # Smooth scrolling
        self._scroll_animation_active = False
        self._scroll_target = 0
        self._scroll_current = 0
    
    def enable_virtual_scrolling(self, enabled: bool = True):
        """Enable or disable virtual scrolling for performance."""
        self._virtual_scrolling = enabled
        if enabled:
            print("Virtual scrolling enabled for chat performance")
        else:
            print("Virtual scrolling disabled")
    
    def smooth_scroll_to(self, target_position: float, duration: float = 0.3):
        """Smoothly scroll to target position."""
        if self._scroll_animation_active:
            return  # Don't start new animation if one is active
        
        self._scroll_target = target_position
        self._scroll_current = self.canvasy(0)  # Current scroll position
        
        self._scroll_animation_active = True
        self._animate_scroll(duration)
    
    def _animate_scroll(self, duration: float):
        """Animate smooth scrolling."""
        steps = max(10, int(duration * 30))  # 30 FPS animation
        step_duration = duration / steps
        
        def scroll_step(current_step):
            if not self._scroll_animation_active or current_step >= steps:
                self._scroll_animation_active = False
                return
            
            # Easing function (ease-out)
            progress = current_step / steps
            eased_progress = 1 - (1 - progress) ** 3
            
            current_pos = self._scroll_current + (self._scroll_target - self._scroll_current) * eased_progress
            
            try:
                self.yview_moveto(current_pos / self.bbox("all")[3] if self.bbox("all") else 0)
            except:
                pass
            
            # Schedule next step
            self.after(int(step_duration * 1000), lambda: scroll_step(current_step + 1))
        
        scroll_step(0)