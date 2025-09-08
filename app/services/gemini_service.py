"""Gemini API service for image analysis and comparison using official Google SDK."""

import base64
import io
import hashlib
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from PIL import Image
import threading
import time
import logging

try:
    import google.generativeai as genai
    from google.generativeai import GenerativeModel
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    SDK_AVAILABLE = True
except ImportError:
    genai = None
    GenerativeModel = None
    HarmCategory = None
    HarmBlockThreshold = None
    SDK_AVAILABLE = False

from ..core.exceptions import WebcamError
from ..core.performance import performance_timer, LRUCache, ThreadPool
from ..core.cache_manager import generate_image_hash

logger = logging.getLogger(__name__)


class GeminiService:
    """Service for interacting with Google's Gemini API for image analysis using official SDK."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash", 
                 timeout: int = 30, temperature: float = 0.7, max_tokens: int = 2048,
                 persona: Optional[str] = None):
        """Initialize the Gemini service with configuration."""
        self.api_key = api_key
        self.model_name = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.persona = persona or ""
        
        # SDK objects
        self.model = None
        self.chat_session = None
        
        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 1.0  # Minimum seconds between requests
        self._rate_limit_enabled = True
        self._requests_per_minute = 15
        
        # Performance optimizations
        self._response_cache = LRUCache(max_size=100)  # Cache API responses
        self._image_hash_cache = LRUCache(max_size=200)  # Cache image hashes
        self._request_queue = []
        self._batch_size = 3
        self._batch_timeout = 2.0  # seconds
        
        # Thread pool for async operations
        self._thread_pool = ThreadPool(max_workers=2, thread_name_prefix="Gemini")
        
        # Connection pooling for better performance
        self._session_pool = []
        self._max_sessions = 3
        
        # Check SDK availability
        if not SDK_AVAILABLE:
            logger.warning("Google Generative AI SDK not available. Install with: pip install google-generativeai")
        
        # Configure API if key is provided
        if api_key:
            self.configure_api(api_key)
        
    def configure_api(self, api_key: str) -> None:
        """Configure Gemini API with SDK."""
        if not SDK_AVAILABLE:
            raise RuntimeError("Google Generative AI SDK not available")
        
        try:
            genai.configure(api_key=api_key)
            self.api_key = api_key
            
            # Create model with safety settings
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            
            generation_config = genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
            
            self.model = GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            logger.info(f"Gemini API configured successfully with model {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
            raise WebcamError(f"Failed to configure Gemini API: {e}")
    
    def set_api_key(self, api_key: str) -> None:
        """Set the API key for the service."""
        self.configure_api(api_key)
    
    def update_configuration(self, **kwargs) -> None:
        """Update service configuration settings."""
        reconfigure = False
        
        if 'api_key' in kwargs and kwargs['api_key'] != self.api_key:
            self.api_key = kwargs['api_key']
            reconfigure = True
        
        if 'model' in kwargs and kwargs['model'] != self.model_name:
            self.model_name = kwargs['model']
            reconfigure = True
            
        if 'temperature' in kwargs:
            self.temperature = kwargs['temperature']
            reconfigure = True
            
        if 'max_tokens' in kwargs:
            self.max_tokens = kwargs['max_tokens']
            reconfigure = True
            
        if 'persona' in kwargs:
            self.persona = kwargs['persona'] or ""
            # Restart chat session if persona changed
            if self.chat_session:
                self.start_chat_session(self.persona)
        
        if 'enable_rate_limiting' in kwargs:
            self._rate_limit_enabled = kwargs['enable_rate_limiting']
            
        if 'requests_per_minute' in kwargs:
            self._requests_per_minute = kwargs['requests_per_minute']
            self._min_request_interval = 60.0 / self._requests_per_minute if self._requests_per_minute > 0 else 1.0
        
        # Reconfigure model if needed
        if reconfigure and self.api_key:
            self.configure_api(self.api_key)
        
    def is_configured(self) -> bool:
        """Check if the service is properly configured with an API key."""
        return bool(self.api_key and self.api_key.strip() and self.model is not None)
    
    def start_chat_session(self, persona: Optional[str] = None) -> None:
        """Start a new chat session with custom persona."""
        if not self.model:
            raise ValueError("API not configured")
        
        try:
            history = []
            if persona and persona.strip():
                # Add persona as system instruction
                history.append({
                    "role": "user",
                    "parts": [f"System: {persona.strip()}"]
                })
                history.append({
                    "role": "model", 
                    "parts": ["I understand. I'll act according to that role and persona."]
                })
            
            self.chat_session = self.model.start_chat(history=history)
            self.persona = persona or ""
            logger.debug("Chat session started with persona")
            
        except Exception as e:
            logger.error(f"Failed to start chat session: {e}")
            raise WebcamError(f"Failed to start chat session: {e}")
    
    @performance_timer("gemini_send_message")
    def send_message(self, message: str, image_data: Optional[bytes] = None) -> str:
        """Send message with optional image to Gemini with caching."""
        # Generate cache key
        cache_key = self._generate_cache_key(message, image_data)
        
        # Try to get from cache first
        cached_response = self._response_cache.get(cache_key)
        if cached_response is not None:
            return cached_response
        
        if not self.chat_session:
            self.start_chat_session(self.persona)
        
        try:
            # Apply rate limiting
            if self._rate_limit_enabled:
                current_time = time.time()
                time_since_last = current_time - self._last_request_time
                if time_since_last < self._min_request_interval:
                    time.sleep(self._min_request_interval - time_since_last)
            
            if image_data:
                # Convert bytes to PIL Image for SDK
                pil_image = Image.open(io.BytesIO(image_data))
                response = self.chat_session.send_message([message, pil_image])
            else:
                # Text-only message
                response = self.chat_session.send_message(message)
            
            self._last_request_time = time.time()
            result = response.text
            
            # Cache the response
            self._response_cache.put(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise WebcamError(f"Gemini API error: {e}")
    
    def _generate_cache_key(self, message: str, image_data: Optional[bytes] = None) -> str:
        """Generate cache key for request."""
        key_parts = [message, self.model_name, str(self.temperature)]
        
        if image_data:
            # Generate hash for image data
            image_hash = hashlib.md5(image_data).hexdigest()
            key_parts.append(image_hash)
        
        combined_key = "|".join(key_parts)
        return hashlib.md5(combined_key.encode()).hexdigest()
    
    def _numpy_to_pil(self, image: np.ndarray) -> Image.Image:
        """Convert numpy image array to PIL Image."""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = image[:, :, ::-1]  # BGR to RGB
        else:
            image_rgb = image
            
        # Convert to PIL Image
        return Image.fromarray(image_rgb.astype(np.uint8))
    
    def _generate_content(self, contents: List[Any]) -> str:
        """Generate content using the SDK."""
        if not self.model:
            raise ValueError("API not configured")
        
        try:
            # Apply rate limiting
            if self._rate_limit_enabled:
                current_time = time.time()
                time_since_last = current_time - self._last_request_time
                if time_since_last < self._min_request_interval:
                    time.sleep(self._min_request_interval - time_since_last)
            
            response = self.model.generate_content(contents)
            self._last_request_time = time.time()
            
            return response.text
            
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            raise WebcamError(f"Content generation failed: {e}")
    
    @performance_timer("gemini_analyze_image")
    def analyze_single_image(self, image: np.ndarray, prompt: str = None) -> str:
        """Analyze a single image with optional custom prompt."""
        if prompt is None:
            prompt = "Describe this image in detail, focusing on objects, their positions, and any notable features."
        
        # Generate cache key for image analysis
        image_hash = generate_image_hash(image)
        cache_key = f"analyze:{image_hash}:{hashlib.md5(prompt.encode()).hexdigest()}"
        
        # Try cache first
        cached_result = self._response_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        pil_image = self._numpy_to_pil(image)
        contents = [prompt, pil_image]
        
        result = self._generate_content(contents)
        
        # Cache result
        self._response_cache.put(cache_key, result)
        
        return result
    
    @performance_timer("gemini_compare_images")
    def compare_images(self, reference_image: np.ndarray, current_image: np.ndarray, 
                      custom_prompt: str = None) -> str:
        """Compare two images and return detailed analysis of differences."""
        if custom_prompt is None:
            custom_prompt = """Compare these two images and identify all differences between them. 
            Focus on:
            1. Objects that are missing, added, or moved
            2. Changes in object positions or orientations
            3. Color or appearance differences
            4. Any other notable changes
            
            Provide a detailed, structured analysis of the differences you observe."""
        
        # Generate cache key for image comparison
        ref_hash = generate_image_hash(reference_image)
        cur_hash = generate_image_hash(current_image)
        prompt_hash = hashlib.md5(custom_prompt.encode()).hexdigest()
        cache_key = f"compare:{ref_hash}:{cur_hash}:{prompt_hash}"
        
        # Try cache first
        cached_result = self._response_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        reference_pil = self._numpy_to_pil(reference_image)
        current_pil = self._numpy_to_pil(current_image)
        
        contents = [
            "Reference image (first image):",
            reference_pil,
            "Current image (second image):",
            current_pil,
            custom_prompt
        ]
        
        result = self._generate_content(contents)
        
        # Cache result
        self._response_cache.put(cache_key, result)
        
        return result
    
    def generate_difference_highlights(self, reference_image: np.ndarray, 
                                     current_image: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        """Generate difference analysis with suggested highlight regions."""
        prompt = """Compare these two images and identify specific differences. 
        For each difference you find, provide:
        1. A description of what changed
        2. The approximate location in the image (as percentages from top-left: x%, y%)
        3. The approximate size of the changed area (width%, height%)
        
        Format your response as a structured analysis that can be used to highlight differences visually.
        Be as specific as possible about locations and sizes of changes."""
        
        analysis = self.compare_images(reference_image, current_image, prompt)
        
        # Parse the response to extract highlighting information
        # This is a simplified parser - in production, you might want more sophisticated parsing
        highlight_regions = self._parse_highlight_regions(analysis)
        
        return analysis, highlight_regions
    
    def _parse_highlight_regions(self, analysis_text: str) -> Dict[str, Any]:
        """Parse analysis text to extract highlight regions (simplified implementation)."""
        # This is a basic implementation - you could enhance this with better NLP
        regions = []
        lines = analysis_text.split('\n')
        
        for line in lines:
            line = line.strip().lower()
            if any(keyword in line for keyword in ['change', 'difference', 'moved', 'added', 'missing']):
                # Simple pattern matching for coordinates (this could be enhanced)
                if '%' in line:
                    # Try to extract percentage values
                    words = line.split()
                    percentages = []
                    for word in words:
                        if '%' in word:
                            try:
                                percent = float(word.replace('%', '').replace(',', ''))
                                percentages.append(percent)
                            except ValueError:
                                continue
                    
                    if len(percentages) >= 2:
                        regions.append({
                            'x': percentages[0] / 100.0,
                            'y': percentages[1] / 100.0,
                            'width': percentages[2] / 100.0 if len(percentages) > 2 else 0.1,
                            'height': percentages[3] / 100.0 if len(percentages) > 3 else 0.1,
                            'description': line
                        })
        
        return {'regions': regions, 'raw_analysis': analysis_text}


class AsyncGeminiService(GeminiService):
    """Asynchronous wrapper for Gemini service to prevent UI blocking."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key, **kwargs)
        self._thread_pool = []
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with backward compatibility."""
        self.update_configuration(**kwargs)
    
    def set_rate_limit(self, enabled: bool, requests_per_minute: int) -> None:
        """Set rate limiting configuration."""
        self.update_configuration(
            enable_rate_limiting=enabled,
            requests_per_minute=requests_per_minute
        )
    
    def set_context_window(self, size: int) -> None:
        """Set context window size (placeholder for future implementation)."""
        # This could be implemented in future versions
        logger.debug(f"Context window size set to {size} (not yet implemented)")
    
    def set_generation_config(self, temperature: float, max_output_tokens: int) -> None:
        """Set generation configuration."""
        self.update_configuration(
            temperature=temperature,
            max_tokens=max_output_tokens
        )
    
    def analyze_single_image_async(self, image: np.ndarray, callback, prompt: str = None):
        """Analyze image asynchronously and call callback with result."""
        def worker():
            try:
                result = self.analyze_single_image(image, prompt)
                callback(result, None)
            except Exception as e:
                callback(None, str(e))
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self._thread_pool.append(thread)
    
    def compare_images_async(self, reference_image: np.ndarray, current_image: np.ndarray, 
                           callback, custom_prompt: str = None):
        """Compare images asynchronously and call callback with result."""
        def worker():
            try:
                result = self.compare_images(reference_image, current_image, custom_prompt)
                callback(result, None)
            except Exception as e:
                callback(None, str(e))
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self._thread_pool.append(thread)
    
    def generate_difference_highlights_async(self, reference_image: np.ndarray, 
                                           current_image: np.ndarray, callback):
        """Generate difference highlights asynchronously and call callback with result."""
        def worker():
            try:
                analysis, regions = self.generate_difference_highlights(reference_image, current_image)
                callback((analysis, regions), None)
            except Exception as e:
                callback(None, str(e))
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self._thread_pool.append(thread)
    
    def cleanup_threads(self):
        """Clean up finished threads."""
        self._thread_pool = [t for t in self._thread_pool if t.is_alive()]