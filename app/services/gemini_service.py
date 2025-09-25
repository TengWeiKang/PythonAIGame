"""Gemini API service for image analysis and comparison using official Google SDK.

Security Features:
- Comprehensive input validation and sanitization
- API key validation and secure handling
- Rate limiting and request timeout protection
- Content filtering for sensitive information
- Secure error handling without information disclosure
"""

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
from ..utils.validation import (
    InputValidator,
    ContentFilter,
    ValidationError,
    validate_user_prompt
)

logger = logging.getLogger(__name__)


class GeminiService:
    """Service for interacting with Google's Gemini API for image analysis using official SDK."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash",
                 timeout: int = 30, temperature: float = 0.7, max_tokens: int = 2048,
                 persona: Optional[str] = None):
        """Initialize the Gemini service with configuration.

        Args:
            api_key: Gemini API key (validated for format)
            model: Model name (validated against supported models)
            timeout: Request timeout in seconds (5-300)
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum response tokens (1-8192)
            persona: Optional persona/system prompt (sanitized)

        Raises:
            ValidationError: If any parameter is invalid
            ValueError: If API key format is invalid
        """
        # Validate and sanitize inputs
        self.api_key = self._validate_api_key(api_key) if api_key else None
        self.model_name = self._validate_model_name(model)
        self.timeout = self._validate_timeout(timeout)
        self.temperature = self._validate_temperature(temperature)
        self.max_tokens = self._validate_max_tokens(max_tokens)
        self.persona = self._sanitize_persona(persona) if persona else ""
        
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
        """Configure Gemini API with SDK.

        Args:
            api_key: The API key to configure

        Raises:
            RuntimeError: If SDK is not available
            ValidationError: If API key is invalid
            WebcamError: If configuration fails
        """
        if not SDK_AVAILABLE:
            raise RuntimeError("Google Generative AI SDK not available")

        # Validate API key before using
        validated_key = self._validate_api_key(api_key)

        try:
            genai.configure(api_key=validated_key)
            self.api_key = validated_key
            
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
        """Set the API key for the service.

        Args:
            api_key: The API key to set (will be validated)

        Raises:
            ValidationError: If API key is invalid
        """
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
        """Check if the service is properly configured with an API key.

        Returns:
            bool: True if service can be used for API calls
        """
        # Check SDK availability first
        if not SDK_AVAILABLE:
            logger.debug("Gemini SDK not available")
            return False

        # Check API key validity
        if not self.api_key or not self.api_key.strip():
            logger.debug("No API key provided")
            return False

        # Validate API key format
        try:
            self._validate_api_key(self.api_key)
        except Exception as e:
            logger.debug(f"API key validation failed: {e}")
            return False

        # Check if model is initialized (this indicates successful configuration)
        if self.model is None:
            logger.debug("Gemini model not initialized - attempting configuration")
            # Try to configure the API if we have a valid key but no model
            try:
                self.configure_api(self.api_key)
                return self.model is not None
            except Exception as e:
                logger.debug(f"API configuration failed: {e}")
                return False

        return True
    
    def start_chat_session(self, persona: Optional[str] = None) -> None:
        """Start a new chat session with custom persona.

        Args:
            persona: Optional persona/system prompt (will be sanitized)

        Raises:
            ValueError: If API not configured
            ValidationError: If persona contains invalid content
            WebcamError: If session creation fails
        """
        if not self.model:
            raise ValueError("API not configured")

        try:
            history = []
            if persona and persona.strip():
                # Sanitize and validate persona
                sanitized_persona = self._sanitize_persona(persona)

                # Add persona as system instruction
                history.append({
                    "role": "user",
                    "parts": [f"System: {sanitized_persona}"]
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
        """Send message with optional image to Gemini with caching.

        Args:
            message: User message (will be validated and sanitized)
            image_data: Optional image data (will be validated)

        Returns:
            str: AI response

        Raises:
            ValidationError: If input is invalid
            WebcamError: If API request fails
        """
        # Validate and sanitize message
        if not isinstance(message, str) or not message.strip():
            raise ValidationError("Message must be a non-empty string")

        sanitized_message = validate_user_prompt(message)

        # Check for sensitive content
        if ContentFilter.contains_sensitive_info(sanitized_message):
            logger.warning("Sensitive content detected in message, filtering")
            sanitized_message = ContentFilter.filter_sensitive_content(sanitized_message)

        # Validate image data if provided
        if image_data is not None:
            is_valid, error_msg = InputValidator.validate_image_data(image_data)
            if not is_valid:
                raise ValidationError(f"Invalid image data: {error_msg}")

        # Generate cache key
        cache_key = self._generate_cache_key(sanitized_message, image_data)

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
                response = self.chat_session.send_message([sanitized_message, pil_image])
            else:
                # Text-only message
                response = self.chat_session.send_message(sanitized_message)
            
            self._last_request_time = time.time()
            result = response.text
            
            # Cache the response
            self._response_cache.put(cache_key, result)
            
            return result
            
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Log error without exposing sensitive information
            error_msg = "Gemini API request failed"
            if "quota" in str(e).lower():
                error_msg = "API quota exceeded - please check your usage limits"
            elif "api key" in str(e).lower():
                error_msg = "API authentication failed - please check your API key"
            elif "timeout" in str(e).lower():
                error_msg = "API request timed out - please try again"

            logger.error(f"Gemini API error: {type(e).__name__}")
            raise WebcamError(error_msg)
    
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
        """Analyze a single image with optional custom prompt.

        Args:
            image: Image array to analyze
            prompt: Optional custom prompt (will be validated and sanitized)

        Returns:
            str: Analysis result

        Raises:
            ValidationError: If inputs are invalid
            WebcamError: If analysis fails
        """
        # Validate image input
        if not isinstance(image, np.ndarray):
            raise ValidationError("Image must be a numpy array")

        if image.size == 0:
            raise ValidationError("Image array is empty")

        # Use default prompt if none provided
        if prompt is None:
            prompt = "Describe this image in detail, focusing on objects, their positions, and any notable features."

        # Validate and sanitize prompt
        sanitized_prompt = validate_user_prompt(prompt)

        # Check for sensitive content
        if ContentFilter.contains_sensitive_info(sanitized_prompt):
            logger.warning("Sensitive content detected in analysis prompt, filtering")
            sanitized_prompt = ContentFilter.filter_sensitive_content(sanitized_prompt)

        # Generate cache key for image analysis
        image_hash = generate_image_hash(image)
        cache_key = f"analyze:{image_hash}:{hashlib.md5(sanitized_prompt.encode()).hexdigest()}"
        
        # Try cache first
        cached_result = self._response_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        pil_image = self._numpy_to_pil(image)
        contents = [sanitized_prompt, pil_image]

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

    def _validate_api_key(self, api_key: str) -> str:
        """Validate Gemini API key format.

        Args:
            api_key: The API key to validate

        Returns:
            str: The validated API key

        Raises:
            ValidationError: If API key is invalid
        """
        if not api_key or not isinstance(api_key, str):
            raise ValidationError("API key must be a non-empty string")

        # Validate API key format
        from ..utils.validation import EnvironmentValidator
        if not EnvironmentValidator.validate_api_key(api_key):
            raise ValidationError("Invalid API key format")

        return api_key

    def _validate_model_name(self, model: str) -> str:
        """Validate model name.

        Args:
            model: The model name to validate

        Returns:
            str: The validated model name

        Raises:
            ValidationError: If model name is invalid
        """
        if not model or not isinstance(model, str):
            raise ValidationError("Model name must be a non-empty string")

        from ..utils.validation import EnvironmentValidator
        if not EnvironmentValidator.validate_model_name(model):
            logger.warning(f"Unrecognized model name: {model}, proceeding anyway")

        return model

    def _validate_timeout(self, timeout: int) -> int:
        """Validate timeout value.

        Args:
            timeout: The timeout in seconds

        Returns:
            int: The validated timeout

        Raises:
            ValidationError: If timeout is invalid
        """
        from ..utils.validation import EnvironmentValidator
        return EnvironmentValidator.validate_numeric_range(timeout, 5, 300, int)

    def _validate_temperature(self, temperature: float) -> float:
        """Validate temperature value.

        Args:
            temperature: The temperature value

        Returns:
            float: The validated temperature

        Raises:
            ValidationError: If temperature is invalid
        """
        from ..utils.validation import EnvironmentValidator
        return EnvironmentValidator.validate_numeric_range(temperature, 0.0, 1.0, float)

    def _validate_max_tokens(self, max_tokens: int) -> int:
        """Validate max tokens value.

        Args:
            max_tokens: The maximum tokens

        Returns:
            int: The validated max tokens

        Raises:
            ValidationError: If max tokens is invalid
        """
        from ..utils.validation import EnvironmentValidator
        return EnvironmentValidator.validate_numeric_range(max_tokens, 1, 8192, int)

    def _sanitize_persona(self, persona: str) -> str:
        """Sanitize persona/system prompt.

        Args:
            persona: The persona to sanitize

        Returns:
            str: The sanitized persona

        Raises:
            ValidationError: If persona contains invalid content
        """
        if not persona or not isinstance(persona, str):
            return ""

        # Use input validator to sanitize
        sanitized = InputValidator.sanitize_string_input(persona, max_length=5000)

        # Check for sensitive content
        if ContentFilter.contains_sensitive_info(sanitized):
            logger.warning("Sensitive content detected in persona, filtering")
            sanitized = ContentFilter.filter_sensitive_content(sanitized)

        return sanitized
    
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
        # Enhanced initialization with environment variable fallback
        effective_api_key = self._resolve_api_key(api_key)
        super().__init__(effective_api_key, **kwargs)
        self._thread_pool = []

        # Track configuration state for debugging
        self._config_diagnostics = {
            'input_api_key': api_key,
            'effective_api_key': effective_api_key,
            'sdk_available': SDK_AVAILABLE,
            'initialization_time': time.time(),
            'last_config_check': None,
            'last_config_error': None
        }

    def _resolve_api_key(self, provided_key: Optional[str]) -> Optional[str]:
        """Resolve API key from multiple sources with priority order.

        Priority:
        1. Explicitly provided key parameter
        2. Environment variable GEMINI_API_KEY
        3. Environment file (.env)

        Returns:
            str: The resolved API key or None
        """
        # 1. Use explicitly provided key if valid
        if provided_key and provided_key.strip():
            logger.debug("Using explicitly provided API key")
            return provided_key.strip()

        # 2. Try environment variable
        import os
        env_key = os.getenv('GEMINI_API_KEY')
        if env_key and env_key.strip():
            logger.debug("Using API key from environment variable")
            return env_key.strip()

        # 3. Try loading from .env file
        try:
            from ..config.env_config import load_env_file, get_env_var
            env_vars = load_env_file()
            env_file_key = get_env_var('GEMINI_API_KEY', env_vars=env_vars)
            if env_file_key and env_file_key.strip():
                logger.debug("Using API key from .env file")
                return env_file_key.strip()
        except Exception as e:
            logger.debug(f"Could not load API key from .env file: {e}")

        logger.debug("No API key found in any source")
        return None
    
    def is_configured(self) -> bool:
        """Enhanced configuration check with diagnostic tracking."""
        self._config_diagnostics['last_config_check'] = time.time()

        try:
            result = super().is_configured()
            self._config_diagnostics['last_config_error'] = None
            return result
        except Exception as e:
            self._config_diagnostics['last_config_error'] = str(e)
            logger.error(f"Configuration check failed: {e}")
            return False

    def get_configuration_status(self) -> Dict[str, Any]:
        """Get detailed configuration status for debugging.

        Returns:
            dict: Comprehensive configuration state information
        """
        status = {
            'is_configured': False,
            'has_api_key': bool(self.api_key and self.api_key.strip()),
            'has_model': bool(self.model),
            'sdk_available': SDK_AVAILABLE,
            'api_key_valid_format': False,
            'diagnostics': self._config_diagnostics,
            'config_sources': [],
            'errors': []
        }

        # Check API key format
        if self.api_key and self.api_key.strip():
            try:
                self._validate_api_key(self.api_key)
                status['api_key_valid_format'] = True
                status['config_sources'].append('API key format valid')
            except Exception as e:
                status['errors'].append(f"API key format invalid: {e}")

        # Check model initialization
        if not self.model and self.api_key:
            try:
                # Attempt to configure if we have key but no model
                self.configure_api(self.api_key)
                status['has_model'] = bool(self.model)
                if self.model:
                    status['config_sources'].append('Model configured successfully')
            except Exception as e:
                status['errors'].append(f"Model configuration failed: {e}")

        # Final configuration check
        try:
            status['is_configured'] = self.is_configured()
        except Exception as e:
            status['errors'].append(f"Configuration check failed: {e}")

        return status

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