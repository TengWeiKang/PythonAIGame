"""Gemini AI service for image analysis and chat."""

import logging
from typing import Optional, List, Dict, Any
import numpy as np
import base64
import io
from PIL import Image

logger = logging.getLogger(__name__)


class GeminiService:
    """Service for interacting with Google Gemini AI API."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-pro",
                 temperature: float = 0.7, max_tokens: int = 2048, timeout: int = 30):
        """Initialize Gemini service.

        Args:
            api_key: Google AI API key
            model: Gemini model name
            temperature: Response temperature (0-1)
            max_tokens: Maximum response tokens
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize Gemini client.

        Returns:
            True if initialized successfully, False otherwise
        """
        try:
            import google.generativeai as genai

            if not self.api_key or self.api_key.strip() == "":
                logger.error("API key is empty")
                return False

            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)
            self._initialized = True
            logger.info(f"Gemini service initialized with model: {self.model}")
            return True

        except ImportError:
            logger.error("google-generativeai package not installed. Install with: pip install google-generativeai")
            return False
        except Exception as e:
            logger.error(f"Error initializing Gemini service: {e}")
            return False

    def analyze_image(self, image: np.ndarray, prompt: str,
                     detections: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """Analyze image with Gemini AI.

        Args:
            image: Image as numpy array (BGR format)
            prompt: Analysis prompt
            detections: Optional YOLO detections to include in context

        Returns:
            AI response text, or None on failure
        """
        if not self._initialized:
            if not self.initialize():
                return None

        try:
            # Convert BGR to RGB
            image_rgb = image[..., ::-1].copy()

            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)

            # Enhance prompt with detections if provided
            enhanced_prompt = self._build_prompt(prompt, detections)

            # Generate response
            response = self._client.generate_content([enhanced_prompt, pil_image])

            if response and response.text:
                return response.text
            else:
                logger.warning("Empty response from Gemini")
                return None

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return None

    def compare_images(self, reference_image: np.ndarray, current_image: np.ndarray,
                      prompt: str, ref_detections: Optional[List[Dict[str, Any]]] = None,
                      curr_detections: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """Compare two images with Gemini AI.

        Args:
            reference_image: Reference image as numpy array
            current_image: Current image as numpy array
            prompt: Comparison prompt
            ref_detections: Detections from reference image
            curr_detections: Detections from current image

        Returns:
            AI response text, or None on failure
        """
        if not self._initialized:
            if not self.initialize():
                return None

        try:
            # Convert images
            ref_rgb = reference_image[..., ::-1].copy()
            curr_rgb = current_image[..., ::-1].copy()

            ref_pil = Image.fromarray(ref_rgb)
            curr_pil = Image.fromarray(curr_rgb)

            # Build enhanced prompt
            enhanced_prompt = self._build_comparison_prompt(prompt, ref_detections, curr_detections)

            # Generate response with both images
            response = self._client.generate_content([
                enhanced_prompt,
                "Reference Image:",
                ref_pil,
                "Current Image:",
                curr_pil
            ])

            if response and response.text:
                return response.text
            else:
                logger.warning("Empty response from Gemini")
                return None

        except Exception as e:
            logger.error(f"Error comparing images: {e}")
            return None

    def chat(self, message: str, context: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
        """Send chat message to Gemini.

        Args:
            message: User message
            context: Optional conversation context (list of {role, content} dicts)

        Returns:
            AI response text, or None on failure
        """
        if not self._initialized:
            if not self.initialize():
                return None

        try:
            # Build conversation history
            if context:
                # Start chat session with history
                chat = self._client.start_chat(history=[])
                response = chat.send_message(message)
            else:
                # Single message
                response = self._client.generate_content(message)

            if response and response.text:
                return response.text
            else:
                logger.warning("Empty response from Gemini")
                return None

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return None

    def _build_prompt(self, base_prompt: str, detections: Optional[List[Dict[str, Any]]]) -> str:
        """Build enhanced prompt with detection information.

        Args:
            base_prompt: Base user prompt
            detections: Optional detection results

        Returns:
            Enhanced prompt string
        """
        if not detections:
            return base_prompt

        # Format detections
        det_summary = self._format_detections(detections)

        enhanced = f"{base_prompt}\n\n"
        enhanced += "YOLO Detection Results:\n"
        enhanced += det_summary
        enhanced += "\n\nPlease analyze the image considering these detected objects."

        return enhanced

    def _build_comparison_prompt(self, base_prompt: str,
                                ref_detections: Optional[List[Dict[str, Any]]],
                                curr_detections: Optional[List[Dict[str, Any]]]) -> str:
        """Build comparison prompt with detection information.

        Args:
            base_prompt: Base user prompt
            ref_detections: Reference image detections
            curr_detections: Current image detections

        Returns:
            Enhanced prompt string
        """
        enhanced = f"{base_prompt}\n\n"

        if ref_detections:
            enhanced += "Reference Image Detections:\n"
            enhanced += self._format_detections(ref_detections)
            enhanced += "\n"

        if curr_detections:
            enhanced += "Current Image Detections:\n"
            enhanced += self._format_detections(curr_detections)
            enhanced += "\n"

        enhanced += "\nPlease compare these images and identify key differences."

        return enhanced

    def _format_detections(self, detections: List[Dict[str, Any]]) -> str:
        """Format detections as readable text.

        Args:
            detections: List of detection dictionaries

        Returns:
            Formatted string
        """
        if not detections:
            return "No objects detected."

        # Count objects by class
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # Format summary
        lines = []
        lines.append(f"Total objects detected: {len(detections)}")
        for class_name, count in sorted(class_counts.items()):
            lines.append(f"- {class_name}: {count}")

        return "\n".join(lines)

    def is_initialized(self) -> bool:
        """Check if service is initialized.

        Returns:
            True if initialized, False otherwise
        """
        return self._initialized

    def update_config(self, model: Optional[str] = None, temperature: Optional[float] = None,
                     max_tokens: Optional[int] = None):
        """Update service configuration.

        Args:
            model: New model name
            temperature: New temperature value
            max_tokens: New max tokens value
        """
        if model is not None:
            self.model = model
            self._initialized = False  # Need to reinitialize

        if temperature is not None:
            self.temperature = max(0.0, min(1.0, temperature))

        if max_tokens is not None:
            self.max_tokens = max(1, max_tokens)