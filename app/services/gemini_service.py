"""Gemini AI service for image analysis and chat."""

import logging
from typing import Optional, List, Dict, Any
import numpy as np
from PIL import Image
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiService:
    """Service for interacting with Google Gemini AI API."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash",
                 temperature: float = 0.7, max_tokens: int = 2048, timeout: int = 30,
                 persona: str = ""):
        """Initialize Gemini service.

        Args:
            api_key: Google AI API key
            model: Gemini model name
            temperature: Response temperature (0-1)
            max_tokens: Maximum response tokens
            timeout: Request timeout in seconds
            persona: Custom AI persona/role instructions
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.persona = persona
        self._client = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize Gemini client using the new google-genai library.

        Returns:
            True if initialized successfully, False otherwise
        """
        try:
            if not self.api_key or self.api_key.strip() == "":
                logger.error("API key is empty")
                return False

            # Create client with API key (new google-genai API)
            self._client = genai.Client(api_key=self.api_key)
            logger.info(f"Gemini service initialized with model: {self.model}")

            self._initialized = True
            return True

        except ImportError:
            logger.error("google-genai package not installed. Install with: pip install google-genai")
            return False
        except Exception as e:
            logger.error(f"Error initializing Gemini service: {e}")
            return False

    def analyze_image(self, image: np.ndarray, prompt: str,
                     detections: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """Analyze image with Gemini AI using the new google-genai library.

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
            logger.info(f"Sending prompt to Gemini API (analyze_image): {enhanced_prompt}")

            # Build config with system instruction
            config = self._build_generation_config()

            # Generate response using new client API
            # New API: client.models.generate_content(model=..., contents=..., config=...)
            response = self._client.models.generate_content(
                model=self.model,
                contents=[enhanced_prompt, pil_image],
                config=config
            )

            if response and response.text:
                return response.text
            else:
                logger.warning("Empty response from Gemini")
                return None

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return None

    def analyze_with_text_only(self, prompt: str,
                               detections: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """Analyze using ONLY text prompt (no images sent to API).

        This method is used in yolo_detection mode where YOLO provides object
        detection results and only these text results are sent to Gemini.

        Args:
            prompt: Analysis prompt
            detections: YOLO detections to include as text context

        Returns:
            AI response text, or None on failure
        """
        if not self._initialized:
            if not self.initialize():
                return None

        try:
            # Build text-only prompt with detection results
            enhanced_prompt = self._build_prompt(prompt, detections)
            logger.info(f"Sending prompt to Gemini API (analyze_with_text_only): {enhanced_prompt}")

            # Build config with system instruction
            config = self._build_generation_config()

            # Generate response with ONLY text (no images) using new client API
            response = self._client.models.generate_content(
                model=self.model,
                contents=enhanced_prompt,
                config=config
            )

            if response and response.text:
                return response.text
            else:
                logger.warning("Empty response from Gemini")
                return None

        except Exception as e:
            logger.error(f"Error analyzing with text only: {e}")
            return None

    def compare_images(self, reference_image: np.ndarray, current_image: np.ndarray,
                      prompt: str, ref_detections: Optional[List[Dict[str, Any]]] = None,
                      curr_detections: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """Compare two images with Gemini AI using the new google-genai library.

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
            logger.info(f"Sending prompt to Gemini API (compare_images): {enhanced_prompt}")

            # Build config with system instruction
            config = self._build_generation_config()

            # Generate response with both images using new client API
            response = self._client.models.generate_content(
                model=self.model,
                contents=[
                    enhanced_prompt,
                    "Reference Image:",
                    ref_pil,
                    "Current Image:",
                    curr_pil
                ],
                config=config
            )

            if response and response.text:
                return response.text
            else:
                logger.warning("Empty response from Gemini")
                return None

        except Exception as e:
            logger.error(f"Error comparing images: {e}")
            return None

    def compare_with_text_only(self, prompt: str,
                               ref_detections: Optional[List[Dict[str, Any]]] = None,
                               curr_detections: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """Compare using ONLY text prompt (no images sent to API).

        This method is used in yolo_detection mode where YOLO provides object
        detection results from both images and only these text results are sent to Gemini.

        Args:
            prompt: Comparison prompt
            ref_detections: Reference image YOLO detections
            curr_detections: Current image YOLO detections

        Returns:
            AI response text, or None on failure
        """
        if not self._initialized:
            if not self.initialize():
                return None

        try:
            # Build text-only comparison prompt with detection results
            enhanced_prompt = self._build_comparison_prompt(prompt, ref_detections, curr_detections)
            logger.info(f"Sending prompt to Gemini API (compare_with_text_only): {enhanced_prompt}")

            # Build config with system instruction
            config = self._build_generation_config()

            # Generate response with ONLY text (no images) using new client API
            response = self._client.models.generate_content(
                model=self.model,
                contents=enhanced_prompt,
                config=config
            )

            if response and response.text:
                return response.text
            else:
                logger.warning("Empty response from Gemini")
                return None

        except Exception as e:
            logger.error(f"Error comparing with text only: {e}")
            return None

    def chat(self, message: str, context: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
        """Send chat message to Gemini using the new google-genai library.

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
            logger.info(f"Sending prompt to Gemini API (chat): {message}")

            # Build config with system instruction
            config = self._build_generation_config()

            # Handle conversation context
            if context:
                # Create chat session with new API
                # New API: client.chats.create(model=..., config=...)
                chat = self._client.chats.create(model=self.model, config=config)
                response = chat.send_message(message)
            else:
                # Single message using generate_content
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=message,
                    config=config
                )

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
            Enhanced prompt string (without persona - that's in system_instruction)
        """
        # Add user's prompt
        enhanced = f"{base_prompt}\n\n"

        # Add detection information if provided
        if detections:
            det_summary = self._format_detections(detections)
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
            Enhanced prompt string (without persona - that's in system_instruction)
        """
        # Add user's prompt
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

    def _build_generation_config(self) -> types.GenerateContentConfig:
        """Build GenerateContentConfig with system instruction (persona) using new google-genai types.

        Returns:
            GenerateContentConfig object with system_instruction if persona exists
        """
        if self.persona and self.persona.strip():
            # Create config with persona as system instruction
            return types.GenerateContentConfig(
                system_instruction=self.persona,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                response_mime_type="text/plain",
                safety_settings=[
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
                ]
            )
        else:
            # Return config without system instruction
            return types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                response_mime_type="text/plain",
                safety_settings=[
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
                ]
            )

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

    def chat_with_images(self, message: str, frame: Optional[np.ndarray] = None,
                        reference: Optional[np.ndarray] = None) -> Optional[str]:
        """Universal chat method that handles any combination of images.

        This is a simplified interface that accepts whatever images are available
        (both, one, or none) and sends them to Gemini for analysis.

        Args:
            message: User message text
            frame: Current frame from webcam (may be None)
            reference: Reference image (may be None)

        Returns:
            AI response text, or None on failure
        """
        if not self._initialized:
            if not self.initialize():
                return None

        try:
            # Build the contents list starting with the message
            contents = [message]

            # Add whatever images are available
            if reference is not None:
                # Convert BGR to RGB and add to contents
                ref_rgb = reference[..., ::-1].copy()
                ref_pil = Image.fromarray(ref_rgb)
                contents.append("Reference Image:")
                contents.append(ref_pil)

            if frame is not None:
                # Convert BGR to RGB and add to contents
                frame_rgb = frame[..., ::-1].copy()
                frame_pil = Image.fromarray(frame_rgb)
                contents.append("Current Frame:")
                contents.append(frame_pil)

            logger.info(f"Sending chat message with {len([c for c in contents if isinstance(c, Image.Image)])} image(s)")

            # Build config with system instruction
            config = self._build_generation_config()

            # Send everything to Gemini
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )

            if response and response.text:
                return response.text
            else:
                logger.warning("Empty response from Gemini")
                return None

        except Exception as e:
            logger.error(f"Error in chat_with_images: {e}")
            return None

    def is_initialized(self) -> bool:
        """Check if service is initialized.

        Returns:
            True if initialized, False otherwise
        """
        return self._initialized

    def update_config(self, model: Optional[str] = None, temperature: Optional[float] = None,
                     max_tokens: Optional[int] = None, persona: Optional[str] = None):
        """Update service configuration.

        Args:
            model: New model name
            temperature: New temperature value
            max_tokens: New max tokens value
            persona: New persona value
        """
        if model is not None:
            self.model = model
            self._initialized = False  # Need to reinitialize

        if temperature is not None:
            self.temperature = max(0.0, min(1.0, temperature))

        if max_tokens is not None:
            self.max_tokens = max(1, max_tokens)

        if persona is not None:
            self.persona = persona
            # No need to reinitialize - persona is now applied in prompts, not model init