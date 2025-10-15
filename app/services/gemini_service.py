"""Gemini AI service for image analysis and chat.

Architecture:
- Images are placed in system context (sent as content parts before user prompt)
- User prompts are purely textual (no image metadata like resolution)
- System context provides persistent reference materials and visual context
- User messages contain only instructions and detection data

This architecture allows the AI to receive images as foundational context
before processing the user's textual query, leading to better contextual understanding.
"""

import logging
from typing import Optional, List, Dict, Any
import numpy as np
from PIL import Image
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiService:
    """Service for interacting with Google Gemini AI API.

    This service implements a context-first architecture where images are
    presented as system context before user prompts, enabling better AI understanding.

    Key features:
    - analyze_image(): Single image analysis or text-only mode
    - compare_images(): Image comparison or text-only comparison mode
    - chat(): Basic text chat
    - chat_with_images(): Flexible chat with optional images

    All methods that accept images can also work in text-only mode by passing None.
    """

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
        self._last_error: Optional[str] = None  # Track last initialization/connection error
        self._connection_status: str = "not_configured"  # not_configured, connecting, ready, error

    def initialize(self) -> bool:
        """Initialize Gemini client using the new google-genai library.

        Returns:
            True if initialized successfully, False otherwise
        """
        try:
            self._connection_status = "connecting"
            self._last_error = None

            if not self.api_key or self.api_key.strip() == "":
                error_msg = "API key is empty"
                logger.error(error_msg)
                self._last_error = error_msg
                self._connection_status = "not_configured"
                return False

            # Create client with API key (new google-genai API)
            self._client = genai.Client(api_key=self.api_key)
            logger.info(f"Gemini service initialized with model: {self.model}")

            self._initialized = True
            self._connection_status = "ready"
            return True

        except ImportError:
            error_msg = "google-genai package not installed. Install with: pip install google-genai"
            logger.error(error_msg)
            self._last_error = error_msg
            self._connection_status = "error"
            return False
        except Exception as e:
            error_msg = f"Error initializing Gemini service: {e}"
            logger.error(error_msg)
            self._last_error = str(e)
            self._connection_status = "error"
            return False

    def analyze_image(self, image: Optional[np.ndarray], prompt: str,
                     class_names: List[str],
                     detections: List[Dict[str, Any]] = None,
                     persona: Optional[str] = None) -> Optional[str]:
        """Analyze image or text-only analysis with Gemini AI.

        Images are placed in system context, user prompts are purely textual.
        This method handles both image analysis and text-only analysis (when image=None).

        Args:
            image: Image as numpy array (BGR format), or None for text-only analysis
            prompt: Analysis instruction (textual)
            class_names: List of all available class names the model can detect (required)
            detections: YOLO detections to include in prompt (None = no detection data, [] = no detections, list = detections found)
            persona: AI persona (optional, uses self.persona if None)

        Returns:
            AI response text, or None on failure
        """
        if not self._initialized:
            if not self.initialize():
                return None

        try:
            # Convert image to PIL if provided
            pil_image = None
            if image is not None:
                # Convert BGR to RGB
                image_rgb = image[..., ::-1].copy()
                # Convert to PIL Image
                pil_image = Image.fromarray(image_rgb)

            # Build config with system context (includes image if provided)
            config = self._build_generation_config(
                persona=persona,
                class_names=class_names,
                current_image=pil_image  # Image goes into system context
            )

            # Build purely textual user prompt
            user_prompt = self._build_prompt(
                base_prompt=prompt,
                detections=detections
            )
            logger.info(f"Sending prompt to Gemini API (analyze_image): {user_prompt}")

            # Assemble content parts: system context + user prompt
            content_parts = config["system_instruction_parts"] + [user_prompt]

            # Generate response using new client API
            response = self._client.models.generate_content(
                model=self.model,
                contents=content_parts,
                config=config["generation_config"]
            )

            # Check response validity with detailed diagnostics
            if response and response.text:
                return response.text
            else:
                # Provide detailed diagnostic information
                diagnostic_msg = self._diagnose_empty_response(response, "analyze_image")
                logger.warning(diagnostic_msg)

                # Check if it's a configuration issue
                if self.max_tokens < 500:
                    logger.error(f"max_tokens is very low ({self.max_tokens}). Recommended: 2048+. Increase in settings.")

                return None

        except Exception as e:
            logger.error(f"Error analyzing image: {e}", exc_info=True)
            return None

    def compare_images(self, reference_image: Optional[np.ndarray], current_image: Optional[np.ndarray],
                      prompt: str, class_names: List[str],
                      ref_detections: List[Dict[str, Any]],
                      ref_width: int, ref_height: int,
                      curr_detections: List[Dict[str, Any]] = None,
                      persona: Optional[str] = None) -> Optional[str]:
        """Compare images or text-only comparison with Gemini AI.

        Both images are placed in system context, user prompts are purely textual.
        This method handles image comparison and text-only comparison (when images=None).

        Args:
            reference_image: Reference image as numpy array, or None for text-only
            current_image: Current image as numpy array, or None for text-only
            prompt: Comparison instruction (textual)
            class_names: List of all available class names the model can detect (required)
            ref_detections: Detections from reference image (required, can be empty list)
            ref_width: Reference image width (required for reference context)
            ref_height: Reference image height (required for reference context)
            curr_detections: Detections from current image (None = no detection data, [] = no detections, list = detections found)
            persona: AI persona (optional, uses self.persona if None)

        Returns:
            AI response text, or None on failure
        """
        if not self._initialized:
            if not self.initialize():
                return None

        try:
            # Convert images to PIL if provided
            ref_pil = None
            curr_pil = None

            if reference_image is not None:
                ref_rgb = reference_image[..., ::-1].copy()
                ref_pil = Image.fromarray(ref_rgb)

            if current_image is not None:
                curr_rgb = current_image[..., ::-1].copy()
                curr_pil = Image.fromarray(curr_rgb)

            # Build config with system context (includes both images and reference detection data)
            config = self._build_generation_config(
                persona=persona,
                class_names=class_names,
                ref_detections=ref_detections,
                ref_width=ref_width,
                ref_height=ref_height,
                ref_image=ref_pil,      # Reference in system context
                current_image=curr_pil   # Current in system context
            )

            # Build purely textual user prompt with current image detection data only
            user_prompt = self._build_comparison_prompt(
                base_prompt=prompt,
                curr_detections=curr_detections
            )
            logger.info(f"Sending prompt to Gemini API (compare_images): {user_prompt}")

            # Assemble content parts: system context + user prompt
            content_parts = config["system_instruction_parts"] + [user_prompt]

            # Generate response using new client API
            response = self._client.models.generate_content(
                model=self.model,
                contents=content_parts,
                config=config["generation_config"]
            )

            # Check response validity with detailed diagnostics
            if response and response.text:
                return response.text
            else:
                # Provide detailed diagnostic information
                diagnostic_msg = self._diagnose_empty_response(response, "compare_images")
                logger.warning(diagnostic_msg)

                # Check if it's a configuration issue
                if self.max_tokens < 500:
                    logger.error(f"max_tokens is very low ({self.max_tokens}). Recommended: 2048+. Increase in settings.")

                return None

        except Exception as e:
            logger.error(f"Error comparing images: {e}", exc_info=True)
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

            # Build config with system context (text only for basic chat)
            config = self._build_generation_config()

            # Handle conversation context
            if context:
                # Create chat session with new API
                # Note: This may need adjustment based on actual API behavior
                # For now, use basic approach with generation_config
                chat = self._client.chats.create(
                    model=self.model,
                    config=config["generation_config"]
                )
                response = chat.send_message(message)
            else:
                # Single message using generate_content
                # For basic chat, system_instruction_parts is just the text
                content_parts = config["system_instruction_parts"] + [message]
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=content_parts,
                    config=config["generation_config"]
                )

            # Check response validity with detailed diagnostics
            if response and response.text:
                return response.text
            else:
                # Provide detailed diagnostic information
                diagnostic_msg = self._diagnose_empty_response(response, "chat")
                logger.warning(diagnostic_msg)

                # Check if it's a configuration issue
                if self.max_tokens < 500:
                    logger.error(f"max_tokens is very low ({self.max_tokens}). Recommended: 2048+. Increase in settings.")

                return None

        except Exception as e:
            logger.error(f"Error in chat: {e}", exc_info=True)
            return None

    def _build_prompt(self, base_prompt: str, detections: List[Dict[str, Any]] = None) -> str:
        """Build purely textual user prompt.

        Images are provided in system context, so this prompt only contains
        textual instructions and detection data.

        Constructs the user message by combining:
        1. Base user instruction (primary directive)
        2. YOLO detection results (if available)

        Args:
            base_prompt: Base user prompt (primary instruction)
            detections: Detection results (None = no detection data, [] = no detections, [items] = detections found)

        Returns:
            Purely textual user prompt string
        """
        # Start with user's instruction
        prompt = f"{base_prompt}"

        # Add detection results if provided
        if detections is not None:
            prompt += "\n\n__CURRENT IMAGE DETECTION DATA__\n"
            if len(detections) > 0:
                det_summary = self._format_detections(detections)
                prompt += f"YOLO Detection Results ({len(detections)} object{'s' if len(detections) != 1 else ''} detected):\n"
                prompt += det_summary
            else:
                prompt += "YOLO Detection Results: No objects detected.\n"

        return prompt

    def _build_comparison_prompt(self, base_prompt: str,
                                curr_detections: List[Dict[str, Any]] = None) -> str:
        """Build purely textual comparison prompt.

        Both reference and current images are in system context.
        This prompt only contains textual instructions and current image detection data.

        Constructs the user message by combining:
        1. Base user instruction (primary directive)
        2. Current image YOLO detection results (if available)

        Args:
            base_prompt: Base user prompt (primary instruction)
            curr_detections: Current image detections (None = no detection data, [] = no detections, [items] = detections found)

        Returns:
            Purely textual comparison prompt string
        """
        # Start with user's instruction
        prompt = f"{base_prompt}"

        # Add current image detection results if provided
        if curr_detections is not None:
            prompt += "\n\n__CURRENT IMAGE DETECTION DATA__\n"
            if len(curr_detections) > 0:
                det_summary = self._format_detections(curr_detections)
                prompt += f"YOLO Detection Results ({len(curr_detections)} object{'s' if len(curr_detections) != 1 else ''} detected):\n"
                prompt += det_summary
            else:
                prompt += "YOLO Detection Results: No objects detected.\n"

        return prompt

    def _build_generation_config(self,
                                persona: Optional[str] = None,
                                class_names: List[str] = None,
                                ref_detections: List[Dict[str, Any]] = None,
                                ref_width: int = None,
                                ref_height: int = None,
                                ref_image = None,
                                current_image = None) -> Dict:
        """Build generation config with images in system context.

        Constructs system context by combining:
        1. System instruction text (persona + reference materials + guidelines)
        2. Reference image (if provided) - as visual context
        3. Current image (if provided) - as visual context

        Images are now part of system context (sent as content parts), not in
        GenerateContentConfig.system_instruction. This allows images to be
        presented upfront as context before the user's textual query.

        Args:
            persona: AI persona/role instructions (optional, uses self.persona if None)
            class_names: List of available class names (YOLO model capabilities)
            ref_detections: Reference image detections (for comparison mode)
            ref_width: Reference image width in pixels (for comparison mode)
            ref_height: Reference image height in pixels (for comparison mode)
            ref_image: Reference image as PIL Image (for comparison mode)
            current_image: Current image as PIL Image (for analysis mode)

        Returns:
            Dict containing:
            - system_instruction_parts: List of [text, ref_image, current_image]
            - generation_config: GenerateContentConfig (without system_instruction)
        """
        # Start with base persona
        persona_text = (persona or self.persona).strip() if (persona or self.persona) else ""
        system_instruction = persona_text

        # Add reference materials section if any reference data is provided
        if class_names or ref_detections is not None or ref_width is not None:
            system_instruction += "\n\n=== REFERENCE MATERIALS ===\n\n"

            # Add available class names (model capabilities)
            if class_names:
                system_instruction += "Available Object Classes (YOLO can detect):\n"
                for class_name in class_names:
                    system_instruction += f"- {class_name}\n"
                system_instruction += "\n"

            # Add reference image detection metadata (not resolution - image speaks for itself)
            if ref_detections is not None:
                system_instruction += "__REFERENCE IMAGE DATA__\n"

                # Add detection results only
                if len(ref_detections) > 0:
                    det_summary = self._format_detections(ref_detections)
                    system_instruction += f"YOLO Detection Results ({len(ref_detections)} object{'s' if len(ref_detections) != 1 else ''} detected):\n"
                    system_instruction += det_summary
                else:
                    system_instruction += "YOLO Detection Results: No objects detected.\n"

                system_instruction += "\nThis is the reference baseline for comparison.\n"

        # Add programmatic instructions for handling queries
        system_instruction += """

=== ANALYSIS INSTRUCTIONS ===
- Images are provided in the context above
- Follow the user's textual instruction as the primary directive
- Analyze the provided images using reference materials as context
- Provide clear, educational feedback
- Detection data (if provided) supplements visual analysis
"""

        # Build system context parts (text + images)
        system_context = [system_instruction]

        # Add reference image to system context if provided
        if ref_image is not None:
            system_context.append(ref_image)

        # Add current image to system context if provided
        if current_image is not None:
            system_context.append(current_image)

        # Return dict with system context and generation config
        return {
            "system_instruction_parts": system_context,
            "generation_config": types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                response_mime_type="text/plain",
                thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disables thinking
            )
        }

    def _format_detections(self, detections: List[Dict[str, Any]]) -> str:
        """Format detections as comprehensive readable text with full details.

        Args:
            detections: List of detection dictionaries with keys:
                - class_name: Object class name
                - confidence: Detection confidence (0-1)
                - bbox: Bounding box [x1, y1, x2, y2] in pixels
                - class_id: Class ID number

        Returns:
            Formatted string with detailed detection information
        """
        if not detections:
            return "No objects detected."

        lines = []

        for idx, det in enumerate(detections, 1):
            class_name = det.get('class_name', 'Unknown')
            confidence = det.get('confidence', 0.0)
            bbox = det.get('bbox', [0, 0, 0, 0])

            # Extract bbox coordinates
            x1, y1, x2, y2 = bbox

            # Calculate center and size
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            # Format detection entry
            lines.append(f"{idx}. Object: {class_name}")
            lines.append(f"   - Bounding Box: ({x1}, {y1}, {x2}, {y2}) pixels")
            lines.append(f"   - Confidence: {confidence:.2f}")
            lines.append(f"   - Position: Center at ({center_x:.0f}, {center_y:.0f})")
            lines.append(f"   - Size: {width:.0f}x{height:.0f} pixels")
            lines.append("")  # Blank line between detections

        return "\n".join(lines)

    def _diagnose_empty_response(self, response, method_name: str) -> str:
        """Diagnose why a Gemini response is empty and return detailed diagnostic message.

        Args:
            response: The GenerateContentResponse object
            method_name: Name of the calling method for context

        Returns:
            Detailed diagnostic message explaining why response is empty
        """
        diagnostics = []

        # Check if response object exists
        if not response:
            return f"[{method_name}] Response object is None or False"

        # Check prompt feedback for blocking
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            diagnostics.append(f"Prompt feedback present: {response.prompt_feedback}")

            # Check for blocked prompt
            if hasattr(response.prompt_feedback, 'block_reason'):
                block_reason = response.prompt_feedback.block_reason
                if block_reason:
                    diagnostics.append(f"PROMPT BLOCKED - Reason: {block_reason}")

            # Check safety ratings on prompt
            if hasattr(response.prompt_feedback, 'safety_ratings'):
                safety_ratings = response.prompt_feedback.safety_ratings
                if safety_ratings:
                    diagnostics.append(f"Prompt safety ratings: {safety_ratings}")

        # Check candidates
        if not hasattr(response, 'candidates') or not response.candidates:
            diagnostics.append("No candidates in response (candidates is None or empty list)")
            return f"[{method_name}] Empty response - " + "; ".join(diagnostics)

        # Analyze first candidate
        candidate = response.candidates[0]
        diagnostics.append(f"Number of candidates: {len(response.candidates)}")

        # Check finish reason
        if hasattr(candidate, 'finish_reason'):
            finish_reason = candidate.finish_reason
            diagnostics.append(f"Finish reason: {finish_reason}")

            # Map finish reasons to explanations
            finish_reason_explanations = {
                'SAFETY': 'Response blocked by safety filters',
                'MAX_TOKENS': 'Response truncated due to token limit (increase max_tokens)',
                'RECITATION': 'Response blocked due to recitation concerns',
                'OTHER': 'Response stopped for other reasons',
                'STOP': 'Normal completion',
            }

            finish_reason_str = str(finish_reason).split('.')[-1] if hasattr(finish_reason, 'name') else str(finish_reason)
            if finish_reason_str in finish_reason_explanations:
                diagnostics.append(f"Explanation: {finish_reason_explanations[finish_reason_str]}")

        # Check safety ratings on candidate
        if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
            diagnostics.append(f"Candidate safety ratings: {candidate.safety_ratings}")

        # Check finish message
        if hasattr(candidate, 'finish_message') and candidate.finish_message:
            diagnostics.append(f"Finish message: {candidate.finish_message}")

        # Check content structure
        if not hasattr(candidate, 'content') or not candidate.content:
            diagnostics.append("Candidate has no content")
        elif not hasattr(candidate.content, 'parts') or not candidate.content.parts:
            diagnostics.append("Candidate content has no parts")
        else:
            # Check what types of parts exist
            parts_info = []
            for part in candidate.content.parts:
                part_dump = part.model_dump(exclude_none=True)
                non_none_fields = [k for k, v in part_dump.items() if v is not None]
                parts_info.append(f"Part fields: {non_none_fields}")

            diagnostics.append(f"Content parts ({len(candidate.content.parts)}): {'; '.join(parts_info)}")

        # Check response.text property
        try:
            text_value = response.text
            if text_value is None:
                diagnostics.append("response.text returned None (no text parts found)")
            elif text_value == '':
                diagnostics.append("response.text returned empty string")
            else:
                diagnostics.append(f"response.text has value but condition failed (length: {len(text_value)})")
        except Exception as e:
            diagnostics.append(f"Error accessing response.text: {e}")

        return f"[{method_name}] Empty response - " + "; ".join(diagnostics)

    def chat_with_images(self, message: str, frame: Optional[np.ndarray] = None,
                        reference: Optional[np.ndarray] = None) -> Optional[str]:
        """Universal chat method with images in system context.

        Images are placed in system context, user message is purely textual.
        This method handles any combination of images (both, one, or none).

        Args:
            message: User message text (textual instruction)
            frame: Current frame from webcam (may be None)
            reference: Reference image (may be None)

        Returns:
            AI response text, or None on failure
        """
        if not self._initialized:
            if not self.initialize():
                return None

        try:
            # Convert images to PIL if provided
            ref_pil = None
            frame_pil = None

            if reference is not None:
                ref_rgb = reference[..., ::-1].copy()
                ref_pil = Image.fromarray(ref_rgb)

            if frame is not None:
                frame_rgb = frame[..., ::-1].copy()
                frame_pil = Image.fromarray(frame_rgb)

            image_count = sum(1 for img in [ref_pil, frame_pil] if img is not None)
            logger.info(f"Sending chat message with {image_count} image(s)")

            # Build config with system context (includes images if provided)
            config = self._build_generation_config(
                ref_image=ref_pil,
                current_image=frame_pil
            )

            # Assemble content parts: system context + user message
            content_parts = config["system_instruction_parts"] + [message]

            # Send to Gemini
            response = self._client.models.generate_content(
                model=self.model,
                contents=content_parts,
                config=config["generation_config"]
            )

            # Check response validity with detailed diagnostics
            if response and response.text:
                return response.text
            else:
                # Provide detailed diagnostic information
                diagnostic_msg = self._diagnose_empty_response(response, "chat_with_images")
                logger.warning(diagnostic_msg)

                # Check if it's a configuration issue
                if self.max_tokens < 500:
                    logger.error(f"max_tokens is very low ({self.max_tokens}). Recommended: 2048+. Increase in settings.")

                return None

        except Exception as e:
            logger.error(f"Error in chat_with_images: {e}", exc_info=True)
            return None

    def is_initialized(self) -> bool:
        """Check if service is initialized.

        Returns:
            True if initialized, False otherwise
        """
        return self._initialized

    def update_config(self, model: Optional[str] = None, temperature: Optional[float] = None,
                     max_tokens: Optional[int] = None, persona: Optional[str] = None,
                     api_key: Optional[str] = None):
        """Update service configuration.

        Args:
            model: New model name
            temperature: New temperature value
            max_tokens: New max tokens value
            persona: New persona value
            api_key: New API key (requires reinitialization)
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

        if api_key is not None:
            self.api_key = api_key
            self._initialized = False  # Need to reinitialize with new API key
            if not api_key or api_key.strip() == "":
                self._connection_status = "not_configured"
            else:
                # API key updated, status will be updated when initialize() is called
                self._connection_status = "not_configured"

    def reinitialize(self) -> bool:
        """Reinitialize the Gemini service (useful after config changes).

        Returns:
            True if reinitialized successfully, False otherwise
        """
        logger.info("Reinitializing Gemini service...")
        self._initialized = False
        self._client = None
        return self.initialize()

    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status and details.

        Returns:
            Dictionary with status information including:
            - status: not_configured, connecting, ready, or error
            - initialized: bool indicating if service is initialized
            - last_error: last error message if any
            - has_api_key: bool indicating if API key is configured
        """
        return {
            'status': self._connection_status,
            'initialized': self._initialized,
            'last_error': self._last_error,
            'has_api_key': bool(self.api_key and self.api_key.strip())
        }

    def test_connection(self) -> tuple[bool, Optional[str]]:
        """Test connection to Gemini API with a simple request.

        Returns:
            Tuple of (success, error_message)
        """
        try:
            if not self._initialized:
                if not self.initialize():
                    return False, self._last_error or "Failed to initialize"

            # Send a minimal test request
            config = types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=10,
                response_mime_type="text/plain"
            )

            response = self._client.models.generate_content(
                model=self.model,
                contents="Say 'OK'",
                config=config
            )

            if response and response.text:
                logger.info("Gemini API connection test successful")
                self._connection_status = "ready"
                self._last_error = None
                return True, None
            else:
                error_msg = "Received empty response from API"
                logger.warning(error_msg)
                self._last_error = error_msg
                self._connection_status = "error"
                return False, error_msg

        except Exception as e:
            error_msg = f"Connection test failed: {str(e)}"
            logger.error(error_msg)
            self._last_error = str(e)
            self._connection_status = "error"
            return False, str(e)