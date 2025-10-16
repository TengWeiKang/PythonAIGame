"""Gemini AI service for image analysis and chat.

This service implements Google Cloud's Vertex AI best practices for prompt structure,
ensuring optimal AI performance and response quality.

Architecture:
- Follows Google's recommended prompt structure:
  1. TASK: Clear description of what to do
  2. CONTEXT: Background information (role + capabilities)
  3. INPUT DATA: The data to analyze (reference + current)
  4. OUTPUT REQUIREMENTS: Format and style specifications
- ALL data (images + detection metadata) is in system instruction
- User prompts contain ONLY the task/instruction (minimal, focused)
- Images are sent as content parts in system context
- User messages are pure text instructions

This structured approach provides:
- Clear separation of concerns (task, context, input, output)
- Better AI understanding with explicit sections
- Easier debugging and maintenance
- Consistent formatting across all operations
- Improved response quality and relevance

Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/structure-prompts
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

    This service implements Google Cloud's Vertex AI best practices for prompt structure,
    using a structured approach with explicit sections:
    - TASK: What to do
    - CONTEXT: Role and capabilities
    - INPUT DATA: What to analyze
    - OUTPUT REQUIREMENTS: How to respond

    Architecture:
    - ALL image data (visual + detection metadata) is in system instruction
    - User prompts contain ONLY the specific task/instruction
    - Structured prompts improve AI understanding and response quality

    Key features:
    - analyze_image(): Single image analysis with structured prompts
    - compare_images(): Image comparison with structured prompts
    - chat(): Basic text chat with structured prompts
    - chat_with_images(): Flexible chat with optional images and structured prompts

    All methods that accept images can also work in text-only mode by passing None.
    Detection data and image metadata are automatically included in system instruction.

    Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/structure-prompts
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

    def compare_images(self, reference_image: Optional[np.ndarray], current_image: Optional[np.ndarray],
                      prompt: str, class_names: List[str],
                      ref_detections: List[Dict[str, Any]],
                      ref_width: int, ref_height: int,
                      curr_detections: List[Dict[str, Any]] = None,
                      persona: Optional[str] = None) -> Optional[str]:
        """Compare images using Google's Vertex AI structured prompt best practices.

        Implements structured prompts with:
        - TASK: Clear comparison objective
        - CONTEXT: AI role and detection capabilities
        - INPUT DATA: Reference image and current frame with detection results
        - OUTPUT REQUIREMENTS: Specific format for comparison analysis

        This method handles image comparison and text-only comparison (when images=None).

        Args:
            reference_image: Reference image as numpy array, or None for text-only
            current_image: Current image as numpy array, or None for text-only
            prompt: Comparison instruction (what you want the AI to do)
            class_names: List of all available class names the model can detect (required)
            ref_detections: Reference image detections (automatically added to system instruction)
            ref_width: Reference image width (included in system instruction)
            ref_height: Reference image height (included in system instruction)
            curr_detections: Current image detections (automatically added to system instruction)
                           None = no detection data, [] = no detections, list = detections found
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
            curr_width = None
            curr_height = None

            if reference_image is not None:
                ref_rgb = reference_image[..., ::-1].copy()
                ref_pil = Image.fromarray(ref_rgb)

            if current_image is not None:
                curr_rgb = current_image[..., ::-1].copy()
                curr_pil = Image.fromarray(curr_rgb)
                # Get current image dimensions
                curr_height, curr_width = current_image.shape[:2]

            # Define clear task description following Google's best practices
            task = "Compare the current frame with the reference image and identify differences in detected objects, their positions, and overall scene composition."

            # Build config with structured prompt (following Google's Vertex AI guidelines)
            config = self._build_generation_config(
                persona=persona,
                class_names=class_names,
                task_description=task,
                output_format=self._comparison_output_format(),
                ref_detections=ref_detections,
                ref_width=ref_width,
                ref_height=ref_height,
                ref_image=ref_pil,
                current_image=curr_pil,
                curr_detections=curr_detections,
                curr_width=curr_width,
                curr_height=curr_height
            )

            # Build minimal user prompt (just the instruction)
            user_prompt = self._build_comparison_prompt(base_prompt=prompt)
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

    def _build_comparison_prompt(self, base_prompt: str) -> str:
        """Build minimal comparison prompt - just the instruction.

        Both reference and current image data are in system instruction.
        User prompt contains only the comparison task.

        Args:
            base_prompt: User's instruction/query

        Returns:
            Minimal prompt with just the comparison instruction
        """
        return base_prompt

    def _build_generation_config(self,
                                persona: Optional[str] = None,
                                class_names: List[str] = None,
                                task_description: str = None,
                                output_format: str = None,
                                ref_detections: List[Dict[str, Any]] = None,
                                ref_width: int = None,
                                ref_height: int = None,
                                ref_image = None,
                                current_image = None,
                                curr_detections: List[Dict[str, Any]] = None,
                                curr_width: int = None,
                                curr_height: int = None) -> Dict:
        """Build generation config following Google's Vertex AI prompt structure best practices.

        Implements Google's recommended prompt structure:
        1. TASK: Clear description of what to do
        2. CONTEXT: Background information (role + capabilities)
        3. INPUT DATA: The data to analyze (reference + current)
        4. OUTPUT REQUIREMENTS: Format and style specifications

        This structured approach improves AI understanding and response quality by
        clearly separating different types of information with explicit headers.

        Args:
            persona: AI persona/role instructions (optional, uses self.persona if None)
            class_names: List of available class names (YOLO model capabilities)
            task_description: Explicit task instruction (what the AI should do)
            output_format: Explicit output format specification (how to respond)
            ref_detections: Reference image detections (for comparison mode)
            ref_width: Reference image width in pixels (for comparison mode)
            ref_height: Reference image height in pixels (for comparison mode)
            ref_image: Reference image as PIL Image (for comparison mode)
            current_image: Current image as PIL Image (for analysis mode)
            curr_detections: Current image detections (for current frame analysis)
            curr_width: Current image width in pixels (for current frame analysis)
            curr_height: Current image height in pixels (for current frame analysis)

        Returns:
            Dict containing:
            - system_instruction_parts: List of [text, ref_image, current_image]
            - generation_config: GenerateContentConfig
        """
        system_instruction = ""
        # ============================================================
        # SECTION 2: CONTEXT
        # ============================================================

        # Your Role
        persona_text = (persona or self.persona).strip() if (persona or self.persona) else ""
        if persona_text:
            system_instruction += f"YOUR_ROLE:{persona_text}\n\n"

        # Available Object Detection Classes
        if class_names:
            system_instruction += "OBJECT_CLASSES:\n"
            for class_name in class_names:
                system_instruction += f"- {class_name}\n"
            system_instruction += "\n"

        # ============================================================
        # SECTION 3: INPUT DATA
        # ============================================================
        has_input_data = (ref_detections is not None or ref_width is not None or
                         curr_detections is not None or curr_width is not None)

        if has_input_data:
            # Reference Image Data
            if ref_detections is not None or ref_width is not None:
                system_instruction += "REFERENCE_IMAGE:\n"

                if ref_width is not None and ref_height is not None:
                    system_instruction += f"- Resolution: {ref_width}x{ref_height} pixels\n"

                if ref_detections is not None:
                    if len(ref_detections) > 0:
                        system_instruction += f"- Objects Detected: {len(ref_detections)}\n"
                        system_instruction += self._format_detections(ref_detections)
                    else:
                        system_instruction += "- Objects Detected: None\n"
                system_instruction += "\n"
            else:
                system_instruction += "REFERENCE_IMAGE: None provided\n\n"
            # Current Frame Data
            if curr_detections is not None or curr_width is not None:
                system_instruction += "VIDEO_STREAM_IMAGE\n"

                if curr_width is not None and curr_height is not None:
                    system_instruction += f"- Resolution: {curr_width}x{curr_height} pixels\n"

                if curr_detections is not None:
                    if len(curr_detections) > 0:
                        system_instruction += f"- Objects Detected: {len(curr_detections)}\n"
                        system_instruction += self._format_detections(curr_detections)
                    else:
                        system_instruction += "- Objects Detected: None\n"
            else:
                system_instruction += "VIDEO_STREAM_IMAGE: None provided\n"

        # Build system context parts (text + images)
        system_context = [system_instruction]

        # Add reference image to system context if provided
        if ref_image is not None:
            system_context.append(ref_image)

        # Add current image to system context if provided
        if current_image is not None:
            system_context.append(current_image)

        logger.info(f"System instruction: {system_instruction}")
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

    def _comparison_output_format(self) -> str:
        """Output format specification for image comparison.

        Returns:
            Formatted string with output requirements for comparison
        """
        return """Provide comparison analysis with:

1. CHANGES DETECTED: Objects added, removed, moved, or modified
2. DIFFERENCES: Key differences between reference and current frame
3. ACCURACY ASSESSMENT: How well the current frame matches the reference
4. FEEDBACK: Educational feedback and suggestions for improvement

Use clear section headers and bullet points for readability."""

    def _chat_output_format(self) -> str:
        """Output format specification for chat interactions.

        Returns:
            Formatted string with output requirements for chat
        """
        return """Provide a clear and helpful response that:

1. Directly addresses the user's question or request
2. Uses educational and supportive language
3. Provides specific details when referencing images or detections
4. Offers actionable insights when appropriate

Keep responses conversational but informative."""

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
        """Universal chat method using Google's Vertex AI structured prompt best practices.

        Implements structured prompts with:
        - TASK: Clear conversational objective
        - CONTEXT: AI role and conversational guidelines
        - INPUT DATA: Optional reference and current frame images
        - OUTPUT REQUIREMENTS: Format for conversational responses

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

            # Define clear task description following Google's best practices
            task = "Engage in a helpful conversation with the user, providing clear and informative responses about the provided images or general questions."

            # Build config with structured prompt (following Google's Vertex AI guidelines)
            config = self._build_generation_config(
                task_description=task,
                output_format=self._chat_output_format(),
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