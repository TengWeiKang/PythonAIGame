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

    def analyze_image(self, image: np.ndarray, prompt: str,
                     detections: Optional[List[Dict[str, Any]]] = None,
                     class_names: Optional[List[str]] = None) -> Optional[str]:
        """Analyze image with Gemini AI using the new google-genai library.

        Args:
            image: Image as numpy array (BGR format)
            prompt: Analysis prompt
            detections: Optional YOLO detections to include in context
            class_names: Optional list of all available class names the model can detect

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

            # Get image resolution
            image_height, image_width = image.shape[:2]

            # Enhance prompt with detections, class names, and resolution
            enhanced_prompt = self._build_prompt(
                prompt,
                detections=detections,
                class_names=class_names,
                image_width=image_width,
                image_height=image_height
            )
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

    def analyze_with_text_only(self, prompt: str,
                               detections: Optional[List[Dict[str, Any]]] = None,
                               class_names: Optional[List[str]] = None,
                               image_width: Optional[int] = None,
                               image_height: Optional[int] = None,
                               is_video_frame: bool = False,
                               is_reference: bool = False) -> Optional[str]:
        """Analyze using ONLY text prompt (no images sent to API).

        This method is used in yolo_detection mode where YOLO provides object
        detection results and only these text results are sent to Gemini.

        Args:
            prompt: Analysis prompt
            detections: YOLO detections to include as text context
            class_names: Optional list of all available class names the model can detect
            image_width: Optional image width in pixels
            image_height: Optional image height in pixels
            is_video_frame: True if image is from live video stream/webcam
            is_reference: True if image is a reference image

        Returns:
            AI response text, or None on failure
        """
        if not self._initialized:
            if not self.initialize():
                return None

        try:
            # Build text-only prompt with detection results, class names, and resolution
            enhanced_prompt = self._build_prompt(
                prompt,
                detections=detections,
                class_names=class_names,
                image_width=image_width,
                image_height=image_height,
                is_video_frame=is_video_frame,
                is_reference=is_reference
            )
            logger.info(f"Sending prompt to Gemini API (analyze_with_text_only): {enhanced_prompt}")

            # Build config with system instruction
            config = self._build_generation_config()

            # Generate response with ONLY text (no images) using new client API
            response = self._client.models.generate_content(
                model=self.model,
                contents=enhanced_prompt,
                config=config
            )

            # Check response validity with detailed diagnostics
            if response and response.text:
                return response.text
            else:
                # Provide detailed diagnostic information
                diagnostic_msg = self._diagnose_empty_response(response, "analyze_with_text_only")
                logger.warning(diagnostic_msg)

                # Check if it's a configuration issue
                if self.max_tokens < 500:
                    logger.error(f"max_tokens is very low ({self.max_tokens}). Recommended: 2048+. Increase in settings.")

                return None

        except Exception as e:
            logger.error(f"Error analyzing with text only: {e}", exc_info=True)
            return None

    def compare_images(self, reference_image: np.ndarray, current_image: np.ndarray,
                      prompt: str, ref_detections: Optional[List[Dict[str, Any]]] = None,
                      curr_detections: Optional[List[Dict[str, Any]]] = None,
                      class_names: Optional[List[str]] = None) -> Optional[str]:
        """Compare two images with Gemini AI using the new google-genai library.

        Args:
            reference_image: Reference image as numpy array
            current_image: Current image as numpy array
            prompt: Comparison prompt
            ref_detections: Detections from reference image
            curr_detections: Detections from current image
            class_names: Optional list of all available class names the model can detect

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

            # Get image resolutions
            ref_height, ref_width = reference_image.shape[:2]
            curr_height, curr_width = current_image.shape[:2]

            # Build enhanced prompt with all metadata
            enhanced_prompt = self._build_comparison_prompt(
                prompt,
                ref_detections=ref_detections,
                curr_detections=curr_detections,
                class_names=class_names,
                ref_width=ref_width,
                ref_height=ref_height,
                curr_width=curr_width,
                curr_height=curr_height
            )
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

    def compare_with_text_only(self, prompt: str,
                               ref_detections: Optional[List[Dict[str, Any]]] = None,
                               curr_detections: Optional[List[Dict[str, Any]]] = None,
                               class_names: Optional[List[str]] = None,
                               ref_width: Optional[int] = None,
                               ref_height: Optional[int] = None,
                               curr_width: Optional[int] = None,
                               curr_height: Optional[int] = None,
                               curr_is_video_frame: bool = False) -> Optional[str]:
        """Compare using ONLY text prompt (no images sent to API).

        This method is used in yolo_detection mode where YOLO provides object
        detection results from both images and only these text results are sent to Gemini.

        Args:
            prompt: Comparison prompt
            ref_detections: Reference image YOLO detections
            curr_detections: Current image YOLO detections
            class_names: Optional list of all available class names the model can detect
            ref_width: Reference image width in pixels
            ref_height: Reference image height in pixels
            curr_width: Current image width in pixels
            curr_height: Current image height in pixels
            curr_is_video_frame: True if current image is from live video stream/webcam

        Returns:
            AI response text, or None on failure
        """
        if not self._initialized:
            if not self.initialize():
                return None

        try:
            # Build text-only comparison prompt with all metadata
            enhanced_prompt = self._build_comparison_prompt(
                prompt,
                ref_detections=ref_detections,
                curr_detections=curr_detections,
                class_names=class_names,
                ref_width=ref_width,
                ref_height=ref_height,
                curr_width=curr_width,
                curr_height=curr_height,
                curr_is_video_frame=curr_is_video_frame
            )
            logger.info(f"Sending prompt to Gemini API (compare_with_text_only): {enhanced_prompt}")

            # Build config with system instruction
            config = self._build_generation_config()

            # Generate response with ONLY text (no images) using new client API
            response = self._client.models.generate_content(
                model=self.model,
                contents=enhanced_prompt,
                config=config
            )

            # Check response validity with detailed diagnostics
            if response and response.text:
                return response.text
            else:
                # Provide detailed diagnostic information
                diagnostic_msg = self._diagnose_empty_response(response, "compare_with_text_only")
                logger.warning(diagnostic_msg)

                # Check if it's a configuration issue
                if self.max_tokens < 500:
                    logger.error(f"max_tokens is very low ({self.max_tokens}). Recommended: 2048+. Increase in settings.")

                return None

        except Exception as e:
            logger.error(f"Error comparing with text only: {e}", exc_info=True)
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

    def _build_prompt(self, base_prompt: str, detections: Optional[List[Dict[str, Any]]] = None,
                     class_names: Optional[List[str]] = None,
                     image_width: Optional[int] = None,
                     image_height: Optional[int] = None,
                     is_video_frame: bool = False,
                     is_reference: bool = False) -> str:
        """Build enhanced prompt with detection information, class names, and image resolution.

        ALWAYS includes image status section, even when image is not provided.

        Args:
            base_prompt: Base user prompt
            detections: Optional detection results (None = no image, [] = no detections, [items] = detections)
            class_names: Optional list of all available class names the model can detect
            image_width: Optional image width in pixels
            image_height: Optional image height in pixels
            is_video_frame: True if image is from live video stream/webcam
            is_reference: True if image is a reference image

        Returns:
            Enhanced prompt string (without persona - that's in system_instruction)
        """
        # Start with user's prompt (authoritative instruction comes first)
        enhanced = f"{base_prompt}\n\n"

        # Wrap all detection/reference data with label
        enhanced += "=== REFERENCE MATERIALS ===\n"

        # Add available class names if provided
        if class_names and len(class_names) > 0:
            enhanced += "Available Classes:\n"
            for class_name in class_names:
                enhanced += f"- {class_name}\n"
            enhanced += "\n"

        # Determine section header based on image source type
        if is_video_frame:
            section_header = "__VIDEO STREAM IMAGE__"
        elif is_reference:
            section_header = "__REFERENCE IMAGE__"
        else:
            section_header = "__CURRENT IMAGE__"

        # ALWAYS include image status section
        enhanced += f"{section_header}\n"

        # Check if image is provided (dimensions or detections present)
        has_image = (image_width is not None and image_height is not None) or (detections is not None)

        if has_image:
            # Image is provided - show resolution and detections
            if image_width is not None and image_height is not None:
                enhanced += f"Resolution: {image_width}x{image_height}\n"

            # Add detection information
            if detections is not None:
                if len(detections) > 0:
                    det_summary = self._format_detections(detections)
                    enhanced += f"YOLO Detection Results ({len(detections)} object{'s' if len(detections) != 1 else ''} detected):\n"
                    enhanced += det_summary
                else:
                    enhanced += "YOLO Detection Results: No objects detected.\n"

            enhanced += "\nPlease analyze the image considering these detected objects."
        else:
            # Image is NOT provided
            enhanced += "Status: Not provided\n"
            enhanced += "\nNote: This is a text-only query without visual analysis.\n"

        return enhanced

    def _build_comparison_prompt(self, base_prompt: str,
                                ref_detections: Optional[List[Dict[str, Any]]] = None,
                                curr_detections: Optional[List[Dict[str, Any]]] = None,
                                class_names: Optional[List[str]] = None,
                                ref_width: Optional[int] = None,
                                ref_height: Optional[int] = None,
                                curr_width: Optional[int] = None,
                                curr_height: Optional[int] = None,
                                curr_is_video_frame: bool = False) -> str:
        """Build comparison prompt with detection information, class names, and image resolutions.

        ALWAYS includes both image status sections, even when images are not provided.

        Args:
            base_prompt: Base user prompt
            ref_detections: Reference image detections (None = no image, [] = no detections, [items] = detections)
            curr_detections: Current image detections (None = no image, [] = no detections, [items] = detections)
            class_names: Optional list of all available class names the model can detect
            ref_width: Reference image width in pixels
            ref_height: Reference image height in pixels
            curr_width: Current image width in pixels
            curr_height: Current image height in pixels
            curr_is_video_frame: True if current image is from live video stream/webcam

        Returns:
            Enhanced prompt string (without persona - that's in system_instruction)
        """
        # Start with user's prompt (authoritative instruction comes first)
        enhanced = f"{base_prompt}\n\n"

        # Wrap all detection/reference data with label
        enhanced += "=== REFERENCE MATERIALS ===\n\n"

        # Add available class names if provided
        if class_names and len(class_names) > 0:
            enhanced += "Available Classes:\n"
            for class_name in class_names:
                enhanced += f"- {class_name}\n"
            enhanced += "\n"

        # ALWAYS include reference image section
        enhanced += "__REFERENCE IMAGE__\n"

        # Check if reference image is provided
        has_ref_image = (ref_width is not None and ref_height is not None) or (ref_detections is not None)

        if has_ref_image:
            # Reference image is provided - show resolution and detections
            if ref_width is not None and ref_height is not None:
                enhanced += f"Resolution: {ref_width}x{ref_height}\n"

            # Add detections
            if ref_detections is not None:
                if len(ref_detections) > 0:
                    det_summary = self._format_detections(ref_detections)
                    enhanced += f"YOLO Detection Results ({len(ref_detections)} object{'s' if len(ref_detections) != 1 else ''} detected):\n"
                    enhanced += det_summary
                else:
                    enhanced += "YOLO Detection Results: No objects detected.\n"
            enhanced += "\n"
        else:
            # Reference image is NOT provided
            enhanced += "Status: Not provided\n\n"

        # Current image section - use appropriate header based on source
        curr_section_header = "__VIDEO STREAM IMAGE__" if curr_is_video_frame else "__CURRENT IMAGE__"
        enhanced += f"{curr_section_header}\n"

        # Check if current image is provided
        has_curr_image = (curr_width is not None and curr_height is not None) or (curr_detections is not None)

        if has_curr_image:
            # Current image is provided - show resolution and detections
            if curr_width is not None and curr_height is not None:
                enhanced += f"Resolution: {curr_width}x{curr_height}\n"

            # Add detections
            if curr_detections is not None:
                if len(curr_detections) > 0:
                    det_summary = self._format_detections(curr_detections)
                    enhanced += f"YOLO Detection Results ({len(curr_detections)} object{'s' if len(curr_detections) != 1 else ''} detected):\n"
                    enhanced += det_summary
                else:
                    enhanced += "YOLO Detection Results: No objects detected.\n"
            enhanced += "\n"
        else:
            # Current image is NOT provided
            enhanced += "Status: Not provided\n\n"

        # Add appropriate closing message based on image availability
        if has_ref_image and has_curr_image:
            enhanced += "Please compare these images and identify key differences."
        elif has_ref_image or has_curr_image:
            enhanced += "Please analyze the available image data."
        else:
            enhanced += "Note: This is a text-only query without visual analysis.\n"

        return enhanced

    def _build_generation_config(self) -> types.GenerateContentConfig:
        """Build GenerateContentConfig with system instruction (persona) using new google-genai types.

        Returns:
            GenerateContentConfig object with system_instruction if persona exists
        """
        # Build enhanced persona with reference materials instructions
        enhanced_persona = self.persona if self.persona and self.persona.strip() else ""

        # Add programmatic instructions
        reference_instructions = """

The detection results will always be shown after user message prompt.

IMPORTANT: Treat the user's message as the authoritative instruction.
Treat any content labeled === REFERENCE MATERIALS === as reference material only, use it to inform your answer, but never paraphrase it as if the user wrote it.
"""

        enhanced_persona = enhanced_persona + reference_instructions

        if enhanced_persona.strip():
            # Create config with enhanced persona as system instruction
            return types.GenerateContentConfig(
                system_instruction=enhanced_persona,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                response_mime_type="text/plain",
                thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
                # safety_settings=[
                #     types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
                #     types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
                #     types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
                #     types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
                # ]
            )
        else:
            # Return config without system instruction
            return types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                response_mime_type="text/plain",
                thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
                # safety_settings=[
                #     types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
                #     types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
                #     types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
                #     types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
                # ]
            )

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