from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import google.auth
from google import genai
from google.cloud import storage
from google.genai import types
from google.oauth2 import service_account

from fathom.constants import ActionType
from fathom.exceptions import VisionError
from fathom.prompts.analysis import ANALYSIS_PROMPT, ANALYSIS_PROMPT_XML
from fathom.schemas.actions import Action, BoundingBox
from fathom.schemas.configuration import GeminiConfig
from fathom.schemas.results import AnalysisResult
from fathom.tools.vision.base import VisionTool

logger = getLogger(__name__)


class GeminiVisionTool(VisionTool):
    """
    Vision tool using Google's Gemini API via the google-genai SDK.
    Analyzes screenshots and recommends actions to achieve intended goals.
    """

    def __init__(self, config: GeminiConfig) -> None:
        """
        Initialize Gemini vision tool.

        Args:
            config: Gemini API configuration.

        Raises:
            ImportError: If google-genai or google-cloud-storage is not installed.
        """

        self.__config = config
        self.__client: Optional[Any] = None
        self.__credentials: Optional[Any] = None

    def __get_client(self) -> Any:
        """
        Initialize and return the Gemini client.

        Returns:
            The initialized Gemini client.

        Raises:
            VisionError: If client initialization fails or credentials are missing.
        """

        if self.__client:
            return self.__client

        project = self.__config.project_id
        location = self.__config.location or "global"

        # Resolve Project ID and Credentials
        if self.__config.credentials_path:
            try:
                path = Path(self.__config.credentials_path)
                if path.exists():
                    self.__credentials = service_account.Credentials.from_service_account_file(
                        str(path),
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                    # Extract project ID from credentials if not set
                    if not project:
                        project = getattr(self.__credentials, "project_id", None)
            except Exception as exception:
                logger.debug(f"Failed to load credentials from file: {exception}")

        # Fallback to environment variables for project
        if not project:
            project = os.environ.get("GEMINI_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")

        # Fallback to default google auth
        if not self.__credentials and not project:
            try:
                credentials, default_project = google.auth.default()
                self.__credentials = credentials
                project = default_project
            except Exception as exception:
                logger.debug(f"Failed to get default project from google.auth: {exception}")

        if not project and not self.__config.api_key:
            raise VisionError("Project ID required for Vertex AI (or set GEMINI_API_KEY)")

        try:
            if self.__config.api_key:
                logger.info("Initializing Gemini Client with API Key")
                self.__client = genai.Client(api_key=self.__config.api_key)
            else:
                logger.info(f"Initializing Gemini Client: project={project}, location={location}")
                self.__client = genai.Client(
                    vertexai=True,
                    project=project,
                    location=location,
                    credentials=self.__credentials,
                )
            return self.__client
        except Exception as exception:
            raise VisionError(f"Failed to initialize Gemini client: {exception}") from exception

    async def __upload_image_to_gcs(self, image_data: bytes) -> str:
        """
        Upload image bytes to GCS and return GCS URI.

        This runs in a thread pool to avoid blocking the async event loop.

        Args:
            image_data: Raw image bytes (PNG).

        Returns:
            The gs:// URI of the uploaded image.

        Raises:
            VisionError: If upload fails.
        """

        credentials = self.__credentials
        project_id = self.__config.project_id
        bucket_name = self.__config.gcs_bucket

        def __upload_sync() -> str:
            try:
                # Instantiate client inside the thread to ensure thread safety
                storage_client = storage.Client(project=project_id, credentials=credentials)
                bucket = storage_client.bucket(bucket_name)

                timestamp = int(time.time() * 1000)
                filename = f"{timestamp}.png"

                blob = bucket.blob(filename)
                blob.upload_from_string(image_data, content_type="image/png")

                gcs_uri = f"gs://{bucket_name}/{filename}"
                logger.debug(f"Uploaded image to GCS: {gcs_uri}")
                return gcs_uri
            except Exception as exception:
                logger.warning(f"Failed to upload to GCS: {exception}")
                raise VisionError(f"GCS upload failed: {exception}") from exception

        return await asyncio.to_thread(__upload_sync)

    async def analyze(
        self,
        intent: str,
        screen: bytes,
        *,
        use_xml: bool = False,
        context: Optional[List[str]] = None,
        failures: Optional[List[str]] = None,
    ) -> AnalysisResult:
        """
        Analyze screen and recommend action.

        Args:
            screen: Screenshot PNG bytes (or annotated image bytes if use_xml is True).
            intent: Goal to achieve.
            context: Recent action descriptions.
            failures: Recently failed actions to avoid.
            use_xml: Whether using XML-based labeling.

        Returns:
            AnalysisResult with recommended action.
        """

        if use_xml:
            prompt = ANALYSIS_PROMPT_XML.format(
                intent=intent,
                context=", ".join(context or []) or "None",
                failures=", ".join(failures or []) or "None",
            )
        else:
            prompt = ANALYSIS_PROMPT.format(
                intent=intent,
                context=", ".join(context or []) or "None",
                failures=", ".join(failures or []) or "None",
            )

        client = self.__get_client()

        # Prepare content
        contents = [prompt]

        # Save screenshot locally for debugging
        try:
            timestamp = int(time.time() * 1000)
            local_filename = f"assets/screenshot/{timestamp}.png"

            path = Path(local_filename)
            path.parent.mkdir(parents=True, exist_ok=True)

            with path.open("wb") as new_file:
                new_file.write(screen)

            logger.debug(f"Saved local screenshot: {local_filename}")
        except Exception as exception:
            logger.warning(f"Failed to save local screenshot: {exception}")

        # Handle Image Upload
        try:
            gcs_uri = await self.__upload_image_to_gcs(screen)
            image_part = types.Part.from_uri(file_uri=gcs_uri, mime_type="image/png")
            contents.append(image_part)
        except Exception as exception:
            logger.warning(f"Falling back to inline image due to GCS error: {exception}")
            image_part = types.Part.from_bytes(data=screen, mime_type="image/png")
            contents.append(image_part)

        # Configuration
        config = types.GenerateContentConfig(
            candidate_count=1,
            response_mime_type="application/json",
            temperature=self.__config.temperature,
            max_output_tokens=self.__config.max_output_tokens,
        )

        max_retries = self.__config.max_retries
        base_delay = self.__config.retry_delay

        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Calling Gemini API (Attempt {attempt + 1}/{max_retries + 1})")

                # Use async client call
                response = await client.aio.models.generate_content(
                    config=config,
                    contents=contents,
                    model=self.__config.model,
                )

                text = self.__extract_text_from_response(response)
                logger.info(f"Gemini Response: {text}")
                data = self.__extract_json_from_text(text)
                return self.__parse_response(data)

            except Exception as exception:
                error_msg = str(exception)
                # Retry on rate limits (429) or resource exhausted
                is_retryable = (
                    "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg
                ) and attempt < max_retries

                if is_retryable:
                    # B311: random.random is used for jitter, not security.
                    delay = (base_delay * (2**attempt)) + (random.random() * 0.5)  # nosec
                    logger.warning(f"Rate limit hit, retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)
                    continue

                logger.exception("Gemini analysis failed", stack_info=True)

                if attempt == max_retries:
                    return AnalysisResult(
                        action=Action(
                            confidence=0.1,
                            action_type=ActionType.WAIT,
                            reasoning=f"Error: {exception}",
                            target="Analysis failed, waiting",
                        ),
                        alternatives=[],
                        is_goal_complete=False,
                        screen_description="Unknown",
                        reasoning=f"Analysis failed: {exception}",
                    )

        # Fallback (should be unreachable)
        return AnalysisResult(
            action=Action(
                target="Analysis failed",
                action_type=ActionType.WAIT,
                reasoning="Unknown fatal error",
            ),
            alternatives=[],
            is_goal_complete=False,
            reasoning="Unknown fatal error",
            screen_description="Error state",
        )

    def __extract_text_from_response(self, response: Any) -> str:
        """
        Extract text from Gemini API response object.

        Args:
            response: The API response object.

        Returns:
            Extracted text content.

        Raises:
            VisionError: If response is empty or invalid.
        """

        try:
            if hasattr(response, "text") and response.text:
                return str(response.text)

            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    parts = candidate.content.parts
                    return "".join(part.text for part in parts if part.text)

            raise VisionError("Empty response from Gemini API")
        except Exception as exception:
            raise VisionError(f"Failed to extract text: {exception}") from exception

    def __extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract and parse JSON from text (handling markdown and repair).

        Args:
            text: Raw text response.

        Returns:
            Parsed JSON dictionary.

        Raises:
            VisionError: If JSON parsing fails.
        """

        # 1. Try ```json block
        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)

        else:
            # 2. Try generic block or just braces
            start = text.find("{")
            end = text.rfind("}")
            # SIM108: Use ternary operator
            json_str = text[start : end + 1] if start != -1 and end != -1 else text

        json_str = json_str.strip()

        # 3. Attempt repair if truncated (basic check)
        open_braces = json_str.count("{")
        close_braces = json_str.count("}")
        if open_braces > close_braces:
            json_str += "}" * (open_braces - close_braces)

        try:
            return cast("Dict[str, Any]", json.loads(json_str))
        except json.JSONDecodeError as exception:
            logger.error(f"JSON Parse Error: {exception}\nText: {text}")
            raise VisionError(f"Invalid JSON: {exception}") from exception

    def __parse_response(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        Parse API response to AnalysisResult.

        Args:
            data: Parsed JSON data.

        Returns:
            AnalysisResult object.
        """

        action_data = data.get("action", {})
        action = self.__parse_action(action_data)

        alternatives = [self.__parse_action(alt) for alt in data.get("alternatives", [])]

        return AnalysisResult(
            action=action,
            alternatives=alternatives,
            reasoning=str(data.get("reasoning", "")),
            screen_description=str(data.get("screen_description", "")),
            is_goal_complete=bool(data.get("is_goal_complete", False)),
        )

    def __parse_action(self, data: Dict[str, Any]) -> Action:
        """
        Parse action from response data.

        Args:
            data: Action dictionary.

        Returns:
            Action object.
        """

        try:
            action_type = ActionType(str(data.get("type", "WAIT")).lower())
        except ValueError:
            action_type = ActionType.WAIT

        bbox = None
        coords = data.get("coordinates")

        if coords and isinstance(coords, dict):
            # Prompt returns normalized 0-1000 coordinates.
            x = int(coords.get("x", 0))
            y = int(coords.get("y", 0))

            # Clamp to safe normalized range
            x = max(0, min(1000, x))
            y = max(0, min(1000, y))

            # Default to a small touch target if no width/height
            width = int(coords.get("width", 100))
            height = int(coords.get("height", 100))

            # Ensure bbox fits within 1000x1000
            if x + width > 1000:
                width = max(1, 1000 - x)
            if y + height > 1000:
                height = max(1, 1000 - y)

            bbox = BoundingBox(
                x=x,
                y=y,
                width=width,
                height=height,
            )

        confidence_raw = data.get("confidence", 0.5)
        confidence = float(confidence_raw) if confidence_raw else 0.5

        return Action(
            bbox=bbox,
            text=data.get("text"),
            action_type=action_type,
            target=str(data.get("target", "")),
            reasoning=str(data.get("reasoning", "")),
            confidence=min(1.0, max(0.0, confidence)),
            label_id=str(data.get("label_id")) if data.get("label_id") else None,
        )

    async def check_completion(self, intent: str, screen: bytes) -> bool:
        """
        Check if intent is complete.

        Args:
            intent: User intent.
            screen: Screenshot bytes.

        Returns:
            True if complete.
        """

        prompt = (
            f"Goal: {intent}\n"
            "Analyze the screen. Is the goal definitively achieved? "
            'Reply with JSON: {"complete": true/false, "reason": "..."}'
        )
        try:
            client = self.__get_client()
            contents = [
                prompt,
                types.Part.from_bytes(data=screen, mime_type="image/png"),
            ]

            response = await client.aio.models.generate_content(
                model=self.__config.model,
                contents=contents,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            data = self.__extract_json_from_text(str(response.text))
            return bool(data.get("complete", False))
        except Exception:
            return False


class MockGeminiVisionTool(VisionTool):
    """
    Mock Gemini vision tool for testing without API calls.
    """

    def __init__(self, *, always_complete: bool = False) -> None:
        """
        Initialize mock vision tool.

        Args:
            always_complete: If True, always returns COMPLETE action.
        """

        self._call_count = 0
        self._always_complete = always_complete

    async def analyze(
        self,
        intent: str,
        screen: bytes,
        *,
        use_xml: bool = False,
        context: Optional[List[str]] = None,
        failures: Optional[List[str]] = None,
    ) -> AnalysisResult:
        """
        Return mock analysis result.
        """

        _ = use_xml
        self._call_count += 1

        if self._always_complete or self._call_count > 5:
            return AnalysisResult(
                action=Action(
                    confidence=0.95,
                    target="Goal achieved",
                    action_type=ActionType.COMPLETE,
                    reasoning="Task completed successfully",
                ),
                alternatives=[],
                is_goal_complete=True,
                reasoning="Goal appears complete",
                screen_description="Success screen",
            )

        action_type = [
            ActionType.TAP,
            ActionType.TAP,
            ActionType.TAP,
            ActionType.TYPE,
            ActionType.SCROLL,
        ][self._call_count % 5]

        return AnalysisResult(
            action=Action(
                confidence=0.8,
                action_type=action_type,
                target=f"Mock target for {intent}",
                bbox=BoundingBox(x=400, y=400, width=200, height=100),
                text="test" if action_type == ActionType.TYPE else None,
                reasoning=f"Mock reasoning step {self._call_count}",
            ),
            alternatives=[],
            is_goal_complete=False,
            screen_description="Mock screen",
            reasoning=f"Mock analysis step {self._call_count}",
        )
