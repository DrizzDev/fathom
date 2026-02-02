from __future__ import annotations

import base64
import json
import re
from typing import Any, Dict, List, Optional, cast

import httpx

from fathom.constants import ActionType
from fathom.exceptions import VisionError
from fathom.prompts.analysis import ANALYSIS_PROMPT
from fathom.schemas.actions import Action, BoundingBox
from fathom.schemas.configuration import GeminiConfig
from fathom.schemas.results import AnalysisResult
from fathom.tools.vision.base import VisionTool


class GeminiVisionTool(VisionTool):
    """
    Vision tool using Google's Gemini API.

    Analyzes screenshots and recommends actions to achieve goals.

    Example:
        ```python
        vision = GeminiVisionTool(GeminiConfig(api_key="..."))
        result = await vision.analyze(
            screen=screenshot_bytes,
            intent="Open settings",
        )
        print(f"Recommended: {result.action}")
        ```
    """

    def __init__(self, config: GeminiConfig) -> None:
        """
        Initialize Gemini vision tool.

        Args:
            config: Gemini API configuration.
        """

        self.__config = config
        self.__http_client: Optional[Any] = None

    async def analyze(
        self,
        screen: bytes,
        intent: str,
        context: Optional[List[str]] = None,
        failures: Optional[List[str]] = None,
    ) -> AnalysisResult:
        """
        Analyze screen and recommend action.

        Args:
            screen: Screenshot PNG bytes.
            intent: Goal to achieve.
            context: Recent action descriptions.
            failures: Recently failed actions to avoid.

        Returns:
            AnalysisResult with recommended action.
        """

        prompt = ANALYSIS_PROMPT.format(
            intent=intent,
            context=", ".join(context or []) or "None",
            failures=", ".join(failures or []) or "None",
        )

        try:
            response = await self.__call_api(screen, prompt)
            return self.__parse_response(response)

        except Exception as exception:
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

    async def check_completion(
        self,
        screen: bytes,
        intent: str,
    ) -> bool:
        """
        Check if intent is complete.

        Args:
            screen: Screenshot bytes.
            intent: User intent.

        Returns:
            True if complete.
        """

        prompt = (
            f"Goal: {intent}\n"
            "Analyze the screen. Is the goal definitively achieved? "
            'Reply with JSON: {"complete": true/false, "reason": "..."}'
        )
        try:
            response = await self.__call_api(screen, prompt)
            return bool(response.get("complete", False))
        except Exception:
            return False

    async def __call_api(self, image: bytes, prompt: str) -> Dict[str, Any]:
        """
        Call Gemini API.

        Args:
            image: Image bytes.
            prompt: Text prompt.

        Returns:
            Parsed API response.
        """

        image_b64 = base64.b64encode(image).decode()

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "data": image_b64,
                                "mime_type": "image/png",
                            }
                        },
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": self.__config.temperature,
                "maxOutputTokens": self.__config.max_output_tokens,
            },
        }

        if self.__config.api_key:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.__config.model}:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.__config.api_key,
            }
        else:
            token = self.__get_access_token()
            location = self.__config.location
            project = self.__config.project_id

            if not project:
                try:
                    import google.auth

                    _, project = google.auth.default()
                except ImportError:
                    pass

            if not project:
                raise VisionError("Project ID required for Vertex AI (or set GEMINI_API_KEY)")

            url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{self.__config.model}:generateContent"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            }

        async with httpx.AsyncClient(timeout=self.__config.timeout) as client:
            response = await client.post(url, json=payload, headers=headers)

            if response.status_code != 200:
                raise VisionError(f"Gemini API error: {response.status_code} - {response.text}")

            data = response.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise VisionError("No response from Gemini")

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            raise VisionError("Empty response from Gemini")

        text = parts[0].get("text", "")

        try:
            return cast("Dict[str, Any]", json.loads(text))
        except json.JSONDecodeError:
            if json_match := re.search(r"\{.*\}", text, re.DOTALL):
                return cast("Dict[str, Any]", json.loads(json_match.group()))

            raise VisionError(f"Invalid JSON response: {text[:200]}") from None

    def __get_access_token(self) -> str:
        """
        Get GCP access token using google-auth.

        Returns:
            Access token string.
        """

        try:
            import google.auth
            from google.auth.transport.requests import Request

            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            credentials.refresh(Request())
            return str(credentials.token)
        except ImportError as exception:
            raise VisionError(
                "google-auth required for Vertex AI: pip install google-auth"
            ) from exception
        except Exception as exception:
            raise VisionError(f"Failed to get GCP credentials: {exception}") from exception

    def __parse_response(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        Parse API response to AnalysisResult.

        Args:
            data: Parsed JSON response.

        Returns:
            AnalysisResult.
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
            data: Action data dictionary.

        Returns:
            Action object.
        """

        action_type_str = str(data.get("type", "WAIT")).upper()

        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            action_type = ActionType.WAIT

        bbox = None
        coords = data.get("coordinates")

        if coords and isinstance(coords, dict):
            x = int(coords.get("x", 500))
            y = int(coords.get("y", 500))
            bbox = BoundingBox(
                width=100,
                height=100,
                x=max(0, x - 50),
                y=max(0, y - 50),
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
        )


class MockGeminiVisionTool(VisionTool):
    """
    Mock Gemini vision tool for testing without API calls. Returns scripted or random actions for testing.
    """

    def __init__(self, *, always_complete: bool = False) -> None:
        """
        Initialize mock vision tool.

        Args:
            always_complete: If True, always returns COMPLETE action.
        """

        self.__call_count = 0
        self.__always_complete = always_complete

    async def analyze(
        self,
        screen: bytes,
        intent: str,
        context: Optional[List[str]] = None,
        failures: Optional[List[str]] = None,
    ) -> AnalysisResult:
        """
        Return mock analysis result.

        Args:
            screen: Screenshot bytes (ignored).
            intent: Goal (used for description).
            context: Context (ignored).
            failures: Failures (ignored).

        Returns:
            Mock AnalysisResult.
        """

        self.__call_count += 1

        if self.__always_complete or self.__call_count > 5:
            return AnalysisResult(
                action=Action(
                    action_type=ActionType.COMPLETE,
                    target="Goal achieved",
                    confidence=0.95,
                    reasoning="Task completed successfully",
                ),
                alternatives=[],
                reasoning="Goal appears complete",
                screen_description="Success screen",
                is_goal_complete=True,
            )

        action_type = [
            ActionType.TAP,
            ActionType.TAP,
            ActionType.TAP,
            ActionType.TYPE,
            ActionType.SCROLL,
        ][self.__call_count % 5]

        return AnalysisResult(
            action=Action(
                confidence=0.8,
                action_type=action_type,
                target=f"Mock target for {intent}",
                bbox=BoundingBox(x=400, y=400, width=200, height=100),
                text="test" if action_type == ActionType.TYPE else None,
                reasoning=f"Mock reasoning step {self.__call_count}",
            ),
            alternatives=[],
            is_goal_complete=False,
            screen_description="Mock screen",
            reasoning=f"Mock analysis step {self.__call_count}",
        )
