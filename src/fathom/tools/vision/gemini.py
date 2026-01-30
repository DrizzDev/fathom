from __future__ import annotations

import base64
import json
import re
from typing import Any, Dict, List, Optional

from fathom.constants import ActionType
from fathom.exceptions import VisionError
from fathom.schemas.actions import Action, BoundingBox
from fathom.schemas.configuration import GeminiConfig
from fathom.schemas.results import AnalysisResult
from fathom.tools.vision.base import VisionTool

ANALYSIS_PROMPT = """Analyze this Android screen and determine the next action to achieve the goal.

GOAL: {intent}

CONTEXT:
- Recent actions: {context}
- Recent failures (avoid repeating): {failures}

OUTPUT FORMAT (JSON):
{{
    "action": {{
        "type": "TAP|TYPE|SWIPE|SCROLL|BACK|HOME|WAIT|COMPLETE",
        "target": "description of what to interact with",
        "coordinates": {{"x": 0-1000, "y": 0-1000}} or null,
        "text": "text to type" or null,
        "confidence": 0.0-1.0
    }},
    "alternatives": [
        // 1-2 alternative actions if uncertain
    ],
    "reasoning": "why this action helps achieve the goal",
    "screen_description": "brief description of current screen",
    "is_goal_complete": true/false
}}

RULES:
1. Coordinates use 0-1000 normalized scale (center of screen = 500,500)
2. Use COMPLETE only when the goal is definitively achieved
3. Confidence should reflect certainty (0.9+ = very confident)
4. Consider recent failures when choosing actions
5. If stuck, try alternative navigation (BACK, SCROLL)
"""


class GeminiVisionTool(VisionTool):
    """Vision tool using Google's Gemini API.

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
        """Initialize Gemini vision tool.

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
        """Analyze screen and recommend action.

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
                    action_type=ActionType.WAIT,
                    target="Analysis failed, waiting",
                    confidence=0.1,
                    reasoning=f"Error: {exception}",
                ),
                alternatives=[],
                reasoning=f"Analysis failed: {exception}",
                screen_description="Unknown",
                is_goal_complete=False,
            )

    async def __call_api(self, image: bytes, prompt: str) -> Dict[str, Any]:
        """Call Gemini API.

        Args:
            image: Image bytes.
            prompt: Text prompt.

        Returns:
            Parsed API response.
        """
        try:
            import httpx
        except ImportError as err:
            raise VisionError("httpx required: pip install httpx") from err

        image_b64 = base64.b64encode(image).decode()

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.__config.model}:generateContent"

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": image_b64,
                            }
                        },
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.__config.temperature,
                "maxOutputTokens": self.__config.max_output_tokens,
                "responseMimeType": "application/json",
            },
        }

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.__config.api_key,
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
            from typing import cast

            return cast("Dict[str, Any]", json.loads(text))
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                from typing import cast

                return cast("Dict[str, Any]", json.loads(json_match.group()))
            raise VisionError(f"Invalid JSON response: {text[:200]}") from None

    def __parse_response(self, data: Dict[str, Any]) -> AnalysisResult:
        """Parse API response to AnalysisResult.

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
        """Parse action from response data.

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
                x=max(0, x - 50),
                y=max(0, y - 50),
                width=100,
                height=100,
            )

        confidence_raw = data.get("confidence", 0.5)
        confidence = float(confidence_raw) if confidence_raw else 0.5

        return Action(
            action_type=action_type,
            target=str(data.get("target", "")),
            bbox=bbox,
            text=data.get("text"),
            confidence=min(1.0, max(0.0, confidence)),
            reasoning=str(data.get("reasoning", "")),
        )


class MockGeminiVisionTool(VisionTool):
    """Mock Gemini vision tool for testing without API calls.

    Returns scripted or random actions for testing.
    """

    def __init__(self, *, always_complete: bool = False) -> None:
        """Initialize mock vision tool.

        Args:
            always_complete: If True, always returns COMPLETE action.
        """
        self.__always_complete = always_complete
        self.__call_count = 0

    async def analyze(
        self,
        screen: bytes,
        intent: str,
        context: Optional[List[str]] = None,
        failures: Optional[List[str]] = None,
    ) -> AnalysisResult:
        """Return mock analysis result.

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
            ActionType.SCROLL,
            ActionType.TAP,
            ActionType.TYPE,
            ActionType.TAP,
        ][self.__call_count % 5]

        return AnalysisResult(
            action=Action(
                action_type=action_type,
                target=f"Mock target for {intent}",
                bbox=BoundingBox(x=400, y=400, width=200, height=100),
                text="test" if action_type == ActionType.TYPE else None,
                confidence=0.8,
                reasoning=f"Mock reasoning step {self.__call_count}",
            ),
            alternatives=[],
            reasoning=f"Mock analysis step {self.__call_count}",
            screen_description="Mock screen",
            is_goal_complete=False,
        )
