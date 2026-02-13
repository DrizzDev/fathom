from __future__ import annotations

from typing import Any, Dict, List, Optional

from fathom.constants import ActionType
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.results import AnalysisResult
from fathom.schemas.screens import ScreenCapture
from fathom.tools.vision.base import VisionTool


class MockGeminiVisionTool(VisionTool):
    """
    Mock vision tool for testing without API calls.
    """

    def __init__(self, *, always_complete: bool = False) -> None:
        self.__always_complete = always_complete
        self.__count = 0

    async def analyze(
        self,
        intent: str,
        capture: ScreenCapture,
        *,
        use_xml: bool = False,
        context: Optional[str] = None,
        failures: Optional[List[str]] = None,
        is_stuck: bool = False,
        last_action: Optional[str] = None,
        elements: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResult:
        """
        Return mock analysis result.
        """
        self.__count += 1

        if self.__always_complete or self.__count > 5:
            return AnalysisResult(
                action=Action(
                    action_type=ActionType.COMPLETE,
                    target="Goal achieved",
                    confidence=0.95,
                    rationale="Task completed successfully",
                ),
                alternatives=[],
                reasoning="Goal appears complete",
                screen_description="Success screen",
                is_goal_complete=True,
            )

        kind = [
            ActionType.TAP,
            ActionType.SCROLL,
            ActionType.TAP,
            ActionType.TYPE,
            ActionType.TAP,
        ][self.__count % 5]

        return AnalysisResult(
            action=Action(
                action_type=kind,
                target=f"Mock target for {intent}",
                bounds=Bounds(x=400, y=400, width=200, height=100),
                text="test" if kind == ActionType.TYPE else None,
                confidence=0.8,
                rationale=f"Mock reasoning step {self.__count}",
            ),
            alternatives=[],
            reasoning=f"Mock analysis step {self.__count}",
            screen_description="Mock screen",
            is_goal_complete=False,
        )

    async def check_completion(self, intent: str, capture: ScreenCapture) -> bool:
        """
        Check mock completion.
        """
        result = await self.analyze(intent=intent, capture=capture)
        return result.is_goal_complete
