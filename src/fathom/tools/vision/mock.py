from __future__ import annotations

from typing import Any, Dict, List, Optional

from fathom.constants import ActionType
from fathom.schemas.actions import Action, BoundingBox
from fathom.schemas.results import AnalysisResult
from fathom.schemas.screens import ScreenCapture
from fathom.tools.vision.base import VisionTool


class MockVisionTool(VisionTool):
    """
    Mock vision tool for testing.
    Returns configurable responses for testing agent behavior.
    """

    def __init__(
        self,
        *,
        complete_after_steps: int = 5,
        default_action: Optional[Action] = None,
    ) -> None:
        """
        Initialize mock vision tool.

        Args:
            default_action: Default action to return. If None, uses tap.
            complete_after_steps: Steps after which to return complete.
        """

        self.__default_action = default_action or Action(
            confidence=0.9,
            target="mock element",
            action_type=ActionType.TAP,
            rationale="Mock default action",
            bbox=BoundingBox(x=500, y=500, width=100, height=100),
        )
        self.__call_count = 0
        self.__history: List[Dict[str, Any]] = []
        self.__complete_after = complete_after_steps

    @property
    def name(self) -> str:
        """
        Tool name.
        """

        return "mock_vision"

    @property
    def provider(self) -> Any:
        """
        Returns a mock provider.
        """

        return None

    @property
    def call_count(self) -> int:
        """
        Number of analyze calls made.
        """

        return self.__call_count

    @property
    def history(self) -> List[Dict[str, Any]]:
        """
        History of analyze calls.
        """

        return self.__history.copy()

    async def analyze(
        self,
        intent: str,
        capture: ScreenCapture,
        *,
        use_xml: bool = False,
        context: Optional[str] = None,
        failures: Optional[List[str]] = None,
        elements: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResult:
        """
        Return mock analysis result.

        Args:
            capture: Ignored in mock.
            intent: Recorded for history.
            context: Recorded for history.
            failures: If present, returns different action.
            use_xml: Whether using XML mode.
            elements: Grounding elements.

        Returns:
            Configured mock result.
        """

        self.__call_count += 1
        self.__history.append(
            {
                "intent": intent,
                "context": context,
                "use_xml": use_xml,
                "failures": failures,
                "elements": elements,
                "screen_size": len(capture.image),
            }
        )

        is_complete = self.__call_count >= self.__complete_after

        if is_complete:
            action = Action(
                confidence=1.0,
                target="Goal achieved",
                action_type=ActionType.COMPLETE,
                rationale="Goal completion threshold reached",
            )
        else:
            action = self.__default_action

        return AnalysisResult(
            action=action,
            is_goal_complete=is_complete,
            screen_description="Mock screen",
            reasoning=f"Mock reasoning for step {self.__call_count}",
            metrics={"memory_retrieval": 0.01, "llm_analysis": 0.05},
        )

    async def check_completion(
        self,
        intent: str,
        capture: ScreenCapture,
    ) -> bool:
        """
        Check if mock should report completion.

        Returns:
            True if call count exceeds complete_after_steps.
        """

        return self.__call_count >= self.__complete_after

    def reset(self) -> None:
        """
        Reset mock state.
        """

        self.__call_count = 0
        self.__history.clear()
