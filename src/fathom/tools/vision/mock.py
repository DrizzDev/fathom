"""Mock vision tool for testing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fathom.constants import ActionType
from fathom.schemas.actions import Action, BoundingBox
from fathom.schemas.results import AnalysisResult
from fathom.tools.vision.base import VisionTool


class MockVisionTool(VisionTool):
    """Mock vision tool for testing.

    Returns configurable responses for testing agent behavior.
    """

    def __init__(
        self,
        *,
        default_action: Optional[Action] = None,
        complete_after_steps: int = 5,
    ) -> None:
        """Initialize mock vision tool.

        Args:
            default_action: Default action to return. If None, uses tap.
            complete_after_steps: Steps after which to return complete.
        """
        self.__default_action = default_action or Action(
            action_type=ActionType.TAP,
            target="mock element",
            bbox=BoundingBox(x=500, y=500, width=100, height=100),
            confidence=0.9,
        )
        self.__complete_after = complete_after_steps
        self.__call_count = 0
        self.__history: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        """Tool name."""
        return "mock_vision"

    @property
    def call_count(self) -> int:
        """Number of analyze calls made."""
        return self.__call_count

    @property
    def history(self) -> List[Dict[str, Any]]:
        """History of analyze calls."""
        return self.__history.copy()

    async def analyze(
        self,
        screen: bytes,
        intent: str,
        *,
        context: Optional[List[str]] = None,
        failures: Optional[List[str]] = None,
    ) -> AnalysisResult:
        """Return mock analysis result.

        Args:
            screen: Ignored in mock.
            intent: Recorded for history.
            context: Recorded for history.
            failures: If present, returns different action.

        Returns:
            Configured mock result.
        """
        self.__call_count += 1
        self.__history.append(
            {
                "intent": intent,
                "context": context,
                "failures": failures,
                "screen_size": len(screen),
            }
        )

        is_complete = self.__call_count >= self.__complete_after

        if is_complete:
            action = Action(
                action_type=ActionType.COMPLETE,
                target="Goal achieved",
                confidence=1.0,
            )
        else:
            action = self.__default_action

        return AnalysisResult(
            action=action,
            reasoning=f"Mock reasoning for step {self.__call_count}",
            is_goal_complete=is_complete,
            screen_description="Mock screen",
        )

    async def check_completion(
        self,
        screen: bytes,
        intent: str,
    ) -> bool:
        """Check if mock should report completion.

        Returns:
            True if call count exceeds complete_after_steps.
        """
        return self.__call_count >= self.__complete_after

    def reset(self) -> None:
        """Reset mock state."""
        self.__call_count = 0
        self.__history.clear()
