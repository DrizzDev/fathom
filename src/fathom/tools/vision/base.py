from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from fathom.schemas.results import AnalysisResult
from fathom.tools.base import Tool


class VisionTool(Tool[AnalysisResult], ABC):
    """Abstract base for vision/LLM tools.

    Vision tools analyze screens and plan actions.
    """

    @property
    def name(self) -> str:
        """
        Tool name.
        """
        return "vision"

    @abstractmethod
    async def analyze(
        self,
        screen: bytes,
        intent: str,
        *,
        context: Optional[List[str]] = None,
        failures: Optional[List[str]] = None,
    ) -> AnalysisResult:
        """Analyze screen and plan next action.

        Args:
            screen: PNG image bytes.
            intent: User intent to achieve.
            context: Recent action history for context.
            failures: Recent failures for recovery.

        Returns:
            Analysis result with recommended action.

        Raises:
            ToolTimeoutError: If analysis times out.
            ToolExecutionError: If analysis fails.
        """
        raise NotImplementedError

    @abstractmethod
    async def check_completion(
        self,
        screen: bytes,
        intent: str,
    ) -> bool:
        """Check if intent is complete on current screen.

        Args:
            screen: PNG image bytes.
            intent: Intent to check completion for.

        Returns:
            True if intent appears complete.
        """
        raise NotImplementedError

    async def execute(self, request: Dict[str, Any]) -> AnalysisResult:
        """Execute via generic interface.

        Args:
            request: Dict with 'screen', 'intent', optional 'context', 'failures'.

        Returns:
            Analysis result.
        """
        return await self.analyze(
            screen=request["screen"],
            intent=request["intent"],
            context=request.get("context"),
            failures=request.get("failures"),
        )
