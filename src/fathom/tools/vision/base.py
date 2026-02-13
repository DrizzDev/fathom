from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from fathom.interfaces import IVisionProvider
from fathom.schemas.results import AnalysisResult
from fathom.schemas.screens import ScreenCapture
from fathom.tools.base import Tool


class VisionTool(Tool[AnalysisResult], ABC):
    """
    Abstract base for vision/LLM tools.
    Vision tools analyze screens and plan actions.
    """

    @property
    def name(self) -> str:
        """
        Tool name.
        """

        return "vision"

    @property
    @abstractmethod
    def provider(self) -> IVisionProvider:
        """
        Returns the underlying vision provider.
        """

        raise NotImplementedError

    @abstractmethod
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
        Analyze screen and recommend action.
        """

        raise NotImplementedError

    async def cleanup(self) -> None:
        """
        Perform any necessary cleanup (e.g., closing connections, deleting caches).
        """

        pass

    @abstractmethod
    async def check_completion(
        self,
        intent: str,
        capture: ScreenCapture,
    ) -> bool:
        """
        Check if intent is complete on current screen.

        Args:
            capture: Screen capture object.
            intent: Intent to check completion for.

        Returns:
            True if intent appears complete.
        """

        raise NotImplementedError

    async def execute(self, request: Dict[str, Any]) -> AnalysisResult:
        """
        Execute via generic interface.

        Args:
            request: Dict with 'capture' (ScreenCapture), 'intent', optional 'context', 'failures'.

        Returns:
            Analysis result.
        """

        return await self.analyze(
            intent=request["intent"],
            capture=request["capture"],
            context=request.get("context"),
            failures=request.get("failures"),
        )
