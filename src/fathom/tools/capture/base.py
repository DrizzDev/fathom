from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.tools.base import Tool


class CaptureTool(Tool[ScreenCapture], ABC):
    """Abstract base for screen capture tools.

    Capture tools handle screenshot acquisition and screen state computation.
    """

    @property
    def name(self) -> str:
        """
        Tool name.
        """
        return "capture"

    @abstractmethod
    async def capture(self) -> ScreenCapture:
        """Capture current screen.

        Returns:
            Screen capture with image and metadata.

        Raises:
            ToolTimeoutError: If capture times out.
            ToolConnectionError: If device connection fails.
        """
        raise NotImplementedError

    @abstractmethod
    async def capture_stable(self, timeout: int = 2000) -> ScreenCapture:
        """Capture screen after waiting for stability.

        Waits until screen stops changing or timeout.

        Args:
            timeout: Maximum wait time for stability in milliseconds.

        Returns:
            Stable screen capture.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_state(self, capture: ScreenCapture) -> ScreenState:
        """Compute screen state from capture.

        Args:
            capture: Screen capture to analyze.

        Returns:
            Computed screen state with hashes.
        """
        raise NotImplementedError

    def are_same_screen(
        self,
        state1: ScreenState,
        state2: ScreenState,
        threshold: int = 10,
    ) -> bool:
        """Check if two screen states represent the same screen.

        Args:
            state1: First screen state.
            state2: Second screen state.
            threshold: Maximum hamming distance for match.

        Returns:
            True if screens are the same.
        """
        return state1.is_same_screen(state2, threshold)

    async def execute(self, request: Dict[str, Any]) -> ScreenCapture:
        """Execute via generic interface.

        Args:
            request: Dict with optional 'stable' and 'timeout'.

        Returns:
            Screen capture.
        """
        if request.get("stable", False):
            return await self.capture_stable(request.get("timeout", 2000))
        return await self.capture()
