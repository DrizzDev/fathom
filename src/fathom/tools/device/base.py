from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from fathom.schemas.results import ActionResult
from fathom.tools.base import Tool


class DeviceTool(Tool[ActionResult], ABC):
    """
    Abstract base for device interaction tools.
    Device tools execute actions on the target device.
    """

    @property
    def name(self) -> str:
        """
        Tool name.
        """

        return "device"

    @abstractmethod
    async def tap(self, x: int, y: int) -> ActionResult:
        """
        Tap at pixel coordinates.

        Args:
            x: X coordinate in pixels.
            y: Y coordinate in pixels.

        Returns:
            Action result.
        """

        raise NotImplementedError

    @abstractmethod
    async def type_text(self, text: str) -> ActionResult:
        """
        Type text into focused element.

        Args:
            text: Text to type.

        Returns:
            Action result.
        """

        raise NotImplementedError

    @abstractmethod
    async def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: int = 300,
    ) -> ActionResult:
        """
        Swipe from one point to another.

        Args:
            x1: Start X coordinate.
            y1: Start Y coordinate.
            x2: End X coordinate.
            y2: End Y coordinate.
            duration: Swipe duration in milliseconds.

        Returns:
            Action result.
        """

        raise NotImplementedError

    @abstractmethod
    async def back(self) -> ActionResult:
        """
        Press back button.

        Returns:
            Action result.
        """

        raise NotImplementedError

    @abstractmethod
    async def home(self) -> ActionResult:
        """
        Press home button.

        Returns:
            Action result.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_screen_size(self) -> Tuple[int, int]:
        """
        Get device screen dimensions.

        Returns:
            Tuple of (width, height) in pixels.
        """

        raise NotImplementedError

    @abstractmethod
    async def screenshot(self) -> Optional[bytes]:
        """
        Capture device screenshot.

        Returns:
            PNG image bytes or None on failure.
        """

        raise NotImplementedError

    @abstractmethod
    async def dump_hierarchy(self) -> Optional[str]:
        """
        Dump UI hierarchy to XML string.
        """

        return None

    async def long_press(
        self,
        x: int,
        y: int,
        duration: int = 1000,
    ) -> ActionResult:
        """
        Long press at coordinates.

        Default implementation uses swipe with same start/end.

        Args:
            x: X coordinate.
            y: Y coordinate.
            duration: Press duration in milliseconds.

        Returns:
            Action result.
        """

        return await self.swipe(x, y, x, y, duration)

    async def execute(self, request: Dict[str, Any]) -> ActionResult:
        """
        Execute via generic interface.

        Args:
            request: Dict with 'action' type and params.

        Returns:
            Action result.
        """

        action = request["action"]

        if action == "tap":
            return await self.tap(request["x"], request["y"])

        elif action == "type":
            return await self.type_text(request["text"])

        elif action == "swipe":
            return await self.swipe(
                request["x1"],
                request["y1"],
                request["x2"],
                request["y2"],
                request.get("duration", 300),
            )

        elif action == "back":
            return await self.back()

        elif action == "home":
            return await self.home()

        else:
            return ActionResult(success=False, duration=0, error=f"Unknown action: {action}")
