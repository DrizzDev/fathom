"""Mock capture tool for testing."""

from __future__ import annotations

import hashlib
import time
from typing import List

from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.tools.capture.base import CaptureTool


class MockCaptureTool(CaptureTool):
    """Mock capture tool for testing.

    Returns configurable screen captures.
    """

    def __init__(
        self,
        *,
        screen_width: int = 1080,
        screen_height: int = 2400,
        change_screen_every: int = 1,
    ) -> None:
        """Initialize mock capture tool.

        Args:
            screen_width: Mock screen width.
            screen_height: Mock screen height.
            change_screen_every: Change screen hash every N captures.
        """
        self.__width = screen_width
        self.__height = screen_height
        self.__change_every = change_screen_every
        self.__capture_count = 0
        self.__activity = "com.mock.app/.MainActivity"
        self.__captures: List[ScreenCapture] = []

    @property
    def name(self) -> str:
        """Tool name."""
        return "mock_capture"

    @property
    def capture_count(self) -> int:
        """Number of captures made."""
        return self.__capture_count

    def set_activity(self, activity: str) -> None:
        """Set mock activity."""
        self.__activity = activity

    async def capture(self) -> ScreenCapture:
        """Return mock screen capture."""
        self.__capture_count += 1
        timestamp = int(time.time() * 1000)

        screen_index = self.__capture_count // self.__change_every
        image = f"mock_screen_{screen_index}".encode() * 100

        capture = ScreenCapture(
            image=image,
            width=self.__width,
            height=self.__height,
            activity=self.__activity,
            timestamp=timestamp,
        )

        self.__captures.append(capture)
        return capture

    async def capture_stable(self, timeout: int = 2000) -> ScreenCapture:
        """Return mock stable capture (same as regular capture)."""
        return await self.capture()

    def compute_state(self, capture: ScreenCapture) -> ScreenState:
        """Compute mock screen state."""
        activity_hash = hashlib.md5(capture.activity.encode()).hexdigest()[:16]  # nosec
        structural_hash = hashlib.md5(capture.image[:50]).hexdigest()[:16]  # nosec
        visual_hash = hashlib.md5(capture.image).hexdigest()[:16]  # nosec

        return ScreenState(
            activity=capture.activity,
            activity_hash=activity_hash,
            structural_hash=structural_hash,
            visual_hash=visual_hash,
            timestamp=capture.timestamp,
        )

    def reset(self) -> None:
        """Reset capture state."""
        self.__capture_count = 0
        self.__captures.clear()
