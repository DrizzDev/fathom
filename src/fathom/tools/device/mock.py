"""Mock device tool for testing."""

from __future__ import annotations

from typing import List, Optional, Tuple

from fathom.schemas.results import ActionResult
from fathom.tools.device.base import DeviceTool


class MockDeviceTool(DeviceTool):
    """Mock device tool for testing.

    Records all actions for verification.
    """

    def __init__(
        self,
        *,
        screen_size: Tuple[int, int] = (1080, 2400),
        fail_on_action: Optional[str] = None,
    ) -> None:
        """Initialize mock device tool.

        Args:
            screen_size: Mock screen dimensions.
            fail_on_action: Action type to fail on (for testing errors).
        """
        self.__screen_size = screen_size
        self.__fail_on_action = fail_on_action
        self.__activity = "com.mock.app/.MainActivity"

        self.__tap_calls: List[Tuple[int, int]] = []
        self.__type_calls: List[str] = []
        self.__swipe_calls: List[Tuple[int, int, int, int, int]] = []
        self.__back_calls: int = 0
        self.__home_calls: int = 0

    @property
    def name(self) -> str:
        """Tool name."""
        return "mock_device"

    @property
    def tap_calls(self) -> List[Tuple[int, int]]:
        """Recorded tap calls."""
        return self.__tap_calls.copy()

    @property
    def type_calls(self) -> List[str]:
        """Recorded type calls."""
        return self.__type_calls.copy()

    def set_activity(self, activity: str) -> None:
        """Set mock activity name."""
        self.__activity = activity

    async def tap(self, x: int, y: int) -> ActionResult:
        """Record tap action."""
        if self.__fail_on_action == "tap":
            return ActionResult(success=False, duration=0, error="Mock tap failure")

        self.__tap_calls.append((x, y))
        return ActionResult(success=True, duration=50)

    async def type_text(self, text: str) -> ActionResult:
        """Record type action."""
        if self.__fail_on_action == "type":
            return ActionResult(success=False, duration=0, error="Mock type failure")

        self.__type_calls.append(text)
        return ActionResult(success=True, duration=len(text) * 10)

    async def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: int = 300,
    ) -> ActionResult:
        """Record swipe action."""
        if self.__fail_on_action == "swipe":
            return ActionResult(success=False, duration=0, error="Mock swipe failure")

        self.__swipe_calls.append((x1, y1, x2, y2, duration))
        return ActionResult(success=True, duration=duration)

    async def back(self) -> ActionResult:
        """Record back action."""
        if self.__fail_on_action == "back":
            return ActionResult(success=False, duration=0, error="Mock back failure")

        self.__back_calls += 1
        return ActionResult(success=True, duration=30)

    async def home(self) -> ActionResult:
        """Record home action."""
        self.__home_calls += 1
        return ActionResult(success=True, duration=30)

    async def get_screen_size(self) -> Tuple[int, int]:
        """Return mock screen size."""
        return self.__screen_size

    async def get_activity(self) -> str:
        """Return mock activity."""
        return self.__activity

    def reset(self) -> None:
        """Reset all recorded calls."""
        self.__tap_calls.clear()
        self.__type_calls.clear()
        self.__swipe_calls.clear()
        self.__back_calls = 0
        self.__home_calls = 0
