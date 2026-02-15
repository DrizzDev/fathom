"""Device port interface for mobile device interactions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from fathom.schemas.actions import Action
from fathom.schemas.results import ActionResult


class DevicePort(ABC):
    """Abstract interface for mobile device interactions."""

    @abstractmethod
    async def tap(self, *, x: int, y: int) -> ActionResult:
        """Tap at screen coordinates."""
        pass

    @abstractmethod
    async def type_text(self, *, text: str) -> ActionResult:
        """Type text into focused element."""
        pass

    @abstractmethod
    async def swipe(
        self, *, x1: int, y1: int, x2: int, y2: int, duration: int = 300
    ) -> ActionResult:
        """Swipe from (x1,y1) to (x2,y2)."""
        pass

    @abstractmethod
    async def back(self) -> ActionResult:
        """Press back button."""
        pass

    @abstractmethod
    async def home(self) -> ActionResult:
        """Press home button."""
        pass

    @abstractmethod
    async def get_screen_size(self) -> Tuple[int, int]:
        """Get screen dimensions (width, height)."""
        pass

    @abstractmethod
    async def capture_screen(self) -> bytes:
        """Capture screenshot as PNG bytes."""
        pass

    @abstractmethod
    async def get_current_package(self) -> str:
        """Get current foreground package name."""
        pass

    @abstractmethod
    async def wait_for_device(self, *, timeout: float) -> bool:
        """Wait for device to be ready."""
        pass
