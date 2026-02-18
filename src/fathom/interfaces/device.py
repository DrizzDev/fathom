from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from fathom.schemas.configuration import ADBConfiguration
from fathom.schemas.results import ActionResult


class DevicePort(ABC):
    """
    Abstract interface for mobile device interactions.
    """

    @property
    @abstractmethod
    def configuration(self) -> Optional[ADBConfiguration]:
        """
        Device configuration.
        """

        raise NotImplementedError

    @abstractmethod
    async def tap(self, *, x: int, y: int) -> ActionResult:
        """
        Tap at screen coordinates.
        """

        raise NotImplementedError

    @abstractmethod
    async def type(self, *, text: str) -> ActionResult:
        """
        Type text into focused element.
        """

        raise NotImplementedError

    @abstractmethod
    async def swipe(
        self, *, x1: int, y1: int, x2: int, y2: int, duration: int = 300
    ) -> ActionResult:
        """
        Swipe from (x1,y1) to (x2,y2).
        """

        raise NotImplementedError

    @abstractmethod
    async def back(self) -> ActionResult:
        """
        Press back button.
        """

        raise NotImplementedError

    @abstractmethod
    async def home(self) -> ActionResult:
        """
        Press home button.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_dimensions(self) -> Tuple[int, int]:
        """
        Get screen dimensions (width, height).
        """

        raise NotImplementedError

    @abstractmethod
    async def capture_screen(self) -> bytes:
        """
        Capture screenshot as PNG bytes.
        """

        raise NotImplementedError

    @abstractmethod
    async def dump_hierarchy(self) -> Optional[str]:
        """
        Dump UI hierarchy to XML string.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_snapshot(self) -> Tuple[bytes, Optional[str]]:
        """
        Capture atomic snapshot (Screenshot + XML) in parallel.
        Returns: (screenshot_bytes, xml_string)
        """

        raise NotImplementedError

    @abstractmethod
    async def get_current_package(self) -> str:
        """
        Get current foreground package name.
        """

        raise NotImplementedError

    @abstractmethod
    async def wait_for_device(self, *, timeout: float) -> bool:
        """
        Wait for device to be ready.
        """

        raise NotImplementedError
