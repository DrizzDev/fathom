from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from fathom.constants.interaction import SwipeSpeed
from fathom.schemas.configuration import DeviceRuntimeConfiguration
from fathom.schemas.results import ActionResult


class DevicePort(ABC):
    """
    Abstract interface for environment action execution.
    """

    @property
    @abstractmethod
    def configuration(self) -> Optional[DeviceRuntimeConfiguration]:
        """
        Platform-neutral device runtime configuration.
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
        self,
        *,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: Optional[int] = None,
        speed: Optional[SwipeSpeed] = None,
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
    async def get_current_package(self) -> str:
        """
        Get the current foreground application identifier.
        """

        raise NotImplementedError

    @abstractmethod
    async def capture_screen(self) -> bytes:
        """
        Capture the current screenshot payload.
        """

        raise NotImplementedError

    @abstractmethod
    async def dump_hierarchy(self) -> Optional[str]:
        """
        Capture the current hierarchy payload.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_snapshot(self) -> Tuple[bytes, Optional[str]]:
        """
        Capture the current screenshot and optional hierarchy payload.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_dimensions(self) -> Tuple[int, int]:
        """
        Get screen dimensions (width, height).
        """

        raise NotImplementedError

    @abstractmethod
    async def wait_for_device(self, *, timeout: float) -> bool:
        """
        Wait for device to be ready.
        """

        raise NotImplementedError
