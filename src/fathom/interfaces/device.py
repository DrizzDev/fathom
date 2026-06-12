from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from fathom.constants.interaction import SwipeSpeed
from fathom.constants.observation import KeyboardVisibility
from fathom.schemas.configuration import DeviceRuntimeConfiguration
from fathom.schemas.observation import KeyboardObservation
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
    async def type(
        self,
        *,
        text: str,
        prefilled: str = "",
        replace: bool = True,
        locator: Optional[str] = None,
    ) -> ActionResult:
        """
        Type text into the focused or identified element.

        When *replace* is True and *prefilled* is non-empty,
        the provider should clear the existing content before typing.
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
    async def launch_package(self, *, package_name: str) -> ActionResult:
        """
        Bring the named application to the foreground.
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

    async def detect_keyboard(self) -> KeyboardObservation:
        """
        Probe the platform for soft-keyboard state; returns UNKNOWN when the adapter cannot determine it.

        Adapters that cannot inspect the IME (e.g., remote gateways without a
        ``dumpsys`` channel) inherit this default. ``UNKNOWN`` instructs the
        swipe planner to skip the keyboard filter, so the original gesture
        dispatches as-is — preserving prior behavior on un-instrumented
        adapters at the cost of losing Glide-Typing protection.
        """

        return KeyboardObservation(visibility=KeyboardVisibility.UNKNOWN)
