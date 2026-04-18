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

    async def launch_configured_package(self) -> None:
        """
        Launch the adapter's pre-configured application.

        Default is a no-op. Adapters with auto-launch capability
        (``ADBDevice``, ``IOSDevice`` on the idb backend) override
        this to launch the package/bundle the user selected in the
        wizard (or via ``--package`` / ``--ios-bundle-identifier``)
        before the agent loop starts. Safe to call repeatedly —
        implementations guard against double-launch.
        """

        return None

    async def terminate_configured_package(self) -> None:
        """
        Terminate the adapter's pre-configured application on run exit.

        Symmetric counterpart to :meth:`launch_configured_package` —
        keeps teardown tidy so we don't leave the app running after
        the agent finishes. Default is a no-op; adapters with a
        ``terminate`` capability override. Failures are logged and
        swallowed; cleanup must never raise.
        """

        return None
