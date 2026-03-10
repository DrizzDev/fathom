from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from fathom.schemas.configuration import DeviceRuntimeConfiguration
from fathom.schemas.screens import ScreenCapture


class PerceptionPort(ABC):
    """
    Abstract interface for runtime perception and screen-state capture.
    """

    @property
    @abstractmethod
    def configuration(self) -> Optional[DeviceRuntimeConfiguration]:
        """
        Return platform-neutral runtime configuration for perception.
        """

        raise NotImplementedError

    @abstractmethod
    async def capture(self) -> ScreenCapture:
        """
        Capture the current screen state, including optional hierarchy data.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_current_application(self) -> str:
        """
        Return the current foreground application identifier.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_dimensions(self) -> Tuple[int, int]:
        """
        Return the current screen dimensions as (width, height).
        """

        raise NotImplementedError
