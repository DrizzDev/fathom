from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from fathom.constants.observation import KeyboardVisibility
from fathom.schemas.configuration import DeviceRuntimeConfiguration
from fathom.schemas.observation import KeyboardObservation
from fathom.schemas.screens import ScreenCapture


class PerceptionPort(ABC):
    """
    Captures screen state and probes soft-keyboard visibility for the runtime.
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

    async def detect_keyboard(
        self, *, capture: Optional[ScreenCapture] = None
    ) -> KeyboardObservation:
        """
        Probe the platform for soft-keyboard visibility and bounds; default UNKNOWN for adapters that cannot.
        """

        _ = capture
        return KeyboardObservation(visibility=KeyboardVisibility.UNKNOWN)
