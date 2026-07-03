from __future__ import annotations

from typing import Dict

from fathom.core.exceptions import InvariantViolation
from fathom.schemas.capture import Capture


class CaptureStore:
    """
    Run-owned registry of values captured by STORE during a single workflow.
    """

    def __init__(self) -> None:
        """
        Start an empty per-run capture registry.
        """

        self.__captures: Dict[str, Capture] = {}

    def write(self, *, capture: Capture) -> None:
        """
        Record a capture, overwriting any prior capture under the same name.
        """

        self.__captures[capture.name] = capture

    def read(self, *, name: str) -> Capture:
        """
        Return the capture stored under a name, failing fast when none exists.
        """

        capture = self.__captures.get(name)
        if capture is None:
            raise InvariantViolation(f"No capture stored under '{name}'.")

        return capture

    def exists(self, *, name: str) -> bool:
        """
        Return whether a capture is stored under a name.
        """

        return name in self.__captures

    def clear(self) -> None:
        """
        Drop every captured value so a new run starts clean.
        """

        self.__captures.clear()
