"""Signal port interface for human-in-the-loop control."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class SignalPort(ABC):
    """Abstract interface for human-in-the-loop control signals."""

    @abstractmethod
    async def check_signal(self) -> Optional[str]:
        """Check for control signal (PAUSE, RESUME, INJECT, ASK)."""
        pass

    @abstractmethod
    async def wait_for_resume(self) -> None:
        """Block until RESUME signal received."""
        pass

    @abstractmethod
    async def request_input(self, *, prompt: str) -> str:
        """Request human input with prompt."""
        pass
