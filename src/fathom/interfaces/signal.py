"""Signal port interface for human-in-the-loop control."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class SignalPort(ABC):
    """Abstract interface for human-in-the-loop control signals."""

    @abstractmethod
    async def check_signal(self) -> Optional[str]:
        """Check for control signal (PAUSE, RESUME, INJECT, ASK)."""
        raise NotImplementedError

    @abstractmethod
    async def wait_for_resume(self) -> None:
        """Block until RESUME signal received."""
        raise NotImplementedError

    @abstractmethod
    async def ask(self, *, prompt: str) -> str:
        """Request human input with prompt."""
        raise NotImplementedError

    @abstractmethod
    def get_injected_context(self) -> Optional[str]:
        """Get injected context and clear it."""
        raise NotImplementedError

    @abstractmethod
    def has_injected_context(self) -> bool:
        """Check if there's injected context available."""
        raise NotImplementedError

    @abstractmethod
    def is_pause_requested(self) -> bool:
        """Check if pause is requested (for immediate cancellation)."""
        raise NotImplementedError
