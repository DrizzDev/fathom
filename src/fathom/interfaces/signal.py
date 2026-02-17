"""Signal port interface for human-in-the-loop control."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class SignalPort(ABC):
    """
    Abstract interface for control signals.
    Defines the contract for external interruptions and context injection.
    """

    @abstractmethod
    async def check_signal(self) -> Optional[str]:
        """
        Non-blocking check for an active signal.
        Returns:
            SignalType value (str) if present, else None.
        """
        raise NotImplementedError

    @abstractmethod
    async def wait_for_pause(self) -> None:
        """
        Block efficiently until a pause signal is received.
        Must use event-driven mechanisms (awaitables), not polling loops.
        """
        raise NotImplementedError

    @abstractmethod
    async def wait_for_resume(self) -> None:
        """
        Block until a resume signal is received.
        """
        raise NotImplementedError

    @abstractmethod
    def get_injected_context(self) -> Optional[str]:
        """
        Retrieve and consume injected user context.
        Returns:
            The context string if available, else None.
        """
        raise NotImplementedError

    @abstractmethod
    async def ask(self, *, prompt: str) -> str:
        """
        Request specific input from the human.
        Args:
            prompt: The question to ask.
        Returns:
            The user's response.
        """
        raise NotImplementedError
