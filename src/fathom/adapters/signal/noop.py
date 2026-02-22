from __future__ import annotations

import asyncio
from typing import Optional

from fathom.interfaces.signal import SignalPort


class NoopSignal(SignalPort):
    """
    No-op adapter for signal port.
    Returns None for all signals, enabling fully autonomous operation.
    """

    async def check_signal(self) -> Optional[str]:
        """
        Check for control signal - always returns None for autonomous mode.
        """

        return None

    async def wait_for_pause(self) -> None:
        """
        Block until a pause signal is received.
        In autonomous mode, pause signals never occur, so block forever.
        """

        # Block forever - pause signals never happen in autonomous mode
        await asyncio.Event().wait()

    async def wait_for_resume(self) -> None:
        """
        Block until RESUME signal - no-op for autonomous mode.
        """

        pass

    async def ask(self, *, prompt: str) -> str:
        """
        Request human input - returns empty string for autonomous mode.
        """

        return ""

    async def get_injected_context(self) -> Optional[str]:
        """
        Get injected context - always returns None for autonomous mode.
        """

        return None

    async def has_injected_context(self) -> bool:
        """
        Check if there's injected context - always returns False for autonomous mode.
        """

        return False

    async def is_pause_requested(self) -> bool:
        """
        Check if pause is requested - always returns False for autonomous mode.
        """

        return False
