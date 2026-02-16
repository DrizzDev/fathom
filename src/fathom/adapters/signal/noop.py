"""No-op signal adapter for autonomous operation."""

from __future__ import annotations

from typing import Optional

from fathom.interfaces.signal import SignalPort


class NoopSignal(SignalPort):
    """
    No-op adapter for signal port.

    Returns None for all signals, enabling fully autonomous operation.
    """

    async def check_signal(self) -> Optional[str]:
        """Check for control signal - always returns None for autonomous mode."""
        return None

    async def wait_for_resume(self) -> None:
        """Block until RESUME signal - no-op for autonomous mode."""
        pass

    async def request_input(self, *, prompt: str) -> str:
        """Request human input - returns empty string for autonomous mode."""
        return ""

    def get_injected_context(self) -> Optional[str]:
        """Get injected context - always returns None for autonomous mode."""
        return None

    def has_injected_context(self) -> bool:
        """Check if there's injected context - always returns False for autonomous mode."""
        return False
