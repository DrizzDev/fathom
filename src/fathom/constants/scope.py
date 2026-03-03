from __future__ import annotations

from enum import StrEnum


class ContextScope(StrEnum):
    """
    Scope of the execution context for memory hydration.
    """

    EXECUTION = "execution"
    CONVERSATION = "conversation"
