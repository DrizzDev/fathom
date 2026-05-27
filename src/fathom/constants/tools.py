from __future__ import annotations

from enum import StrEnum
from typing import FrozenSet


class ToolName(StrEnum):
    """Canonical tool identifiers exposed to the language model."""

    ASK_USER = "ask_user"
    EXECUTE_UI = "execute_ui"
    VERIFY_GOAL = "verify_goal"
    STORE_MEMORY = "store_memory"
    RECALL_MEMORY = "recall_memory"
    VALIDATE_STATE = "validate_state"


BASE_TOOLS: FrozenSet[ToolName] = frozenset(
    {ToolName.EXECUTE_UI, ToolName.STORE_MEMORY, ToolName.RECALL_MEMORY}
)

VERIFICATION_TOOLS: FrozenSet[ToolName] = frozenset({ToolName.VERIFY_GOAL, ToolName.VALIDATE_STATE})

VERIFICATION_KEYWORDS: FrozenSet[str] = frozenset({"verify", "check", "confirm", "validate"})
