from __future__ import annotations

from enum import StrEnum
from typing import Final


class ToolName(StrEnum):
    """
    Canonical tool identifiers exposed to the language model.
    """

    ASK_USER = "ask_user"
    EXECUTE_UI = "execute_ui"
    VERIFY_GOAL = "verify_goal"
    STORE_MEMORY = "store_memory"
    RECALL_MEMORY = "recall_memory"
    VALIDATE_STATE = "validate_state"


class TurnMode(StrEnum):
    """
    Per-turn flag controlling which optional tool group the LLM may invoke.

    Multiple flags can be active simultaneously. Action tools live in
    :data:`BASE_TOOLS` and are always available regardless of any mode flag;
    the mode set is purely an *additive* exposure signal for optional groups.
    """

    VERIFY = "verify"


BASE_TOOLS: Final[frozenset[ToolName]] = frozenset(
    {ToolName.EXECUTE_UI, ToolName.STORE_MEMORY, ToolName.RECALL_MEMORY}
)
