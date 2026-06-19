from __future__ import annotations

from typing import Final, FrozenSet

# Tokens that flag an action as potentially destructive when present in its
# target description or rationale. The supervisor blocks any action whose
# textual context overlaps this set so the planner is forced to escalate.
UNSAFE_ACTION_KEYWORDS: Final[FrozenSet[str]] = frozenset(
    {
        "wipe",
        "purge",
        "erase",
        "format",
        "delete account",
        "factory reset",
        "remove account",
    }
)
