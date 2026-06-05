from __future__ import annotations

from enum import StrEnum
from typing import Final


class DirectiveKind(StrEnum):
    """
    Classification of an operator-issued directive (HITL response or
    remote-operator instruction) used to bypass autonomous-mode guards.
    """

    COMPLETE = "COMPLETE"
    NAVIGATE = "NAVIGATE"
    FREE_FORM = "FREE_FORM"
    RETRY_ACTION = "RETRY_ACTION"


DIRECTIVE_DEFAULT_TTL_TURNS: Final[int] = 2
