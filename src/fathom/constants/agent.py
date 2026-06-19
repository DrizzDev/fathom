from __future__ import annotations

from enum import StrEnum
from typing import Final


class DirectiveKind(StrEnum):
    """
    Classification of an operator-issued directive.
    """

    ABORT = "ABORT"
    NAVIGATE = "NAVIGATE"
    FREE_FORM = "FREE_FORM"
    RETRY_ACTION = "RETRY_ACTION"


DIRECTIVE_DEFAULT_TTL_TURNS: Final[int] = 2
