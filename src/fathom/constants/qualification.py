from __future__ import annotations

from enum import StrEnum

from fathom.constants.links import FATHOM_DOCS_URL

DEFAULT_REJECTION_MESSAGE: str = (
    "Fathom can only execute actions inside mobile apps. "
    "Try describing a task to perform — for example, exploring the login flow, "
    "completing the checkout flow, or finding a specific setting in the app. "
    f"See {FATHOM_DOCS_URL} for help and examples."
)


class QualificationLabel(StrEnum):
    """
    Binary executability classification for a user intent.
    """

    EXECUTABLE = "EXECUTABLE"
    NOT_EXECUTABLE = "NOT_EXECUTABLE"


class RationaleCategory(StrEnum):
    """
    Reasoning bucket for a qualification verdict.
    """

    EMPTY = "empty"
    OTHER = "other"
    UI_TASK = "ui_task"
    CREATIVE = "creative"
    GIBBERISH = "gibberish"
    AMBIGUOUS = "ambiguous"
    PERMISSIVE = "permissive"
    UNSPECIFIED = "unspecified"
    INFORMATIONAL = "informational"
    CONVERSATIONAL = "conversational"
    QUALIFIER_ERROR = "qualifier_error"
