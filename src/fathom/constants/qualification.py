from __future__ import annotations

from enum import StrEnum
from typing import Literal

from fathom.constants.links import FATHOM_DOCS_URL

DEFAULT_REJECTION_MESSAGE: str = (
    "Fathom can only execute actions inside mobile apps. "
    "Try describing a task to perform — for example, exploring the login flow, "
    "completing the checkout flow, or finding a specific setting in the app. "
    f"See {FATHOM_DOCS_URL} for help and examples."
)

# Qualifier inference defaults.
DEFAULT_QUALIFIER_MODEL: str = "gemini-2.5-flash-lite"
# Per-attempt wall-clock cap that bounds tail latency.
DEFAULT_QUALIFIER_TIMEOUT: float = 5.0
# Retries after the initial attempt; adapter handles backoff + jitter.
DEFAULT_QUALIFIER_MAX_RETRIES: int = 2
# Deterministic output for a binary classifier — no creativity needed.
DEFAULT_QUALIFIER_TEMPERATURE: float = 0.0
# Short, one-off prompts; cache reuse wastes more than it saves at qualifier scale.
DEFAULT_QUALIFIER_USE_CACHE: bool = False
# Minimum reasoning depth — classification doesn't benefit from chain-of-thought.
DEFAULT_QUALIFIER_THINKING_LEVEL: Literal["minimal", "low", "medium", "high"] = "low"


class QualificationLabel(StrEnum):
    """Binary executability classification for a user intent."""

    EXECUTABLE = "EXECUTABLE"
    NOT_EXECUTABLE = "NOT_EXECUTABLE"


class RationaleCategory(StrEnum):
    """Reasoning bucket for a qualification verdict."""

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
