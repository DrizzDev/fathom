from __future__ import annotations

from enum import StrEnum


class PhaseKind(StrEnum):
    """
    Workflow phases visible to the client during a run.
    """

    QUALIFYING = "QUALIFYING"
    DECOMPOSING = "DECOMPOSING"

    PLANNING = "PLANNING"
    GROUNDING = "GROUNDING"
    UNDERSTANDING = "UNDERSTANDING"

    ACTING = "ACTING"
    OBSERVING = "OBSERVING"
    VERIFYING = "VERIFYING"
    AUTHORING = "AUTHORING"
