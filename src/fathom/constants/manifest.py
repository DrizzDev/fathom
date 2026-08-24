from __future__ import annotations

from enum import StrEnum
from typing import Final

MANIFEST_VERSION: Final[int] = 1


class ModelRole(StrEnum):
    """
    A model-consuming role Fathom composes for a run.
    """

    PLANNER = "PLANNER"
    ARCHITECT = "ARCHITECT"
    QUALIFIER = "QUALIFIER"
    SUMMARIZER = "SUMMARIZER"
    ABORT = "ABORT"
    LOCALIZER = "LOCALIZER"
    EMBEDDER = "EMBEDDER"
    ORACLE = "ORACLE"


class RoleBinding(StrEnum):
    """
    How a role obtained its adapter relative to the other roles in the run.
    """

    INDEPENDENT = "INDEPENDENT"
    DERIVED = "DERIVED"
    SHARED = "SHARED"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


SHARED_ARCHITECT_NOTE: Final[str] = (
    "derive() returned the planner adapter itself, so the architect is not a deterministic "
    "sibling and runs at the planner's temperature and cache setting."
)
SHARED_QUALIFIER_NOTE: Final[str] = (
    "Qualifier runs on the planner adapter, so its inference configuration does not apply; a "
    "dedicated adapter is built only when the runner is composed with an assembly."
)
UNDECLARED_BINDING_NOTE: Final[str] = (
    "Port does not expose the adapter it was composed with, so its relationship to the planner "
    "is not determinable."
)
UNREAD_CONFIGURATION_NOTE: Final[str] = (
    "Composition ignores this role's configured model and enabled flag; the role runs on the "
    "planner adapter."
)
