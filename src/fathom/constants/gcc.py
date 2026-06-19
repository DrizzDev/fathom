from enum import StrEnum
from typing import Final

# Active sub-goal count at which RECORD emits a GCC BRANCH event so the
# decomposer surfaces parallel work streams instead of a flat list.
GCC_BRANCHING_ACTIVE_COUNT: Final[int] = 15


class GCCTier(StrEnum):
    """
    The three tiers of the GCC reasoning hierarchy.
    """

    ROADMAP = "roadmap"  # Tier 1: Global intent and milestones
    COMMIT = "commit"  # Tier 2: Summarized progress units
    LOG = "log"  # Tier 3: Fine-grained execution traces


class GCCCommand(StrEnum):
    """
    Formal operations supported by the Git-Context-Controller.
    """

    MERGE = "MERGE"
    COMMIT = "COMMIT"
    BRANCH = "BRANCH"
    CONTEXT = "CONTEXT"
