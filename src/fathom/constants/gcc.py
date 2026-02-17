"""
Constants for the Generative Context Construction (GCC) framework.
"""

from enum import StrEnum


class GCCTier(StrEnum):
    """
    The three tiers of the GCC reasoning hierarchy.
    """
    ROADMAP = "roadmap"     # Tier 1: Global intent and milestones
    COMMIT = "commit"       # Tier 2: Summarized progress units
    LOG = "log"             # Tier 3: Fine-grained execution traces


class GCCCommand(StrEnum):
    """
    Formal operations supported by the Git-Context-Controller.
    """
    COMMIT = "COMMIT"
    BRANCH = "BRANCH"
    MERGE = "MERGE"
    CONTEXT = "CONTEXT"
