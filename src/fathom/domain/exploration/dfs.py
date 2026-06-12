"""
DFS exploration state types: the phase machine and frontier queue entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple

from fathom.schemas.actions import Action


class BFSPhase(Enum):
    """
    State machine phases for DFS-driven exploration.

    SCAN      — On the target screen. VLM identifies and taps the next untried
                element.  If the tap navigates to a new screen, DFS follows it
                (stays in SCAN on the new screen).
    BACKTRACK — Current screen fully scanned.  Press BACK to return to the
                parent screen.  If the parent is also exhausted, keep
                backtracking until we find a screen with untried elements.
    ADVANCE   — Recovery only.  When BACKTRACK reaches the root and all
                screens on the DFS path are scanned, but the KG has orphaned
                unexplored screens that were skipped (e.g. due to BACK
                overshooting), navigate to them via path replay.
    """

    SCAN = "scan"
    BACKTRACK = "backtrack"
    ADVANCE = "advance"


@dataclass
class BFSQueueEntry:
    """
    An entry in the BFS frontier queue.

    Stores enough information to navigate back to this screen from any
    position in the app using the simple-BACK strategy.
    """

    screen_hash: str
    parent_hash: str
    action_from_parent: Action
    depth: int
    path_from_root: List[Tuple[str, Action]] = field(default_factory=list)
