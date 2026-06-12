"""
Constants for the application-exploration strategy.
"""

from __future__ import annotations

from enum import StrEnum


class BFSPhase(StrEnum):
    """
    Phases of the DFS-driven exploration state machine.

    SCAN      - On the target screen; the model taps the next untried element,
                following any navigation onto the new screen (stays in SCAN).
    BACKTRACK - Current screen exhausted; press BACK toward the parent and keep
                backtracking until a screen with untried elements is found.
    ADVANCE   - Recovery only; replay a saved path from the root to an orphaned
                unexplored screen that earlier backtracking skipped.
    """

    SCAN = "scan"
    BACKTRACK = "backtrack"
    ADVANCE = "advance"


class GraphFormat(StrEnum):
    """
    Serialisation formats for exporting the explored screen graph.
    """

    JSON = "json"
    DOT = "dot"
    MERMAID = "mermaid"


class CriticalScreenKind(StrEnum):
    """
    Why a screen is structurally significant in the navigation graph.
    """

    HUB = "hub"
    BOTTLENECK = "bottleneck"


class RecommendationLevel(StrEnum):
    """
    Severity of a coverage recommendation in an exploration report.
    """

    OK = "ok"
    NOTE = "note"
    WARNING = "warning"


# Longest screen label rendered in a diagram before it is truncated.
MAX_SCREEN_LABEL_LENGTH: int = 48

# A screen counts as a hub once its combined inbound + outbound edges reach this.
HUB_CONNECTIVITY_THRESHOLD: int = 5

# Fraction of all screens that must be able to reach a screen for it to count
# as a bottleneck.
BOTTLENECK_REACH_RATIO: float = 0.5

# Coverage is flagged low when this fraction of screens remains under-explored.
LOW_COVERAGE_RATIO: float = 0.3

# Connectivity is flagged low when the average edges-per-screen falls below this.
MIN_AVERAGE_CONNECTIVITY: float = 1.5

# Screens included in the "most visited" ranking of a report.
TOP_SCREEN_LIMIT: int = 10

# Navigation cycles included in a report before the list is truncated.
MAX_REPORTED_CYCLES: int = 20

# Loop-detection threshold set high enough that the detector never fires:
# DFS exploration revisits screens by design (backtracking returns to parents),
# so the stuck-loop heuristic must stay disabled for the whole run.
DISABLED_LOOP_THRESHOLD: int = 1_000_000


# Quantization cell (on the normalized 0-1000 grid) for coordinate-bucket dedup:
# two taps aimed at the same visual element land in the same bucket despite label drift.
COORD_BUCKET_GRID_SIZE: int = 50

# Depth bounds for the pure graph-search algorithms over the screen graph.
PATH_SEARCH_MAX_DEPTH: int = 50
ALL_PATHS_SEARCH_MAX_DEPTH: int = 10
