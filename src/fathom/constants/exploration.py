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


class ExpectedOutcome(StrEnum):
    """
    The screen effect the model predicts an action will produce.
    """

    NEW_SCREEN = "new_screen"
    IN_SCREEN_CHANGE = "in_screen_change"
    DIALOG_OR_POPUP = "dialog_or_popup"
    SCROLL_CONTENT = "scroll_content"
    DISMISS_OVERLAY = "dismiss_overlay"
    GO_BACK = "go_back"

    @property
    def implies_transition(self) -> bool:
        """
        Whether the prediction expects the screen to visibly change.
        """

        return self in _TRANSITION_OUTCOMES


# Predictions that should leave the device on a visibly different screen; a tap
# that claims one of these yet leaves the screen unchanged is likely inert.
_TRANSITION_OUTCOMES: frozenset[ExpectedOutcome] = frozenset(
    {
        ExpectedOutcome.NEW_SCREEN,
        ExpectedOutcome.DIALOG_OR_POPUP,
        ExpectedOutcome.DISMISS_OVERLAY,
        ExpectedOutcome.GO_BACK,
    }
)


class FocusRelevance(StrEnum):
    """
    How a screen relates to the user-supplied exploration focus.

    ON_FOCUS     - The screen is part of the focused section, flow, or feature.
    LEADS_TOWARD - Not the focus itself, but a route that heads toward it.
    OFF_FOCUS    - Unrelated to the focus.
    UNSCOPED     - No focus was configured; exploration is broad-coverage.
    """

    ON_FOCUS = "on_focus"
    LEADS_TOWARD = "leads_toward"
    OFF_FOCUS = "off_focus"
    UNSCOPED = "unscoped"

    @property
    def recovery_priority(self) -> int:
        """
        Frontier-recovery rank; lower is recovered first (on-focus before off-focus).

        On a broad-coverage run every screen is UNSCOPED, so this collapses to a
        constant and recovery falls back to pure nearest-first ordering.
        """

        return _RECOVERY_PRIORITY[self]


# Order in which the frontier-recovery pass reaches unscanned screens on a
# focused run: on-focus first, off-focus last, but every screen stays reachable.
_RECOVERY_PRIORITY: dict[FocusRelevance, int] = {
    FocusRelevance.ON_FOCUS: 0,
    FocusRelevance.LEADS_TOWARD: 1,
    FocusRelevance.UNSCOPED: 2,
    FocusRelevance.OFF_FOCUS: 3,
}


# The default exploration goal used when no specific focus is requested; a
# generic goal keeps the model in broad-coverage mode rather than steering it
# toward one flow (see EXPLORATION_FOCUS_DIRECTIVE).
DEFAULT_EXPLORATION_INTENT: str = "Explore application structure"

# Telemetry event name carrying a per-step exploration progress snapshot; the
# TUI telemetry adapter renders it as live header state, the console adapter
# logs it as a structured line.
EXPLORATION_PROGRESS_EVENT: str = "exploration.progress"

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

# Safety bound on consecutive DFS routing cycles that complete no exploration
# step. A healthy run records a step (or backtracks) within a couple of cycles,
# so crossing this means the phase machine is wedged -- e.g. a screen yields no
# usable capture and scan keeps routing back without advancing -- and the run
# ends with CompletionReason.STUCK instead of spinning to the graph recursion
# limit. Generous multiple of the healthy maximum, far below any recursion cap.
MAX_ROUTES_WITHOUT_PROGRESS: int = 50


# Coverage-plateau bound: consecutive completed steps that surface no new screen
# before the run ends with CompletionReason.COVERAGE_PLATEAU. Unlike the routing
# watchdog above (which fires when steps stop happening), this fires when steps
# keep happening but exploration has stopped discovering, so the crawl does not
# spend its remaining budget re-treading screens it has already mapped.
MAX_STEPS_WITHOUT_NEW_SCREEN: int = 15


# Most recent executed actions surfaced back into the scan context so the model
# can see whether its latest moves advanced exploration (new screen / no-op /
# failed) and avoid re-issuing ineffective taps the per-screen dedup cannot catch.
RECENT_ACTION_WINDOW: int = 3


# Re-prompts allowed when the traversal guard vetoes an action that would enter a
# sensitive area (payment, auth, destructive); after these the screen is treated
# as exhausted so the crawl describes it but backtracks instead of acting in.
MAX_SENSITIVE_ACTION_RETRIES: int = 2


# Quantization cell (on the normalized 0-1000 grid) for coordinate-bucket dedup:
# two taps aimed at the same visual element land in the same bucket despite label drift.
COORD_BUCKET_GRID_SIZE: int = 50

# Depth bounds for the pure graph-search algorithms over the screen graph.
PATH_SEARCH_MAX_DEPTH: int = 50
ALL_PATHS_SEARCH_MAX_DEPTH: int = 10
