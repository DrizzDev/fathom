from __future__ import annotations

from enum import StrEnum


class ExplorationEvent(StrEnum):
    """
    Telemetry events emitted by the exploration graph.

    Consumed by the live TUI to update header counters and render
    body panels. The CLI passes a no-op bus when ``--tui`` is off,
    so emit sites are always safe to call.
    """

    WORKFLOW_STARTED = "WORKFLOW_STARTED"
    PHASE_TRANSITION = "PHASE_TRANSITION"

    SCREEN_CAPTURED = "SCREEN_CAPTURED"
    SCREEN_DISCOVERED = "SCREEN_DISCOVERED"
    SCREEN_REVISITED = "SCREEN_REVISITED"

    ACTION_PLANNED = "ACTION_PLANNED"
    ACTION_EXECUTED = "ACTION_EXECUTED"

    LLM_CALL_COMPLETED = "LLM_CALL_COMPLETED"

    NAVIGATION_STARTED = "NAVIGATION_STARTED"
    BACKTRACK = "BACKTRACK"

    STEP_COMPLETED = "STEP_COMPLETED"

    WORKFLOW_COMPLETED = "WORKFLOW_COMPLETED"
    WORKFLOW_CANCELLED = "WORKFLOW_CANCELLED"
    ERROR = "ERROR"
