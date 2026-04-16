from __future__ import annotations

from enum import StrEnum


class FathomEvent(StrEnum):
    """
    Telemetry events for Fathom execution.
    """

    REASONING = "REASONING"
    PLANNED_ACTION = "PLANNED_ACTION"

    STEP_AUDITED = "STEP_AUDITED"
    STEP_COMPLETED = "STEP_COMPLETED"
    SCRIPT_GENERATED = "SCRIPT_GENERATED"

    # High-signal cue events surfaced in the CLI / demo TUI regardless
    # of log level. Emitted by the intent strategy + graph nodes so
    # the user sees classifier + decomposer + sub-goal transitions
    # without enabling verbose logging.
    INTENT_CLASSIFIED = "INTENT_CLASSIFIED"
    DECOMPOSITION_COMPLETE = "DECOMPOSITION_COMPLETE"
    SUB_GOAL_STARTED = "SUB_GOAL_STARTED"
    SUB_GOAL_COMPLETED = "SUB_GOAL_COMPLETED"

    PROMPT_BUILT = "PROMPT_BUILT"
    CONTEXT_CAPTURED = "CONTEXT_CAPTURED"

    LATENCY_PHASE = "LATENCY_PHASE"
    LLM_CALL_COMPLETED = "LLM_CALL_COMPLETED"

    HITL_RECEIVED = "HITL_RECEIVED"
    HITL_REQUESTED = "HITL_REQUESTED"

    WORKFLOW_PAUSED = "WORKFLOW_PAUSED"
    WORKFLOW_RESUMED = "WORKFLOW_RESUMED"

    WORKFLOW_CANCELLED = "WORKFLOW_CANCELLED"
    WORKFLOW_COMPLETED = "WORKFLOW_COMPLETED"
