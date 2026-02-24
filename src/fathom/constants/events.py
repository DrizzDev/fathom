from __future__ import annotations

from enum import StrEnum


class FathomEvent(StrEnum):
    """
    Strongly typed telemetry events for Fathom execution.
    """

    STEP_COMPLETED = "STEP_COMPLETED"
    REASONING = "REASONING"
    PLANNED_ACTION = "PLANNED_ACTION"
    HITL_REQUESTED = "HITL_REQUESTED"
    WORKFLOW_PAUSED = "WORKFLOW_PAUSED"
    WORKFLOW_RESUMED = "WORKFLOW_RESUMED"
    WORKFLOW_CANCELLED = "WORKFLOW_CANCELLED"
    WORKFLOW_COMPLETED = "WORKFLOW_COMPLETED"
    SCRIPT_GENERATED = "SCRIPT_GENERATED"
