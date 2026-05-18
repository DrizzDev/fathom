from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class RuntimeEventKind(StrEnum):
    """
    Runtime event kinds used for replay, telemetry, and external projection.
    """

    DECISION_MADE = "decision.made"
    SCREEN_OBSERVED = "screen.observed"
    TARGET_LOCALIZED = "target.localized"

    ACTION_EXECUTED = "action.executed"
    ACTION_SUPERVISED = "action.supervised"

    HEALING_DECIDED = "healing.decided"
    OUTCOME_OBSERVED = "outcome.observed"
    HEALING_REQUESTED = "healing.requested"

    TASK_UPDATED = "task.updated"
    RECOVERY_DISPATCHED = "recovery.dispatched"
    VERIFICATION_COMPLETED = "verification.completed"


class RuntimeEvent(BaseModel):
    """
    Append-only runtime fact emitted by the execution engine.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow: str = Field(description="Workflow identifier.")
    identifier: str = Field(description="Stable event identifier.")

    step: int = Field(ge=0, description="Runtime step index.")
    kind: RuntimeEventKind = Field(description="Runtime event kind.")
    created: int = Field(
        ge=0,
        description=(
            "UNIX epoch milliseconds at event creation. Stored as int to match "
            "ScreenState.timestamp and ScreenCapture.timestamp across the repo."
        ),
    )
    payload: JsonValue = Field(description="Structured event payload.")
