from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.schemas.subgoal import SubGoal


class CompletionLogContext(BaseModel):
    """
    Correlation context attached to every structured log event in the completion pipeline.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: str = Field(
        description="Run-level correlation identifier shared across every event in one workflow.",
    )
    sub_goal: SubGoal = Field(
        description="Snapshot of the sub-goal under adjudication on this turn.",
    )
    step_number: int = Field(
        description="Per-run monotonic step counter at the time the event was emitted.",
    )
    screen_hash_pre: Optional[str] = Field(
        default=None,
        description="Visual hash of the screen before the action this turn, when available.",
    )
    screen_hash_post: Optional[str] = Field(
        default=None,
        description="Visual hash of the screen after the action this turn, when available.",
    )
