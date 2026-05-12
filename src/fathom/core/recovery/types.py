from __future__ import annotations

from enum import StrEnum
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from fathom.schemas.screens import ScreenCapture
from fathom.schemas.subgoal import SubGoal


class RecoveryTrigger(StrEnum):
    """
    Enumerates the points at which the graph asks the recovery coordinator
    for help. New triggers can be added without touching existing strategies.
    """

    ACTION_BLOCKED = "ACTION_BLOCKED"
    VERIFY_REJECTED = "VERIFY_REJECTED"


class RecoveryRequest(BaseModel):
    """
    Frozen input handed to a recovery strategy when its trigger fires.
    Carries enough context to reason about the stuck state without
    granting access to mutable agent state.
    """

    trigger: RecoveryTrigger = Field(description="Trigger that produced this request")
    capture: ScreenCapture = Field(description="Current screen capture for visual grounding")

    reason: str = Field(description="Free-text failure reason from the graph node")
    hint: Optional[str] = Field(
        default=None, description="Optional concrete hint such as the blocked action"
    )

    pending_sub_goals: List[str] = Field(description="Remaining unfinished sub-goal descriptions")
    stuck_sub_goal: str = Field(
        description="Description of the sub-goal the agent failed to complete"
    )

    recent_actions: List[str] = Field(
        default_factory=list, description="Most-recent action descriptors, oldest first"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


class ReplanOutcome(BaseModel):
    """
    Strategy decided to replace the pending sub-goal tail.
    The graph node applies the result via ``AgentState.replan_pending_sub_goals``.
    """

    summary: str = Field(description="Short human-readable summary for telemetry/logs")
    kind: Literal["replan"] = Field(default="replan", description="Outcome discriminator")
    new_sub_goals: List[SubGoal] = Field(description="Replacement sub-goals in execution order")

    model_config = ConfigDict(frozen=True)


class NoopOutcome(BaseModel):
    """
    Strategy considered the situation but declined to act. Coordinator
    treats this as "fall through to the standard rejection path."
    """

    kind: Literal["noop"] = Field(default="noop", description="Outcome discriminator")
    summary: str = Field(description="Short human-readable summary for telemetry/logs")

    model_config = ConfigDict(frozen=True)


RecoveryOutcome = Annotated[
    Union[ReplanOutcome, NoopOutcome],
    Field(discriminator="kind"),
]
