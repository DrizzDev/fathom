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

    Each value is a distinct stuck signal:

    - ``ACTION_BLOCKED``: the planner determined the previously-planned
      approach is unreachable from the current screen.
    - ``VERIFY_REJECTED``: final verification refused the run as complete.
    - ``LOOP_DETECTED``: the loop detector observed repetition across
      screens or actions strong enough to indicate the agent is cycling.
    - ``NO_PROGRESS``: the action-effect classifier marked the last N
      actions as producing no visual progress.
    - ``TARGET_UNRESOLVED``: the resolution layer could not map the
      agent's named target to a tappable manifest element or screen
      coordinate.
    - ``SUBGOAL_BUDGET_EXCEEDED``: the active sub-goal exhausted its
      per-sub-goal step budget without its success criterion being met.
    - ``REPORT_UNACTIONABLE``: the agent itself reported that the active
      sub-goal does not match the current screen.
    """

    ACTION_BLOCKED = "ACTION_BLOCKED"
    VERIFY_REJECTED = "VERIFY_REJECTED"
    LOOP_DETECTED = "LOOP_DETECTED"
    NO_PROGRESS = "NO_PROGRESS"
    TARGET_UNRESOLVED = "TARGET_UNRESOLVED"
    SUBGOAL_BUDGET_EXCEEDED = "SUBGOAL_BUDGET_EXCEEDED"
    REPORT_UNACTIONABLE = "REPORT_UNACTIONABLE"


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
