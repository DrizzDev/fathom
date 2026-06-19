from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field

from fathom.constants import ActionType
from fathom.constants.subgoal import (
    DEFAULT_SUB_GOAL_MAX_STEPS,
)


class SubGoalStatus(StrEnum):
    """
    Lifecycle states for a sub-goal.
    """

    FAILED = "FAILED"
    PENDING = "PENDING"
    COMPLETE = "COMPLETE"
    IN_PROGRESS = "IN_PROGRESS"


class SubGoalKind(StrEnum):
    """
    Classification used to select the completion gate strategy for a sub-goal.
    """

    ACTION = "action"
    VALIDATION = "validation"


class SubGoal(BaseModel):
    """
    Represents a single sub-goal in a decomposed intent.
    Sub-goals must be executed sequentially without skipping.
    """

    description: str = Field(description="Task description for this sub-goal")
    index: int = Field(description="Position in the decomposition sequence (0-based)")

    criterion: str | None = Field(
        default=None,
        description="Observable terminal state criterion for compatibility with execution tasks.",
    )
    directive: Optional[ActionType] = Field(
        default=None,
        description=(
            "Structured action the planner must emit to satisfy this sub-goal. "
            "The completion gate compares the planner-emitted action_type "
            "against this value to detect divergence. Optional only for "
            "backward compatibility with checkpoints written before the "
            "directive contract existed; new decompositions always populate it."
        ),
    )
    status: SubGoalStatus = Field(
        default=SubGoalStatus.PENDING, description="Current lifecycle status"
    )

    kind: SubGoalKind = Field(
        default=SubGoalKind.ACTION,
        description=(
            "Classification used by the completion gate: ACTION sub-goals require a "
            "state-mutating action and screen evolution to advance; VALIDATION sub-goals "
            "advance on an asserted completion claim alone."
        ),
    )

    # Completion signals (tracked for multi-signal verification)
    flagged_complete: bool = Field(default=False, description="Model raised the completion flag")
    trace_verified: bool = Field(
        default=False, description="Trace/action history confirms completion"
    )
    rationale_verified: bool = Field(
        default=False, description="Rationale tokens match sub-goal completion keywords"
    )
    completion_verified: bool = Field(
        default=False, description="Final verification that sub-goal is complete"
    )

    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Decomposer confidence (0.0=low, 1.0=perfect)"
    )

    max_steps: int = Field(
        ge=1,
        default=DEFAULT_SUB_GOAL_MAX_STEPS,
        description="Maximum graph iterations the agent may spend on this sub-goal.",
    )

    deferral_count: int = Field(
        ge=0,
        default=0,
        description=(
            "Consecutive escalations deferred by the escalation gate while "
            "this sub-goal was active. Resets on observable progress or is "
            "implicitly reset by sub-goal advance (the next sub-goal starts "
            "at 0). Bounded by the gate's deferral_limit so deferral cannot "
            "hide a genuinely stuck flow indefinitely."
        ),
    )
    completion_claim_streak: int = Field(
        ge=0,
        default=0,
        description=(
            "Consecutive planner emits of ``validate`` + ``flagged_complete`` "
            "against a non-validate directive while this sub-goal was active. "
            "Counts the divergence pattern that arises when the app skips an "
            "intermediate screen and the named action is no longer reachable "
            "but the criterion is already satisfied. Bounded by "
            "IMPLICIT_COMPLETION_THRESHOLD; on reach the completion gate "
            "accepts the divergence as an implicit completion and advances."
        ),
    )

    def mark_in_progress(self) -> None:
        """
        Mark sub-goal as currently being executed.
        """

        self.status = SubGoalStatus.IN_PROGRESS

    def mark_complete(
        self,
        trace_verified: bool = False,
        flagged_complete: bool = False,
        rationale_verified: bool = False,
    ) -> None:
        """
        Mark sub-goal as complete with multi-signal verification.
        """

        self.trace_verified = trace_verified
        self.flagged_complete = flagged_complete
        self.rationale_verified = rationale_verified

        self.completion_verified = True
        self.status = SubGoalStatus.COMPLETE

    def is_complete(self) -> bool:
        """
        Check if sub-goal is in COMPLETE state.
        """

        return self.status == SubGoalStatus.COMPLETE

    def __repr__(self) -> str:
        return (
            f"SubGoal(idx={self.index}, status={self.status.value}, "
            f"desc='{self.description[:30]}...', confidence={self.confidence:.2f})"
        )
