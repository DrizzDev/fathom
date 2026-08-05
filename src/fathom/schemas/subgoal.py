from __future__ import annotations

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.subgoal import DEFAULT_SUB_GOAL_MAX_STEPS
from fathom.schemas.base.common import NonBlank, SealedModel
from fathom.schemas.steps import StepResult
from fathom.schemas.success import Success


class PendingProof(SealedModel):
    """
    A correlated successful command receipt held on a goal awaiting its later visual postcondition.
    """

    receipt: StepResult = Field(
        description="The matching executed command step proving the primitive ran, pending its postcondition."
    )


class SubGoalStatus(StrEnum):
    """
    Lifecycle states for a sub-goal.
    """

    FAILED = "FAILED"
    PENDING = "PENDING"
    COMPLETE = "COMPLETE"
    IN_PROGRESS = "IN_PROGRESS"


class SubGoal(SealedModel):
    """
    Immutable definition of one decomposed sub-goal.
    """

    index: int = Field(ge=0, description="Position in the decomposition sequence.")
    objective: NonBlank = Field(description="Observable objective this sub-goal must achieve.")
    success: Success = Field(description="The single typed definition of this sub-goal's success.")


class Progress(BaseModel):
    """
    Mutable execution state for one sub-goal.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    status: SubGoalStatus = Field(
        default=SubGoalStatus.PENDING, description="Current lifecycle status."
    )
    attempts: int = Field(default=0, ge=0, description="Device actions spent on this sub-goal.")
    recovery: int = Field(
        default=0, ge=0, description="Escalation deferrals accrued on this sub-goal."
    )
    limit: int = Field(
        default=DEFAULT_SUB_GOAL_MAX_STEPS,
        gt=0,
        description="Device-action budget before the sub-goal is over budget.",
    )
    proof: Optional[PendingProof] = Field(
        default=None,
        description="Additive checkpointed P5 evidence: a correlated command receipt ignored by live decisions until P6.",
    )


class GoalState(BaseModel):
    """
    Pairs an immutable sub-goal definition with its mutable progress.
    """

    model_config = ConfigDict(extra="forbid")

    goal: SubGoal = Field(description="Immutable sub-goal definition.")
    progress: Progress = Field(default_factory=Progress, description="Mutable execution state.")

    @property
    def index(self) -> int:
        """
        Position of this sub-goal in the decomposition sequence.
        """

        return self.goal.index

    @property
    def objective(self) -> str:
        """
        Observable objective this sub-goal must achieve.
        """

        return self.goal.objective

    @property
    def success(self) -> Success:
        """
        Typed definition of this sub-goal's success.
        """

        return self.goal.success

    @property
    def deferral_count(self) -> int:
        """
        Escalation deferrals accrued on this sub-goal.
        """

        return self.progress.recovery

    @property
    def over_budget(self) -> bool:
        """
        Whether device actions on this sub-goal have reached its budget.
        """

        return self.progress.attempts >= self.progress.limit

    def is_pending(self) -> bool:
        """
        Whether the sub-goal has not yet been started.
        """

        return self.progress.status is SubGoalStatus.PENDING

    def is_complete(self) -> bool:
        """
        Whether the sub-goal has been completed.
        """

        return self.progress.status is SubGoalStatus.COMPLETE

    def mark_in_progress(self) -> None:
        """
        Transition the sub-goal into the in-progress state.
        """

        self.progress.status = SubGoalStatus.IN_PROGRESS

    def mark_complete(self) -> None:
        """
        Transition the sub-goal into the complete state.
        """

        self.progress.status = SubGoalStatus.COMPLETE

    def mark_failed(self) -> None:
        """
        Transition the sub-goal into the failed state.
        """

        self.progress.status = SubGoalStatus.FAILED

    def record_attempt(self) -> None:
        """
        Count one device action spent against this sub-goal's budget.
        """

        self.progress.attempts += 1

    def record_deferral(self) -> None:
        """
        Count one escalation deferral against this sub-goal.
        """

        self.progress.recovery += 1

    def clear_deferrals(self) -> None:
        """
        Reset this sub-goal's escalation-deferral count.
        """

        if self.progress.recovery != 0:
            self.progress.recovery = 0
