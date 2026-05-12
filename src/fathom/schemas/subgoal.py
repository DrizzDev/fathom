from enum import StrEnum

from pydantic import BaseModel, Field


class SubGoalStatus(StrEnum):
    """
    Lifecycle states for a sub-goal.
    """

    FAILED = "FAILED"
    PENDING = "PENDING"
    COMPLETE = "COMPLETE"
    IN_PROGRESS = "IN_PROGRESS"


class SubGoal(BaseModel):
    """
    Represents a single sub-goal in a decomposed intent.
    Sub-goals must be executed sequentially without skipping.
    """

    index: int = Field(description="Position in the decomposition sequence (0-based)")
    description: str = Field(description="Task description for this sub-goal")
    status: SubGoalStatus = Field(
        default=SubGoalStatus.PENDING, description="Current lifecycle status"
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
