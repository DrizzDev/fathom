from enum import StrEnum

from pydantic import BaseModel, Field

# Default per-sub-goal step budget. The agent gets this many ANALYZE →
# EXECUTE → RECORD cycles per sub-goal before the recovery coordinator
# is dispatched with ``SUBGOAL_BUDGET_EXCEEDED``. Hard cap regardless of
# whether the loop detector or no-progress classifier has fired.
#
# Sized generously so popup-heavy iOS flows (Allow / Skip / coachmark
# chains) don't blow the budget on legitimate work. A run that needs
# more than this many steps to satisfy a single sub-goal almost always
# indicates the sub-goal was over-decomposed or names a target the
# screen doesn't expose — both healed at the decomposer (Phase 2).
DEFAULT_SUB_GOAL_MAX_STEPS: int = 8


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

    max_steps: int = Field(
        default=DEFAULT_SUB_GOAL_MAX_STEPS,
        ge=1,
        description=(
            "Maximum graph iterations (ANALYZE → EXECUTE → RECORD) the "
            "agent may spend on this sub-goal before the recovery "
            "coordinator is dispatched with SUBGOAL_BUDGET_EXCEEDED. "
            "Decomposers may override per-sub-goal; default keeps "
            "popup-heavy flows comfortable."
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
