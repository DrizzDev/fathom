from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.subgoal import (
    DEFAULT_SUB_GOAL_MAX_STEPS,
    INPUT_SUB_GOAL_MAX_STEPS,
    SCROLL_SUB_GOAL_MAX_STEPS,
    TAP_SUB_GOAL_MAX_STEPS,
    VALIDATE_SUB_GOAL_MAX_STEPS,
    WAIT_SUB_GOAL_MAX_STEPS,
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


class RequiredActionFamily(StrEnum):
    """
    Structured action-family contract carried by one sub-goal.
    """

    UNSPECIFIED = "unspecified"
    SCROLL = "scroll"
    TAP = "tap"
    INPUT = "input"
    WAIT = "wait"
    VALIDATE = "validate"


class ScrollAxis(StrEnum):
    """
    Explicit scroll axis declared by the planner when one is known.
    """

    UNSPECIFIED = "unspecified"
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


class ExecutionContract(BaseModel):
    """
    Structured execution constraints that strict mode must preserve.
    """

    model_config = ConfigDict(frozen=True)

    required_action_family: RequiredActionFamily = Field(
        default=RequiredActionFamily.UNSPECIFIED,
        description="Primary action family the planner must preserve for this sub-goal.",
    )
    scroll_axis: ScrollAxis = Field(
        default=ScrollAxis.UNSPECIFIED,
        description="Explicit scroll axis when the task truly constrains one.",
    )
    surface: str | None = Field(
        default=None,
        description=(
            "Specific section, container, or on-screen area that the planner must preserve "
            "for this sub-goal when the user names one."
        ),
    )


def default_max_steps_for_execution_contract(*, contract: ExecutionContract) -> int:
    """
    Return the default step budget for one execution contract.
    """

    family = contract.required_action_family
    if family is RequiredActionFamily.SCROLL:
        return SCROLL_SUB_GOAL_MAX_STEPS
    if family is RequiredActionFamily.TAP:
        return TAP_SUB_GOAL_MAX_STEPS
    if family is RequiredActionFamily.INPUT:
        return INPUT_SUB_GOAL_MAX_STEPS
    if family is RequiredActionFamily.WAIT:
        return WAIT_SUB_GOAL_MAX_STEPS
    if family is RequiredActionFamily.VALIDATE:
        return VALIDATE_SUB_GOAL_MAX_STEPS
    return DEFAULT_SUB_GOAL_MAX_STEPS


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
    execution_contract: ExecutionContract = Field(
        default_factory=ExecutionContract,
        description="Structured execution constraints preserved by strict mode.",
    )
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
            "Maximum graph iterations (ANALYZE → EXECUTE → RECORD) the agent may spend on this sub-goal before the recovery "
            "coordinator is dispatched with SUBGOAL_BUDGET_EXCEEDED. Decomposers may override per-sub-goal; default keeps popup-heavy flows comfortable."
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
