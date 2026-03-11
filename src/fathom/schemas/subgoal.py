"""Sub-goal schema for intent decomposition and sequential execution tracking."""

from enum import Enum

from pydantic import BaseModel, Field


class SubGoalStatus(str, Enum):
    """Lifecycle states for a sub-goal."""

    PENDING = "PENDING"  # Not yet started
    IN_PROGRESS = "IN_PROGRESS"  # Currently being executed
    COMPLETE = "COMPLETE"  # Successfully completed
    FAILED = "FAILED"  # Execution attempt failed


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
    llm_signaled: bool = Field(
        default=False, description="LLM explicitly indicated completion via rationale"
    )
    trace_verified: bool = Field(
        default=False, description="Trace/action history confirms completion"
    )
    rationale_verified: bool = Field(
        default=False, description="LLM rationale tokens match sub-goal completion keywords"
    )
    completion_verified: bool = Field(
        default=False, description="Final verification that sub-goal is complete"
    )

    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Decomposer confidence (0.0=low, 1.0=perfect)"
    )

    def mark_in_progress(self) -> None:
        """Mark sub-goal as currently being executed."""
        self.status = SubGoalStatus.IN_PROGRESS

    def mark_complete(
        self,
        llm_signal: bool = False,
        trace_verified: bool = False,
        rationale_verified: bool = False,
    ) -> None:
        """
        Mark sub-goal as complete with multi-signal verification.

        Args:
            llm_signal: LLM provided explicit completion signal
            trace_verified: Trace/action analysis confirmed completion
            rationale_verified: Rationale token matching confirmed completion
        """
        self.llm_signaled = llm_signal
        self.trace_verified = trace_verified
        self.rationale_verified = rationale_verified
        self.completion_verified = True
        self.status = SubGoalStatus.COMPLETE

    def is_complete(self) -> bool:
        """Check if sub-goal is in COMPLETE state."""
        return self.status == SubGoalStatus.COMPLETE

    def __repr__(self) -> str:
        return (
            f"SubGoal(idx={self.index}, status={self.status.value}, "
            f"desc='{self.description[:30]}...', confidence={self.confidence:.2f})"
        )
