"""State schemas for Fathom workflows."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from fathom.schemas.steps import StepRecord, StepResult


class WorkflowState(BaseModel):
    """Fully serializable workflow state for checkpointing.

    This model is designed to be Temporal-compatible with all state
    serializable via model_dump(mode='json').
    """

    workflow_id: str = Field(description="Unique workflow identifier")
    intent: str = Field(description="Intent being executed")
    step_count: int = Field(default=0, ge=0, description="Total steps executed")

    recent_screens: List[str] = Field(
        default_factory=list,
        description="Recent screen hashes (bounded)",
    )
    recent_actions: List[str] = Field(
        default_factory=list,
        description="Recent action descriptions (bounded)",
    )
    completed_steps: List[StepRecord] = Field(
        default_factory=list,
        description="All completed step records",
    )

    is_complete: bool = Field(default=False)
    is_stuck: bool = Field(default=False)
    is_cancelled: bool = Field(default=False)
    final_message: Optional[str] = Field(default=None)

    max_history: int = Field(default=10, ge=1, le=100)

    def record_step(self, result: StepResult) -> None:
        """Record a completed step.

        Args:
            result: Result of the executed step.
        """
        self.step_count += 1
        self.recent_screens.append(result.post_hash)
        self.recent_actions.append(result.step.action.to_description())
        self.completed_steps.append(result.to_record())

        if len(self.recent_screens) > self.max_history:
            self.recent_screens = self.recent_screens[-self.max_history :]
        if len(self.recent_actions) > self.max_history:
            self.recent_actions = self.recent_actions[-self.max_history :]

        self.__detect_stuck()

    def __detect_stuck(self) -> None:
        """Detect if agent is stuck in a loop."""
        if len(self.recent_screens) >= 3:
            last_three = self.recent_screens[-3:]
            if len(set(last_three)) == 1:
                self.is_stuck = True

    def checkpoint(self) -> Dict[str, object]:
        """Return checkpoint data for Temporal persistence."""
        return self.model_dump(mode="json")


class ExecutionContext(BaseModel):
    """Execution context passed between steps.

    Designed to be serializable for workflow checkpointing.
    """

    workflow_id: str = Field(description="Parent workflow identifier")
    intent: str = Field(description="Current intent")
    step_count: int = Field(default=0, ge=0)
    max_history: int = Field(default=10, ge=1, le=100)

    recent_actions: List[str] = Field(default_factory=list)
    recent_screens: List[str] = Field(default_factory=list)
    failures: List[str] = Field(default_factory=list)
    clarifications: List[str] = Field(default_factory=list)

    def add_action(self, description: str) -> None:
        """Add an action to recent history."""
        self.recent_actions.append(description)
        if len(self.recent_actions) > self.max_history:
            self.recent_actions = self.recent_actions[-self.max_history :]
        self.step_count += 1

    def add_screen(self, screen_hash: str) -> None:
        """Add a screen hash to recent history."""
        self.recent_screens.append(screen_hash)
        if len(self.recent_screens) > self.max_history:
            self.recent_screens = self.recent_screens[-self.max_history :]

    def add_failure(self, message: str) -> None:
        """Record a failure for context."""
        self.failures.append(message)
        if len(self.failures) > 5:
            self.failures = self.failures[-5:]

    def to_dict(self) -> Dict[str, object]:
        """Serialize for passing to tools."""
        return self.model_dump(mode="json")
