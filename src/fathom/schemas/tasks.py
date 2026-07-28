from __future__ import annotations

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants import ActionType
from fathom.constants.capability import CompletionMode
from fathom.schemas.base.common import SealedModel
from fathom.schemas.subgoal import SubGoalKind


class ExecutionTaskState(StrEnum):
    """
    Runtime state for an execution task.
    """

    ACTIVE = "active"
    FAILED = "failed"
    BLOCKED = "blocked"
    PENDING = "pending"
    SUCCEEDED = "succeeded"


class TaskKind(StrEnum):
    """
    Classifies an execution task as an action or a validation step.
    """

    ACTION = "action"
    VALIDATION = "validation"


class TaskAttemptState(BaseModel):
    """
    Attempt accounting for one execution task.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    count: int = Field(ge=0, description="Number of action attempts already used.")
    limit: int = Field(gt=0, description="Maximum action attempts before escalation.")


class ExecutionTask(BaseModel):
    """
    Observable task unit used by the runtime plan.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    objective: str = Field(description="Human-readable task objective.")
    identifier: str = Field(description="Stable runtime task identifier.")

    state: ExecutionTaskState = Field(description="Current task state.")
    criterion: str = Field(description="Natural-language terminal state criterion.")
    attempts: TaskAttemptState = Field(description="Attempt accounting for the task.")
    parent: Optional[str] = Field(default=None, description="Parent task identifier.")
    kind: TaskKind = Field(
        default=TaskKind.ACTION,
        description="Whether this task expects an action or only validates observed state.",
    )

    @property
    def over_budget(self) -> bool:
        """
        Return whether the task has exhausted its attempt budget.
        """

        return self.attempts.count >= self.attempts.limit


class Task(SealedModel):
    """
    Consolidated execution task: what to do, how it is proven complete, and its decomposition slot.
    """

    index: int = Field(ge=0, description="Position in the decomposition sequence.")
    description: str = Field(description="What this task accomplishes.")

    kind: SubGoalKind = Field(description="Completion-gate strategy family for the task.")
    completion: CompletionMode = Field(
        description="Typed proof requirement projected once at decomposition.",
    )

    criterion: Optional[str] = Field(
        default=None,
        description="Observable terminal state criterion.",
    )
    directive: Optional[ActionType] = Field(
        default=None,
        description="Structured action the planner must emit to satisfy this task.",
    )
