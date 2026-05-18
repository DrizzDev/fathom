from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.completion import CompletionEvidence
from fathom.schemas.tasks import ExecutionTaskState


class CompletionVerdict(BaseModel):
    """
    Verdict returned by the completion service for one task.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    complete: bool = Field(description="Whether the task is observably complete.")
    next_state: ExecutionTaskState = Field(description="Target lifecycle state for the task.")
    reason: str = Field(description="Actionable reason for the verdict.")
    missing: List[CompletionEvidence] = Field(
        default_factory=list,
        description="Evidence dimensions required for completion but not observed.",
    )
