from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fathom.constants.collaboration import JobState
from fathom.schemas.interaction import Outcome


class JobHandlerResult(BaseModel):
    """
    Terminal result returned by a job handler after executing one claimed job.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: JobState = Field(description="Terminal state to persist for the job.")
    outcome: Outcome = Field(description="Machine-readable job outcome.")

    @model_validator(mode="after")
    def require_terminal_state(self) -> JobHandlerResult:
        """
        Require handlers to return a terminal job state.
        """

        if self.state not in (JobState.COMPLETED, JobState.FAILED, JobState.ABANDONED):
            raise ValueError("Job handler result state must be terminal.")

        return self
