from __future__ import annotations

from typing import Optional

from pydantic import Field

from fathom.constants.authoring import AuthoringKind, AuthoringStatus
from fathom.schemas.authoring.artifact import AuthoringArtifact
from fathom.schemas.base import SealedModel


class AuthoringDraft(SealedModel):
    """
    Persisted result of a step or run authoring task.
    """

    execution_id: str = Field(min_length=1, description="Execution the draft belongs to.")

    status: AuthoringStatus = Field(description="Authoring outcome for this draft.")
    kind: AuthoringKind = Field(description="Authoring task kind that produced the draft.")

    artifact: Optional[AuthoringArtifact] = Field(
        default=None, description="Generated artifact when authoring succeeded."
    )
    reason: Optional[str] = Field(default=None, description="Skip or failure reason.")
    step_index: Optional[int] = Field(
        default=None, ge=0, description="Evidence step index for step-scoped drafts."
    )

    @property
    def generated(self) -> bool:
        """
        Return whether the draft carries generated content.
        """

        return self.status is AuthoringStatus.GENERATED and self.artifact is not None
