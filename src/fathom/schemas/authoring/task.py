from __future__ import annotations

from typing import Optional

from pydantic import Field, model_validator

from fathom.constants.authoring import AuthoringKind, AuthoringStatus
from fathom.constants.dialect import DialectName
from fathom.schemas.authoring.artifact import AuthoringArtifact
from fathom.schemas.authoring.evidence import AuthoringEvidence
from fathom.schemas.base import SealedModel
from fathom.schemas.flow import Flow, Report


class AuthoringTask(SealedModel):
    """
    One script-authoring request handled by the single AuthoringAgent.
    """

    kind: AuthoringKind = Field(description="Authoring mode for this task.")
    execution_id: str = Field(min_length=1, description="Execution being authored.")

    intent: str = Field(min_length=1, description="User intent for the run.")
    step_number: int = Field(ge=0, description="Current execution step count.")

    dialect: DialectName = Field(default=DialectName.DRIZZ, description="Target script dialect.")
    evidence: AuthoringEvidence = Field(description="Evidence available to the authoring agent.")

    draft: Optional[Flow] = Field(default=None, description="Existing flow draft when available.")
    review: Optional[Report] = Field(default=None, description="Existing review when available.")

    @model_validator(mode="after")
    def __evidence_matches_kind(self) -> "AuthoringTask":
        """
        Ensure the task kind matches the supplied evidence view.
        """

        if self.kind is AuthoringKind.RUN and self.evidence.run is None:
            raise ValueError("RUN authoring requires run evidence.")

        if self.kind is AuthoringKind.STEP and self.evidence.step is None:
            raise ValueError("STEP authoring requires step evidence.")

        if self.kind is AuthoringKind.REPAIR and self.evidence.repair is None:
            raise ValueError("REPAIR authoring requires repair evidence.")

        return self


class AuthoringResponse(SealedModel):
    """
    Outcome of an authoring task before dialect/finalization selection.
    """

    artifact: Optional[AuthoringArtifact] = Field(
        default=None, description="Authored artifact when generated."
    )
    reason: Optional[str] = Field(default=None, description="Why authoring skipped or failed.")
    status: AuthoringStatus = Field(
        description="Whether authoring produced a script, skipped, or failed."
    )

    @property
    def script(self) -> Optional[str]:
        """
        Return rendered text for callers that still consume script text.
        """

        if self.artifact is None:
            return None

        return self.artifact.content

    @property
    def has_script(self) -> bool:
        """
        Whether the response carries generated artifact content.
        """

        return self.artifact is not None
