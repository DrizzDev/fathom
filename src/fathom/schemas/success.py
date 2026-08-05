from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import Field, model_validator

from fathom.constants.success import SuccessKind
from fathom.schemas.base.common import NonBlank, SealedModel
from fathom.schemas.capture import CaptureIdentity
from fathom.schemas.requirement import CommandRequirement


class ObservationRequirement(SealedModel):
    """
    An observable screen state a turn must confirm; carries no UI tactic.
    """

    assertion: NonBlank = Field(
        description="Observable state that must hold; carries no action verb or UI tactic."
    )


class SourceLocation(SealedModel):
    """
    Character offsets of a cited span within the original intent.
    """

    start: int = Field(ge=0, description="Inclusive start offset of the cited span in the intent.")
    end: int = Field(gt=0, description="Exclusive end offset of the cited span in the intent.")

    @model_validator(mode="after")
    def __ordered(self) -> "SourceLocation":
        """
        Reject a location whose end does not lie after its start.
        """

        if self.end <= self.start:
            raise ValueError("Source location end must lie after its start.")

        return self


class SourceSpan(SealedModel):
    """
    A verbatim intent span with its exact location within the original intent.
    """

    quote: NonBlank = Field(description="Verbatim intent text that names the requested operation.")
    location: SourceLocation = Field(description="Offsets of the quote within the original intent.")

    @model_validator(mode="after")
    def __consistent(self) -> "SourceSpan":
        """
        Keep the quote length consistent with its span; the binder proves it against the intent.
        """

        if self.location.end - self.location.start != len(self.quote):
            raise ValueError("Source span length must equal its quote length.")

        return self


class ObservedSuccess(SealedModel):
    """
    Success defined by an observable screen state, never by an implied UI tactic.
    """

    kind: Literal[SuccessKind.OBSERVED] = Field(
        default=SuccessKind.OBSERVED, description="Discriminator for the observed variant."
    )
    observation: ObservationRequirement = Field(
        description="Observable state that defines completion."
    )


class CommandSuccess(SealedModel):
    """
    Success defined by an explicit user-requested primitive with cited provenance.
    """

    kind: Literal[SuccessKind.COMMAND] = Field(
        default=SuccessKind.COMMAND, description="Discriminator for the command variant."
    )
    requirement: CommandRequirement = Field(
        description="Canonical user-requested operation and its typed parameters."
    )
    source: Optional[SourceSpan] = Field(
        default=None,
        description="Diagnostic intent span that cited the primitive; provenance only, never authority.",
    )
    postcondition: Optional[ObservationRequirement] = Field(
        default=None, description="Optional observable postcondition also required for completion."
    )


class CaptureSuccess(SealedModel):
    """
    Success defined by an exact capture identity.
    """

    kind: Literal[SuccessKind.CAPTURE] = Field(
        default=SuccessKind.CAPTURE, description="Discriminator for the capture variant."
    )
    target: CaptureIdentity = Field(
        description="Capture identity; the variable name is the sole identity key."
    )
    subject: NonBlank = Field(
        description="Descriptive context of what is captured; never an identity key."
    )


Success = Annotated[
    Union[ObservedSuccess, CommandSuccess, CaptureSuccess],
    Field(discriminator="kind", description="Exactly one typed definition of a sub-goal's success."),
]
