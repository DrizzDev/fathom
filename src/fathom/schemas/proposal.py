from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import Field

from fathom.constants.command import CommandBindingOutcome, CommandRejection
from fathom.constants.success import CaptureNameProvenance, SuccessKind
from fathom.schemas.base.common import NonBlank, SealedModel
from fathom.schemas.requirement import (
    NavigationRequirement,
    PressRequirement,
    ScrollRequirement,
    SwipeRequirement,
    TypeRequirement,
    WaitRequirement,
)
from fathom.schemas.success import CommandSuccess, ObservationRequirement

# Non-discriminated requirement union for the untrusted LLM proposal schema. The operation
# Literal keeps parsing unambiguous, and this emits JSON-Schema ``anyOf`` (which Gemini's
# structured-output schema accepts) rather than the ``oneOf``+``discriminator`` a discriminated
# union produces (which Gemini rejects). The canonical discriminated ``CommandRequirement`` is
# still used everywhere trusted (``Success``).
ProposalRequirement = Union[
    PressRequirement,
    TypeRequirement,
    ScrollRequirement,
    SwipeRequirement,
    WaitRequirement,
    NavigationRequirement,
]


class ObservedProposal(SealedModel):
    """
    Untrusted decomposition proposal for an observable outcome.
    """

    kind: Literal[SuccessKind.OBSERVED] = Field(
        default=SuccessKind.OBSERVED, description="Discriminator for the observed proposal."
    )
    assertion: NonBlank = Field(description="Observable state the model proposes as the objective.")


class CommandProposal(SealedModel):
    """
    Untrusted decomposition proposal for a command: its requirement and cited quote, unbound.
    """

    kind: Literal[SuccessKind.COMMAND] = Field(
        default=SuccessKind.COMMAND, description="Discriminator for the command proposal."
    )
    requirement: ProposalRequirement = Field(
        description="Model-proposed canonical operation and parameters."
    )
    quote: NonBlank = Field(description="Exact intent text the model cited for this command.")
    postcondition: Optional[ObservationRequirement] = Field(
        default=None, description="Optional observable postcondition also required for completion."
    )


class CaptureProposal(SealedModel):
    """
    Untrusted decomposition proposal for a capture: its subject, name, and the name's provenance.
    """

    kind: Literal[SuccessKind.CAPTURE] = Field(
        default=SuccessKind.CAPTURE, description="Discriminator for the capture proposal."
    )
    subject: NonBlank = Field(description="What the model proposes to capture.")
    name: NonBlank = Field(
        description="Capture variable name: the user's exact name, or a model-proposed identifier."
    )
    provenance: CaptureNameProvenance = Field(
        description="USER when the intent named the variable, MODEL when the model proposed it."
    )


# Non-discriminated so the JSON schema emits ``anyOf`` for Gemini structured output; the ``kind``
# Literal on each variant keeps validation unambiguous. Canonical ``Success`` stays discriminated.
DecompositionProposal = Union[ObservedProposal, CommandProposal, CaptureProposal]


class BoundCommand(SealedModel):
    """
    A command proposal bound to a canonical CommandSuccess by trusted provenance.
    """

    outcome: Literal[CommandBindingOutcome.BOUND] = Field(
        default=CommandBindingOutcome.BOUND, description="Discriminator for a bound result."
    )
    success: CommandSuccess = Field(description="The canonical, source-bound command success.")


class RejectedCommand(SealedModel):
    """
    A command proposal that failed structural provenance or catalog admission.
    """

    outcome: Literal[CommandBindingOutcome.REJECTED] = Field(
        default=CommandBindingOutcome.REJECTED, description="Discriminator for a rejected result."
    )
    reason: CommandRejection = Field(description="Why the proposal was rejected.")


CommandBinding = Annotated[
    Union[BoundCommand, RejectedCommand],
    Field(discriminator="outcome", description="Typed outcome of binding a command proposal."),
]
