from __future__ import annotations

from typing import FrozenSet

from pydantic import Field

from fathom.constants import ActionType
from fathom.constants.capability import (
    CompletionMode,
    ExecutionChannel,
    PayloadField,
    RecordMode,
    RetryMode,
    TargetRequirement,
)
from fathom.schemas.base import SealedModel


class CommandContract(SealedModel):
    """
    The structured inputs a command requires to be grounded and validated.
    """

    target: TargetRequirement = Field(
        default=TargetRequirement.NONE, description="On-screen target the command must ground to."
    )
    required: FrozenSet[PayloadField] = Field(
        default_factory=frozenset, description="Mandatory structured payload fields."
    )


class CommandProfile(SealedModel):
    """
    The execution-side capability declaration for one command.
    """

    channel: ExecutionChannel = Field(description="How the command reaches the world.")
    completion: CompletionMode = Field(
        description="How a sub-goal directed by this command completes."
    )
    records: RecordMode = Field(
        description="Persisted event category for the command's step record."
    )
    retry: RetryMode = Field(description="Retry policy that wraps the command's execution.")
    contract: CommandContract = Field(
        default_factory=CommandContract, description="Required grounding and payload contract."
    )


class CommandAvailabilityConfig(SealedModel):
    """
    Typed availability configuration selecting which commands are enabled for a runtime.
    """

    disabled: FrozenSet[ActionType] = Field(
        default_factory=frozenset, description="Commands switched off for this runtime."
    )
