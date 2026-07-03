from __future__ import annotations

from pydantic import Field

from fathom.constants.authoring import AuthoringMode
from fathom.schemas.base import SealedModel


class StepAuthoringConfiguration(SealedModel):
    """
    Configuration for per-step rich authoring.
    """

    mode: AuthoringMode = Field(
        default=AuthoringMode.DISABLED,
        description="Whether per-step rich authoring is disabled, asynchronous, or synchronous.",
    )


class RunConfiguration(SealedModel):
    """
    Configuration for whole-run authoring.
    """

    enabled: bool = Field(
        default=True,
        description="Whether whole-run authoring is attempted before baseline fallback.",
    )


class AuthoringConfiguration(SealedModel):
    """
    Configuration for script authoring modes.
    """

    attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum authoring attempts before falling back or failing.",
    )
    step: StepAuthoringConfiguration = Field(
        default_factory=StepAuthoringConfiguration,
        description="Per-step authoring configuration.",
    )
    run: RunConfiguration = Field(
        default_factory=RunConfiguration,
        description="Whole-run authoring configuration.",
    )
