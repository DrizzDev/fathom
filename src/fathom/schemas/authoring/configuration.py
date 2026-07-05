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


class AuthoringArtifactConfiguration(SealedModel):
    """
    Configuration for artifact payloads attached to authoring model requests.
    """

    include_images: bool = Field(
        default=True,
        description="Whether screenshot and trace image artifacts are attached to authoring requests.",
    )
    include_manifests: bool = Field(
        default=True,
        description="Whether manifest or UI-tree artifacts are embedded in authoring requests.",
    )
    include_text: bool = Field(
        default=True,
        description="Whether text artifacts such as logs or OCR output are embedded in authoring requests.",
    )

    max_images: int = Field(
        ge=0,
        default=2,
        description="Maximum image artifacts attached to one authoring request.",
    )
    max_text_artifacts: int = Field(
        ge=0,
        default=3,
        description="Maximum text or manifest artifacts embedded in one authoring request.",
    )
    max_text_characters: int = Field(
        ge=0,
        default=8000,
        description="Maximum characters read from each text or manifest artifact.",
    )


class AuthoringConfiguration(SealedModel):
    """
    Configuration for script authoring modes.
    """

    attempts: int = Field(
        ge=1,
        default=3,
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
    artifacts: AuthoringArtifactConfiguration = Field(
        default_factory=AuthoringArtifactConfiguration,
        description="Artifact payload limits for authoring model requests.",
    )
