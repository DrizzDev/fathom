from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.storage import StorageBackend


class ScreenArtifact(BaseModel):
    """
    Persisted reference to one screen image.
    """

    model_config = ConfigDict(frozen=True)

    uri: str = Field(description="Identifier returned by the storage adapter")
    storage_backend: StorageBackend = Field(
        default=StorageBackend.LOCAL,
        description="Canonical storage backend that produced the URI",
    )
    captured_at: Optional[int] = Field(
        default=None, description="Capture timestamp in milliseconds"
    )
    visual_hash: Optional[str] = Field(
        default=None, description="Perceptual hash of the captured image when available"
    )
    width: Optional[int] = Field(default=None, description="Screen width in pixels")
    height: Optional[int] = Field(default=None, description="Screen height in pixels")
    mime_type: str = Field(default="image/png", description="MIME type of the artifact")


class ScreenArtifactBundle(BaseModel):
    """
    Pre-action and post-action screen artifacts for one step.
    """

    model_config = ConfigDict(frozen=True)

    before: Optional[ScreenArtifact] = Field(default=None, description="Pre-action screen artifact")
    after: Optional[ScreenArtifact] = Field(default=None, description="Post-action screen artifact")


class StepArtifacts(BaseModel):
    """
    Namespaced artifact references produced by one step.
    """

    model_config = ConfigDict(frozen=True)

    screen: Optional[ScreenArtifactBundle] = Field(
        default=None, description="Before/after screen captures for the step"
    )
