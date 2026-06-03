from __future__ import annotations

from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.storage import StorageBackend


class ScreenArtifact(BaseModel):
    """
    Reference to one screen image carried back to the caller.

    Either ``uri`` or ``image`` should be set: ``uri`` when the bytes
    live behind a storage backend the consumer can dereference, and
    ``image`` when the producer hands the raw bytes through directly.
    ``image`` is excluded from the default serialization so telemetry
    and JSON dumps stay small; in-process consumers read it off the
    Pydantic instance.
    """

    model_config = ConfigDict(frozen=True)

    uri: Optional[str] = Field(
        default=None,
        description="Stable identifier returned by the storage adapter when available",
    )
    image: Optional[bytes] = Field(
        repr=False,
        default=None,
        exclude=True,
        description="Raw image bytes for in-process consumers",
    )
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
    Pre-action, post-action, annotated, and gesture-trace screen artifacts for one step.
    """

    model_config = ConfigDict(frozen=True)

    before: Optional[ScreenArtifact] = Field(default=None, description="Pre-action screen artifact")
    after: Optional[ScreenArtifact] = Field(default=None, description="Post-action screen artifact")

    annotated: Optional[ScreenArtifact] = Field(
        default=None, description="XML-annotated screen artifact rendered during grounding"
    )
    traces: Optional[Tuple[ScreenArtifact, ...]] = Field(
        default=None,
        description="Rendered gesture-trace artifacts; one per dispatched attempt. None when no trace path ran.",
    )


class StepArtifacts(BaseModel):
    """
    Namespaced artifact references produced by one step.
    """

    model_config = ConfigDict(frozen=True)

    screen: Optional[ScreenArtifactBundle] = Field(
        default=None,
        description="Before, after, annotated, and gesture-trace screen captures for the step",
    )
