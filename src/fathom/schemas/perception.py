from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from fathom.schemas.localization import EnsembleMemberName


class DocumentAiCredentials(BaseModel):
    """
    Provider credentials for the Document AI OCR adapter loaded from env vars.

    ``credentials`` carries the same service-account material that the
    Gemini adapter consumes (either an inline dict from
    ``GOOGLE_APPLICATION_CREDENTIALS_JSON`` or a file path from
    ``GOOGLE_APPLICATION_CREDENTIALS``). The adapter materializes it
    into a :class:`google.oauth2.service_account.Credentials` instance
    so Document AI authenticates against the same identity as Gemini —
    no reliance on ambient ``gcloud`` ADC or GCE metadata.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    project: str = Field(min_length=1, description="GCP project identifier.")
    location: str = Field(min_length=1, description="Document AI location id.")
    processor: str = Field(min_length=1, description="Document AI processor id.")
    credentials: Optional[Union[Dict[str, Any], str]] = Field(
        default=None,
        description=(
            "Service-account credentials: inline JSON dict or absolute "
            "path to the key file. None falls back to library defaults."
        ),
    )


class OcrConfiguration(BaseModel):
    """
    Boot-time configuration for the OCR subsystem.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Whether OCR enrichment is active for this run.",
    )
    document_ai: Optional[DocumentAiCredentials] = Field(
        default=None,
        description="Document AI credentials when the provider is enabled.",
    )


class CvConfiguration(BaseModel):
    """
    Boot-time configuration for the OpenCV visual-control labeler.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(
        default=False,
        description=(
            "Whether the OpenCV visual-control labeler runs alongside the XML "
            "manifest. Off by default — the labeler often duplicates real icons "
            "with anonymous 'VisualControl' boxes that confuse the LLM."
        ),
    )


class IconConfiguration(BaseModel):
    """
    Boot-time configuration for the icon-template detector.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(
        default=False,
        description=(
            "Whether the icon-template detector runs in the observation pipeline. "
            "Useful only when the templates registry has entries for the target app."
        ),
    )


class OverlayConfiguration(BaseModel):
    """
    Boot-time configuration for the pixel-overlay detector.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(
        default=False,
        description=(
            "Whether the pixel-overlay detector runs in the observation pipeline. "
            "Detects modal dimming layers when no element-level overlay is present."
        ),
    )


class KeyboardConfiguration(BaseModel):
    """
    Boot-time configuration for keyboard detection.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(
        default=False,
        description=(
            "Whether keyboard detection contributes to perception output for this run. "
            "Execution policy decisions belong to runtime configuration, not perception."
        ),
    )


class JournalConfiguration(BaseModel):
    """
    Boot-time configuration for the runtime-journal adapter.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    local_enabled: bool = Field(
        default=False,
        description="Whether the local JSONL runtime-journal adapter is active.",
    )


class PerceptionConfiguration(BaseModel):
    """
    Boot-time configuration for the perception layer (OCR, CV, icon, overlay).

    Every perception subsystem is opt-in. When all four flags are False
    the runtime falls back to the original XML+LLM-only flow: drawer
    extracts elements from the platform XML, LLM grounds against that
    manifest, snap-to-label resolves coordinates. Each ``enabled`` flag
    turns its subsystem into a fallback contributor that augments the
    manifest when the primary path has insufficient evidence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ocr: OcrConfiguration = Field(
        default_factory=OcrConfiguration,
        description="OCR subsystem configuration.",
    )
    cv: CvConfiguration = Field(
        default_factory=CvConfiguration,
        description="OpenCV visual-control labeler configuration.",
    )
    icon: IconConfiguration = Field(
        default_factory=IconConfiguration,
        description="Icon-template detector configuration.",
    )
    overlay: OverlayConfiguration = Field(
        default_factory=OverlayConfiguration,
        description="Pixel-overlay detector configuration.",
    )
    keyboard: KeyboardConfiguration = Field(
        default_factory=KeyboardConfiguration,
        description="Keyboard detection configuration.",
    )
    journal: JournalConfiguration = Field(
        default_factory=JournalConfiguration,
        description="Runtime-journal adapter configuration.",
    )


class LocalizationEnsembleConfiguration(BaseModel):
    """
    Boot-time configuration for the ensemble vision-localizer layer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Whether the ensemble vision localizer is active for this run.",
    )
    members: Tuple[EnsembleMemberName, ...] = Field(
        default_factory=tuple,
        description="Ordered tuple of enabled member names.",
    )
