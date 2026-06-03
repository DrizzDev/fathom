from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Final, Literal, Mapping, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.artifact import ArtifactDirectory, ArtifactQueue
from fathom.schemas.actions import Action
from fathom.schemas.completion import CompletionVerdict
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.results import ActionTraceAttempt
from fathom.schemas.screens import ScreenCapture


class ArtifactKind(StrEnum):
    """
    Categories of persistable artifacts produced during a run.
    """

    TRACE = "trace"
    SCRIPT = "script"
    ANNOTATED = "annotated"
    PERCEPTION = "perception"
    SCREENSHOT = "screenshot"
    VERIFICATION = "verification"
    HIERARCHY_XML = "hierarchy_xml"
    OCR_RAW = "ocr_raw"
    CV_PERCEPTION = "cv_perception"
    OCR_PERCEPTION = "ocr_perception"
    ICON_PERCEPTION = "icon_perception"
    VISION_PERCEPTION = "vision_perception"
    OVERLAY_PERCEPTION = "overlay_perception"


class ArtifactCategory:
    """
    Single source of truth mapping :class:`ArtifactKind` onto one of the five canonical asset directories.

    Used by :class:`SharedPathManager` for EFS path resolution and by :class:`CloudSink` for the storage category metadata
    so cloud and local writers cannot drift apart and corrupt files (e.g. routing XML bytes through the screenshot directory).
    """

    __MAPPING: Final[Mapping[ArtifactKind, str]] = {
        ArtifactKind.TRACE: ArtifactDirectory.TRACES,
        ArtifactKind.SCRIPT: ArtifactDirectory.HISTORY,
        ArtifactKind.HIERARCHY_XML: ArtifactDirectory.XMLS,
        ArtifactKind.OCR_RAW: ArtifactDirectory.XMLS,
        ArtifactKind.VERIFICATION: ArtifactDirectory.TRACES,
        ArtifactKind.ANNOTATED: ArtifactDirectory.ANNOTATED,
        ArtifactKind.PERCEPTION: ArtifactDirectory.ANNOTATED,
        ArtifactKind.SCREENSHOT: ArtifactDirectory.SCREENSHOT,
        ArtifactKind.CV_PERCEPTION: ArtifactDirectory.ANNOTATED,
        ArtifactKind.OCR_PERCEPTION: ArtifactDirectory.ANNOTATED,
        ArtifactKind.ICON_PERCEPTION: ArtifactDirectory.ANNOTATED,
        ArtifactKind.VISION_PERCEPTION: ArtifactDirectory.ANNOTATED,
        ArtifactKind.OVERLAY_PERCEPTION: ArtifactDirectory.ANNOTATED,
    }

    @classmethod
    def for_(cls, *, kind: ArtifactKind) -> str:
        """
        Resolve the canonical asset directory for one :class:`ArtifactKind`.
        """

        return cls.__MAPPING[kind]


class QueueConfiguration(BaseModel):
    """
    Bounded-queue tuning for :class:`ArtifactPipeline` background work.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    capacity: int = Field(
        gt=0,
        default=ArtifactQueue.CAPACITY,
        description="Maximum in-flight background uploads before back-pressure.",
    )
    drain_timeout: float = Field(
        gt=0.0,
        default=ArtifactQueue.DRAIN_TIMEOUT_SECONDS,
        description="Seconds the pipeline waits at shutdown before giving up.",
    )


class LocalArtifactPolicy(BaseModel):
    """
    Retention policy for the EFS-staged artifact copy.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cleanup: bool = Field(
        default=True,
        description=(
            "Whether the pipeline unlinks the EFS-staged file after the sink "
            "acknowledges cleanup. Hosts that read the staged path after the "
            "sink ack (e.g. enricher's healing bridge queueing a delayed "
            "Cloudinary upload) disable this and own a fallback sweep."
        ),
    )


class PipelineConfiguration(BaseModel):
    """
    Top-level :class:`ArtifactPipeline` configuration aggregate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    queue: QueueConfiguration = Field(
        default_factory=QueueConfiguration,
        description="Bounded-queue tuning for the background upload worker.",
    )
    local: LocalArtifactPolicy = Field(
        default_factory=LocalArtifactPolicy,
        description="Retention policy for the EFS-staged artifact copy.",
    )


class ScreenshotPayload(BaseModel):
    """
    Raw device capture artifact — written as-is by the pipeline.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[ArtifactKind.SCREENSHOT] = Field(
        default=ArtifactKind.SCREENSHOT,
        description="Discriminator value routing the record to the passthrough renderer.",
    )
    capture: ScreenCapture = Field(description="Raw screenshot captured from the device.")


class AnnotatedPayload(BaseModel):
    """
    XML-annotated image (LLM-facing) — already rendered upstream.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[ArtifactKind.ANNOTATED] = Field(
        default=ArtifactKind.ANNOTATED,
        description="Discriminator value routing the record to the passthrough renderer.",
    )
    capture: ScreenCapture = Field(description="Screen capture with manifest annotations applied.")


class PerceptionPayload(BaseModel):
    """
    Merged-perception debug image — rendered from the observation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[ArtifactKind.PERCEPTION] = Field(
        default=ArtifactKind.PERCEPTION,
        description="Discriminator value routing the record to the perception renderer.",
    )
    capture: ScreenCapture = Field(description="Capture used as the rendering canvas.")
    observation: ScreenObservation = Field(
        description="Merged OCR/icon/overlay/ensemble elements drawn over the canvas.",
    )


class OcrPerceptionPayload(BaseModel):
    """
    OCR-only debug image — same shape as :class:`PerceptionPayload`
    but routed to a renderer that filters the observation to elements whose ``source`` is :attr:`ElementSource.OCR`.

    Kept as a distinct payload (rather than a flag on PerceptionPayload) so the renderer registry stays a flat strategy map keyed by kind.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[ArtifactKind.OCR_PERCEPTION] = Field(
        default=ArtifactKind.OCR_PERCEPTION,
        description="Discriminator value routing the record to the OCR-only renderer.",
    )
    capture: ScreenCapture = Field(description="Capture used as the rendering canvas.")
    observation: ScreenObservation = Field(
        description="Observation; the renderer projects to OCR-source elements only.",
    )


class CvPerceptionPayload(BaseModel):
    """
    CV-only debug image projecting :attr:`ElementSource.CV` elements only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[ArtifactKind.CV_PERCEPTION] = Field(
        default=ArtifactKind.CV_PERCEPTION,
        description="Discriminator routing the record to the CV-only renderer.",
    )
    capture: ScreenCapture = Field(description="Capture used as the rendering canvas.")
    observation: ScreenObservation = Field(
        description="Observation; the renderer projects to CV-source elements only.",
    )


class IconPerceptionPayload(BaseModel):
    """
    Icon-only debug image projecting :attr:`ElementSource.ICON` elements only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[ArtifactKind.ICON_PERCEPTION] = Field(
        default=ArtifactKind.ICON_PERCEPTION,
        description="Discriminator routing the record to the icon-only renderer.",
    )
    capture: ScreenCapture = Field(description="Capture used as the rendering canvas.")
    observation: ScreenObservation = Field(
        description="Observation; the renderer projects to ICON-source elements only.",
    )


class VisionPerceptionPayload(BaseModel):
    """
    Vision-only debug image projecting :attr:`ElementSource.VISION` elements only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[ArtifactKind.VISION_PERCEPTION] = Field(
        default=ArtifactKind.VISION_PERCEPTION,
        description="Discriminator routing the record to the vision-only renderer.",
    )
    capture: ScreenCapture = Field(description="Capture used as the rendering canvas.")
    observation: ScreenObservation = Field(
        description="Observation; the renderer projects to VISION-source elements only.",
    )


class OverlayPerceptionPayload(BaseModel):
    """
    Overlay-only debug image projecting :class:`OverlayObservation` rectangles.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[ArtifactKind.OVERLAY_PERCEPTION] = Field(
        default=ArtifactKind.OVERLAY_PERCEPTION,
        description="Discriminator routing the record to the overlay-only renderer.",
    )
    capture: ScreenCapture = Field(description="Capture used as the rendering canvas.")
    observation: ScreenObservation = Field(
        description="Observation; the renderer projects to overlay rectangles only.",
    )


class TracePayload(BaseModel):
    """
    Post-action trace image — action coordinates drawn on pre-capture.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[ArtifactKind.TRACE] = Field(
        default=ArtifactKind.TRACE,
        description="Discriminator value routing the record to the trace renderer.",
    )
    capture: ScreenCapture = Field(
        description="Pre-action screenshot the trace overlay is drawn on."
    )
    coords: Tuple[int, ...] = Field(description="Action coordinates drawn onto the trace image.")
    action: Action = Field(description="Action whose execution this trace records.")
    attempt: ActionTraceAttempt | None = Field(
        default=None,
        description="Attempt metadata when this trace belongs to a multi-attempt device action.",
    )


class VerificationPayload(BaseModel):
    """
    Verifier-stage artifact — verdict overlay drawn on the inspected capture.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[ArtifactKind.VERIFICATION] = Field(
        default=ArtifactKind.VERIFICATION,
        description="Discriminator value routing the record to the verification renderer.",
    )
    capture: ScreenCapture = Field(description="Capture the verifier inspected.")
    verdict: CompletionVerdict = Field(description="Verifier verdict overlaid on the artifact.")


class HierarchyXmlPayload(BaseModel):
    """
    XML hierarchy dump — UTF-8 text written verbatim by the pipeline.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[ArtifactKind.HIERARCHY_XML] = Field(
        default=ArtifactKind.HIERARCHY_XML,
        description="Discriminator value routing the record to the passthrough renderer.",
    )
    content: str = Field(min_length=1, description="Raw XML hierarchy text.")


class OcrRawPayload(BaseModel):
    """
    Raw OCR provider response — UTF-8 JSON written verbatim.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[ArtifactKind.OCR_RAW] = Field(
        default=ArtifactKind.OCR_RAW,
        description="Discriminator value routing the record to the passthrough renderer.",
    )
    content: str = Field(min_length=1, description="Raw OCR provider response JSON.")


class ScriptPayload(BaseModel):
    """
    Generated automation script — UTF-8 text written verbatim.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[ArtifactKind.SCRIPT] = Field(
        default=ArtifactKind.SCRIPT,
        description="Discriminator value routing the record to the passthrough renderer.",
    )
    content: str = Field(min_length=1, description="Final automation script source.")


ArtifactPayload = Annotated[
    Union[
        TracePayload,
        ScriptPayload,
        AnnotatedPayload,
        PerceptionPayload,
        ScreenshotPayload,
        VerificationPayload,
        HierarchyXmlPayload,
        OcrRawPayload,
        CvPerceptionPayload,
        OcrPerceptionPayload,
        IconPerceptionPayload,
        VisionPerceptionPayload,
        OverlayPerceptionPayload,
    ],
    Field(discriminator="kind", description="Discriminated union over every artifact payload."),
]


class ArtifactMetadata(BaseModel):
    """
    Identity-and-routing fields persisted to the EFS sidecar JSON.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str = Field(min_length=1, description="Workflow / session identifier.")
    kind: ArtifactKind = Field(description="Artifact category routing the record to its sink path.")

    step_number: int = Field(ge=0, description="Zero-based step index inside the run.")
    package_name: str = Field(min_length=1, description="Active package the record belongs to.")

    created: int = Field(ge=0, description="Epoch milliseconds at emit time.")


class ArtifactRecord(BaseModel):
    """
    Emit-time record carrying the typed payload the renderer consumes.

    Producers build this at the lifecycle seam. The pipeline derives an :class:`ArtifactMetadata` slice from it for sink persistence
    and sidecar storage; bytes-heavy payload fields never travel to disk twice.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str = Field(min_length=1, description="Workflow / session identifier.")
    step_number: int = Field(ge=0, description="Zero-based step index inside the run.")
    package_name: str = Field(min_length=1, description="Active package the record belongs to.")

    created: int = Field(ge=0, description="Epoch milliseconds at emit time.")
    payload: ArtifactPayload = Field(description="Typed payload describing the artifact contents.")

    def metadata(self) -> ArtifactMetadata:
        """
        Project this record onto its sink-facing identity slice.
        """

        return ArtifactMetadata(
            created=self.created,
            kind=self.payload.kind,
            session_id=self.session_id,
            step_number=self.step_number,
            package_name=self.package_name,
        )


class ArtifactReceipt(BaseModel):
    """
    Outcome returned by every :class:`ArtifactSinkPort.persist` call.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    identifier: str = Field(
        description="Stable artifact identifier (cloud URL or local path).",
    )
    local_cleanup: bool = Field(
        description="Whether the pipeline should unlink the staged EFS files on success.",
    )
