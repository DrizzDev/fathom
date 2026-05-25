from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from pathlib import Path

from fathom.adapters.artifact.cloud import CloudSink
from fathom.adapters.artifact.efs import EfsSink
from fathom.adapters.icon.noop import NoopIconDetector
from fathom.adapters.icon.template import TemplateIconDetector
from fathom.adapters.journal.jsonl import JsonRuntimeJournal
from fathom.adapters.journal.noop import NoopRuntimeJournal
from fathom.adapters.localization.document import DocumentAiLayoutLocalizer
from fathom.adapters.localization.gemini import GeminiVisionLocalizer
from fathom.adapters.ocr.document import DocumentAiOcr
from fathom.adapters.ocr.noop import NoopOcr
from fathom.adapters.perception.overlay.noop import NoopOverlayDetector
from fathom.adapters.perception.overlay.pixel import PixelOverlayDetector
from fathom.adapters.storage.cloud import CloudStorage
from fathom.base.paths import SharedPathManager
from fathom.constants.artifact import StorageBackend
from fathom.core.artifact.drawing import BoxDrawer
from fathom.core.artifact.pipeline import ArtifactPipeline
from fathom.core.artifact.renderer import (
    OverlayPerceptionRenderer,
    PassthroughRenderer,
    PerceptionRenderer,
    SourceFilteredPerceptionRenderer,
    TraceRenderer,
    VerificationRenderer,
)
from fathom.core.config.loader import RuntimeConfigLoader
from fathom.core.localization.ensemble import EnsembleLocalizerService
from fathom.infrastructure.storage.cloud import GCSImageStorage
from fathom.interfaces.artifact import ArtifactRendererPort, ArtifactSinkPort
from fathom.interfaces.icon import IconDetectorPort
from fathom.interfaces.journal import RuntimeJournalPort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.localization import TargetLocalizerPort
from fathom.interfaces.ocr import OcrPort
from fathom.interfaces.overlay import OverlayDetectorPort
from fathom.interfaces.storage import StoragePort
from fathom.schemas.artifact import ArtifactKind, PipelineConfig
from fathom.schemas.configuration import StorageConfiguration
from fathom.schemas.localization import EnsembleMemberName
from fathom.schemas.observation import ElementSource
from fathom.schemas.ocr import DocumentAiConfiguration
from fathom.schemas.perception import PerceptionConfiguration

logger = getLogger(__name__)


class AdapterAssembly:
    """
    Composition root that instantiates the right adapter mix per run.

    Lives in :mod:`fathom.runtime` because it imports from ``adapters/``
    and is the only place those concrete classes meet — domain code in
    ``core/`` depends only on the ports under ``interfaces/`` and never
    on the adapters directly.
    """

    def __init__(
        self,
        *,
        loader: RuntimeConfigLoader,
        llm: LLMPort,
        workflow_id: str,
        journal_directory: Optional[Path] = None,
    ) -> None:
        """
        Initialize the assembly with config loader, LLM port, run id, and journal output dir.
        """

        self.__loader = loader
        self.__llm = llm
        self.__workflow_id = workflow_id
        self.__journal_directory = journal_directory
        self.__perception = loader.perception()
        self.__localization = loader.localization()

    @property
    def perception_configuration(self) -> PerceptionConfiguration:
        """
        Expose the loaded perception configuration to downstream wiring.

        Strategies that need to thread perception toggles into domain
        services (e.g. ``cv_enabled`` for :class:`HierarchyService`)
        read this rather than rebuilding the configuration from the
        loader.
        """

        return self.__perception

    def ocr(self) -> OcrPort:
        """
        Build the OCR adapter (Document AI when enabled + credentialed, otherwise noop).
        """

        if not self.__perception.ocr.enabled:
            return NoopOcr()

        if (credentials := self.__perception.ocr.document_ai) is None:
            logger.warning(
                "OCR requested but Document AI credentials missing; falling back to noop",
                extra={
                    **self.__log_context(),
                    "event": "factory.ocr.fallback",
                },
            )
            return NoopOcr()

        configuration = DocumentAiConfiguration(
            project=credentials.project,
            location=credentials.location,
            processor=credentials.processor,
            credentials=credentials.credentials,
        )
        logger.info(
            "Document AI OCR adapter assembled",
            extra={
                **self.__log_context(),
                "event": "factory.ocr.document_ai.assembled",
                "processor": credentials.processor,
                "location": credentials.location,
            },
        )
        return DocumentAiOcr(configuration=configuration, workflow_id=self.__workflow_id)

    def icons(self) -> IconDetectorPort:
        """
        Build the icon detector when explicitly enabled; otherwise noop.

        The template registry ships empty until a future entry populates
        it, so a "default-on" icon detector adds latency without value.
        Gated behind :attr:`IconConfiguration.enabled` so the original
        XML+LLM flow runs without it.
        """

        if not self.__perception.icon.enabled:
            return NoopIconDetector()

        return TemplateIconDetector(workflow_id=self.__workflow_id)

    def overlay(self) -> OverlayDetectorPort:
        """
        Build the pixel overlay detector when explicitly enabled; otherwise noop.
        """

        if not self.__perception.overlay.enabled:
            return NoopOverlayDetector()

        return PixelOverlayDetector(workflow_id=self.__workflow_id)

    def ensemble(self) -> EnsembleLocalizerService:
        """
        Build the localization ensemble from the configured member set.
        """

        if not self.__localization.enabled or not self.__localization.members:
            return EnsembleLocalizerService(workflow_id=self.__workflow_id)

        members: List[TargetLocalizerPort] = []
        for name in self.__localization.members:
            if (member := self.__build_member(name=name)) is not None:
                members.append(member)

        logger.info(
            "Ensemble localizer assembled",
            extra={
                **self.__log_context(),
                "event": "factory.ensemble.assembled",
                "members.configured": [m.value for m in self.__localization.members],
                "members.active": [member.name for member in members],
            },
        )
        return EnsembleLocalizerService(
            members=tuple(members),
            workflow_id=self.__workflow_id,
        )

    def pipeline(
        self,
        *,
        path_manager: SharedPathManager,
        storage_configuration: StorageConfiguration,
    ) -> ArtifactPipeline:
        """
        Build the artifact pipeline wired against the configured backends.

        When ``"CLOUD"`` is enabled and a bucket is configured, the
        pipeline uploads to the cloud :class:`StoragePort` and clears
        the EFS-staged copy on success. Otherwise the pipeline retains
        the EFS-staged copy via :class:`EfsSink` — the local copy IS
        the durable artifact in that mode.
        """

        sink = self.__resolve_sink(configuration=storage_configuration)
        drawer = BoxDrawer()

        renderers: Dict[ArtifactKind, ArtifactRendererPort] = {
            ArtifactKind.SCREENSHOT: PassthroughRenderer(kind=ArtifactKind.SCREENSHOT),
            ArtifactKind.ANNOTATED: PassthroughRenderer(kind=ArtifactKind.ANNOTATED),
            ArtifactKind.HIERARCHY_XML: PassthroughRenderer(kind=ArtifactKind.HIERARCHY_XML),
            ArtifactKind.OCR_RAW: PassthroughRenderer(kind=ArtifactKind.OCR_RAW),
            ArtifactKind.SCRIPT: PassthroughRenderer(kind=ArtifactKind.SCRIPT),
            ArtifactKind.PERCEPTION: PerceptionRenderer(drawer=drawer),
            ArtifactKind.OCR_PERCEPTION: SourceFilteredPerceptionRenderer(
                kind=ArtifactKind.OCR_PERCEPTION,
                source=ElementSource.OCR,
                drawer=drawer,
            ),
            ArtifactKind.CV_PERCEPTION: SourceFilteredPerceptionRenderer(
                kind=ArtifactKind.CV_PERCEPTION,
                source=ElementSource.CV,
                drawer=drawer,
            ),
            ArtifactKind.ICON_PERCEPTION: SourceFilteredPerceptionRenderer(
                kind=ArtifactKind.ICON_PERCEPTION,
                source=ElementSource.ICON,
                drawer=drawer,
            ),
            ArtifactKind.VISION_PERCEPTION: SourceFilteredPerceptionRenderer(
                kind=ArtifactKind.VISION_PERCEPTION,
                source=ElementSource.VISION,
                drawer=drawer,
            ),
            ArtifactKind.OVERLAY_PERCEPTION: OverlayPerceptionRenderer(drawer=drawer),
            ArtifactKind.TRACE: TraceRenderer(),
            ArtifactKind.VERIFICATION: VerificationRenderer(),
        }

        logger.info(
            "Artifact pipeline assembled",
            extra={
                **self.__log_context(),
                "event": "factory.pipeline.assembled",
                "artifact.sink": type(sink).__name__,
            },
        )
        return ArtifactPipeline(
            config=PipelineConfig(),
            renderers=renderers,
            sink=sink,
            path_manager=path_manager,
            workflow_id=self.__workflow_id,
        )

    def journal(self) -> RuntimeJournalPort:
        """
        Build the runtime journal (local JSONL when configured, otherwise noop).
        """

        if not self.__perception.journal.local_enabled or self.__journal_directory is None:
            return NoopRuntimeJournal()

        path = self.__journal_directory / f"{self.__workflow_id}.jsonl"
        logger.info(
            "Local JSONL runtime journal assembled",
            extra={
                **self.__log_context(),
                "event": "factory.journal.jsonl.assembled",
                "journal.path": str(path),
            },
        )
        return JsonRuntimeJournal(path=path)

    def __build_member(self, *, name: EnsembleMemberName) -> Optional[TargetLocalizerPort]:
        """
        Build one ensemble member by its typed name.

        ``DOCUMENT_AI_LAYOUT`` is a downstream consumer of OCR tokens —
        it has nothing to localize against when OCR is disabled or
        unconfigured. Drop it at composition so the ensemble never
        carries a member that can only ever return ``None``.
        """

        if name == EnsembleMemberName.GEMINI_VISION:
            return GeminiVisionLocalizer(llm=self.__llm, workflow_id=self.__workflow_id)
        if name == EnsembleMemberName.DOCUMENT_AI_LAYOUT:
            if not self.__perception.ocr.enabled or self.__perception.ocr.document_ai is None:
                logger.info(
                    "Skipping document_ai_layout ensemble member: OCR not configured",
                    extra={
                        **self.__log_context(),
                        "event": "factory.ensemble.member.skipped",
                        "member": EnsembleMemberName.DOCUMENT_AI_LAYOUT.value,
                        "reason": "ocr.disabled.or.unconfigured",
                    },
                )
                return None
            return DocumentAiLayoutLocalizer(workflow_id=self.__workflow_id)
        return None

    def __resolve_sink(self, *, configuration: StorageConfiguration) -> ArtifactSinkPort:
        """
        Pick the artifact sink based on the configured storage backends.

        Cloud-enabled deployments upload via :class:`CloudSink` wrapping
        a fresh cloud-only :class:`StoragePort` — wrapping the runtime's
        composite storage instead would re-write the same local file the
        producer just wrote and race downstream readers.

        Local-only deployments fall back to :class:`EfsSink`, which
        keeps the EFS-staged copy as the durable artifact and reports
        no remote cleanup so the pipeline never deletes the local file.
        """

        if self.__cloud_enabled(configuration=configuration):
            return self.__cloud_sink(configuration=configuration)

        return EfsSink()

    @staticmethod
    def __cloud_enabled(*, configuration: StorageConfiguration) -> bool:
        """
        Whether the cloud backend is both selected and fully configured.
        """

        return StorageBackend.CLOUD in configuration.backends and bool(configuration.storage_bucket)

    def __cloud_sink(self, *, configuration: StorageConfiguration) -> CloudSink:
        """
        Build a :class:`CloudSink` over an isolated cloud-only storage.
        """

        storage: StoragePort = CloudStorage(
            storage=GCSImageStorage(configuration=configuration),
        )
        return CloudSink(storage=storage, workflow_id=self.__workflow_id)

    def __log_context(self) -> Dict[str, Any]:
        """
        Return shared structured-logging context for adapter-assembly entries.
        """

        return {
            "component": "runtime.adapters",
            "workflow.id": self.__workflow_id,
        }
