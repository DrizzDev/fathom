"""
Writes the graph exports and analysis report for a completed exploration run.
"""

from __future__ import annotations

from logging import getLogger
from pathlib import Path  # noqa: TC003
from typing import List, Optional, Tuple

from fathom.constants.artifact import ArtifactDirectory
from fathom.core.services.exporter.graph import GraphExporter
from fathom.core.services.exporter.report import (
    ExplorationReportGenerator,
    MarkdownReportRenderer,
)
from fathom.core.services.exporter.snapshot import ExplorationSnapshotBuilder
from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph
from fathom.interfaces.storage import StoragePort

logger = getLogger(__name__)


class ExplorationArtifactWriter:
    """
    Renders and persists every export and report for an exploration run.

    Each artifact is written to the local report directory (the durable copy) and,
    when a :class:`StoragePort` is configured, uploaded under the ``reports`` storage
    category so a run's report and graph exports sit beside its screenshots in the
    cloud. Cloud upload is best-effort: a failed upload neither loses the local copy
    nor aborts the run.
    """

    def __init__(
        self,
        *,
        storage: Optional[StoragePort] = None,
        snapshot_builder: Optional[ExplorationSnapshotBuilder] = None,
        exporter: Optional[GraphExporter] = None,
        renderer: Optional[MarkdownReportRenderer] = None,
    ) -> None:
        self.__storage = storage
        self.__snapshot_builder = snapshot_builder or ExplorationSnapshotBuilder()
        self.__exporter = exporter or GraphExporter()
        self.__renderer = renderer or MarkdownReportRenderer()

    async def write(
        self,
        *,
        graph: KnowledgeGraph,
        directory: Path,
        workflow: str,
        package: str,
        generated_at: str,
        duration: float = 0.0,
    ) -> List[Path]:
        """
        Writes each graph format and the Markdown report locally, uploads them to the
        configured storage under the ``reports`` category, and returns the local paths.
        """

        directory.mkdir(parents=True, exist_ok=True)
        rendered = self.__render(
            graph=graph,
            workflow=workflow,
            package=package,
            generated_at=generated_at,
            duration=duration,
        )

        written: List[Path] = []
        for filename, content in rendered:
            destination = directory / filename
            destination.write_text(content, encoding="utf-8")
            written.append(destination)
            await self.__upload(
                filename=filename, content=content, workflow=workflow, package=package
            )

        return written

    def __render(
        self,
        *,
        graph: KnowledgeGraph,
        workflow: str,
        package: str,
        generated_at: str,
        duration: float,
    ) -> List[Tuple[str, str]]:
        """
        Render every graph export and the Markdown report into ``(filename, text)`` pairs.
        """

        snapshot = self.__snapshot_builder.build(graph=graph)
        artifacts: List[Tuple[str, str]] = [
            (
                f"graph.{graph_format.value}",
                self.__exporter.render(snapshot=snapshot, graph_format=graph_format),
            )
            for graph_format in self.__exporter.formats
        ]

        report = ExplorationReportGenerator(graph=graph).generate(
            workflow=workflow, package=package, generated_at=generated_at, duration=duration
        )
        artifacts.append(("report.md", self.__renderer.render(report=report)))

        return artifacts

    async def __upload(self, *, filename: str, content: str, workflow: str, package: str) -> None:
        """
        Best-effort upload of one rendered artifact to the ``reports`` storage category.

        Mirrors the cloud artifact sink's policy: the local copy is the durable record,
        so an upload failure (or an empty identifier from a fully-failed composite
        backend) is logged and swallowed rather than propagated to the run.
        """

        if self.__storage is None:
            return

        try:
            identifier = await self.__storage.save(
                data=content.encode("utf-8"),
                metadata={
                    "category": ArtifactDirectory.REPORTS,
                    "filename": filename,
                    "session_id": workflow,
                    "package_name": package,
                },
            )
        except Exception as exception:  # best-effort persistence boundary, as in CloudSink
            logger.warning(
                "Report artifact upload raised; local copy retained",
                extra={
                    "component": "exporter.artifacts",
                    "event": "report.upload.failed",
                    "error.message": str(exception),
                    "artifact.filename": filename,
                    "workflow.id": workflow,
                },
            )
            return

        if not identifier:
            logger.warning(
                "Report artifact upload returned no identifier; local copy retained",
                extra={
                    "component": "exporter.artifacts",
                    "event": "report.upload.dropped",
                    "artifact.filename": filename,
                    "workflow.id": workflow,
                },
            )
            return

        logger.info(
            "Report artifact uploaded",
            extra={
                "component": "exporter.artifacts",
                "event": "report.upload.succeeded",
                "artifact.filename": filename,
                "artifact.identifier": identifier,
                "workflow.id": workflow,
            },
        )
