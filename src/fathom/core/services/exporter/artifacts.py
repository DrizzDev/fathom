"""
Writes the graph exports and analysis report for a completed exploration run.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, List, Optional

from fathom.core.services.exporter.defect import BugReportRenderer
from fathom.core.services.exporter.document import (
    ScreenDocumentExporter,
    ScreenDocumentRenderer,
)
from fathom.core.services.exporter.graph import GraphExporter
from fathom.core.services.exporter.report import (
    ExplorationReportGenerator,
    MarkdownReportRenderer,
)
from fathom.core.services.exporter.snapshot import ExplorationSnapshotBuilder
from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph
from fathom.schemas.report import ReportMetadata

if TYPE_CHECKING:
    from fathom.schemas.defect import BugReport


class ExplorationArtifactWriter:
    """
    Renders and persists every export and report for an exploration run.
    """

    def __init__(
        self,
        *,
        snapshot_builder: Optional[ExplorationSnapshotBuilder] = None,
        exporter: Optional[GraphExporter] = None,
        renderer: Optional[MarkdownReportRenderer] = None,
        bug_renderer: Optional[BugReportRenderer] = None,
        document_exporter: Optional[ScreenDocumentExporter] = None,
        document_renderer: Optional[ScreenDocumentRenderer] = None,
    ) -> None:
        self.__snapshot_builder = snapshot_builder or ExplorationSnapshotBuilder()
        self.__exporter = exporter or GraphExporter()
        self.__renderer = renderer or MarkdownReportRenderer()
        self.__bug_renderer = bug_renderer or BugReportRenderer()
        self.__document_exporter = document_exporter or ScreenDocumentExporter()
        self.__document_renderer = document_renderer or ScreenDocumentRenderer()

    def write(
        self,
        *,
        graph: KnowledgeGraph,
        directory: Path,
        workflow: str,
        package: str,
        generated_at: str,
        duration: float = 0.0,
        bug_report: Optional[BugReport] = None,
    ) -> List[Path]:
        """
        Writes each graph format, the Markdown report, and any bug report; returns the paths.
        """

        directory.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []

        snapshot = self.__snapshot_builder.build(graph=graph)
        for graph_format in self.__exporter.formats:
            destination = directory / f"graph.{graph_format.value}"
            destination.write_text(
                self.__exporter.render(snapshot=snapshot, graph_format=graph_format),
                encoding="utf-8",
            )
            written.append(destination)

        report = ExplorationReportGenerator(graph=graph).generate(
            workflow=workflow, package=package, generated_at=generated_at, duration=duration
        )
        report_path = directory / "report.md"
        report_path.write_text(self.__renderer.render(report=report), encoding="utf-8")
        written.append(report_path)

        if bug_report is not None:
            written.extend(self.__write_bug_report(directory=directory, bug_report=bug_report))

        written.extend(
            self.__write_screen_documents(
                graph=graph,
                directory=directory,
                metadata=ReportMetadata(
                    workflow=workflow,
                    package=package,
                    generated_at=generated_at,
                    duration=duration,
                ),
                bug_report=bug_report,
            )
        )

        return written

    def __write_bug_report(self, *, directory: Path, bug_report: BugReport) -> List[Path]:
        """
        Writes the bug report as Markdown and JSON.
        """

        markdown_path = directory / "bug_report.md"
        markdown_path.write_text(self.__bug_renderer.render(report=bug_report), encoding="utf-8")

        json_path = directory / "bug_report.json"
        json_path.write_text(bug_report.model_dump_json(indent=2), encoding="utf-8")

        return [markdown_path, json_path]

    def __write_screen_documents(
        self,
        *,
        graph: KnowledgeGraph,
        directory: Path,
        metadata: ReportMetadata,
        bug_report: Optional[BugReport],
    ) -> List[Path]:
        """
        Writes one image-free Markdown document per logical screen plus an index.
        """

        defects = bug_report.defects if bug_report is not None else []
        index = self.__document_exporter.build(graph=graph, defects=defects, metadata=metadata)

        screens_directory = directory / "screens"
        screens_directory.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []

        index_path = screens_directory / "index.md"
        index_path.write_text(self.__document_renderer.render_index(index=index), encoding="utf-8")
        written.append(index_path)

        for document in index.documents:
            document_path = screens_directory / f"{document.slug}.md"
            document_path.write_text(
                self.__document_renderer.render_screen(document=document), encoding="utf-8"
            )
            written.append(document_path)

        return written
