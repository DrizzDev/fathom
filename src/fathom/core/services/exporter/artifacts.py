"""
Writes the graph exports and analysis report for a completed exploration run.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import List, Optional

from fathom.core.services.exporter.graph import GraphExporter
from fathom.core.services.exporter.report import (
    ExplorationReportGenerator,
    MarkdownReportRenderer,
)
from fathom.core.services.exporter.snapshot import ExplorationSnapshotBuilder
from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph


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
    ) -> None:
        self.__snapshot_builder = snapshot_builder or ExplorationSnapshotBuilder()
        self.__exporter = exporter or GraphExporter()
        self.__renderer = renderer or MarkdownReportRenderer()

    def write(
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
        Writes each graph format and the Markdown report, returning the paths.
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

        return written
