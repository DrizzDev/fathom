from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Optional

from fathom.constants import ActionType
from fathom.constants.defect import DefectSignal, DefectSource
from fathom.constants.document import SCREEN_DOCUMENT_SCHEMA_VERSION
from fathom.core.defect.aggregator import DefectAggregator
from fathom.core.services.exporter.artifacts import ExplorationArtifactWriter
from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph
from fathom.schemas.actions import Action
from fathom.schemas.defect import Defect, DefectEvidence
from fathom.schemas.document import DocumentIndex
from fathom.schemas.report import ReportMetadata
from fathom.schemas.screens import ScreenState


class _NullProvider:
    """IMemoryProvider stand-in whose writes are no-ops."""

    async def store_observation(self, screen: ScreenState, description: Optional[str]) -> None:
        return None

    async def store_transition(
        self, source_hash: str, action: Action, destination_hash: str
    ) -> None:
        return None


class TestExplorationArtifactWriter(unittest.IsolatedAsyncioTestCase):
    """The writer persists every graph format and the Markdown report."""

    @staticmethod
    def __screen(*, key: str, visual_hash: str) -> ScreenState:
        return ScreenState(
            activity=f"com.app/.{key.title()}",
            timestamp=0,
            activity_hash=f"act-{key}",
            visual_hash=visual_hash,
        )

    async def __graph(self) -> KnowledgeGraph:
        graph = KnowledgeGraph(provider=_NullProvider())
        await graph.add_screen(
            state=self.__screen(key="home", visual_hash="0000000000000000"), description="Home"
        )
        await graph.add_screen(
            state=self.__screen(key="cart", visual_hash="ffffffffffffffff"), description="Cart"
        )
        await graph.record_transition(
            source_hash="0000000000000000",
            action=Action(
                action_type=ActionType.TAP, rationale="r", natural_language_target="Cart"
            ),
            destination_hash="ffffffffffffffff",
        )
        return graph

    async def test_write_emits_graph_formats_and_report(self) -> None:
        graph = await self.__graph()

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "reports"
            written = ExplorationArtifactWriter().write(
                graph=graph,
                directory=directory,
                workflow="wf",
                package="com.app",
                generated_at="2026-06-12T00:00:00",
                duration=1.0,
            )

            top_level = {path.name for path in written if path.parent == directory}
            self.assertEqual(top_level, {"graph.json", "graph.dot", "graph.mermaid", "report.md"})
            self.assertTrue(all(path.exists() for path in written))
            self.assertIn("digraph exploration", (directory / "graph.dot").read_text())
            self.assertIn("# Exploration Report", (directory / "report.md").read_text())

    async def test_write_emits_per_screen_documents(self) -> None:
        graph = await self.__graph()

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "reports"
            written = ExplorationArtifactWriter().write(
                graph=graph,
                directory=directory,
                workflow="wf",
                package="com.app",
                generated_at="2026-06-12T00:00:00",
                duration=1.0,
            )

            screens = {path.name for path in written if path.parent == directory / "screens"}
            self.assertIn("index.md", screens)
            self.assertIn(
                "# Screen Documentation", (directory / "screens" / "index.md").read_text()
            )
            # The typed artifact plus one Markdown doc per logical screen (home + cart).
            self.assertIn("index.json", screens)
            self.assertEqual(len(screens), 4)
            for screen_doc in screens:
                if screen_doc.endswith(".md"):
                    self.assertNotIn("![", (directory / "screens" / screen_doc).read_text())

    async def test_write_emits_versioned_json_artifact(self) -> None:
        graph = await self.__graph()

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "reports"
            ExplorationArtifactWriter().write(
                graph=graph,
                directory=directory,
                workflow="wf",
                package="com.app",
                generated_at="2026-06-12T00:00:00",
                duration=1.0,
            )

            artifact = directory / "screens" / "index.json"
            self.assertTrue(artifact.exists())
            index = DocumentIndex.model_validate_json(artifact.read_text())
            self.assertEqual(index.schema_version, SCREEN_DOCUMENT_SCHEMA_VERSION)
            self.assertEqual(index.metadata.package, "com.app")
            self.assertGreaterEqual(len(index.documents), 1)

    async def test_write_emits_bug_report_when_provided(self) -> None:
        graph = await self.__graph()
        defect = Defect.from_signal(
            signal=DefectSignal.CRASH,
            source=DefectSource.INLINE,
            summary="App left the package",
            evidence=DefectEvidence(screen="0000000000000000"),
        )
        bug_report = DefectAggregator().build(
            defects=[defect],
            metadata=ReportMetadata(
                workflow="wf", package="com.app", generated_at="2026-06-12T00:00:00"
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "reports"
            written = ExplorationArtifactWriter().write(
                graph=graph,
                directory=directory,
                workflow="wf",
                package="com.app",
                generated_at="2026-06-12T00:00:00",
                duration=1.0,
                bug_report=bug_report,
            )

            names = {path.name for path in written}
            self.assertIn("bug_report.md", names)
            self.assertIn("bug_report.json", names)
            self.assertIn("App left the package", (directory / "bug_report.md").read_text())


if __name__ == "__main__":
    unittest.main()
