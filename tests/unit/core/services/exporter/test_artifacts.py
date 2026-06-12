from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Optional

from fathom.constants import ActionType
from fathom.core.services.exporter.artifacts import ExplorationArtifactWriter
from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph
from fathom.schemas.actions import Action
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

            names = {path.name for path in written}
            self.assertEqual(names, {"graph.json", "graph.dot", "graph.mermaid", "report.md"})
            self.assertTrue(all(path.exists() for path in written))
            self.assertIn("digraph exploration", (directory / "graph.dot").read_text())
            self.assertIn("# Exploration Report", (directory / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
