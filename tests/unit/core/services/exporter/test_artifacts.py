from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from fathom.constants import ActionType
from fathom.core.services.exporter.artifacts import ExplorationArtifactWriter
from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph
from fathom.interfaces.storage import StoragePort
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


class _RecordingStorage(StoragePort):
    """StoragePort stand-in that records every save for assertions."""

    def __init__(self) -> None:
        self.saved: List[Dict[str, Any]] = []

    async def save(self, *, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        record = {"data": data, "metadata": metadata or {}}
        self.saved.append(record)
        return (
            f"gs://bucket/{record['metadata'].get('category')}/{record['metadata'].get('filename')}"
        )


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
            written = await ExplorationArtifactWriter().write(
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

    async def test_write_uploads_every_artifact_under_reports_category(self) -> None:
        graph = await self.__graph()
        storage = _RecordingStorage()

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "reports"
            written = await ExplorationArtifactWriter(storage=storage).write(
                graph=graph,
                directory=directory,
                workflow="wf",
                package="com.app",
                generated_at="2026-06-12T00:00:00",
                duration=1.0,
            )

            uploaded_names = {record["metadata"]["filename"] for record in storage.saved}
            self.assertEqual(
                uploaded_names, {"graph.json", "graph.dot", "graph.mermaid", "report.md"}
            )
            self.assertEqual(
                {record["metadata"]["category"] for record in storage.saved}, {"reports"}
            )
            self.assertTrue(
                all(record["metadata"]["session_id"] == "wf" for record in storage.saved)
            )
            self.assertTrue(
                all(record["metadata"]["package_name"] == "com.app" for record in storage.saved)
            )
            # Each uploaded payload matches the bytes written locally (no drift).
            for path in written:
                local_bytes = path.read_bytes()
                match = next(
                    record
                    for record in storage.saved
                    if record["metadata"]["filename"] == path.name
                )
                self.assertEqual(match["data"], local_bytes)
            # The local copies are still the durable record.
            self.assertTrue(all(path.exists() for path in written))

    async def test_write_keeps_local_copies_when_upload_fails(self) -> None:
        graph = await self.__graph()

        class _FailingStorage(StoragePort):
            """StoragePort whose every save raises, to prove uploads are best-effort."""

            async def save(self, *, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
                raise RuntimeError("simulated cloud outage")

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "reports"
            written = await ExplorationArtifactWriter(storage=_FailingStorage()).write(
                graph=graph,
                directory=directory,
                workflow="wf",
                package="com.app",
                generated_at="2026-06-12T00:00:00",
                duration=1.0,
            )

            self.assertEqual(
                {path.name for path in written},
                {"graph.json", "graph.dot", "graph.mermaid", "report.md"},
            )
            self.assertTrue(all(path.exists() for path in written))


if __name__ == "__main__":
    unittest.main()
