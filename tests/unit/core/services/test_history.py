from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

from fathom.base.paths import SharedPathManager
from fathom.core.services.exporter import ScriptExporter
from fathom.core.services.history import HistoryService
from fathom.interfaces.storage import StoragePort
from fathom.settings.env import FathomSettings


class _RecordingStorage(StoragePort):
    """In-memory StoragePort that records save calls for assertion."""

    def __init__(self) -> None:
        self.saves: List[Dict[str, Any]] = []

    async def save(self, *, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        self.saves.append({"data": data, "metadata": dict(metadata or {})})
        return "recorded://" + str(len(self.saves))


class HistoryServiceStorageRoutingTest(unittest.IsolatedAsyncioTestCase):
    """
    ``HistoryService`` writes three artifacts (``history.json``,
    ``history.yaml``, ``script.txt``). After the V1 refactor every
    write must route through the injected ``StoragePort`` — no direct
    filesystem writes in core.
    """

    def __build_service(self, *, tmp_dir: Path) -> tuple[HistoryService, _RecordingStorage]:
        settings = FathomSettings(assets_path=tmp_dir)
        path_manager = SharedPathManager(settings=settings)
        storage = _RecordingStorage()
        service = HistoryService(
            workflow_id="session-abc",
            package_name="com.example.app",
            exporter=ScriptExporter(llm=None, use_cache=False),
            path_manager=path_manager,
            storage=storage,
        )
        return service, storage

    async def test_save_json_routes_through_storage(self) -> None:
        """
        ``__save_json`` must push bytes through the injected port with
        ``category=history`` and the history.json filename — never write
        to disk directly.
        """

        with tempfile.TemporaryDirectory() as tmp:
            service, storage = self.__build_service(tmp_dir=Path(tmp))

            await service._HistoryService__save_json(  # type: ignore[attr-defined]
                data={"workflow_id": "session-abc", "history": []},
                package_name="com.example.app",
            )

            self.assertEqual(len(storage.saves), 1)
            entry = storage.saves[0]
            self.assertEqual(entry["metadata"]["filename"], "history.json")
            self.assertEqual(entry["metadata"]["category"], "history")
            self.assertEqual(entry["metadata"]["session_id"], "session-abc")
            self.assertEqual(entry["metadata"]["package_name"], "com.example.app")
            # Data must be UTF-8 encoded JSON.
            self.assertIn(b'"workflow_id": "session-abc"', entry["data"])

    async def test_save_yaml_routes_through_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service, storage = self.__build_service(tmp_dir=Path(tmp))

            await service._HistoryService__save_yaml(  # type: ignore[attr-defined]
                history=[{"action_type": "tap", "activity": "com.example.app", "center": [10, 20]}],
                package_name="com.example.app",
            )

            self.assertEqual(len(storage.saves), 1)
            entry = storage.saves[0]
            self.assertEqual(entry["metadata"]["filename"], "history.yaml")
            self.assertEqual(entry["metadata"]["category"], "history")

    async def test_update_script_routes_through_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service, storage = self.__build_service(tmp_dir=Path(tmp))
            # Stub the exporter so we don't hit the LLM path.
            service._HistoryService__exporter.export_with_llm = AsyncMock(  # type: ignore[attr-defined]
                return_value="open the app\ntap button\n",
            )
            # Seed a tiny history list.
            history = [{"action_type": "tap", "activity": "com.example.app"}]

            await service._HistoryService__update_script(  # type: ignore[attr-defined]
                history=history,
                intent="demo",
                package_name="com.example.app",
            )

            script_saves = [
                entry
                for entry in storage.saves
                if entry["metadata"].get("filename") == "script.txt"
            ]
            self.assertEqual(len(script_saves), 1)
            self.assertEqual(script_saves[0]["data"], b"open the app\ntap button\n")


if __name__ == "__main__":
    unittest.main()
