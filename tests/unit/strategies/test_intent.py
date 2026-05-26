from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from fathom.adapters.checkpoint import SqliteCheckpointStore
from fathom.runtime.checkpoint_serde import CheckpointSerdeFactory
from fathom.schemas.checkpoint import SqliteCheckpointPolicy
from fathom.strategies.intent import (
    CHECKPOINT_ALLOWED_JSON_MODULES,
    CHECKPOINT_ALLOWED_MSGPACK_MODULES,
)


class CheckpointStoreSerdeTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover SqliteCheckpointStore serde wiring through the new checkpoint port.
    """

    @staticmethod
    def __module_available(name: str) -> bool:
        """
        Return True when an optional checkpointer dependency module is importable.
        """

        try:
            return importlib.util.find_spec(name) is not None
        except ModuleNotFoundError:
            return False

    async def test_adapter_passes_fathom_serde_with_allowed_modules(self) -> None:
        """
        SqliteCheckpointStore.open hands LangGraph the CheckpointSerdeFactory-built serde with the fathom allow-list.
        """

        sqlite_available = self.__module_available(
            "langgraph.checkpoint.sqlite.aio"
        ) and self.__module_available("aiosqlite")
        if not sqlite_available:
            self.skipTest("langgraph-checkpoint-sqlite / aiosqlite not installed")

        serde = CheckpointSerdeFactory.build()

        with tempfile.TemporaryDirectory() as directory:
            store = SqliteCheckpointStore(
                directory=Path(directory),
                policy=SqliteCheckpointPolicy(),
                serde=serde,
            )

            async with store.open(workflow_id="workflow-allowlist-test") as checkpointer:
                self.assertEqual(type(checkpointer).__name__, "AsyncSqliteSaver")
                self.assertEqual(type(checkpointer.serde).__name__, "JsonPlusSerializer")
                allowed_modules = getattr(
                    checkpointer.serde,
                    "_allowed_json_modules",
                    checkpointer.serde._allowed_modules,
                )
                self.assertEqual(allowed_modules, set(CHECKPOINT_ALLOWED_JSON_MODULES))
                if hasattr(checkpointer.serde, "_allowed_msgpack_modules"):
                    self.assertEqual(
                        checkpointer.serde._allowed_msgpack_modules,
                        set(CHECKPOINT_ALLOWED_MSGPACK_MODULES),
                    )
