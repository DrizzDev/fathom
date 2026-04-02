from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable, cast

from fathom.strategies.intent import (
    CHECKPOINT_ALLOWED_JSON_MODULES,
    CHECKPOINT_ALLOWED_MSGPACK_MODULES,
    IntentStrategy,
)


class IntentStrategyTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover IntentStrategy persistence setup.
    """

    async def test_build_checkpointer_context_configures_allowed_modules(self) -> None:
        """
        Build the SQLite checkpointer with the Fathom serde allowlist.
        """

        strategy = object.__new__(IntentStrategy)
        context_builder = cast(
            "Callable[[Path], Any]",
            strategy.__getattribute__("_IntentStrategy__build_checkpointer_context"),
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoints.db"

            async with context_builder(checkpoint_path) as checkpointer:
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
