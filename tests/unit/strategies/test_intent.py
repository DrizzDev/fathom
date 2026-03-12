from pathlib import Path
from typing import Any, Callable, cast

import pytest

from fathom.strategies.intent import (
    CHECKPOINT_ALLOWED_JSON_MODULES,
    CHECKPOINT_ALLOWED_MSGPACK_MODULES,
    IntentStrategy,
)


@pytest.mark.asyncio
async def test_build_checkpointer_context_configures_allowed_modules(
    tmp_path: Path,
) -> None:
    """Build the SQLite checkpointer with the Fathom serde allowlist."""

    strategy = object.__new__(IntentStrategy)
    checkpoint_path = tmp_path / "checkpoints.db"
    context_builder = cast(
        "Callable[[Path], Any]",
        strategy.__getattribute__("_IntentStrategy__build_checkpointer_context"),
    )

    async with context_builder(checkpoint_path) as checkpointer:
        assert type(checkpointer).__name__ == "AsyncSqliteSaver"
        assert type(checkpointer.serde).__name__ == "JsonPlusSerializer"
        assert checkpointer.serde._allowed_modules == set(CHECKPOINT_ALLOWED_JSON_MODULES)
        if hasattr(checkpointer.serde, "_allowed_msgpack_modules"):
            assert checkpointer.serde._allowed_msgpack_modules == set(
                CHECKPOINT_ALLOWED_MSGPACK_MODULES
            )
