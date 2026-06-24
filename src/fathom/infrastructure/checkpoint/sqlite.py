"""
SQLite-backed persistence for the exploration DFS checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003
from typing import Optional

import aiosqlite

from fathom.core.runtime.checkpoint import CheckpointCodec
from fathom.interfaces.checkpoint import ExplorationCheckpointPort
from fathom.schemas.checkpoint import ExplorationCheckpoint

# Schema version of the persisted checkpoint payload; a load of any other version
# is discarded so an incompatible older checkpoint never crashes a resume.
_CHECKPOINT_VERSION = 1


class SqliteExplorationCheckpointRepository(ExplorationCheckpointPort):
    """
    Persists the latest exploration DFS checkpoint per workflow in SQLite.
    """

    def __init__(self, *, database_path: Path) -> None:
        self.__path = database_path
        self.__path.parent.mkdir(parents=True, exist_ok=True)
        self.__initialized = False
        self.__codec = CheckpointCodec()

    async def __initialize(self) -> None:
        """
        Creates the single-row checkpoint table when absent.
        """

        if self.__initialized:
            return

        async with aiosqlite.connect(self.__path) as db:
            await db.execute(
                "CREATE TABLE IF NOT EXISTS exploration_checkpoint ("
                "workflow_id TEXT PRIMARY KEY, "
                "version INTEGER NOT NULL, "
                "payload TEXT NOT NULL"
                ")"
            )
            await db.commit()

        self.__initialized = True

    async def save(self, *, workflow_id: str, checkpoint: ExplorationCheckpoint) -> None:
        """
        Persists the latest checkpoint, replacing any prior one for the workflow.
        """

        await self.__initialize()
        envelope = self.__codec.encode(
            payload=checkpoint.model_dump(mode="json"), version=_CHECKPOINT_VERSION
        )

        async with aiosqlite.connect(self.__path) as db:
            await db.execute(
                "INSERT INTO exploration_checkpoint (workflow_id, version, payload) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(workflow_id) DO UPDATE SET "
                "version = excluded.version, payload = excluded.payload",
                (workflow_id, envelope.version, json.dumps(envelope.payload)),
            )
            await db.commit()

    async def load(self, *, workflow_id: str) -> Optional[ExplorationCheckpoint]:
        """
        Loads the saved checkpoint, or None when absent or of an incompatible version.
        """

        await self.__initialize()

        async with (
            aiosqlite.connect(self.__path) as db,
            db.execute(
                "SELECT version, payload FROM exploration_checkpoint WHERE workflow_id = ?",
                (workflow_id,),
            ) as cursor,
        ):
            row = await cursor.fetchone()

        if row is None or int(row[0]) != _CHECKPOINT_VERSION:
            return None
        return ExplorationCheckpoint.model_validate(json.loads(row[1]))
