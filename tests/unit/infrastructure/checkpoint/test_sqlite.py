from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import aiosqlite

from fathom.constants import ActionType
from fathom.constants.exploration import BFSPhase
from fathom.infrastructure.checkpoint.sqlite import SqliteExplorationCheckpointRepository
from fathom.schemas.actions import Action
from fathom.schemas.checkpoint import ExplorationCheckpoint


class SqliteExplorationCheckpointRepositoryTest(unittest.IsolatedAsyncioTestCase):
    """The repository round-trips the DFS checkpoint and guards the schema version."""

    def setUp(self) -> None:
        self.__tmp = TemporaryDirectory()
        self.__db = Path(self.__tmp.name) / "exploration.db"
        self.__repo = SqliteExplorationCheckpointRepository(database_path=self.__db)

    def tearDown(self) -> None:
        self.__tmp.cleanup()

    @staticmethod
    def __checkpoint() -> ExplorationCheckpoint:
        return ExplorationCheckpoint(
            phase=BFSPhase.BACKTRACK,
            root_hash="root",
            current_path=[
                ("hash_a", Action(action_type=ActionType.TAP, target="a", rationale="r"))
            ],
            fully_scanned=["a", "b"],
            exhaustion_retries={"a": 1},
        )

    async def test_save_then_load_round_trips(self) -> None:
        await self.__repo.save(workflow_id="wf", checkpoint=self.__checkpoint())

        self.assertEqual(await self.__repo.load(workflow_id="wf"), self.__checkpoint())

    async def test_load_absent_workflow_is_none(self) -> None:
        self.assertIsNone(await self.__repo.load(workflow_id="missing"))

    async def test_save_replaces_the_prior_checkpoint(self) -> None:
        await self.__repo.save(workflow_id="wf", checkpoint=self.__checkpoint())
        await self.__repo.save(
            workflow_id="wf",
            checkpoint=self.__checkpoint().model_copy(update={"root_hash": "root2"}),
        )

        loaded = await self.__repo.load(workflow_id="wf")

        assert loaded is not None
        self.assertEqual(loaded.root_hash, "root2")

    async def test_incompatible_version_is_discarded(self) -> None:
        await self.__repo.save(workflow_id="wf", checkpoint=self.__checkpoint())
        async with aiosqlite.connect(self.__db) as db:
            await db.execute(
                "UPDATE exploration_checkpoint SET version = 99 WHERE workflow_id = 'wf'"
            )
            await db.commit()

        self.assertIsNone(await self.__repo.load(workflow_id="wf"))


if __name__ == "__main__":
    unittest.main()
