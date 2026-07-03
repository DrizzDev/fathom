from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fathom.authoring.adapters.store import FileAuthoringDraftStore
from fathom.constants.authoring import AuthoringKind, AuthoringStatus
from fathom.interfaces.paths import HistoryPaths
from fathom.schemas.authoring.draft import AuthoringDraft


class StubHistoryPaths(HistoryPaths):
    """
    Test path resolver returning one temporary directory.
    """

    def __init__(self, *, directory: Path) -> None:
        """
        Store the directory returned for every workflow.
        """

        self.__directory = directory

    def get_history_directory(self, *, session_id: str) -> Path:
        """
        Return the configured temporary directory.
        """

        _ = session_id
        return self.__directory


class FileAuthoringDraftStoreTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover file-backed authoring draft persistence.
    """

    async def test_save_and_list_round_trip_drafts(self) -> None:
        """
        Draft store persists authoring drafts in workflow order.
        """

        with TemporaryDirectory() as temporary:
            store = FileAuthoringDraftStore(
                path_manager=StubHistoryPaths(directory=Path(temporary))
            )
            draft = AuthoringDraft(
                step_index=2,
                reason="disabled",
                kind=AuthoringKind.STEP,
                workflow_id="workflow-1",
                status=AuthoringStatus.SKIPPED,
            )

            await store.save(draft=draft)

            loaded = await store.list(workflow_id="workflow-1")

            self.assertEqual(loaded, (draft,))
