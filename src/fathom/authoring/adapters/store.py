from __future__ import annotations

import json
from typing import TYPE_CHECKING, List, Tuple

from fathom.constants.authoring import AUTHORING_DRAFTS_FILENAME
from fathom.interfaces.authoring import AuthoringDraftStore
from fathom.interfaces.paths import HistoryPaths
from fathom.schemas.authoring.draft import AuthoringDraft

if TYPE_CHECKING:
    from pathlib import Path


class FileAuthoringDraftStore(AuthoringDraftStore):
    """
    File-backed authoring draft store scoped to execution history directories.
    """

    def __init__(self, *, path_manager: HistoryPaths) -> None:
        """
        Bind the history path resolver used for execution-scoped draft files.
        """

        self.__path_manager = path_manager

    async def save(self, *, draft: AuthoringDraft) -> None:
        """
        Persist one authoring draft by replacing the workflow sidecar atomically.
        """

        drafts = list(await self.list(execution_id=draft.execution_id))
        drafts.append(draft)

        path = self.__path(execution_id=draft.execution_id)
        payload = [item.model_dump(mode="json", exclude_none=True) for item in drafts]

        self.__write(path=path, content=json.dumps(payload, ensure_ascii=False, indent=2))

    async def list(self, *, execution_id: str) -> Tuple[AuthoringDraft, ...]:
        """
        Return drafts recorded for one execution.
        """

        path = self.__path(execution_id=execution_id)
        if not path.exists():
            return ()

        with path.open(mode="r") as handle:
            raw = json.load(fp=handle)

        if not isinstance(raw, list):
            return ()

        drafts: List[AuthoringDraft] = []

        for item in raw:
            if isinstance(item, dict):
                drafts.append(AuthoringDraft.model_validate(item))

        return tuple(drafts)

    def __path(self, *, execution_id: str) -> "Path":
        """
        Return the draft sidecar path for one execution.
        """

        return self.__path_manager.get_history_directory(session_id=execution_id).joinpath(
            AUTHORING_DRAFTS_FILENAME
        )

    @staticmethod
    def __write(*, path: "Path", content: str) -> None:
        """
        Atomically replace a text file.
        """

        path.parent.mkdir(parents=True, exist_ok=True)

        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(data=content, encoding="utf-8")
        temporary.replace(path)
