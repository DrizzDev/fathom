from __future__ import annotations

from typing import Protocol, Tuple

from fathom.schemas.authoring import AuthoringResponse, AuthoringTask
from fathom.schemas.authoring.draft import AuthoringDraft
from fathom.schemas.flow import RunObjective


class AuthoringPort(Protocol):
    """
    Port for producing authored script output for a task.
    """

    async def author(self, *, task: AuthoringTask) -> AuthoringResponse:
        """
        Author script output for the supplied task.
        """

        ...


class AuthoringDraftStore(Protocol):
    """
    Port for persisting and loading authoring drafts.
    """

    async def save(self, *, draft: AuthoringDraft) -> None:
        """
        Persist one authoring draft.
        """

        ...

    async def list(self, *, execution_id: str) -> Tuple[AuthoringDraft, ...]:
        """
        Return drafts recorded for one execution.
        """

        ...


class AuthoringScheduler(Protocol):
    """
    Port for scheduling optional background authoring work.
    """

    def schedule_step(self, *, execution_id: str, objective: RunObjective, step_index: int) -> None:
        """
        Schedule optional authoring for one persisted execution step.
        """

        ...

    async def drain(self) -> None:
        """
        Wait for scheduled authoring work to settle.
        """

        ...
