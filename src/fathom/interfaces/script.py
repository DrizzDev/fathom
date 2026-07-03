from __future__ import annotations

from abc import ABC, abstractmethod

from fathom.schemas.flow import RunObjective


class ScriptRefresher(ABC):
    """
    Port that refreshes the available script artifact after an execution's recorded history changes.
    """

    @abstractmethod
    def schedule(self, *, execution_id: str, objective: RunObjective) -> None:
        """
        Schedule a non-blocking, coalesced refresh of the script artifact for the execution.
        """

        raise NotImplementedError

    @abstractmethod
    async def drain(self) -> None:
        """
        Await any in-flight refresh so finalization observes the latest artifact.
        """

        raise NotImplementedError
