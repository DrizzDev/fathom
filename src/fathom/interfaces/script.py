from __future__ import annotations

from abc import ABC, abstractmethod

from fathom.schemas.flow import RunObjective


class ScriptRefresher(ABC):
    """
    Port that refreshes the available script artifact after a run's recorded history changes.
    """

    @abstractmethod
    def schedule(self, *, run: str, objective: RunObjective) -> None:
        """
        Schedule a non-blocking, coalesced refresh of the script artifact for the run.
        """

        raise NotImplementedError

    @abstractmethod
    async def drain(self) -> None:
        """
        Await any in-flight refresh so finalization observes the latest artifact.
        """

        raise NotImplementedError
