from __future__ import annotations

from abc import ABC, abstractmethod

from fathom.schemas.flow import Evidence, RunObjective


class EvidenceSource(ABC):
    """
    Port that supplies a run's recorded evidence to script generation.
    """

    @abstractmethod
    async def read(self, *, run: str, objective: RunObjective) -> Evidence:
        """
        Return the full evidence aggregate for a run and its objective.
        """

        raise NotImplementedError
