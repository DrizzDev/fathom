from __future__ import annotations

from abc import ABC, abstractmethod

from fathom.schemas.flow import Evidence, RunObjective


class EvidenceSource(ABC):
    """
    Port that supplies an execution's recorded evidence to script generation.
    """

    @abstractmethod
    async def read(self, *, execution_id: str, objective: RunObjective) -> Evidence:
        """
        Return the full evidence aggregate for an execution and its objective.
        """

        raise NotImplementedError
