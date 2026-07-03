from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from fathom.schemas.flow import Evidence, Flow, Issue


class FlowGenerator(ABC):
    """
    Port that produces a target-neutral flow from recorded evidence.
    """

    @abstractmethod
    async def generate(self, *, evidence: Evidence, feedback: Tuple[Issue, ...] = ()) -> Flow:
        """
        Produce a flow for the evidence, incorporating prior gate feedback when repairing.
        """

        raise NotImplementedError
