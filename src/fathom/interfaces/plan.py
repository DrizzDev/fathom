from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from fathom.schemas.plan import Plan


class PlanStore(ABC):
    """
    Durable read and seed of one run's accepted plan, framework-neutral.
    """

    @abstractmethod
    async def read(self, *, run: object, workflow: str) -> Optional[Plan]:
        """
        Return the plan already committed for this run, or None on a fresh run.
        """

        raise NotImplementedError

    @abstractmethod
    async def seed(self, *, run: object, workflow: str, plan: Plan) -> None:
        """
        Durably persist the accepted plan before the execution graph runs.
        """

        raise NotImplementedError
