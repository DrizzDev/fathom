from __future__ import annotations

from abc import ABC, abstractmethod

from fathom.schemas.events import RuntimeEvent


class RuntimeJournalPort(ABC):
    """
    Records append-only runtime events for replay and external projection.
    """

    @abstractmethod
    async def record(self, *, event: RuntimeEvent) -> None:
        """
        Record one runtime event.
        """

        raise NotImplementedError
