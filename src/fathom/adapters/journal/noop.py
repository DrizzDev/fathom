from __future__ import annotations

from fathom.interfaces.journal import RuntimeJournalPort
from fathom.schemas.events import RuntimeEvent


class NoopRuntimeJournal(RuntimeJournalPort):
    """
    Runtime journal adapter that intentionally discards events.
    """

    async def record(self, *, event: RuntimeEvent) -> None:
        """
        Accept one runtime event without persisting it.
        """

        _ = event
