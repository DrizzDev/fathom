from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from fathom.interfaces.journal import RuntimeJournalPort
from fathom.schemas.events import RuntimeEvent

if TYPE_CHECKING:
    from pathlib import Path


class JsonRuntimeJournal(RuntimeJournalPort):
    """
    Runtime journal adapter that appends events to a local JSONL file.
    """

    def __init__(self, *, path: Path) -> None:
        """
        Initialize the journal with its output path.
        """

        self.__path = path

    async def record(self, *, event: RuntimeEvent) -> None:
        """
        Append one runtime event to the JSONL file.
        """

        await asyncio.to_thread(self.__write, event=event)

    def __write(self, *, event: RuntimeEvent) -> None:
        """
        Write one event line synchronously.
        """

        self.__path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(event.model_dump(mode="json"), sort_keys=True)

        with self.__path.open(mode="a", encoding="utf-8") as handle:
            handle.write(f"{payload}\n")
