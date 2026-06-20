from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from typing import Any, List

from fathom.constants.collaboration import (
    Audience,
    JobKind,
    JobState,
    MessageKind,
)
from fathom.core.services.projector import MemoryProjectorHandler
from fathom.schemas.interaction import (
    Content,
    Identity,
    Job,
    Message,
    MessageCursorQuery,
    MessagePage,
    Metadata,
    Timing,
)


class _RecordingMemory:
    """
    Minimal MemoryPort stub that captures the last set call.
    """

    def __init__(self) -> None:
        """
        Initialise captured key/value fields.
        """

        self.last_value: str | None = None
        self.last_key: str | None = None

    async def set(self, *, key: str, value: str) -> None:
        """
        Capture the projected memory payload.
        """

        self.last_key = key
        self.last_value = value


class _PaginatedInteraction:
    """
    Interaction stub that returns multiple pages of messages, exercising the
    projector's pagination loop.
    """

    def __init__(self, *, pages: List[List[Message]]) -> None:
        """
        Store the message pages the fake interaction will return.
        """

        self.__pages = pages
        self.calls = 0

    async def list_messages(self, *, query: MessageCursorQuery) -> MessagePage:
        """
        Return the next configured page and increment the call counter.
        """

        index = self.calls
        self.calls += 1
        if index >= len(self.__pages):
            return MessagePage(items=(), next=None, total=0)
        items = tuple(self.__pages[index])
        next_cursor = f"cursor-{index + 1}" if index + 1 < len(self.__pages) else None
        return MessagePage(items=items, next=next_cursor, total=len(items))


class TestMemoryProjectorPagination(unittest.IsolatedAsyncioTestCase):
    """
    Verify the projector walks every page of task messages instead of
    silently truncating to the first page.
    """

    def __message(self, *, identifier: str) -> Message:
        """
        Build one message fixture.
        """

        return Message(
            identity=Identity(id=identifier, tenant="tenant-1"),
            thread="thread-1",
            task="task-1",
            author="actor-1",
            sequence=1,
            kind=MessageKind.NOTE,
            audience=Audience.THREAD,
            content=Content(body={"text": identifier}),
            created=datetime(2026, 5, 4, tzinfo=timezone.utc),
            metadata=Metadata(),
        )

    def __job(self) -> Job:
        """
        Build the memory projection job fixture.
        """

        return Job(
            identity=Identity(id="job-1", tenant="tenant-1"),
            thread="thread-1",
            task="task-1",
            kind=JobKind.MEMORY,
            state=JobState.CLAIMED,
            attempts=1,
            owner="worker-1",
            payload=Metadata(),
            timing=Timing(
                created=datetime(2026, 5, 4, tzinfo=timezone.utc),
                updated=datetime(2026, 5, 4, tzinfo=timezone.utc),
            ),
            available=datetime(2026, 5, 4, tzinfo=timezone.utc),
            metadata=Metadata(),
        )

    async def test_projector_walks_every_page(self) -> None:
        """
        Three pages of messages must all be projected; previously only the
        first page was read and the rest were silently dropped.
        """

        pages = [
            [self.__message(identifier=f"m-{i}") for i in range(0, 3)],
            [self.__message(identifier=f"m-{i}") for i in range(3, 6)],
            [self.__message(identifier=f"m-{i}") for i in range(6, 7)],
        ]
        interaction: Any = _PaginatedInteraction(pages=pages)
        memory = _RecordingMemory()
        handler = MemoryProjectorHandler(interaction=interaction, memory=memory)

        result = await handler.handle(job=self.__job())

        self.assertEqual(JobState.COMPLETED, result.state)
        self.assertEqual(3, interaction.calls)
        assert memory.last_value is not None
        payload = json.loads(memory.last_value)
        self.assertEqual(7, len(payload["messages"]))
