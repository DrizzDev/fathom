from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fathom.adapters.interaction.pypika.sqlite import SQLiteInteraction
from fathom.constants.collaboration import (
    ActorKind,
    Audience,
    MembershipRole,
    MessageKind,
)
from fathom.schemas.interaction import (
    Content,
    CreateActor,
    CreateThread,
    Identity,
    JoinThread,
    MessageQuery,
    RecordMessage,
)


class TestThreadSequenceAllocator(unittest.IsolatedAsyncioTestCase):
    """
    Verify the new sequences allocator returns monotonic per-thread
    sequences for messages and events, and that scopes are independent.
    """

    def setUp(self) -> None:
        """
        Create an isolated SQLite interaction store for each allocator test.
        """

        self.__directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.__directory.cleanup)
        path = Path(self.__directory.name) / "sequence.db"
        self.__interaction = SQLiteInteraction(path=path)
        self.__now = datetime(2026, 5, 4, tzinfo=timezone.utc)

    async def __seed(self) -> None:
        """
        Seed one actor, thread, and membership.
        """

        await self.__interaction.create_actor(
            request=CreateActor(
                identity=Identity(id="actor-1", tenant="tenant-1"),
                kind=ActorKind.HUMAN,
                name="A",
                created=self.__now,
            )
        )
        await self.__interaction.create_thread(
            request=CreateThread(
                identity=Identity(id="thread-1", tenant="tenant-1"),
                title="t",
                creator="actor-1",
                created=self.__now,
            )
        )
        await self.__interaction.join_thread(
            request=JoinThread(
                identity=Identity(id="m-1", tenant="tenant-1"),
                thread="thread-1",
                actor="actor-1",
                role=MembershipRole.OWNER,
                joined=self.__now,
            )
        )

    async def __record(self, *, identifier: str) -> None:
        """
        Record one note message with a caller-provided identity.
        """

        await self.__interaction.record_message(
            request=RecordMessage(
                identity=Identity(id=identifier, tenant="tenant-1"),
                thread="thread-1",
                author="actor-1",
                kind=MessageKind.NOTE,
                audience=Audience.THREAD,
                content=Content(body={"text": identifier}),
                created=self.__now,
            )
        )

    async def test_message_sequences_are_monotonic_per_thread(self) -> None:
        """
        Every recorded message must receive a unique, monotonically
        increasing sequence within its thread.
        """

        await self.__seed()
        for index in range(5):
            await self.__record(identifier=f"m-{index}")

        messages = await self.__interaction.get_messages(
            query=MessageQuery(tenant="tenant-1", thread="thread-1")
        )

        sequences = [message.sequence for message in messages]
        self.assertEqual([1, 2, 3, 4, 5], sorted(sequences))

    async def test_message_and_event_sequences_are_independent(self) -> None:
        """
        Messages and lifecycle events draw from separate per-thread
        sequence spaces; their numbering must not interleave.
        """

        await self.__seed()
        await self.__record(identifier="m-1")
        await self.__record(identifier="m-2")

        # The seeding above generated several lifecycle events
        # (THREAD_CREATED, ACTOR_JOINED, MESSAGE_RECORDED ...). Re-record to
        # add more events and verify the message sequence stays at 3, not
        # interleaved with event sequences.
        await self.__record(identifier="m-3")

        messages = await self.__interaction.get_messages(
            query=MessageQuery(tenant="tenant-1", thread="thread-1")
        )
        self.assertEqual([1, 2, 3], sorted(message.sequence for message in messages))
