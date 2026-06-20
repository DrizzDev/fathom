from __future__ import annotations

import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fathom.adapters.interaction.pypika.sqlite import SQLiteInteraction
from fathom.constants.collaboration import (
    ActorKind,
    ArtifactBackend,
    ArtifactKind,
    Audience,
    EventKind,
    Label,
    MembershipRole,
    MessageKind,
    TaskKind,
    TaskState,
)
from fathom.core.exceptions import ThreadConflictError
from fathom.schemas.interaction import (
    Assignment,
    Content,
    CreateActor,
    CreateThread,
    EventQuery,
    Identity,
    JoinThread,
    Lineage,
    LinkArtifact,
    MessageQuery,
    OpenTask,
    Plan,
    RecordMessage,
    Sanitize,
)


class TestCorrectnessPass(unittest.IsolatedAsyncioTestCase):
    """
    Targeted regression coverage for B1 / B2 / B10 / B14 / B16 fixes.
    """

    def setUp(self) -> None:
        """
        Create an isolated SQLite interaction store for each regression test.
        """

        self.__directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.__directory.cleanup)
        path = Path(self.__directory.name) / "correctness.db"
        self.__path = path
        self.__interaction = SQLiteInteraction(path=path)
        self.__now = datetime(2026, 5, 4, 10, 0, 0, tzinfo=timezone.utc)

    async def __seed_thread(self) -> None:
        """
        Seed one actor, thread, membership, and running task.
        """

        await self.__interaction.create_actor(
            request=CreateActor(
                identity=Identity(id="actor-1", tenant="tenant-1"),
                kind=ActorKind.HUMAN,
                name="Aman",
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
        await self.__interaction.open_task(
            request=OpenTask(
                identity=Identity(id="task-1", tenant="tenant-1"),
                thread="thread-1",
                assignment=Assignment(creator="actor-1", assignee="actor-1"),
                lineage=Lineage(),
                kind=TaskKind.AGENT,
                state=TaskState.RUNNING,
                plan=Plan(objective="x"),
                created=self.__now,
            )
        )

    # B1 ----------------------------------------------------------------

    async def test_b1_record_message_none_sequence_is_auto_allocated(self) -> None:
        """
        Passing sequence=None defers allocation to the store; subsequent
        messages receive monotonic per-thread sequences.
        """

        await self.__seed_thread()
        for index in range(3):
            await self.__interaction.record_message(
                request=RecordMessage(
                    identity=Identity(id=f"m-{index}", tenant="tenant-1"),
                    thread="thread-1",
                    author="actor-1",
                    sequence=None,
                    kind=MessageKind.NOTE,
                    audience=Audience.THREAD,
                    content=Content(body={"text": f"m-{index}"}),
                    created=self.__now,
                )
            )

        messages = await self.__interaction.get_messages(
            query=MessageQuery(tenant="tenant-1", thread="thread-1")
        )
        sequences = sorted(message.sequence for message in messages)
        # Three new messages allocated 1..3; thread also has the THREAD_CREATED
        # event sequence space which is independent.
        self.assertEqual([1, 2, 3], sequences)

    async def test_b1_record_message_explicit_sequence_is_honored(self) -> None:
        """
        An explicit caller-supplied sequence is persisted as-is.
        """

        await self.__seed_thread()
        await self.__interaction.record_message(
            request=RecordMessage(
                identity=Identity(id="m-explicit", tenant="tenant-1"),
                thread="thread-1",
                author="actor-1",
                sequence=42,
                kind=MessageKind.NOTE,
                audience=Audience.THREAD,
                content=Content(body={"text": "explicit"}),
                created=self.__now,
            )
        )

        messages = await self.__interaction.get_messages(
            query=MessageQuery(tenant="tenant-1", thread="thread-1")
        )
        self.assertEqual([42], [message.sequence for message in messages])

    # B2 ----------------------------------------------------------------

    async def test_b2_event_ids_are_unique_per_emit(self) -> None:
        """
        Two events emitted for the same (kind, subject) pair receive distinct
        ids — the per-thread sequence becomes the uniqueness component so
        future RUNNING -> BLOCKED -> RUNNING -> BLOCKED transitions cannot
        collide on the events PK.
        """

        await self.__seed_thread()

        # Recording two messages emits MESSAGE_RECORDED twice; subjects are
        # the message ids (distinct) but the event id format embeds the
        # allocated sequence so even same-subject emissions can't collide.
        for index in range(3):
            await self.__interaction.record_message(
                request=RecordMessage(
                    identity=Identity(id=f"m-{index}", tenant="tenant-1"),
                    thread="thread-1",
                    author="actor-1",
                    kind=MessageKind.NOTE,
                    audience=Audience.THREAD,
                    content=Content(body={"text": f"m-{index}"}),
                    created=self.__now,
                )
            )

        events = await self.__interaction.get_events(
            query=EventQuery(tenant="tenant-1", thread="thread-1")
        )
        message_event_ids = [
            event.identity.id for event in events if event.kind == EventKind.MESSAGE_RECORDED
        ]
        self.assertEqual(len(message_event_ids), len(set(message_event_ids)))
        # Event ids are opaque UUIDs derived from (kind, subject, sequence) via
        # InteractionIdentity.stable; clients must not parse them, so the only
        # contract is well-formedness + per-emit uniqueness.
        for event_id in message_event_ids:
            uuid.UUID(event_id)

    # B10 ---------------------------------------------------------------

    async def test_b10_create_thread_raises_typed_conflict(self) -> None:
        """
        A second create_thread with a different title for the same id
        raises the typed ThreadConflictError; the recorder relies on this
        to disambiguate concurrent racers from generic InteractionErrors.
        """

        await self.__interaction.create_actor(
            request=CreateActor(
                identity=Identity(id="actor-1", tenant="tenant-1"),
                kind=ActorKind.HUMAN,
                name="Aman",
                created=self.__now,
            )
        )
        await self.__interaction.create_thread(
            request=CreateThread(
                identity=Identity(id="thread-x", tenant="tenant-1"),
                title="first",
                creator="actor-1",
                created=self.__now,
            )
        )

        with self.assertRaises(ThreadConflictError) as context:
            await self.__interaction.create_thread(
                request=CreateThread(
                    identity=Identity(id="thread-x", tenant="tenant-1"),
                    title="second",
                    creator="actor-1",
                    created=self.__now,
                )
            )

        self.assertEqual("thread-x", context.exception.thread)

    # B14 ---------------------------------------------------------------

    async def test_b14_sanitize_replay_with_reordered_labels_returns_existing(self) -> None:
        """
        Sanitizing the same content twice with labels supplied in different
        tuple order must succeed (returns the stored message) instead of
        falsely raising "different content".
        """

        await self.__seed_thread()
        await self.__interaction.record_message(
            request=RecordMessage(
                identity=Identity(id="msg-1", tenant="tenant-1"),
                thread="thread-1",
                author="actor-1",
                kind=MessageKind.NOTE,
                audience=Audience.THREAD,
                content=Content(
                    body={"text": "raw"},
                    labels=(Label.PRIVACY_OTP, Label.RETENTION_SHORT),
                ),
                created=self.__now,
            )
        )

        sanitized = self.__now.replace(second=10)
        original_request = Sanitize(
            tenant="tenant-1",
            message="msg-1",
            content=Content(
                body={"text": "[redacted]"},
                labels=(Label.PRIVACY_OTP, Label.RETENTION_SHORT),
                sanitizer="policy@1",
            ),
            sanitized=sanitized,
        )
        await self.__interaction.sanitize_message(request=original_request)

        # Replay with labels in reverse order — this used to falsely conflict.
        replay_request = original_request.model_copy(
            update={
                "content": Content(
                    body={"text": "[redacted]"},
                    labels=(Label.RETENTION_SHORT, Label.PRIVACY_OTP),
                    sanitizer="policy@1",
                ),
            }
        )
        message = await self.__interaction.sanitize_message(request=replay_request)
        self.assertEqual("msg-1", message.identity.id)

    async def test_b14_artifact_replay_with_reordered_labels_returns_existing(self) -> None:
        """
        Artifact replay also tolerates label reordering.
        """

        await self.__seed_thread()
        first = LinkArtifact(
            identity=Identity(id="art-1", tenant="tenant-1"),
            thread="thread-1",
            task="task-1",
            producer="actor-1",
            kind=ArtifactKind.SCREENSHOT,
            uri="/tmp/x.png",
            backend=ArtifactBackend.LOCAL,
            labels=(Label.RETENTION_SHORT, Label.DISPLAY_DEBUG),
            created=self.__now,
        )
        await self.__interaction.link_artifact(request=first)

        second = first.model_copy(update={"labels": (Label.DISPLAY_DEBUG, Label.RETENTION_SHORT)})
        replay = await self.__interaction.link_artifact(request=second)
        self.assertEqual("art-1", replay.identity.id)

    # B16 ---------------------------------------------------------------

    async def test_b16_fts_backfill_is_idempotent_after_partial_state(self) -> None:
        """
        Simulate a partial backfill by deleting one row from search
        and re-running the migration helper; the missing row must be
        re-inserted (no duplicate-key error, no permanent gap).
        """

        await self.__seed_thread()
        for index in range(3):
            await self.__interaction.record_message(
                request=RecordMessage(
                    identity=Identity(id=f"m-{index}", tenant="tenant-1"),
                    thread="thread-1",
                    author="actor-1",
                    kind=MessageKind.NOTE,
                    audience=Audience.THREAD,
                    content=Content(body={"text": f"m-{index}"}),
                    created=self.__now,
                )
            )

        import aiosqlite

        from fathom.infrastructure.interaction.pypika.sqlite import schema as schema_module

        # Tear out one FTS row to simulate a crash-truncated backfill.
        async with aiosqlite.connect(self.__path) as connection:
            await connection.execute(
                "DELETE FROM search WHERE rowid = (SELECT rowid FROM messages WHERE id = 'm-1')"
            )
            await connection.commit()

            # Re-run the anti-join backfill and confirm it restores the row.
            await connection.execute(schema_module.SEARCH_BACKFILL)
            await connection.commit()

            async with connection.execute("SELECT COUNT(*) FROM search") as cursor:
                row = await cursor.fetchone()

        assert row is not None
        self.assertEqual(3, int(row[0]))
