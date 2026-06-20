from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple

from fathom.adapters.interaction.pypika.sqlite import SQLiteInteraction
from fathom.adapters.signing.noop import NoopSigner
from fathom.constants.collaboration import (
    ActorKind,
    Audience,
    Label,
    MembershipRole,
    MessageKind,
    TaskKind,
    TaskState,
)
from fathom.core.services.conversation import ConversationService
from fathom.schemas.conversation import TimelineQuery
from fathom.schemas.interaction import (
    Assignment,
    Content,
    CreateActor,
    CreateThread,
    Identity,
    JoinThread,
    Lineage,
    OpenTask,
    Plan,
    RecordMessage,
)


class TestTimelinePaginationCorrectness(unittest.IsolatedAsyncioTestCase):
    """
    Verify the consume-emit walk holds the global limit and never skips or
    duplicates entries across pages, including pages of all-hidden rows.
    """

    def setUp(self) -> None:
        """
        Create an isolated conversation service for each pagination test.
        """

        self.__directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.__directory.cleanup)
        path = Path(self.__directory.name) / "timeline.db"
        self.__interaction = SQLiteInteraction(path=path)
        self.__service = ConversationService(signer=NoopSigner(), interaction=self.__interaction)
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

    async def __record_message(self, *, identifier: str, second: int, hidden: bool = False) -> None:
        """
        Record one timeline message at a deterministic timestamp.
        """

        labels: Tuple[Label, ...] = (Label.DISPLAY_HIDDEN,) if hidden else ()
        await self.__interaction.record_message(
            request=RecordMessage(
                identity=Identity(id=identifier, tenant="tenant-1"),
                thread="thread-1",
                task="task-1",
                author="actor-1",
                kind=MessageKind.NOTE,
                audience=Audience.THREAD,
                content=Content(body={"text": identifier}, labels=labels),
                created=self.__now + timedelta(seconds=second),
            )
        )

    async def test_global_limit_caps_total_emitted_entries(self) -> None:
        """
        Page must hold at most query.limit entries even when multiple kinds
        each have rows available — replaces the previous 4*limit regression.
        """

        await self.__seed_thread()
        for second in range(6):
            await self.__record_message(identifier=f"m-{second}", second=second)

        page = await self.__service.timeline(
            query=TimelineQuery(tenant="tenant-1", thread="thread-1", limit=2)
        )

        self.assertLessEqual(len(page.entries), 2)
        self.assertIsNotNone(page.next)

    async def test_pages_walk_every_entry_exactly_once(self) -> None:
        """
        Repeatedly fetch pages until cursor exhausts; every emitted id appears
        once and no id is duplicated across pages.
        """

        await self.__seed_thread()
        for second in range(7):
            await self.__record_message(identifier=f"m-{second}", second=second)

        seen: List[str] = []
        cursor: str | None = None
        for _ in range(20):
            page = await self.__service.timeline(
                query=TimelineQuery(tenant="tenant-1", thread="thread-1", limit=2, cursor=cursor)
            )
            seen.extend(entry.id for entry in page.entries)
            if page.next is None:
                break
            cursor = page.next

        self.assertEqual(sorted(seen), [f"m-{i}" for i in range(7)])
        self.assertEqual(len(seen), len(set(seen)))

    async def test_all_hidden_page_still_advances_cursor(self) -> None:
        """
        A page made entirely of hidden rows must still advance the cursor —
        the next call returns the next visible entries instead of looping.
        """

        await self.__seed_thread()
        # First two entries are hidden, third is visible.
        await self.__record_message(identifier="m-hidden-1", second=0, hidden=True)
        await self.__record_message(identifier="m-hidden-2", second=1, hidden=True)
        await self.__record_message(identifier="m-visible", second=2)

        first = await self.__service.timeline(
            query=TimelineQuery(tenant="tenant-1", thread="thread-1", limit=1)
        )

        # The walk consumes both hidden entries and emits the visible one.
        self.assertEqual(["m-visible"], [entry.id for entry in first.entries])

    async def test_terminal_page_returns_null_cursor(self) -> None:
        """
        Once every kind is exhausted, the composite cursor collapses to None.
        """

        await self.__seed_thread()
        await self.__record_message(identifier="m-only", second=0)

        first = await self.__service.timeline(
            query=TimelineQuery(tenant="tenant-1", thread="thread-1", limit=10)
        )

        self.assertEqual(["m-only"], [entry.id for entry in first.entries])
        self.assertIsNone(first.next)


class TestTimelineSortOrderEndToEnd(unittest.IsolatedAsyncioTestCase):
    """
    End-to-end SQLite-backed coverage of the timeline + per-kind list sort
    order. The default is DESC (chat-style: newest first); ASC is opt-in.
    """

    def setUp(self) -> None:
        """
        Build a fresh service per test, anchored at a deterministic timestamp.
        """

        self.__directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.__directory.cleanup)
        path = Path(self.__directory.name) / "sort_order.db"
        self.__interaction = SQLiteInteraction(path=path)
        self.__service = ConversationService(signer=NoopSigner(), interaction=self.__interaction)
        self.__now = datetime(2026, 5, 4, 10, 0, 0, tzinfo=timezone.utc)

    async def asyncTearDown(self) -> None:
        """
        Release the SQLite backend opened in setUp.
        """

        await self.__interaction.aclose()

    async def __seed_thread(self) -> None:
        """
        Seed one actor, thread, membership, and a running task.
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
                title="ordering",
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

    async def __record_message(self, *, identifier: str, second: int) -> None:
        """
        Record one timeline message at a deterministic timestamp.
        """

        await self.__interaction.record_message(
            request=RecordMessage(
                identity=Identity(id=identifier, tenant="tenant-1"),
                thread="thread-1",
                task="task-1",
                author="actor-1",
                kind=MessageKind.NOTE,
                audience=Audience.THREAD,
                content=Content(body={"text": identifier}),
                created=self.__now + timedelta(seconds=second),
            )
        )

    async def test_timeline_defaults_to_desc_when_order_omitted(self) -> None:
        """
        With no `order` field set, the page returns newest first.
        """

        await self.__seed_thread()
        for second in range(5):
            await self.__record_message(identifier=f"m-{second}", second=second)

        page = await self.__service.timeline(
            query=TimelineQuery(tenant="tenant-1", thread="thread-1", limit=10),
        )

        self.assertEqual(
            ["m-4", "m-3", "m-2", "m-1", "m-0"],
            [entry.id for entry in page.entries],
        )

    async def test_timeline_asc_opt_in_returns_oldest_first(self) -> None:
        """
        Passing order=ASC restores the oldest-first behaviour.
        """

        from fathom.schemas.interaction import SortOrder

        await self.__seed_thread()
        for second in range(5):
            await self.__record_message(identifier=f"m-{second}", second=second)

        page = await self.__service.timeline(
            query=TimelineQuery(
                tenant="tenant-1",
                thread="thread-1",
                limit=10,
                order=SortOrder.ASC,
            ),
        )

        self.assertEqual(
            ["m-0", "m-1", "m-2", "m-3", "m-4"],
            [entry.id for entry in page.entries],
        )

    async def test_desc_cursor_advances_to_older_pages(self) -> None:
        """
        Under DESC, the page cursor must walk toward older items so the
        chat-style scroll-up pattern fetches the prior page.
        """

        await self.__seed_thread()
        for second in range(5):
            await self.__record_message(identifier=f"m-{second}", second=second)

        first = await self.__service.timeline(
            query=TimelineQuery(tenant="tenant-1", thread="thread-1", limit=2),
        )
        second = await self.__service.timeline(
            query=TimelineQuery(
                tenant="tenant-1",
                thread="thread-1",
                limit=2,
                cursor=first.next,
            ),
        )

        self.assertEqual(["m-4", "m-3"], [entry.id for entry in first.entries])
        self.assertEqual(["m-2", "m-1"], [entry.id for entry in second.entries])

    async def test_asc_cursor_advances_to_newer_pages(self) -> None:
        """
        Under ASC the cursor must walk toward newer items — the historical
        traversal direction is preserved as the opt-in path.
        """

        from fathom.schemas.interaction import SortOrder

        await self.__seed_thread()
        for second in range(5):
            await self.__record_message(identifier=f"m-{second}", second=second)

        first = await self.__service.timeline(
            query=TimelineQuery(
                tenant="tenant-1",
                thread="thread-1",
                limit=2,
                order=SortOrder.ASC,
            ),
        )
        second = await self.__service.timeline(
            query=TimelineQuery(
                tenant="tenant-1",
                thread="thread-1",
                limit=2,
                cursor=first.next,
                order=SortOrder.ASC,
            ),
        )

        self.assertEqual(["m-0", "m-1"], [entry.id for entry in first.entries])
        self.assertEqual(["m-2", "m-3"], [entry.id for entry in second.entries])

    async def test_desc_walk_emits_every_entry_exactly_once(self) -> None:
        """
        Walking pages under DESC exhausts the data set with no duplicates.
        """

        await self.__seed_thread()
        for second in range(7):
            await self.__record_message(identifier=f"m-{second}", second=second)

        seen: List[str] = []
        cursor: str | None = None
        for _ in range(20):
            page = await self.__service.timeline(
                query=TimelineQuery(
                    tenant="tenant-1",
                    thread="thread-1",
                    limit=2,
                    cursor=cursor,
                ),
            )
            seen.extend(entry.id for entry in page.entries)
            if page.next is None:
                break
            cursor = page.next

        self.assertEqual(
            ["m-6", "m-5", "m-4", "m-3", "m-2", "m-1", "m-0"],
            seen,
        )

    async def test_desc_interleaves_messages_and_artifacts_newest_first(self) -> None:
        """
        Mixed-kind composition under DESC interleaves messages + artifacts in
        a single newest-first stream, not grouped by kind.
        """

        from fathom.constants.collaboration import ArtifactBackend, ArtifactKind
        from fathom.schemas.interaction import LinkArtifact

        await self.__seed_thread()
        await self.__record_message(identifier="m-1", second=1)
        await self.__interaction.link_artifact(
            request=LinkArtifact(
                identity=Identity(id="artifact-2", tenant="tenant-1"),
                thread="thread-1",
                task="task-1",
                producer="actor-1",
                kind=ArtifactKind.SCREENSHOT,
                uri="/tmp/a-2.png",
                backend=ArtifactBackend.LOCAL,
                mime="image/png",
                created=self.__now + timedelta(seconds=2),
            ),
        )
        await self.__record_message(identifier="m-3", second=3)
        await self.__interaction.link_artifact(
            request=LinkArtifact(
                identity=Identity(id="artifact-4", tenant="tenant-1"),
                thread="thread-1",
                task="task-1",
                producer="actor-1",
                kind=ArtifactKind.SCREENSHOT,
                uri="/tmp/a-4.png",
                backend=ArtifactBackend.LOCAL,
                mime="image/png",
                created=self.__now + timedelta(seconds=4),
            ),
        )

        page = await self.__service.timeline(
            query=TimelineQuery(tenant="tenant-1", thread="thread-1", limit=10),
        )

        self.assertEqual(
            ["artifact-4", "m-3", "artifact-2", "m-1"],
            [entry.id for entry in page.entries],
        )


class TestTimelineComposerSortOrder(unittest.TestCase):
    """
    Direct unit tests for the merge-sort direction in TimelineComposer.
    """

    def __build(self, *, order):
        """
        Build a composer view from in-memory rows with the requested order.
        """

        from fathom.constants.collaboration import (
            ArtifactBackend,
            ArtifactKind,
            Audience,
            MessageKind,
            ThreadState,
        )
        from fathom.conversation.cursor import CompositeTimelineCursor
        from fathom.conversation.timeline import TimelineComposer
        from fathom.schemas.conversation import TimelineQuery
        from fathom.schemas.interaction import (
            Artifact,
            Content,
            Identity,
            Message,
            Thread,
            Timing,
        )

        anchor = datetime(2026, 5, 4, 10, 0, 0, tzinfo=timezone.utc)
        thread = Thread(
            identity=Identity(id="thread-1", tenant="tenant-1"),
            title="t",
            state=ThreadState.ACTIVE,
            creator="actor-1",
            timing=Timing(created=anchor, updated=anchor),
        )
        messages = (
            Message(
                identity=Identity(id="m-1", tenant="tenant-1"),
                thread="thread-1",
                task="task-1",
                author="actor-1",
                kind=MessageKind.NOTE,
                audience=Audience.THREAD,
                content=Content(body={"text": "first"}),
                sequence=1,
                created=anchor + timedelta(seconds=1),
            ),
            Message(
                identity=Identity(id="m-2", tenant="tenant-1"),
                thread="thread-1",
                task="task-1",
                author="actor-1",
                kind=MessageKind.NOTE,
                audience=Audience.THREAD,
                content=Content(body={"text": "third"}),
                sequence=2,
                created=anchor + timedelta(seconds=3),
            ),
        )
        artifacts = (
            Artifact(
                identity=Identity(id="artifact-1", tenant="tenant-1"),
                thread="thread-1",
                task="task-1",
                producer="actor-1",
                kind=ArtifactKind.SCREENSHOT,
                uri="/tmp/a.png",
                backend=ArtifactBackend.LOCAL,
                mime="image/png",
                created=anchor + timedelta(seconds=2),
            ),
        )

        composer = TimelineComposer()
        return composer.build(
            thread=thread,
            messages=messages,
            events=(),
            artifacts=artifacts,
            contexts=(),
            inbound=CompositeTimelineCursor(),
            has_more={"messages": False, "events": False, "artifacts": False, "contexts": False},
            total=3,
            query=TimelineQuery(
                tenant="tenant-1",
                thread="thread-1",
                order=order,
                limit=10,
            ),
        )

    def test_composer_emits_desc_when_order_is_desc(self) -> None:
        """
        DESC order puts the newest entry first, regardless of source kind.
        """

        from fathom.schemas.interaction import SortOrder

        view = self.__build(order=SortOrder.DESC)
        self.assertEqual(["m-2", "artifact-1", "m-1"], [e.id for e in view.entries])

    def test_composer_emits_asc_when_order_is_asc(self) -> None:
        """
        ASC order preserves the historical oldest-first traversal.
        """

        from fathom.schemas.interaction import SortOrder

        view = self.__build(order=SortOrder.ASC)
        self.assertEqual(["m-1", "artifact-1", "m-2"], [e.id for e in view.entries])
