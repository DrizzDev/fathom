from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fathom.adapters.interaction.pypika.sqlite import SQLiteInteraction
from fathom.adapters.signing.noop import NoopSigner
from fathom.constants.collaboration import (
    ActorKind,
    ArtifactBackend,
    ArtifactKind,
    Audience,
    ContextPurpose,
    MembershipRole,
    MessageKind,
    TaskCode,
    TaskKind,
    TaskState,
    ThreadState,
)
from fathom.constants.conversation import EntryKind, Visibility
from fathom.core.exceptions import InteractionError, ThreadNotFoundError
from fathom.core.services.conversation import ConversationService
from fathom.schemas.conversation import (
    ActorInput,
    ArtifactAttach,
    ArtifactListQuery,
    ContextRecord,
    ConversationListQuery,
    ConversationThreadQuery,
    ConversationTransition,
    MessageAppend,
    MessageListQuery,
    RunScriptQuery,
    ScriptsQuery,
    TaskFinish,
    TaskStart,
    TaskTreeQuery,
    ThreadCreate,
    TimelineQuery,
)
from fathom.schemas.interaction import (
    Assignment,
    BuildContext,
    Content,
    CreateActor,
    CreateThread,
    Identity,
    JoinThread,
    Lineage,
    LinkArtifact,
    OpenTask,
    Plan,
    RecordMessage,
    References,
    SaveScript,
    SortOrder,
    TaskQuery,
    ThreadQuery,
)


class TestConversationService(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for client-facing conversation read service.
    """

    def setUp(self) -> None:
        """
        Create an isolated conversation service backed by SQLite.
        """

        self.__temporary_directory = tempfile.TemporaryDirectory()

        self.addCleanup(self.__temporary_directory.cleanup)
        path = Path(self.__temporary_directory.name) / "conversation.db"

        self.__interaction = SQLiteInteraction(path=path)
        self.__service = ConversationService(signer=NoopSigner(), interaction=self.__interaction)
        self.__now = datetime(2026, 4, 28, 10, 0, 0, tzinfo=timezone.utc)

    async def test_get_thread_returns_client_view(self) -> None:
        """
        Load a stored ledger thread as a client-facing thread view.
        """

        await self.__create_records()

        view = await self.__service.get(
            query=ConversationThreadQuery(tenant="tenant-1", thread="thread-1")
        )

        self.assertEqual("thread-1", view.id)
        self.assertEqual("Buy milk", view.title)

    async def test_archive_hides_thread_until_unarchived(self) -> None:
        """
        Archived conversations should leave public reads but remain restorable.
        """

        await self.__create_records()

        archived = await self.__service.archive(
            request=ConversationTransition(
                tenant="tenant-1",
                thread="thread-1",
                updated=self.__now.replace(minute=1),
            )
        )
        default_page = await self.__service.list(query=ConversationListQuery(tenant="tenant-1"))
        active_page = await self.__service.list(
            query=ConversationListQuery(tenant="tenant-1", state=ThreadState.ACTIVE.value)
        )
        archived_page = await self.__service.list(
            query=ConversationListQuery(tenant="tenant-1", state=ThreadState.ARCHIVED.value)
        )

        with self.assertRaises(ThreadNotFoundError):
            await self.__service.get(
                query=ConversationThreadQuery(tenant="tenant-1", thread="thread-1")
            )
        with self.assertRaises(ThreadNotFoundError):
            await self.__service.messages(
                query=MessageListQuery(tenant="tenant-1", thread="thread-1")
            )
        with self.assertRaises(ThreadNotFoundError):
            await self.__service.artifacts(
                query=ArtifactListQuery(tenant="tenant-1", thread="thread-1")
            )
        with self.assertRaises(ThreadNotFoundError):
            await self.__service.tasks(query=TaskTreeQuery(tenant="tenant-1", thread="thread-1"))
        with self.assertRaises(ThreadNotFoundError):
            await self.__service.timeline(query=TimelineQuery(tenant="tenant-1", thread="thread-1"))
        with self.assertRaises(ThreadNotFoundError):
            await self.__service.list_scripts(
                query=ScriptsQuery(tenant="tenant-1", thread="thread-1")
            )
        with self.assertRaises(ThreadNotFoundError):
            await self.__service.script(
                query=RunScriptQuery(tenant="tenant-1", thread="thread-1", task="task-1")
            )

        unarchived = await self.__service.unarchive(
            request=ConversationTransition(
                tenant="tenant-1",
                thread="thread-1",
                updated=self.__now.replace(minute=2),
            )
        )
        restored_page = await self.__service.list(query=ConversationListQuery(tenant="tenant-1"))

        self.assertEqual(ThreadState.ARCHIVED, archived.state)
        self.assertEqual((), default_page.items)
        self.assertEqual((), active_page.items)
        self.assertEqual(("thread-1",), tuple(item.id for item in archived_page.items))
        self.assertEqual(ThreadState.ACTIVE, unarchived.state)
        self.assertEqual(("thread-1",), tuple(item.id for item in restored_page.items))

    async def test_delete_hides_thread_and_thread_owned_records(self) -> None:
        """
        Deleted conversations should 404 and their child collections should disappear.
        """

        await self.__create_records()

        deleted = await self.__service.delete(
            request=ConversationTransition(
                tenant="tenant-1",
                thread="thread-1",
                updated=self.__now.replace(minute=3),
            )
        )
        page = await self.__service.list(query=ConversationListQuery(tenant="tenant-1"))
        tasks = await self.__interaction.get_tasks(
            query=TaskQuery(tenant="tenant-1", thread="thread-1")
        )

        self.assertEqual(ThreadState.DELETED, deleted.state)
        self.assertEqual((), page.items)
        self.assertEqual([], tasks)
        with self.assertRaises(ThreadNotFoundError):
            await self.__service.get(
                query=ConversationThreadQuery(tenant="tenant-1", thread="thread-1")
            )

    async def test_create_thread_creates_creator_membership(self) -> None:
        """
        Create a thread with its creator and owner membership.
        """

        view = await self.__service.create(
            request=ThreadCreate(
                id="thread-2",
                tenant="tenant-1",
                title="Plan groceries",
                creator=ActorInput(id="actor-2", kind=ActorKind.HUMAN, name="Aman"),
                member="member-2",
                created=self.__now,
            )
        )

        message = await self.__service.append(
            request=MessageAppend(
                id="message-2",
                tenant="tenant-1",
                thread="thread-2",
                author="actor-2",
                kind=MessageKind.REQUEST,
                body={"text": "Plan groceries"},
                created=self.__now.replace(second=1),
            )
        )

        self.assertEqual("thread-2", view.id)
        self.assertEqual(EntryKind.MESSAGE, message.kind)
        self.assertEqual("actor-2", message.actor)

    async def test_create_thread_rolls_back_when_creator_membership_fails(self) -> None:
        """
        Keep conversation creation atomic when creator membership cannot be stored.
        """

        await self.__create_records()

        with self.assertRaises(InteractionError):
            await self.__service.create(
                request=ThreadCreate(
                    id="thread-rollback",
                    tenant="tenant-1",
                    title="Plan rollback",
                    creator=ActorInput(id="actor-1", kind=ActorKind.HUMAN, name="Aman"),
                    member="member-1",
                    created=self.__now.replace(second=5),
                )
            )

        thread = await self.__interaction.get_thread(
            query=ThreadQuery(tenant="tenant-1", thread="thread-rollback")
        )

        self.assertIsNone(thread)

    async def test_create_thread_rejects_creator_without_membership(self) -> None:
        """
        Require caller-owned identifiers for creator membership records.
        """

        with self.assertRaises(ValueError):
            ThreadCreate(
                id="thread-2",
                tenant="tenant-1",
                creator=ActorInput(id="actor-2", kind=ActorKind.HUMAN, name="Aman"),
                created=self.__now,
            )

    async def test_append_message_returns_renderable_entry(self) -> None:
        """
        Append a message through the service and return a timeline entry.
        """

        await self.__create_records()

        entry = await self.__service.append(
            request=MessageAppend(
                id="message-2",
                tenant="tenant-1",
                thread="thread-1",
                task="task-1",
                author="actor-1",
                kind=MessageKind.NOTE,
                body={"text": "Use wallet balance"},
                created=self.__now.replace(second=4),
            )
        )

        self.assertEqual("message-2", entry.id)
        self.assertEqual(EntryKind.MESSAGE, entry.kind)
        self.assertEqual({"text": "Use wallet balance"}, entry.payload["body"])

    async def test_task_write_use_cases_return_task_views(self) -> None:
        """
        Start and finish a task through the conversation service.
        """

        await self.__create_records()

        started = await self.__service.start(
            request=TaskStart(
                id="task-2",
                tenant="tenant-1",
                thread="thread-1",
                creator="actor-1",
                assignee="actor-1",
                parent="task-1",
                root="task-1",
                kind=TaskKind.DELEGATION,
                objective="Verify checkout",
                created=self.__now.replace(second=4),
            )
        )
        finished = await self.__service.finish(
            request=TaskFinish(
                tenant="tenant-1",
                task="task-2",
                state=TaskState.SUCCEEDED,
                code=TaskCode.COMPLETED,
                summary="Checkout verified",
                ended=self.__now.replace(second=5),
                elapsed=1000,
            )
        )

        self.assertEqual("task-2", started.id)
        self.assertEqual("running", started.state)
        self.assertEqual("succeeded", finished.state)
        self.assertEqual("Checkout verified", finished.summary)

    async def test_artifact_and_context_write_use_cases_return_timeline_entries(self) -> None:
        """
        Attach artifacts and record context recipes through the service.
        """

        await self.__create_records()

        artifact = await self.__service.attach(
            request=ArtifactAttach(
                id="artifact-2",
                tenant="tenant-1",
                thread="thread-1",
                task="task-1",
                producer="actor-1",
                kind=ArtifactKind.REPORT,
                uri="/tmp/report.json",
                backend=ArtifactBackend.LOCAL,
                mime="application/json",
                created=self.__now.replace(second=4),
            )
        )
        context = await self.__service.record(
            request=ContextRecord(
                id="context-2",
                tenant="tenant-1",
                thread="thread-1",
                task="task-1",
                consumer="actor-1",
                purpose=ContextPurpose.EXECUTION,
                builder="conversation@1",
                messages=("message-1",),
                artifacts=("artifact-2",),
                created=self.__now.replace(second=5),
            )
        )

        self.assertEqual(EntryKind.ARTIFACT, artifact.kind)
        self.assertEqual(Visibility.USER, artifact.visibility)
        self.assertEqual(EntryKind.CONTEXT, context.kind)
        self.assertEqual(Visibility.AUDIT, context.visibility)

    async def test_build_timeline_hides_internal_entries_by_default(self) -> None:
        """
        Render only user-visible entries for the default timeline mode.
        """

        await self.__create_records()

        timeline = await self.__service.timeline(
            query=TimelineQuery(tenant="tenant-1", thread="thread-1")
        )

        self.assertEqual(
            [EntryKind.ARTIFACT, EntryKind.MESSAGE],
            [entry.kind for entry in timeline.entries],
        )
        self.assertEqual(2, timeline.total)

    async def test_build_timeline_exposes_debug_and_audit_modes(self) -> None:
        """
        Include lifecycle events in debug mode and context recipes in audit mode.
        """

        await self.__create_records()

        debug = await self.__service.timeline(
            query=TimelineQuery(tenant="tenant-1", thread="thread-1", mode=Visibility.DEBUG)
        )
        audit = await self.__service.timeline(
            query=TimelineQuery(tenant="tenant-1", thread="thread-1", mode=Visibility.AUDIT)
        )

        self.assertIn(EntryKind.EVENT, [entry.kind for entry in debug.entries])
        self.assertNotIn(EntryKind.CONTEXT, [entry.kind for entry in debug.entries])
        self.assertIn(EntryKind.CONTEXT, [entry.kind for entry in audit.entries])

    async def test_build_timeline_filters_by_task(self) -> None:
        """
        Render entries scoped to one task when a task filter is supplied.
        """

        await self.__create_records()
        await self.__interaction.record_message(
            request=RecordMessage(
                identity=self.__identity(id="message-thread"),
                thread="thread-1",
                author="actor-1",
                sequence=None,
                kind=MessageKind.NOTE,
                audience=Audience.THREAD,
                content=Content(body={"text": "Thread-only note"}),
                created=self.__now.replace(second=4),
            )
        )

        timeline = await self.__service.timeline(
            query=TimelineQuery(tenant="tenant-1", thread="thread-1", task="task-1")
        )

        self.assertEqual(["artifact-1", "message-1"], [entry.id for entry in timeline.entries])

    async def test_get_run_script_returns_latest_revision(self) -> None:
        """
        When a run produced multiple script records, the convenience endpoint must return the most recently updated script per the API contract.
        """

        await self.__create_records()

        for second, identifier in enumerate(["script-old", "script-mid", "script-new"]):
            await self.__interaction.save_script(
                request=SaveScript(
                    identity=self.__identity(id=identifier),
                    thread="thread-1",
                    task="task-1",
                    content=f"OPEN_APP {identifier}",
                    summary=f"version of {identifier}",
                    created=self.__now.replace(second=second + 1),
                )
            )

        result = await self.__service.script(
            query=RunScriptQuery(tenant="tenant-1", thread="thread-1", task="task-1")
        )

        assert result is not None
        self.assertEqual("script-new", result.id)

    async def test_build_timeline_paginates_with_cursor(self) -> None:
        """
        Render timeline pages with an opaque cursor that walks every entry once.
        """

        await self.__create_records()
        await self.__interaction.record_message(
            request=RecordMessage(
                identity=self.__identity(id="message-2"),
                thread="thread-1",
                author="actor-1",
                sequence=None,
                kind=MessageKind.NOTE,
                audience=Audience.THREAD,
                content=Content(body={"text": "Second note"}),
                created=self.__now.replace(second=4),
            )
        )

        first = await self.__service.timeline(
            query=TimelineQuery(tenant="tenant-1", thread="thread-1", limit=1)
        )
        first_ids = {entry.id for entry in first.entries}
        self.assertIn("message-2", first_ids)
        self.assertIsNotNone(first.next)

        second = await self.__service.timeline(
            query=TimelineQuery(
                tenant="tenant-1",
                thread="thread-1",
                limit=2,
                cursor=first.next,
            )
        )
        second_ids = {entry.id for entry in second.entries}
        self.assertEqual(set(), first_ids & second_ids)
        self.assertIn("message-1", second_ids)

    async def test_get_task_tree_renders_nested_tasks(self) -> None:
        """
        Render parent and child tasks as a nested client-facing tree.
        """

        await self.__create_records()
        await self.__interaction.open_task(
            request=OpenTask(
                identity=self.__identity(id="task-2"),
                thread="thread-1",
                assignment=Assignment(creator="actor-1", assignee="actor-1"),
                lineage=Lineage(parent="task-1", root="task-1"),
                kind=TaskKind.DELEGATION,
                state=TaskState.RUNNING,
                plan=Plan(objective="Verify checkout"),
                created=self.__now.replace(second=5),
            )
        )

        tree = await self.__service.tasks(query=TaskTreeQuery(tenant="tenant-1", thread="thread-1"))

        self.assertEqual(2, tree.total)
        self.assertEqual("task-1", tree.roots[0].id)
        self.assertEqual("task-2", tree.roots[0].children[0].id)
        self.assertEqual("Verify checkout", tree.roots[0].children[0].objective)

    async def test_list_scripts_returns_paginated_view(self) -> None:
        """
        list_scripts maps the durable script page into a client-facing ScriptPage.
        """

        await self.__create_records()

        for index in range(3):
            await self.__interaction.save_script(
                request=SaveScript(
                    identity=self.__identity(id=f"script-{index}"),
                    thread="thread-1",
                    task="task-1",
                    content=f"OPEN_APP {index}",
                    summary=f"draft {index}",
                    created=self.__now.replace(second=index + 1),
                )
            )

        first_page = await self.__service.list_scripts(
            query=ScriptsQuery(tenant="tenant-1", thread="thread-1", limit=2)
        )
        second_page = await self.__service.list_scripts(
            query=ScriptsQuery(
                tenant="tenant-1",
                thread="thread-1",
                limit=2,
                cursor=first_page.next,
            )
        )

        self.assertEqual(3, first_page.total)
        self.assertEqual(2, len(first_page.items))
        self.assertIsNotNone(first_page.next)
        self.assertEqual(1, len(second_page.items))
        self.assertIsNone(second_page.next)
        seen = {item.id for item in first_page.items} | {item.id for item in second_page.items}
        self.assertEqual({"script-0", "script-1", "script-2"}, seen)

    async def test_list_scripts_rejects_missing_thread(self) -> None:
        """
        list_scripts surfaces InteractionError when the thread does not exist.
        """

        with self.assertRaises(InteractionError):
            await self.__service.list_scripts(
                query=ScriptsQuery(tenant="tenant-1", thread="missing", limit=10)
            )

    async def test_script_returns_none_when_no_script_for_task(self) -> None:
        """
        The single-run script lookup returns None instead of raising when the run never produced a script.
        """

        await self.__create_records()

        result = await self.__service.script(
            query=RunScriptQuery(tenant="tenant-1", thread="thread-1", task="task-1")
        )

        self.assertIsNone(result)

    async def test_get_thread_rejects_missing_thread(self) -> None:
        """
        Fail clearly when a requested conversation thread does not exist.
        """

        with self.assertRaises(InteractionError):
            await self.__service.get(
                query=ConversationThreadQuery(tenant="tenant-1", thread="missing")
            )

    async def __create_records(self) -> None:
        """
        Create one thread with message, event, artifact, and context records.
        """

        await self.__interaction.create_actor(
            request=CreateActor(
                identity=self.__identity(id="actor-1"),
                kind=ActorKind.HUMAN,
                name="Aman",
                created=self.__now,
            )
        )
        await self.__interaction.create_thread(
            request=CreateThread(
                identity=self.__identity(id="thread-1"),
                title="Buy milk",
                creator="actor-1",
                created=self.__now,
            )
        )
        await self.__interaction.join_thread(
            request=JoinThread(
                identity=self.__identity(id="member-1"),
                thread="thread-1",
                actor="actor-1",
                role=MembershipRole.OWNER,
                joined=self.__now,
            )
        )
        await self.__interaction.open_task(
            request=OpenTask(
                identity=self.__identity(id="task-1"),
                thread="thread-1",
                assignment=Assignment(creator="actor-1", assignee="actor-1"),
                kind=TaskKind.FATHOM,
                state=TaskState.RUNNING,
                plan=Plan(objective="Buy milk"),
                created=self.__now,
            )
        )
        await self.__interaction.record_message(
            request=RecordMessage(
                identity=self.__identity(id="message-1"),
                thread="thread-1",
                task="task-1",
                author="actor-1",
                sequence=None,
                kind=MessageKind.REQUEST,
                audience=Audience.THREAD,
                content=Content(body={"text": "Buy milk"}),
                created=self.__now.replace(second=1),
            )
        )
        await self.__interaction.link_artifact(
            request=LinkArtifact(
                identity=self.__identity(id="artifact-1"),
                thread="thread-1",
                task="task-1",
                producer="actor-1",
                kind=ArtifactKind.SCREENSHOT,
                uri="/tmp/screenshot.png",
                backend=ArtifactBackend.LOCAL,
                mime="image/png",
                created=self.__now.replace(second=2),
            )
        )
        await self.__interaction.build_context(
            request=BuildContext(
                identity=self.__identity(id="context-1"),
                thread="thread-1",
                task="task-1",
                consumer="actor-1",
                purpose=ContextPurpose.EXECUTION,
                builder="test@1",
                references=References(messages=("message-1",), artifacts=("artifact-1",)),
                created=self.__now.replace(second=3),
            )
        )

    def __identity(self, *, id: str) -> Identity:
        """
        Create a tenant-scoped identity for tests.
        """

        return Identity(id=id, tenant="tenant-1")


class TestConversationServiceMessageOrdering(unittest.IsolatedAsyncioTestCase):
    """
    Cover the sort-order contract of `ConversationService.messages()`.
    DESC is the default; ASC is opt-in; cursor pagination respects direction.
    """

    def setUp(self) -> None:
        """
        Build a fresh SQLite-backed service anchored at a deterministic timestamp.
        """

        self.__temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.__temporary_directory.cleanup)
        path = Path(self.__temporary_directory.name) / "messages_order.db"
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
        Seed actor + thread + membership + running task.
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
                title="messages-order",
                creator="actor-1",
                created=self.__now,
            )
        )
        await self.__interaction.join_thread(
            request=JoinThread(
                identity=Identity(id="member-1", tenant="tenant-1"),
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

    async def __record(self, *, identifier: str, second: int) -> None:
        """
        Record one NOTE message at a deterministic timestamp.
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

    async def test_default_order_returns_newest_first(self) -> None:
        """
        Without an explicit `order` the service returns the newest message first.
        """

        await self.__seed_thread()
        for second in range(4):
            await self.__record(identifier=f"m-{second}", second=second)

        page = await self.__service.messages(
            query=MessageListQuery(tenant="tenant-1", thread="thread-1", limit=10),
        )

        self.assertEqual(["m-3", "m-2", "m-1", "m-0"], [m.id for m in page.items])

    async def test_ascending_opt_in_returns_oldest_first(self) -> None:
        """
        Passing order=ASC restores the historical oldest-first behaviour.
        """

        await self.__seed_thread()
        for second in range(4):
            await self.__record(identifier=f"m-{second}", second=second)

        page = await self.__service.messages(
            query=MessageListQuery(
                tenant="tenant-1",
                thread="thread-1",
                limit=10,
                order=SortOrder.ASC,
            ),
        )

        self.assertEqual(["m-0", "m-1", "m-2", "m-3"], [m.id for m in page.items])

    async def test_descending_cursor_walks_toward_older_messages(self) -> None:
        """
        Under DESC the next-page cursor must surface progressively older items.
        """

        await self.__seed_thread()
        for second in range(5):
            await self.__record(identifier=f"m-{second}", second=second)

        first = await self.__service.messages(
            query=MessageListQuery(tenant="tenant-1", thread="thread-1", limit=2),
        )
        second = await self.__service.messages(
            query=MessageListQuery(
                tenant="tenant-1",
                thread="thread-1",
                limit=2,
                cursor=first.next,
            ),
        )

        self.assertEqual(["m-4", "m-3"], [m.id for m in first.items])
        self.assertEqual(["m-2", "m-1"], [m.id for m in second.items])


class TestConversationServiceArtifactOrdering(unittest.IsolatedAsyncioTestCase):
    """
    Cover the sort-order contract of `ConversationService.artifacts()`.
    """

    def setUp(self) -> None:
        """
        Build a fresh SQLite-backed service anchored at a deterministic timestamp.
        """

        self.__temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.__temporary_directory.cleanup)
        path = Path(self.__temporary_directory.name) / "artifacts_order.db"
        self.__interaction = SQLiteInteraction(path=path)
        self.__service = ConversationService(signer=NoopSigner(), interaction=self.__interaction)
        self.__now = datetime(2026, 5, 4, 10, 0, 0, tzinfo=timezone.utc)

    async def asyncTearDown(self) -> None:
        """
        Release the SQLite backend opened in setUp.
        """

        await self.__interaction.aclose()

    async def __seed_thread_with_artifacts(self, *, count: int) -> None:
        """
        Seed actor + thread + task + `count` artifacts at staggered timestamps.
        """

        await self.__interaction.create_actor(
            request=CreateActor(
                identity=Identity(id="actor-1", tenant="tenant-1"),
                kind=ActorKind.HUMAN,
                name="Aman",
                created=self.__now,
            ),
        )
        await self.__interaction.create_thread(
            request=CreateThread(
                identity=Identity(id="thread-1", tenant="tenant-1"),
                title="artifacts-order",
                creator="actor-1",
                created=self.__now,
            ),
        )
        await self.__interaction.join_thread(
            request=JoinThread(
                identity=Identity(id="member-1", tenant="tenant-1"),
                thread="thread-1",
                actor="actor-1",
                role=MembershipRole.OWNER,
                joined=self.__now,
            ),
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
            ),
        )
        for second in range(count):
            await self.__interaction.link_artifact(
                request=LinkArtifact(
                    identity=Identity(id=f"artifact-{second}", tenant="tenant-1"),
                    thread="thread-1",
                    task="task-1",
                    producer="actor-1",
                    kind=ArtifactKind.SCREENSHOT,
                    uri=f"/tmp/{second}.png",
                    backend=ArtifactBackend.LOCAL,
                    mime="image/png",
                    created=self.__now + timedelta(seconds=second),
                ),
            )

    async def test_default_order_returns_newest_first(self) -> None:
        """
        Without an explicit `order` the service returns the newest artifact first.
        """

        await self.__seed_thread_with_artifacts(count=3)

        page = await self.__service.artifacts(
            query=ArtifactListQuery(tenant="tenant-1", thread="thread-1", limit=10),
        )

        self.assertEqual(
            ["artifact-2", "artifact-1", "artifact-0"],
            [a.id for a in page.items],
        )

    async def test_ascending_opt_in_returns_oldest_first(self) -> None:
        """
        Passing order=ASC restores the oldest-first behaviour.
        """

        await self.__seed_thread_with_artifacts(count=3)

        page = await self.__service.artifacts(
            query=ArtifactListQuery(
                tenant="tenant-1",
                thread="thread-1",
                limit=10,
                order=SortOrder.ASC,
            ),
        )

        self.assertEqual(
            ["artifact-0", "artifact-1", "artifact-2"],
            [a.id for a in page.items],
        )
