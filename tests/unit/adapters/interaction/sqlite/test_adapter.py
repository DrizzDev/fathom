from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Set, Tuple

import aiosqlite
from pydantic import JsonValue

from fathom.adapters.interaction.pypika.sqlite import SQLiteInteraction
from fathom.constants.collaboration import (
    ActorKind,
    ArtifactBackend,
    ArtifactKind,
    Audience,
    ContextPurpose,
    EventKind,
    IdempotencyState,
    JobCode,
    JobKind,
    JobState,
    Label,
    MembershipRole,
    MessageKind,
    PolicyScope,
    TaskCode,
    TaskKind,
    TaskState,
)
from fathom.constants.storage import SQLiteJournalMode, SQLiteSynchronous
from fathom.core.exceptions import InteractionError, JobLeaseLostError, TaskConflictError
from fathom.infrastructure.interaction.pypika.sqlite.migration import Migration
from fathom.infrastructure.interaction.pypika.sqlite.unit import Unit
from fathom.schemas.configuration import SQLiteInteractionConfiguration
from fathom.schemas.interaction import (
    ArtifactQuery,
    Assignment,
    BeginRequest,
    BuildContext,
    ClaimJob,
    Content,
    ContextQuery,
    CreateActor,
    CreateThread,
    EventQuery,
    FinishJob,
    FinishRequest,
    FinishTask,
    Governance,
    IdempotencyQuery,
    Identity,
    JobQuery,
    JoinThread,
    Lineage,
    LinkArtifact,
    MemoryReference,
    MessageQuery,
    Metadata,
    OpenTask,
    Outcome,
    Plan,
    PolicyQuery,
    RecordMessage,
    RecoverJob,
    References,
    RescheduleJob,
    Sanitize,
    SavePolicy,
    SaveScript,
    ScheduleJob,
    ScriptListQuery,
    ScriptQuery,
    ScriptVersionQuery,
    TaskQuery,
    Terminal,
    ThreadListQuery,
    ThreadQuery,
)


class TestSQLiteInteraction(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for the SQLite interaction adapter.
    """

    def setUp(self) -> None:
        """
        Create an isolated SQLite database for each test.
        """

        self.__temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.__temporary_directory.cleanup)
        path = Path(self.__temporary_directory.name) / "interaction.db"
        self.__interaction = SQLiteInteraction(path=path)
        self.__path = path
        self.__now = datetime(2026, 4, 27, 10, 0, 0, tzinfo=timezone.utc)

    async def test_spine_round_trip(self) -> None:
        """
        Create the full PR 1 spine and read it back through the adapter.
        """

        await self.__create_spine(state=TaskState.RUNNING)

        message = await self.__interaction.record_message(
            request=RecordMessage(
                identity=self.__identity(id="message-1"),
                thread="thread-1",
                task="task-1",
                author="actor-1",
                sequence=1,
                kind=MessageKind.REQUEST,
                audience=Audience.THREAD,
                content=Content(body={"text": "Buy milk"}, labels=(Label.PRIVACY_UPI,)),
                created=self.__now,
            )
        )

        finished = await self.__interaction.finish_task(
            request=FinishTask(
                tenant="tenant-1",
                task="task-1",
                state=TaskState.SUCCEEDED,
                terminal=Terminal(code=TaskCode.COMPLETED, detail="Completed"),
                summary="Milk purchase finished.",
                ended=self.__now,
                elapsed=1200,
            )
        )

        messages = await self.__interaction.get_messages(
            query=MessageQuery(tenant="tenant-1", thread="thread-1", task="task-1")
        )

        self.assertEqual("message-1", message.identity.id)
        self.assertEqual(TaskState.SUCCEEDED, finished.state)
        self.assertEqual("Milk purchase finished.", finished.summary)
        self.assertEqual([message], messages)

    async def test_queries_are_tenant_scoped(self) -> None:
        """
        Ensure thread and message reads cannot cross tenant boundaries.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.record_message(
            request=RecordMessage(
                identity=self.__identity(id="message-1"),
                thread="thread-1",
                task="task-1",
                author="actor-1",
                sequence=1,
                kind=MessageKind.REQUEST,
                audience=Audience.THREAD,
                content=Content(body={"text": "Buy milk"}),
                created=self.__now,
            )
        )

        thread = await self.__interaction.get_thread(
            query=ThreadQuery(tenant="tenant-2", thread="thread-1")
        )
        messages = await self.__interaction.get_messages(
            query=MessageQuery(tenant="tenant-2", thread="thread-1", task="task-1")
        )

        self.assertIsNone(thread)
        self.assertEqual([], messages)

    async def test_identities_are_tenant_scoped(self) -> None:
        """
        Allow different tenants to use the same natural entity identifiers.
        """

        await self.__create_spine(state=TaskState.RUNNING, tenant="tenant-1")
        await self.__create_spine(state=TaskState.RUNNING, tenant="tenant-2")
        await self.__interaction.record_message(
            request=RecordMessage(
                identity=self.__identity(id="message-1", tenant="tenant-1"),
                thread="thread-1",
                task="task-1",
                author="actor-1",
                sequence=1,
                kind=MessageKind.REQUEST,
                audience=Audience.THREAD,
                content=Content(body={"text": "Tenant one"}),
                created=self.__now,
            )
        )
        await self.__interaction.record_message(
            request=RecordMessage(
                identity=self.__identity(id="message-1", tenant="tenant-2"),
                thread="thread-1",
                task="task-1",
                author="actor-1",
                sequence=1,
                kind=MessageKind.REQUEST,
                audience=Audience.THREAD,
                content=Content(body={"text": "Tenant two"}),
                created=self.__now,
            )
        )

        tenant_one = await self.__interaction.get_messages(
            query=MessageQuery(tenant="tenant-1", thread="thread-1", task="task-1")
        )
        tenant_two = await self.__interaction.get_messages(
            query=MessageQuery(tenant="tenant-2", thread="thread-1", task="task-1")
        )

        self.assertEqual({"text": "Tenant one"}, tenant_one[0].content.body)
        self.assertEqual({"text": "Tenant two"}, tenant_two[0].content.body)

    async def test_thread_level_messages_are_queryable(self) -> None:
        """
        Load messages that belong to a thread without requiring a task filter.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        message = await self.__interaction.record_message(
            request=RecordMessage(
                identity=self.__identity(id="message-1"),
                thread="thread-1",
                author="actor-1",
                sequence=1,
                kind=MessageKind.NOTE,
                audience=Audience.THREAD,
                content=Content(body={"text": "General thread note"}),
                created=self.__now,
            )
        )

        messages = await self.__interaction.get_messages(
            query=MessageQuery(tenant="tenant-1", thread="thread-1")
        )

        self.assertEqual([message], messages)

    async def test_record_message_allocates_sequence_when_zero(self) -> None:
        """
        Allocate stable thread message sequence inside the store when requested.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        first = await self.__interaction.record_message(
            request=RecordMessage(
                identity=self.__identity(id="message-1"),
                thread="thread-1",
                author="actor-1",
                sequence=None,
                kind=MessageKind.NOTE,
                audience=Audience.THREAD,
                content=Content(body={"text": "First"}),
                created=self.__now,
            )
        )
        second = await self.__interaction.record_message(
            request=RecordMessage(
                identity=self.__identity(id="message-2"),
                thread="thread-1",
                author="actor-1",
                sequence=None,
                kind=MessageKind.NOTE,
                audience=Audience.THREAD,
                content=Content(body={"text": "Second"}),
                created=self.__now,
            )
        )

        self.assertEqual([1, 2], [first.sequence, second.sequence])

    async def test_integrity_errors_are_translated(self) -> None:
        """
        Convert storage integrity failures into interaction boundary errors.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        request = RecordMessage(
            identity=self.__identity(id="message-1"),
            thread="thread-1",
            task="task-1",
            author="actor-1",
            sequence=1,
            kind=MessageKind.REQUEST,
            audience=Audience.THREAD,
            content=Content(body={"text": "First"}),
            created=self.__now,
        )
        await self.__interaction.record_message(request=request)

        with self.assertRaises(InteractionError):
            await self.__interaction.record_message(
                request=RecordMessage(
                    identity=self.__identity(id="message-2"),
                    thread="thread-1",
                    task="task-1",
                    author="actor-1",
                    sequence=1,
                    kind=MessageKind.REQUEST,
                    audience=Audience.THREAD,
                    content=Content(body={"text": "Duplicate sequence"}),
                    created=self.__now,
                )
            )

    async def test_sanitize_message_updates_content_and_records_event(self) -> None:
        """
        Replace message content with sanitized content exactly once.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.record_message(
            request=RecordMessage(
                identity=self.__identity(id="message-1"),
                thread="thread-1",
                task="task-1",
                author="actor-1",
                sequence=1,
                kind=MessageKind.ANSWER,
                audience=Audience.TASK,
                content=Content(body={"text": "OTP is 482901"}, labels=(Label.PRIVACY_OTP,)),
                created=self.__now,
            )
        )
        request = Sanitize(
            tenant="tenant-1",
            message="message-1",
            content=Content(
                body={"text": "OTP is [redacted]"},
                labels=(Label.PRIVACY_OTP, Label.DISPLAY_HIDDEN),
                sanitizer="privacy.default",
            ),
            sanitized=self.__now,
        )

        sanitized = await self.__interaction.sanitize_message(request=request)
        replay = await self.__interaction.sanitize_message(request=request)
        events = await self.__interaction.get_events(
            query=EventQuery(tenant="tenant-1", thread="thread-1", task="task-1")
        )

        self.assertEqual(sanitized, replay)
        self.assertEqual({"text": "OTP is [redacted]"}, sanitized.content.body)
        self.assertEqual(self.__now, sanitized.content.sanitized)
        self.assertEqual(EventKind.CONTENT_SANITIZED, events[-1].kind)

    async def test_sanitize_message_rejects_conflicting_content(self) -> None:
        """
        Reject a second sanitization that changes the stored sanitized content.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.record_message(
            request=RecordMessage(
                identity=self.__identity(id="message-1"),
                thread="thread-1",
                task="task-1",
                author="actor-1",
                sequence=1,
                kind=MessageKind.ANSWER,
                audience=Audience.TASK,
                content=Content(body={"text": "OTP is 482901"}, labels=(Label.PRIVACY_OTP,)),
                created=self.__now,
            )
        )
        await self.__interaction.sanitize_message(
            request=Sanitize(
                tenant="tenant-1",
                message="message-1",
                content=Content(
                    body={"text": "OTP is [redacted]"},
                    labels=(Label.PRIVACY_OTP,),
                    sanitizer="privacy.default",
                ),
                sanitized=self.__now,
            )
        )

        with self.assertRaises(InteractionError):
            await self.__interaction.sanitize_message(
                request=Sanitize(
                    tenant="tenant-1",
                    message="message-1",
                    content=Content(
                        body={"text": "Different"},
                        labels=(Label.PRIVACY_OTP,),
                        sanitizer="privacy.default",
                    ),
                    sanitized=self.__now,
                )
            )

    async def test_sanitize_message_requires_sanitizer(self) -> None:
        """
        Reject sanitized content that does not name the sanitizer.
        """

        with self.assertRaises(InteractionError):
            await self.__interaction.sanitize_message(
                request=Sanitize(
                    tenant="tenant-1",
                    message="message-1",
                    content=Content(body={"text": "[redacted]"}),
                    sanitized=self.__now,
                )
            )

    async def test_events_are_recorded_for_lifecycle_operations(self) -> None:
        """
        Record ordered lifecycle events for thread, membership, task, message, and finish.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.record_message(
            request=RecordMessage(
                identity=self.__identity(id="message-1"),
                thread="thread-1",
                task="task-1",
                author="actor-1",
                sequence=1,
                kind=MessageKind.REQUEST,
                audience=Audience.THREAD,
                content=Content(body={"text": "Buy milk"}),
                created=self.__now,
            )
        )
        await self.__interaction.finish_task(
            request=FinishTask(
                tenant="tenant-1",
                task="task-1",
                state=TaskState.SUCCEEDED,
                terminal=Terminal(code=TaskCode.COMPLETED),
                ended=self.__now,
                elapsed=1200,
            )
        )

        events = await self.__interaction.get_events(
            query=EventQuery(tenant="tenant-1", thread="thread-1")
        )

        self.assertEqual(
            [
                EventKind.THREAD_CREATED,
                EventKind.ACTOR_JOINED,
                EventKind.TASK_OPENED,
                EventKind.MESSAGE_RECORDED,
                EventKind.TASK_SUCCEEDED,
            ],
            [event.kind for event in events],
        )
        self.assertEqual([1, 2, 3, 4, 5], [event.sequence for event in events])
        self.assertEqual("actor-1", events[3].actor)
        self.assertEqual({"kind": "request", "audience": "thread"}, events[3].payload.entries)

    async def test_join_thread_reuses_existing_actor_membership(self) -> None:
        """
        Reuse the active actor membership when callers provide a different membership id.
        """

        await self.__create_spine(state=TaskState.RUNNING)

        membership = await self.__interaction.join_thread(
            request=JoinThread(
                identity=self.__identity(id="membership-requester"),
                thread="thread-1",
                actor="actor-1",
                role=MembershipRole.REQUESTER,
                joined=self.__now,
            )
        )
        events = await self.__interaction.get_events(
            query=EventQuery(tenant="tenant-1", thread="thread-1")
        )

        self.assertEqual("membership-1", membership.identity.id)
        self.assertEqual(MembershipRole.OWNER, membership.role)
        self.assertEqual(
            1,
            sum(1 for event in events if event.kind == EventKind.ACTOR_JOINED),
        )

    async def test_event_queries_are_tenant_and_task_scoped(self) -> None:
        """
        Ensure event reads cannot cross tenant or task boundaries.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.record_message(
            request=RecordMessage(
                identity=self.__identity(id="message-1"),
                thread="thread-1",
                task="task-1",
                author="actor-1",
                sequence=1,
                kind=MessageKind.REQUEST,
                audience=Audience.THREAD,
                content=Content(body={"text": "Buy milk"}),
                created=self.__now,
            )
        )

        other_tenant = await self.__interaction.get_events(
            query=EventQuery(tenant="tenant-2", thread="thread-1")
        )
        task_events = await self.__interaction.get_events(
            query=EventQuery(tenant="tenant-1", thread="thread-1", task="task-1")
        )

        self.assertEqual([], other_tenant)
        self.assertEqual(
            [EventKind.TASK_OPENED, EventKind.MESSAGE_RECORDED],
            [event.kind for event in task_events],
        )

    async def test_failed_message_write_does_not_record_event(self) -> None:
        """
        Keep events transactional with the operation they describe.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.create_actor(
            request=CreateActor(
                identity=self.__identity(id="actor-2"),
                kind=ActorKind.AGENT,
                name="Unjoined Agent",
                created=self.__now,
            )
        )
        before = await self.__interaction.get_events(
            query=EventQuery(tenant="tenant-1", thread="thread-1")
        )

        with self.assertRaises(InteractionError):
            await self.__interaction.record_message(
                request=RecordMessage(
                    identity=self.__identity(id="message-1"),
                    thread="thread-1",
                    task="task-1",
                    author="actor-2",
                    sequence=1,
                    kind=MessageKind.NOTE,
                    audience=Audience.THREAD,
                    content=Content(body={"text": "I should not be accepted."}),
                    created=self.__now,
                )
            )

        after = await self.__interaction.get_events(
            query=EventQuery(tenant="tenant-1", thread="thread-1")
        )

        self.assertEqual(before, after)

    async def test_create_thread_rejects_unknown_creator(self) -> None:
        """
        Fail clearly when a thread creator does not exist.
        """

        with self.assertRaises(InteractionError):
            await self.__interaction.create_thread(
                request=CreateThread(
                    identity=self.__identity(id="thread-1"),
                    title="Milk order",
                    creator="missing-actor",
                    created=self.__now,
                )
            )

    async def test_policy_round_trip(self) -> None:
        """
        Save and load a tenant policy.
        """

        policy = await self.__interaction.save_policy(
            request=SavePolicy(
                identity=self.__identity(id="policy-1"),
                scope=PolicyScope.TENANT,
                name="default",
                region="in",
                governance=Governance(
                    retention=self.__metadata(entries={"messages": 90}),
                    labels=self.__metadata(entries={"otp": "privacy.otp"}),
                    sanitizers=self.__metadata(entries={"privacy.otp": "mask"}),
                ),
                created=self.__now,
            )
        )

        loaded = await self.__interaction.get_policy(
            query=PolicyQuery(tenant="tenant-1", name="default")
        )

        self.assertEqual(policy, loaded)
        self.assertEqual({"messages": 90}, policy.governance.retention.entries)
        self.assertEqual({"privacy.otp": "mask"}, policy.governance.sanitizers.entries)

    async def test_policy_queries_are_workspace_scoped(self) -> None:
        """
        Keep tenant and workspace policy reads separate.
        """

        await self.__interaction.save_policy(
            request=SavePolicy(
                identity=Identity(id="policy-1", tenant="tenant-1", workspace="workspace-1"),
                scope=PolicyScope.WORKSPACE,
                name="default",
                governance=Governance(retention=self.__metadata(entries={"messages": 30})),
                created=self.__now,
            )
        )

        tenant_policy = await self.__interaction.get_policy(
            query=PolicyQuery(tenant="tenant-1", name="default")
        )
        workspace_policy = await self.__interaction.get_policy(
            query=PolicyQuery(tenant="tenant-1", workspace="workspace-1", name="default")
        )

        self.assertIsNone(tenant_policy)
        assert workspace_policy is not None
        self.assertEqual({"messages": 30}, workspace_policy.governance.retention.entries)

    async def test_save_policy_retry_returns_existing_policy(self) -> None:
        """
        Treat repeated policy saving with the same identity as an idempotent retry.
        """

        request = SavePolicy(
            identity=self.__identity(id="policy-1"),
            scope=PolicyScope.TENANT,
            name="default",
            created=self.__now,
        )

        first = await self.__interaction.save_policy(request=request)
        second = await self.__interaction.save_policy(request=request)

        self.assertEqual(first, second)

    async def test_save_policy_rejects_conflicting_retry(self) -> None:
        """
        Reject repeated policy identity when the payload differs.
        """

        await self.__interaction.save_policy(
            request=SavePolicy(
                identity=self.__identity(id="policy-1"),
                scope=PolicyScope.TENANT,
                name="default",
                governance=Governance(retention=self.__metadata(entries={"messages": 90})),
                created=self.__now,
            )
        )

        with self.assertRaises(InteractionError):
            await self.__interaction.save_policy(
                request=SavePolicy(
                    identity=self.__identity(id="policy-1"),
                    scope=PolicyScope.TENANT,
                    name="default",
                    governance=Governance(retention=self.__metadata(entries={"messages": 30})),
                    created=self.__now,
                )
            )

    async def test_save_policy_validates_scope_boundary(self) -> None:
        """
        Reject policy scope that contradicts workspace ownership.
        """

        with self.assertRaises(InteractionError):
            await self.__interaction.save_policy(
                request=SavePolicy(
                    identity=Identity(id="policy-1", tenant="tenant-1", workspace="workspace-1"),
                    scope=PolicyScope.TENANT,
                    name="default",
                    created=self.__now,
                )
            )

        with self.assertRaises(InteractionError):
            await self.__interaction.save_policy(
                request=SavePolicy(
                    identity=self.__identity(id="policy-2"),
                    scope=PolicyScope.WORKSPACE,
                    name="default",
                    created=self.__now,
                )
            )

    async def test_job_round_trip_and_claim(self) -> None:
        """
        Schedule and claim one available job.
        """

        await self.__create_spine(state=TaskState.RUNNING)

        scheduled = await self.__interaction.schedule_job(
            request=ScheduleJob(
                identity=self.__identity(id="job-1"),
                thread="thread-1",
                task="task-1",
                kind=JobKind.SANITIZE,
                available=self.__now,
                payload=self.__metadata(entries={"target": "message-1"}),
                created=self.__now,
            )
        )
        claimed = await self.__interaction.claim_job(
            request=ClaimJob(
                tenant="tenant-1",
                owner="worker-1",
                claimed=self.__now,
                kind=JobKind.SANITIZE,
            )
        )
        events = await self.__interaction.get_events(
            query=EventQuery(tenant="tenant-1", thread="thread-1", task="task-1")
        )

        self.assertEqual(JobState.PENDING, scheduled.state)
        assert claimed is not None
        self.assertEqual("job-1", claimed.identity.id)
        self.assertEqual(JobState.CLAIMED, claimed.state)
        self.assertEqual(1, claimed.attempts)
        self.assertEqual("worker-1", claimed.owner)
        self.assertEqual(EventKind.JOB_SCHEDULED, events[-1].kind)

    async def test_job_claim_skips_future_or_claimed_jobs(self) -> None:
        """
        Claim only pending jobs that are currently available.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        future = self.__now.replace(hour=11)
        await self.__interaction.schedule_job(
            request=ScheduleJob(
                identity=self.__identity(id="job-1"),
                thread="thread-1",
                kind=JobKind.MEMORY,
                available=future,
                created=self.__now,
            )
        )

        missing = await self.__interaction.claim_job(
            request=ClaimJob(tenant="tenant-1", owner="worker-1", claimed=self.__now)
        )
        claimed = await self.__interaction.claim_job(
            request=ClaimJob(tenant="tenant-1", owner="worker-1", claimed=future)
        )
        repeated = await self.__interaction.claim_job(
            request=ClaimJob(tenant="tenant-1", owner="worker-2", claimed=future)
        )

        self.assertIsNone(missing)
        assert claimed is not None
        self.assertEqual("job-1", claimed.identity.id)
        self.assertIsNone(repeated)

    async def test_finish_job_records_event_and_rejects_conflicting_retry(self) -> None:
        """
        Finish a claimed job and reject conflicting repeated outcomes.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.schedule_job(
            request=ScheduleJob(
                identity=self.__identity(id="job-1"),
                thread="thread-1",
                task="task-1",
                kind=JobKind.ARTIFACT,
                available=self.__now,
                created=self.__now,
            )
        )
        await self.__interaction.claim_job(
            request=ClaimJob(tenant="tenant-1", owner="worker-1", claimed=self.__now)
        )
        request = FinishJob(
            tenant="tenant-1",
            job="job-1",
            owner="worker-1",
            state=JobState.COMPLETED,
            outcome=Outcome(code=JobCode.COMPLETED),
            finished=self.__now,
        )

        finished = await self.__interaction.finish_job(request=request)
        replay = await self.__interaction.finish_job(request=request)
        events = await self.__interaction.get_events(
            query=EventQuery(tenant="tenant-1", thread="thread-1", task="task-1")
        )

        self.assertEqual(finished, replay)
        self.assertEqual(JobState.COMPLETED, finished.state)
        self.assertEqual(EventKind.JOB_COMPLETED, events[-1].kind)
        with self.assertRaises(InteractionError):
            await self.__interaction.finish_job(
                request=FinishJob(
                    tenant="tenant-1",
                    job="job-1",
                    owner="worker-1",
                    state=JobState.COMPLETED,
                    outcome=Outcome(code=JobCode.UNKNOWN_ERROR),
                    finished=self.__now,
                )
            )

    async def test_finish_job_rejects_unclaimed_job(self) -> None:
        """
        Reject terminal completion for a job that has not been claimed.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.schedule_job(
            request=ScheduleJob(
                identity=self.__identity(id="job-1"),
                thread="thread-1",
                kind=JobKind.RECOVERY,
                available=self.__now,
                created=self.__now,
            )
        )

        with self.assertRaises(InteractionError):
            await self.__interaction.finish_job(
                request=FinishJob(
                    tenant="tenant-1",
                    job="job-1",
                    owner="worker-1",
                    state=JobState.COMPLETED,
                    outcome=Outcome(code=JobCode.COMPLETED),
                    finished=self.__now,
                )
            )

    async def test_finish_job_with_zero_owner_match_raises_lease_lost(self) -> None:
        """
        Reject a finalize whose owner does not match the stored lease, even
        when validate_job_finish would otherwise accept the transition. This
        test pins the rowcount-zero defense added after the silent no-op
        regression.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.schedule_job(
            request=ScheduleJob(
                identity=self.__identity(id="job-1"),
                thread="thread-1",
                task="task-1",
                kind=JobKind.CONTEXT,
                available=self.__now,
                created=self.__now,
            )
        )
        await self.__interaction.claim_job(
            request=ClaimJob(tenant="tenant-1", owner="worker-real", claimed=self.__now)
        )

        with self.assertRaises(JobLeaseLostError):
            await self.__interaction.finish_job(
                request=FinishJob(
                    tenant="tenant-1",
                    job="job-1",
                    owner="worker-imposter",
                    state=JobState.COMPLETED,
                    outcome=Outcome(code=JobCode.COMPLETED),
                    finished=self.__now,
                )
            )

    async def test_reschedule_job_rejects_wrong_owner(self) -> None:
        """
        Only the owning worker may reschedule a claimed job. A different
        worker attempting to reschedule must hit JobLeaseLostError.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.schedule_job(
            request=ScheduleJob(
                identity=self.__identity(id="job-1"),
                thread="thread-1",
                task="task-1",
                kind=JobKind.CONTEXT,
                available=self.__now,
                created=self.__now,
            )
        )
        claimed = await self.__interaction.claim_job(
            request=ClaimJob(tenant="tenant-1", owner="worker-real", claimed=self.__now)
        )
        assert claimed is not None

        with self.assertRaises(JobLeaseLostError):
            await self.__interaction.reschedule_job(
                request=RescheduleJob(
                    tenant="tenant-1",
                    job="job-1",
                    owner="worker-imposter",
                    attempts=claimed.attempts,
                    available=self.__now,
                    rescheduled=self.__now,
                )
            )

    async def test_finish_job_rejects_lease_lost_to_other_worker(self) -> None:
        """
        Refuse to finish a job once its lease was recovered and re-claimed.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.schedule_job(
            request=ScheduleJob(
                identity=self.__identity(id="job-1"),
                thread="thread-1",
                task="task-1",
                kind=JobKind.CONTEXT,
                available=self.__now,
                created=self.__now,
            )
        )
        await self.__interaction.claim_job(
            request=ClaimJob(tenant="tenant-1", owner="worker-1", claimed=self.__now)
        )
        await self.__interaction.recover_jobs(
            request=RecoverJob(tenant="tenant-1", before=self.__now, available=self.__now)
        )
        await self.__interaction.claim_job(
            request=ClaimJob(tenant="tenant-1", owner="worker-2", claimed=self.__now)
        )

        with self.assertRaises(JobLeaseLostError) as context:
            await self.__interaction.finish_job(
                request=FinishJob(
                    tenant="tenant-1",
                    job="job-1",
                    owner="worker-1",
                    state=JobState.COMPLETED,
                    outcome=Outcome(code=JobCode.COMPLETED),
                    finished=self.__now,
                )
            )

        self.assertEqual("job-1", context.exception.job)

    async def test_finish_task_surfaces_typed_conflict(self) -> None:
        """
        Surface a typed TaskConflictError when terminal outcome differs.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.finish_task(
            request=FinishTask(
                tenant="tenant-1",
                task="task-1",
                state=TaskState.SUCCEEDED,
                terminal=Terminal(code=TaskCode.COMPLETED),
                summary="ok",
                ended=self.__now,
                elapsed=100,
            )
        )

        with self.assertRaises(TaskConflictError) as context:
            await self.__interaction.finish_task(
                request=FinishTask(
                    tenant="tenant-1",
                    task="task-1",
                    state=TaskState.SUCCEEDED,
                    terminal=Terminal(code=TaskCode.UNKNOWN_ERROR),
                    summary="conflict",
                    ended=self.__now,
                    elapsed=100,
                )
            )

        self.assertEqual("task-1", context.exception.task)

    async def test_recover_jobs_releases_stale_claims(self) -> None:
        """
        Recover stale claimed jobs so another worker can retry them.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.schedule_job(
            request=ScheduleJob(
                identity=self.__identity(id="job-1"),
                thread="thread-1",
                task="task-1",
                kind=JobKind.CONTEXT,
                available=self.__now,
                created=self.__now,
            )
        )
        await self.__interaction.claim_job(
            request=ClaimJob(tenant="tenant-1", owner="worker-1", claimed=self.__now)
        )

        recovered = await self.__interaction.recover_jobs(
            request=RecoverJob(
                tenant="tenant-1",
                before=self.__now,
                available=self.__now,
                kind=JobKind.CONTEXT,
            )
        )
        claimed = await self.__interaction.claim_job(
            request=ClaimJob(tenant="tenant-1", owner="worker-2", claimed=self.__now)
        )

        self.assertEqual(1, len(recovered))
        self.assertEqual(JobState.PENDING, recovered[0].state)
        assert claimed is not None
        self.assertEqual("worker-2", claimed.owner)

    async def test_recover_jobs_can_recover_same_job_more_than_once(self) -> None:
        """
        Record distinct recovery events for repeated stale claims of one job.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.schedule_job(
            request=ScheduleJob(
                identity=self.__identity(id="job-1"),
                thread="thread-1",
                task="task-1",
                kind=JobKind.CONTEXT,
                available=self.__now,
                created=self.__now,
            )
        )
        await self.__interaction.claim_job(
            request=ClaimJob(tenant="tenant-1", owner="worker-1", claimed=self.__now)
        )
        await self.__interaction.recover_jobs(
            request=RecoverJob(tenant="tenant-1", before=self.__now, available=self.__now)
        )
        await self.__interaction.claim_job(
            request=ClaimJob(tenant="tenant-1", owner="worker-2", claimed=self.__now)
        )

        recovered = await self.__interaction.recover_jobs(
            request=RecoverJob(tenant="tenant-1", before=self.__now, available=self.__now)
        )
        events = await self.__interaction.get_events(
            query=EventQuery(tenant="tenant-1", thread="thread-1", task="task-1")
        )

        self.assertEqual(1, len(recovered))
        self.assertEqual(
            [EventKind.RECOVERY_LOST, EventKind.RECOVERY_LOST],
            [event.kind for event in events if event.kind == EventKind.RECOVERY_LOST],
        )

    async def test_schedule_job_rejects_conflicting_retry(self) -> None:
        """
        Reject repeated job identity when the payload differs.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.schedule_job(
            request=ScheduleJob(
                identity=self.__identity(id="job-1"),
                thread="thread-1",
                kind=JobKind.MEMORY,
                available=self.__now,
                payload=self.__metadata(entries={"memory": "a"}),
                created=self.__now,
            )
        )

        with self.assertRaises(InteractionError):
            await self.__interaction.schedule_job(
                request=ScheduleJob(
                    identity=self.__identity(id="job-1"),
                    thread="thread-1",
                    kind=JobKind.MEMORY,
                    available=self.__now,
                    payload=self.__metadata(entries={"memory": "b"}),
                    created=self.__now,
                )
            )

    async def test_get_jobs_combines_filters(self) -> None:
        """
        Combine thread, state, and kind filters when reading tenant jobs.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.schedule_job(
            request=ScheduleJob(
                identity=self.__identity(id="job-pending-mem"),
                thread="thread-1",
                kind=JobKind.MEMORY,
                available=self.__now,
                created=self.__now,
            )
        )
        await self.__interaction.schedule_job(
            request=ScheduleJob(
                identity=self.__identity(id="job-pending-ctx"),
                thread="thread-1",
                kind=JobKind.CONTEXT,
                available=self.__now,
                created=self.__now,
            )
        )

        memory_pending = await self.__interaction.get_jobs(
            query=JobQuery(
                tenant="tenant-1",
                thread="thread-1",
                state=JobState.PENDING,
                kind=JobKind.MEMORY,
            )
        )

        self.assertEqual(["job-pending-mem"], [job.identity.id for job in memory_pending])

    async def test_save_policy_rejects_duplicate_scoped_name(self) -> None:
        """
        Reject ambiguous policies with the same tenant, workspace, and name.
        """

        await self.__interaction.save_policy(
            request=SavePolicy(
                identity=self.__identity(id="policy-1"),
                scope=PolicyScope.TENANT,
                name="default",
                created=self.__now,
            )
        )

        with self.assertRaises(InteractionError):
            await self.__interaction.save_policy(
                request=SavePolicy(
                    identity=self.__identity(id="policy-2"),
                    scope=PolicyScope.TENANT,
                    name="default",
                    created=self.__now,
                )
            )

    async def test_finish_task_retry_does_not_duplicate_event(self) -> None:
        """
        Treat repeated terminal completion as an idempotent retry.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        request = FinishTask(
            tenant="tenant-1",
            task="task-1",
            state=TaskState.SUCCEEDED,
            terminal=Terminal(code=TaskCode.COMPLETED),
            ended=self.__now,
            elapsed=1200,
        )

        first = await self.__interaction.finish_task(request=request)
        second = await self.__interaction.finish_task(request=request)
        events = await self.__interaction.get_events(
            query=EventQuery(tenant="tenant-1", thread="thread-1", task="task-1")
        )

        self.assertEqual(first, second)
        self.assertEqual(
            [EventKind.TASK_OPENED, EventKind.TASK_SUCCEEDED],
            [event.kind for event in events],
        )

    async def test_finish_task_rejects_conflicting_retry(self) -> None:
        """
        Reject repeated terminal completion when the stored outcome differs.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.finish_task(
            request=FinishTask(
                tenant="tenant-1",
                task="task-1",
                state=TaskState.SUCCEEDED,
                terminal=Terminal(code=TaskCode.COMPLETED),
                summary="Done.",
                ended=self.__now,
                elapsed=1200,
            )
        )

        with self.assertRaises(InteractionError):
            await self.__interaction.finish_task(
                request=FinishTask(
                    tenant="tenant-1",
                    task="task-1",
                    state=TaskState.SUCCEEDED,
                    terminal=Terminal(code=TaskCode.COMPLETED),
                    summary="Different.",
                    ended=self.__now,
                    elapsed=1200,
                )
            )

    async def test_task_lineage_is_persisted(self) -> None:
        """
        Persist task lineage for delegated child work.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        child = await self.__interaction.open_task(
            request=OpenTask(
                identity=self.__identity(id="task-2"),
                thread="thread-1",
                assignment=Assignment(creator="actor-1", assignee="actor-1"),
                lineage=Lineage(parent="task-1", root="task-1"),
                kind=TaskKind.DELEGATION,
                state=TaskState.RUNNING,
                plan=Plan(objective="Verify milk purchase"),
                created=self.__now,
            )
        )

        self.assertEqual("task-1", child.lineage.parent)
        self.assertEqual("task-1", child.lineage.root)

    async def test_get_tasks_loads_thread_scoped_task_tree_records(self) -> None:
        """
        Load all tasks for a thread without crossing tenant or thread boundaries.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.open_task(
            request=OpenTask(
                identity=self.__identity(id="task-2"),
                thread="thread-1",
                assignment=Assignment(creator="actor-1", assignee="actor-1"),
                lineage=Lineage(parent="task-1", root="task-1"),
                kind=TaskKind.DELEGATION,
                state=TaskState.RUNNING,
                plan=Plan(objective="Verify milk purchase"),
                created=self.__now,
            )
        )

        tasks = await self.__interaction.get_tasks(
            query=TaskQuery(tenant="tenant-1", thread="thread-1")
        )
        other = await self.__interaction.get_tasks(
            query=TaskQuery(tenant="tenant-2", thread="thread-1")
        )

        self.assertEqual(["task-1", "task-2"], [task.identity.id for task in tasks])
        self.assertEqual([], other)

    async def test_open_task_rejects_assignment_outside_thread(self) -> None:
        """
        Reject task assignments to actors without active thread membership.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.create_actor(
            request=CreateActor(
                identity=self.__identity(id="actor-2"),
                kind=ActorKind.AGENT,
                name="Unjoined Agent",
                created=self.__now,
            )
        )

        with self.assertRaises(InteractionError):
            await self.__interaction.open_task(
                request=OpenTask(
                    identity=self.__identity(id="task-2"),
                    thread="thread-1",
                    assignment=Assignment(creator="actor-1", assignee="actor-2"),
                    lineage=Lineage(parent="task-1", root="task-1"),
                    kind=TaskKind.DELEGATION,
                    state=TaskState.RUNNING,
                    plan=Plan(objective="Verify milk purchase"),
                    created=self.__now,
                )
            )

    async def test_labels_are_persisted(self) -> None:
        """
        Persist labels attached to message content.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.record_message(
            request=RecordMessage(
                identity=self.__identity(id="message-1"),
                thread="thread-1",
                task="task-1",
                author="actor-1",
                sequence=1,
                kind=MessageKind.ANSWER,
                audience=Audience.TASK,
                content=Content(
                    body={"text": "482901"},
                    labels=(Label.PRIVACY_OTP, Label.RETENTION_SHORT),
                ),
                created=self.__now,
            )
        )

        messages = await self.__interaction.get_messages(
            query=MessageQuery(tenant="tenant-1", thread="thread-1", task="task-1")
        )

        self.assertEqual(
            (Label.PRIVACY_OTP, Label.RETENTION_SHORT),
            messages[0].content.labels,
        )

    async def test_artifact_round_trip_records_event(self) -> None:
        """
        Link an artifact and record the corresponding lifecycle event.
        """

        await self.__create_spine(state=TaskState.RUNNING)

        artifact = await self.__interaction.link_artifact(
            request=LinkArtifact(
                identity=self.__identity(id="artifact-1"),
                thread="thread-1",
                task="task-1",
                producer="actor-1",
                kind=ArtifactKind.SCREENSHOT,
                uri="/tmp/screenshot.png",
                backend=ArtifactBackend.LOCAL,
                mime="image/png",
                size=512,
                retention="short",
                labels=(Label.RETENTION_SHORT,),
                created=self.__now,
            )
        )

        artifacts = await self.__interaction.get_artifacts(
            query=ArtifactQuery(tenant="tenant-1", thread="thread-1", task="task-1")
        )
        events = await self.__interaction.get_events(
            query=EventQuery(tenant="tenant-1", thread="thread-1", task="task-1")
        )

        self.assertEqual("artifact-1", artifact.identity.id)
        self.assertEqual([artifact], artifacts)
        self.assertEqual(ArtifactKind.SCREENSHOT, artifacts[0].kind)
        self.assertEqual((Label.RETENTION_SHORT,), artifacts[0].labels)
        self.assertEqual(EventKind.ARTIFACT_LINKED, events[-1].kind)
        self.assertEqual({"kind": "screenshot", "backend": "local"}, events[-1].payload.entries)

    async def test_script_round_trip_records_content_and_versions(self) -> None:
        """
        Save generated script content as a first-class reusable script.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.link_artifact(
            request=LinkArtifact(
                identity=self.__identity(id="artifact-script"),
                thread="thread-1",
                task="task-1",
                producer="actor-1",
                kind=ArtifactKind.SCRIPT,
                uri="/tmp/script.txt",
                backend=ArtifactBackend.LOCAL,
                mime="text/plain",
                size=17,
                created=self.__now,
            )
        )

        script = await self.__interaction.save_script(
            request=SaveScript(
                identity=self.__identity(id="script-1"),
                thread="thread-1",
                task="task-1",
                artifact="artifact-script",
                title="Checkout flow",
                content="OPEN_APP example",
                summary="Generated script export.",
                actor="actor-1",
                created=self.__now,
            )
        )
        updated = await self.__interaction.save_script(
            request=SaveScript(
                identity=self.__identity(id="script-1"),
                thread="thread-1",
                task="task-1",
                artifact="artifact-script",
                title="Checkout flow",
                content="OPEN_APP example\nValidate success.",
                summary="Append validation.",
                actor="actor-1",
                created=self.__now,
            )
        )

        scripts = await self.__interaction.get_scripts(
            query=ScriptQuery(tenant="tenant-1", thread="thread-1")
        )
        versions = await self.__interaction.get_script_versions(
            query=ScriptVersionQuery(tenant="tenant-1", script="script-1")
        )

        self.assertEqual("OPEN_APP example", script.content)
        self.assertEqual("OPEN_APP example\nValidate success.", updated.content)
        self.assertEqual(2, updated.revision)
        self.assertEqual([updated], scripts)
        self.assertEqual([1, 2], [version.version for version in versions])
        self.assertEqual(
            ["OPEN_APP example", "OPEN_APP example\nValidate success."],
            [version.content for version in versions],
        )
        self.assertTrue(all(version.checksum for version in versions))

    async def test_save_script_updates_title_without_new_version(self) -> None:
        """
        Same content with a changed title updates the row but does not bump revision or insert a version.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.link_artifact(
            request=LinkArtifact(
                identity=self.__identity(id="artifact-script"),
                thread="thread-1",
                task="task-1",
                producer="actor-1",
                kind=ArtifactKind.SCRIPT,
                uri="/tmp/script.txt",
                backend=ArtifactBackend.LOCAL,
                mime="text/plain",
                size=17,
                created=self.__now,
            )
        )
        original = await self.__interaction.save_script(
            request=SaveScript(
                identity=self.__identity(id="script-1"),
                thread="thread-1",
                task="task-1",
                artifact="artifact-script",
                title="Checkout flow",
                content="OPEN_APP example",
                summary="initial",
                actor="actor-1",
                created=self.__now,
            )
        )

        renamed = await self.__interaction.save_script(
            request=SaveScript(
                identity=self.__identity(id="script-1"),
                thread="thread-1",
                task="task-1",
                artifact="artifact-script",
                title="Renamed checkout flow",
                content="OPEN_APP example",
                summary="rename",
                actor="actor-1",
                created=self.__now,
            )
        )
        versions = await self.__interaction.get_script_versions(
            query=ScriptVersionQuery(tenant="tenant-1", script="script-1")
        )

        self.assertEqual("Checkout flow", original.title)
        self.assertEqual("Renamed checkout flow", renamed.title)
        self.assertEqual(1, original.revision)
        self.assertEqual(1, renamed.revision)
        self.assertEqual([1], [version.version for version in versions])

    async def test_save_script_bumps_revision_on_content_change(self) -> None:
        """
        Changed content bumps revision and inserts a new immutable version row.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.link_artifact(
            request=LinkArtifact(
                identity=self.__identity(id="artifact-script"),
                thread="thread-1",
                task="task-1",
                producer="actor-1",
                kind=ArtifactKind.SCRIPT,
                uri="/tmp/script.txt",
                backend=ArtifactBackend.LOCAL,
                mime="text/plain",
                size=17,
                created=self.__now,
            )
        )
        await self.__interaction.save_script(
            request=SaveScript(
                identity=self.__identity(id="script-1"),
                thread="thread-1",
                task="task-1",
                artifact="artifact-script",
                title="Checkout flow",
                content="OPEN_APP example",
                summary="initial",
                actor="actor-1",
                created=self.__now,
            )
        )
        updated = await self.__interaction.save_script(
            request=SaveScript(
                identity=self.__identity(id="script-1"),
                thread="thread-1",
                task="task-1",
                artifact="artifact-script",
                title="Checkout flow",
                content="OPEN_APP example\nValidate success.",
                summary="append",
                actor="actor-1",
                created=self.__now,
            )
        )
        versions = await self.__interaction.get_script_versions(
            query=ScriptVersionQuery(tenant="tenant-1", script="script-1")
        )

        self.assertEqual(2, updated.revision)
        self.assertEqual([1, 2], [version.version for version in versions])

    async def test_save_script_concurrent_inserts_keep_one_row_and_one_version(self) -> None:
        """
        Concurrent inserts of the same identity end with exactly one row at revision 1 and one version.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.link_artifact(
            request=LinkArtifact(
                identity=self.__identity(id="artifact-script"),
                thread="thread-1",
                task="task-1",
                producer="actor-1",
                kind=ArtifactKind.SCRIPT,
                uri="/tmp/script.txt",
                backend=ArtifactBackend.LOCAL,
                mime="text/plain",
                size=17,
                created=self.__now,
            )
        )

        async def __racing_save() -> None:
            """
            Race condition driver: identical SaveScript from two tasks.
            """

            await self.__interaction.save_script(
                request=SaveScript(
                    identity=self.__identity(id="script-race"),
                    thread="thread-1",
                    task="task-1",
                    artifact="artifact-script",
                    title="Race",
                    content="OPEN_APP race",
                    summary="race",
                    actor="actor-1",
                    created=self.__now,
                )
            )

        await asyncio.gather(__racing_save(), __racing_save())

        scripts = await self.__interaction.get_scripts(
            query=ScriptQuery(tenant="tenant-1", script="script-race")
        )
        versions = await self.__interaction.get_script_versions(
            query=ScriptVersionQuery(tenant="tenant-1", script="script-race")
        )

        self.assertEqual(1, len(scripts))
        self.assertEqual(1, scripts[0].revision)
        self.assertEqual(1, len(versions))

    async def test_list_scripts_orders_by_updated_with_cursor(self) -> None:
        """
        Cursor pagination orders scripts by updated_at desc with stable tiebreaker.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.link_artifact(
            request=LinkArtifact(
                identity=self.__identity(id="artifact-script"),
                thread="thread-1",
                task="task-1",
                producer="actor-1",
                kind=ArtifactKind.SCRIPT,
                uri="/tmp/script.txt",
                backend=ArtifactBackend.LOCAL,
                mime="text/plain",
                size=17,
                created=self.__now,
            )
        )
        for index in range(3):
            await self.__interaction.save_script(
                request=SaveScript(
                    identity=self.__identity(id=f"script-{index}"),
                    thread="thread-1",
                    task="task-1",
                    artifact="artifact-script",
                    title=f"Script {index}",
                    content=f"OPEN_APP example {index}",
                    summary="Generated script export.",
                    actor="actor-1",
                    created=self.__now,
                )
            )

        first = await self.__interaction.list_scripts(
            query=ScriptListQuery(
                tenant="tenant-1",
                thread="thread-1",
                limit=2,
            )
        )
        self.assertEqual(3, first.total)
        self.assertEqual(2, len(first.items))
        self.assertIsNotNone(first.next)

        second = await self.__interaction.list_scripts(
            query=ScriptListQuery(
                tenant="tenant-1",
                thread="thread-1",
                limit=2,
                cursor=first.next,
            )
        )
        self.assertEqual(1, len(second.items))
        self.assertIsNone(second.next)

        seen = {item.identity.id for item in first.items} | {
            item.identity.id for item in second.items
        }
        self.assertEqual({"script-0", "script-1", "script-2"}, seen)

    async def test_list_scripts_filters_by_task_and_tenant(self) -> None:
        """
        Scripts cannot leak across tenant or task boundaries through list_scripts.
        """

        await self.__create_spine(state=TaskState.RUNNING, tenant="tenant-1")
        await self.__create_spine(state=TaskState.RUNNING, tenant="tenant-2")
        await self.__interaction.link_artifact(
            request=LinkArtifact(
                identity=self.__identity(id="artifact-script-1"),
                thread="thread-1",
                task="task-1",
                producer="actor-1",
                kind=ArtifactKind.SCRIPT,
                uri="/tmp/script.txt",
                backend=ArtifactBackend.LOCAL,
                mime="text/plain",
                size=17,
                created=self.__now,
            )
        )
        await self.__interaction.save_script(
            request=SaveScript(
                identity=self.__identity(id="script-1"),
                thread="thread-1",
                task="task-1",
                artifact="artifact-script-1",
                title="Tenant one",
                content="OPEN_APP one",
                summary="Tenant one summary.",
                actor="actor-1",
                created=self.__now,
            )
        )

        other_tenant = await self.__interaction.list_scripts(
            query=ScriptListQuery(tenant="tenant-2", thread="thread-1", limit=10)
        )
        wrong_task = await self.__interaction.list_scripts(
            query=ScriptListQuery(
                tenant="tenant-1",
                thread="thread-1",
                task="task-2",
                limit=10,
            )
        )
        same_tenant_task = await self.__interaction.list_scripts(
            query=ScriptListQuery(
                tenant="tenant-1",
                thread="thread-1",
                task="task-1",
                limit=10,
            )
        )

        self.assertEqual((), other_tenant.items)
        self.assertEqual((), wrong_task.items)
        self.assertEqual(1, len(same_tenant_task.items))
        self.assertEqual("script-1", same_tenant_task.items[0].identity.id)

    async def test_list_threads_title_filter_escapes_like_wildcards(self) -> None:
        """
        Title filter treats SQL LIKE wildcards as literal characters via ESCAPE.
        """

        await self.__interaction.create_actor(
            request=CreateActor(
                identity=self.__identity(id="actor-1"),
                kind=ActorKind.HUMAN,
                name="Aman",
                created=self.__now,
            )
        )
        titles = ("50% discount on milk", "50 cents", "_drafts", "Final report")
        for index, title in enumerate(titles):
            await self.__interaction.create_thread(
                request=CreateThread(
                    identity=self.__identity(id=f"thread-{index}"),
                    title=title,
                    creator="actor-1",
                    created=self.__now,
                )
            )

        percent = await self.__interaction.list_threads(
            query=ThreadListQuery(tenant="tenant-1", title="50%", limit=10)
        )
        underscore = await self.__interaction.list_threads(
            query=ThreadListQuery(tenant="tenant-1", title="_dr", limit=10)
        )
        prefix = await self.__interaction.list_threads(
            query=ThreadListQuery(tenant="tenant-1", title="50", limit=10)
        )

        self.assertEqual(
            {"thread-0"},
            {thread.identity.id for thread in percent.items},
        )
        self.assertEqual(
            {"thread-2"},
            {thread.identity.id for thread in underscore.items},
        )
        self.assertEqual(
            {"thread-0", "thread-1"},
            {thread.identity.id for thread in prefix.items},
        )

    async def test_artifact_queries_are_tenant_and_task_scoped(self) -> None:
        """
        Ensure artifact reads cannot cross tenant or task boundaries.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.link_artifact(
            request=LinkArtifact(
                identity=self.__identity(id="artifact-1"),
                thread="thread-1",
                task="task-1",
                producer="actor-1",
                kind=ArtifactKind.TRACE,
                uri="/tmp/trace.json",
                backend=ArtifactBackend.LOCAL,
                created=self.__now,
            )
        )

        other_tenant = await self.__interaction.get_artifacts(
            query=ArtifactQuery(tenant="tenant-2", thread="thread-1")
        )
        other_task = await self.__interaction.get_artifacts(
            query=ArtifactQuery(tenant="tenant-1", thread="thread-1", task="task-2")
        )
        thread_artifacts = await self.__interaction.get_artifacts(
            query=ArtifactQuery(tenant="tenant-1", thread="thread-1")
        )

        self.assertEqual([], other_tenant)
        self.assertEqual([], other_task)
        self.assertEqual(1, len(thread_artifacts))

    async def test_link_artifact_retry_does_not_duplicate_event(self) -> None:
        """
        Treat repeated artifact linking with the same identity as an idempotent retry.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        request = LinkArtifact(
            identity=self.__identity(id="artifact-1"),
            thread="thread-1",
            task="task-1",
            producer="actor-1",
            kind=ArtifactKind.TRACE,
            uri="/tmp/trace.json",
            backend=ArtifactBackend.LOCAL,
            created=self.__now,
        )

        first = await self.__interaction.link_artifact(request=request)
        second = await self.__interaction.link_artifact(request=request)
        artifacts = await self.__interaction.get_artifacts(
            query=ArtifactQuery(tenant="tenant-1", thread="thread-1", task="task-1")
        )
        events = await self.__interaction.get_events(
            query=EventQuery(tenant="tenant-1", thread="thread-1", task="task-1")
        )

        self.assertEqual(first, second)
        self.assertEqual([first], artifacts)
        self.assertEqual(
            [EventKind.TASK_OPENED, EventKind.ARTIFACT_LINKED],
            [event.kind for event in events],
        )

    async def test_link_artifact_rejects_conflicting_retry(self) -> None:
        """
        Reject repeated artifact identity when the payload differs.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.link_artifact(
            request=LinkArtifact(
                identity=self.__identity(id="artifact-1"),
                thread="thread-1",
                task="task-1",
                producer="actor-1",
                kind=ArtifactKind.TRACE,
                uri="/tmp/trace.json",
                backend=ArtifactBackend.LOCAL,
                created=self.__now,
            )
        )

        with self.assertRaises(InteractionError):
            await self.__interaction.link_artifact(
                request=LinkArtifact(
                    identity=self.__identity(id="artifact-1"),
                    thread="thread-1",
                    task="task-1",
                    producer="actor-1",
                    kind=ArtifactKind.TRACE,
                    uri="/tmp/other.json",
                    backend=ArtifactBackend.LOCAL,
                    created=self.__now,
                )
            )

    async def test_failed_artifact_write_does_not_record_event(self) -> None:
        """
        Keep artifact events transactional with artifact writes.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        before = await self.__interaction.get_events(
            query=EventQuery(tenant="tenant-1", thread="thread-1")
        )

        with self.assertRaises(InteractionError):
            await self.__interaction.link_artifact(
                request=LinkArtifact(
                    identity=self.__identity(id="artifact-1"),
                    thread="thread-1",
                    task="missing-task",
                    producer="actor-1",
                    kind=ArtifactKind.SCREENSHOT,
                    uri="/tmp/screenshot.png",
                    backend=ArtifactBackend.LOCAL,
                    created=self.__now,
                )
            )

        after = await self.__interaction.get_events(
            query=EventQuery(tenant="tenant-1", thread="thread-1")
        )

        self.assertEqual(before, after)

    async def test_task_scoped_records_reject_foreign_thread_task(self) -> None:
        """
        Reject artifacts, jobs, and contexts that pair a thread with another thread's task.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.create_thread(
            request=CreateThread(
                identity=self.__identity(id="thread-2"),
                title="Other thread",
                creator="actor-1",
                created=self.__now,
            )
        )
        await self.__interaction.join_thread(
            request=JoinThread(
                identity=self.__identity(id="membership-2"),
                thread="thread-2",
                actor="actor-1",
                role=MembershipRole.OWNER,
                joined=self.__now,
            )
        )
        await self.__interaction.open_task(
            request=OpenTask(
                identity=self.__identity(id="task-2"),
                thread="thread-2",
                assignment=Assignment(creator="actor-1", assignee="actor-1"),
                kind=TaskKind.FATHOM,
                state=TaskState.RUNNING,
                plan=Plan(objective="Other work"),
                created=self.__now,
            )
        )

        with self.assertRaises(InteractionError):
            await self.__interaction.link_artifact(
                request=LinkArtifact(
                    identity=self.__identity(id="artifact-foreign"),
                    thread="thread-1",
                    task="task-2",
                    producer="actor-1",
                    kind=ArtifactKind.SCREENSHOT,
                    uri="/tmp/screenshot.png",
                    backend=ArtifactBackend.LOCAL,
                    created=self.__now,
                )
            )

        with self.assertRaises(InteractionError):
            await self.__interaction.schedule_job(
                request=ScheduleJob(
                    identity=self.__identity(id="job-foreign"),
                    thread="thread-1",
                    task="task-2",
                    kind=JobKind.CONTEXT,
                    available=self.__now,
                    created=self.__now,
                )
            )

        with self.assertRaises(InteractionError):
            await self.__interaction.build_context(
                request=BuildContext(
                    identity=self.__identity(id="context-foreign"),
                    thread="thread-1",
                    task="task-2",
                    purpose=ContextPurpose.EXECUTION,
                    builder="execution@1",
                    references=References(),
                    created=self.__now,
                )
            )

    async def test_record_message_rejects_actor_outside_thread(self) -> None:
        """
        Reject messages from actors without active thread membership.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.create_actor(
            request=CreateActor(
                identity=self.__identity(id="actor-2"),
                kind=ActorKind.AGENT,
                name="Unjoined Agent",
                created=self.__now,
            )
        )

        with self.assertRaises(InteractionError):
            await self.__interaction.record_message(
                request=RecordMessage(
                    identity=self.__identity(id="message-1"),
                    thread="thread-1",
                    task="task-1",
                    author="actor-2",
                    sequence=1,
                    kind=MessageKind.NOTE,
                    audience=Audience.THREAD,
                    content=Content(body={"text": "I should not be accepted."}),
                    created=self.__now,
                )
            )

    async def test_finish_task_rejects_invalid_transition(self) -> None:
        """
        Reject terminal completion from an invalid lifecycle state.
        """

        await self.__create_spine(state=TaskState.QUEUED)

        with self.assertRaises(InteractionError):
            await self.__interaction.finish_task(
                request=FinishTask(
                    tenant="tenant-1",
                    task="task-1",
                    state=TaskState.SUCCEEDED,
                    terminal=Terminal(code=TaskCode.COMPLETED),
                    ended=self.__now,
                    elapsed=1,
                )
            )

    async def test_context_round_trip_records_event(self) -> None:
        """
        Persist a reference-based context recipe and record the lifecycle event.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.record_message(
            request=RecordMessage(
                identity=self.__identity(id="message-1"),
                thread="thread-1",
                task="task-1",
                author="actor-1",
                sequence=1,
                kind=MessageKind.REQUEST,
                audience=Audience.THREAD,
                content=Content(body={"text": "Buy milk"}),
                created=self.__now,
            )
        )

        context = await self.__interaction.build_context(
            request=BuildContext(
                identity=self.__identity(id="context-1"),
                thread="thread-1",
                task="task-1",
                consumer="actor-1",
                purpose=ContextPurpose.EXECUTION,
                builder="execution@1",
                references=References(
                    messages=("message-1",),
                    memories=(MemoryReference(system="fathom_recall", reference="recall-1"),),
                ),
                budget=Metadata(entries={"tokens": 4000}),
                filters=Metadata(entries={"labels.exclude": ["privacy.otp"]}),
                hash="hash-1",
                provider="gemini",
                model="gemini-2-pro",
                created=self.__now,
            )
        )
        contexts = await self.__interaction.get_contexts(
            query=ContextQuery(tenant="tenant-1", thread="thread-1", task="task-1")
        )
        events = await self.__interaction.get_events(
            query=EventQuery(tenant="tenant-1", thread="thread-1", task="task-1")
        )

        self.assertEqual([context], contexts)
        self.assertEqual(("message-1",), context.references.messages)
        self.assertEqual(
            (MemoryReference(system="fathom_recall", reference="recall-1"),),
            context.references.memories,
        )
        self.assertEqual(EventKind.CONTEXT_BUILT, events[-1].kind)
        self.assertEqual(
            {"purpose": "execution", "builder": "execution@1"},
            events[-1].payload.entries,
        )

    async def test_get_contexts_filters_by_purpose(self) -> None:
        """
        Combine task and purpose filters when reading contexts.
        """

        await self.__create_spine(state=TaskState.RUNNING)

        for index, purpose in enumerate((ContextPurpose.EXECUTION, ContextPurpose.DIGEST), start=1):
            await self.__interaction.build_context(
                request=BuildContext(
                    identity=self.__identity(id=f"context-{index}"),
                    thread="thread-1",
                    task="task-1",
                    purpose=purpose,
                    builder=f"{purpose.value}@1",
                    references=References(),
                    created=self.__now,
                )
            )

        digests = await self.__interaction.get_contexts(
            query=ContextQuery(
                tenant="tenant-1",
                thread="thread-1",
                task="task-1",
                purpose=ContextPurpose.DIGEST,
            )
        )

        self.assertEqual(["context-2"], [context.identity.id for context in digests])

    async def test_build_context_retry_does_not_duplicate_event(self) -> None:
        """
        Treat repeated context building with the same identity as an idempotent retry.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        request = BuildContext(
            identity=self.__identity(id="context-1"),
            thread="thread-1",
            task="task-1",
            purpose=ContextPurpose.EXECUTION,
            builder="execution@1",
            references=References(),
            created=self.__now,
        )

        first = await self.__interaction.build_context(request=request)
        second = await self.__interaction.build_context(request=request)
        events = await self.__interaction.get_events(
            query=EventQuery(tenant="tenant-1", thread="thread-1", task="task-1")
        )

        self.assertEqual(first, second)
        self.assertEqual(
            [EventKind.TASK_OPENED, EventKind.CONTEXT_BUILT],
            [event.kind for event in events],
        )

    async def test_build_context_rejects_conflicting_retry(self) -> None:
        """
        Reject repeated context identity when the recipe differs.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.build_context(
            request=BuildContext(
                identity=self.__identity(id="context-1"),
                thread="thread-1",
                task="task-1",
                purpose=ContextPurpose.EXECUTION,
                builder="execution@1",
                references=References(),
                created=self.__now,
            )
        )

        with self.assertRaises(InteractionError):
            await self.__interaction.build_context(
                request=BuildContext(
                    identity=self.__identity(id="context-1"),
                    thread="thread-1",
                    task="task-1",
                    purpose=ContextPurpose.EXECUTION,
                    builder="execution@2",
                    references=References(),
                    created=self.__now,
                )
            )

    async def test_build_context_rejects_missing_local_references(self) -> None:
        """
        Reject context recipes that point at missing local records.
        """

        await self.__create_spine(state=TaskState.RUNNING)

        with self.assertRaises(InteractionError):
            await self.__interaction.build_context(
                request=BuildContext(
                    identity=self.__identity(id="context-1"),
                    thread="thread-1",
                    task="task-1",
                    purpose=ContextPurpose.EXECUTION,
                    builder="execution@1",
                    references=References(messages=("missing-message",)),
                    created=self.__now,
                )
            )

    async def test_begin_request_is_idempotent_for_same_hash(self) -> None:
        """
        Replay an idempotent request with the same hash without creating a new record.
        """

        request = BeginRequest(
            tenant="tenant-1",
            key="request-1",
            hash="hash-a",
            created=self.__now,
            expires=self.__now.replace(hour=12),
        )

        first = await self.__interaction.begin_request(request=request)
        second = await self.__interaction.begin_request(request=request)

        self.assertEqual(first, second)
        self.assertEqual(IdempotencyState.STARTED, first.state)

    async def test_begin_request_rejects_different_hash(self) -> None:
        """
        Reject reuse of an requests key with a different request payload hash.
        """

        await self.__interaction.begin_request(
            request=BeginRequest(
                tenant="tenant-1",
                key="request-1",
                hash="hash-a",
                created=self.__now,
                expires=self.__now.replace(hour=12),
            )
        )

        with self.assertRaises(InteractionError):
            await self.__interaction.begin_request(
                request=BeginRequest(
                    tenant="tenant-1",
                    key="request-1",
                    hash="hash-b",
                    created=self.__now,
                    expires=self.__now.replace(hour=12),
                )
            )

    async def test_begin_request_reuses_expired_key(self) -> None:
        """
        Allow an expired requests key to start a new request.
        """

        await self.__interaction.begin_request(
            request=BeginRequest(
                tenant="tenant-1",
                key="request-1",
                hash="hash-a",
                created=self.__now,
                expires=self.__now.replace(hour=11),
            )
        )

        replacement = await self.__interaction.begin_request(
            request=BeginRequest(
                tenant="tenant-1",
                key="request-1",
                hash="hash-b",
                created=self.__now.replace(hour=12),
                expires=self.__now.replace(hour=13),
            )
        )

        self.assertEqual("hash-b", replacement.hash)

    async def test_finish_request_records_terminal_state(self) -> None:
        """
        Move an active requests record to a terminal state with cached response.
        """

        await self.__interaction.begin_request(
            request=BeginRequest(
                tenant="tenant-1",
                key="request-1",
                hash="hash-a",
                created=self.__now,
                expires=self.__now.replace(hour=12),
            )
        )

        finished = await self.__interaction.finish_request(
            request=FinishRequest(
                tenant="tenant-1",
                key="request-1",
                state=IdempotencyState.COMPLETED,
                response={"thread": "thread-1"},
                finished=self.__now,
            )
        )
        loaded = await self.__interaction.get_idempotency(
            query=IdempotencyQuery(tenant="tenant-1", key="request-1")
        )

        self.assertEqual(IdempotencyState.COMPLETED, finished.state)
        self.assertEqual({"thread": "thread-1"}, finished.response)
        self.assertEqual(finished, loaded)

    async def test_finish_request_rejects_conflicting_replay(self) -> None:
        """
        Reject a terminal replay with a different cached response.
        """

        await self.__interaction.begin_request(
            request=BeginRequest(
                tenant="tenant-1",
                key="request-1",
                hash="hash-a",
                created=self.__now,
                expires=self.__now.replace(hour=12),
            )
        )
        await self.__interaction.finish_request(
            request=FinishRequest(
                tenant="tenant-1",
                key="request-1",
                state=IdempotencyState.COMPLETED,
                response={"thread": "thread-1"},
                finished=self.__now,
            )
        )

        with self.assertRaises(InteractionError):
            await self.__interaction.finish_request(
                request=FinishRequest(
                    tenant="tenant-1",
                    key="request-1",
                    state=IdempotencyState.COMPLETED,
                    response={"thread": "thread-2"},
                    finished=self.__now,
                )
            )

    async def test_finish_request_rejects_unknown_record(self) -> None:
        """
        Reject finishing an requests record that was never started.
        """

        with self.assertRaises(InteractionError):
            await self.__interaction.finish_request(
                request=FinishRequest(
                    tenant="tenant-1",
                    key="missing-request",
                    state=IdempotencyState.COMPLETED,
                    finished=self.__now,
                )
            )

    async def test_migrations_are_idempotent(self) -> None:
        """
        Run schema initialization more than once without changing version state.
        """

        unit = Unit(configuration=SQLiteInteractionConfiguration(path=self.__path))

        await unit.initialize()
        await unit.initialize()

        async with (
            aiosqlite.connect(self.__path) as database,
            database.execute("PRAGMA user_version") as cursor,
        ):
            row = await cursor.fetchone()

        self.assertEqual(Migration.CURRENT, row[0] if row else 0)

    async def test_structural_foreign_keys_are_declared(self) -> None:
        """
        Ensure concrete relationship columns are backed by SQLite FKs.
        """

        unit = Unit(configuration=SQLiteInteractionConfiguration(path=self.__path))

        await unit.initialize()

        async with aiosqlite.connect(self.__path) as database:
            thread_foreign_keys = await self.__foreign_keys(
                database=database,
                table="threads",
            )
            task_foreign_keys = await self.__foreign_keys(database=database, table="tasks")
            sequence_foreign_keys = await self.__foreign_keys(
                database=database,
                table="sequences",
            )

        self.assertIn(("creator", "actors", "id"), thread_foreign_keys)
        self.assertIn(("origin", "messages", "id"), task_foreign_keys)
        self.assertIn(("thread", "threads", "id"), sequence_foreign_keys)

    async def test_concurrent_initialization_is_safe(self) -> None:
        """
        Initialize the same unit concurrently without corrupting schema version state.
        """

        unit = Unit(configuration=SQLiteInteractionConfiguration(path=self.__path))

        await asyncio.gather(*(unit.initialize() for _ in range(5)))

        async with (
            aiosqlite.connect(self.__path) as database,
            database.execute("PRAGMA user_version") as cursor,
        ):
            row = await cursor.fetchone()

        self.assertEqual(Migration.CURRENT, row[0] if row else 0)

    async def __foreign_keys(
        self,
        *,
        database: aiosqlite.Connection,
        table: str,
    ) -> Set[Tuple[str, str, str]]:
        """
        Return child column, parent table, and parent column triples.
        """

        async with database.execute(f"PRAGMA foreign_key_list({table})") as cursor:  # nosec B608
            rows = await cursor.fetchall()

        return {(str(row[3]), str(row[2]), str(row[4])) for row in rows}

    async def test_pragmas_applied_from_configuration(self) -> None:
        """
        SQLite file-persistent tunables reflect the configured PRAGMAs.

        Per-connection tunables (foreign_keys, busy_timeout) cannot be
        observed from a fresh connection; they are exercised indirectly by
        the rest of the suite which opens sessions through the unit.
        """

        configuration = SQLiteInteractionConfiguration(
            path=self.__path,
            journal_mode=SQLiteJournalMode.WAL,
            synchronous=SQLiteSynchronous.NORMAL,
            busy_timeout=1234,
            allow_wal_on_shared_filesystem=True,
        )
        unit = Unit(configuration=configuration)
        await unit.initialize()

        async with (
            aiosqlite.connect(self.__path) as database,
            database.execute("PRAGMA journal_mode") as cursor,
        ):
            journal = await cursor.fetchone()

        assert journal is not None
        self.assertEqual("wal", str(journal[0]).lower())

    async def test_search_supports_full_text_search(self) -> None:
        """
        FTS5 mirror returns messages whose body text matches a search term.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.record_message(
            request=RecordMessage(
                identity=self.__identity(id="message-milk"),
                thread="thread-1",
                task="task-1",
                author="actor-1",
                sequence=1,
                kind=MessageKind.REQUEST,
                audience=Audience.THREAD,
                content=Content(body={"text": "Buy milk from BigBasket"}),
                created=self.__now,
            )
        )
        await self.__interaction.record_message(
            request=RecordMessage(
                identity=self.__identity(id="message-bread"),
                thread="thread-1",
                task="task-1",
                author="actor-1",
                sequence=2,
                kind=MessageKind.REQUEST,
                audience=Audience.THREAD,
                content=Content(body={"text": "Order bread from corner store"}),
                created=self.__now,
            )
        )

        async with (
            aiosqlite.connect(self.__path) as database,
            database.execute(
                "SELECT m.id FROM search f "
                "JOIN messages m ON m.rowid = f.rowid "
                "WHERE search MATCH ? "
                "ORDER BY rank",
                ("milk",),
            ) as cursor,
        ):
            rows = await cursor.fetchall()

        self.assertEqual(["message-milk"], [str(row[0]) for row in rows])

    async def test_search_indexes_non_text_message_payload_fields(self) -> None:
        """
        FTS5 mirror indexes request, result, and audit payload fields.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        await self.__interaction.record_message(
            request=RecordMessage(
                identity=self.__identity(id="message-request"),
                thread="thread-1",
                task="task-1",
                author="actor-1",
                kind=MessageKind.REQUEST,
                audience=Audience.THREAD,
                content=Content(
                    body={
                        "intent": "Open ChatGPT app and search Delhi places",
                        "workflow": "workflow-1",
                        "evidence": ["search tab visible"],
                    }
                ),
                created=self.__now,
            )
        )

        async with (
            aiosqlite.connect(self.__path) as database,
            database.execute(
                "SELECT m.id FROM search f "
                "JOIN messages m ON m.rowid = f.rowid "
                "WHERE search MATCH ?",
                ("Delhi",),
            ) as cursor,
        ):
            rows = await cursor.fetchall()

        self.assertEqual(["message-request"], [str(row[0]) for row in rows])

    async def test_thread_updated_advances_when_events_are_recorded(self) -> None:
        """
        Thread activity timestamp follows later lifecycle writes.
        """

        await self.__create_spine(state=TaskState.RUNNING)
        later = datetime(2026, 4, 27, 10, 5, 0, tzinfo=timezone.utc)
        await self.__interaction.record_message(
            request=RecordMessage(
                identity=self.__identity(id="message-later"),
                thread="thread-1",
                task="task-1",
                author="actor-1",
                kind=MessageKind.NOTE,
                audience=Audience.THREAD,
                content=Content(body={"summary": "Later activity"}),
                created=later,
            )
        )

        thread = await self.__interaction.get_thread(
            query=ThreadQuery(tenant="tenant-1", thread="thread-1")
        )

        self.assertEqual(later, thread.timing.updated)

    async def test_active_partial_indexes_exist(self) -> None:
        """
        Migration v9 creates the partial indexes for the active hot paths.
        """

        unit = Unit(configuration=SQLiteInteractionConfiguration(path=self.__path))
        await unit.initialize()

        async with (
            aiosqlite.connect(self.__path) as database,
            database.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_%active%'"
            ) as cursor,
        ):
            rows = {str(row[0]) for row in await cursor.fetchall()}

        self.assertIn("idx_threads_active_updated", rows)
        self.assertIn("idx_tasks_thread_active", rows)
        self.assertIn("idx_artifacts_thread_active", rows)
        self.assertIn("idx_messages_active", rows)

    async def __create_spine(self, *, state: TaskState, tenant: str = "tenant-1") -> None:
        """
        Create a thread, actor, membership, and task.
        """

        await self.__interaction.create_actor(
            request=CreateActor(
                identity=self.__identity(id="actor-1", tenant=tenant),
                kind=ActorKind.HUMAN,
                name="Aman",
                created=self.__now,
            )
        )
        await self.__interaction.create_thread(
            request=CreateThread(
                identity=self.__identity(id="thread-1", tenant=tenant),
                title="Milk order",
                creator="actor-1",
                created=self.__now,
            )
        )
        await self.__interaction.join_thread(
            request=JoinThread(
                identity=self.__identity(id="membership-1", tenant=tenant),
                thread="thread-1",
                actor="actor-1",
                role=MembershipRole.OWNER,
                joined=self.__now,
            )
        )
        await self.__interaction.open_task(
            request=OpenTask(
                identity=self.__identity(id="task-1", tenant=tenant),
                thread="thread-1",
                assignment=Assignment(creator="actor-1", assignee="actor-1"),
                kind=TaskKind.FATHOM,
                state=state,
                plan=Plan(objective="Buy milk"),
                created=self.__now,
            )
        )

    def __identity(self, *, id: str, tenant: str = "tenant-1") -> Identity:
        """
        Create a tenant-scoped identity for tests.
        """

        return Identity(id=id, tenant=tenant)

    def __metadata(self, *, entries: Dict[str, JsonValue]) -> Metadata:
        """
        Create metadata for tests.
        """

        return Metadata(entries=entries)
