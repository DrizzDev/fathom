from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
from uuid import UUID, uuid4

import pytest
from tests.unit.infrastructure.interaction.orm.repositories.factories import (
    InteractionRepositoryFactory,
)
from tests.unit.infrastructure.interaction.orm.support import InteractionPostgresSchema
from tortoise.exceptions import IntegrityError

from fathom.constants.collaboration import (
    SYSTEM_INTERACTION_ACTOR_ID,
    EventKind,
    JobCode,
    JobKind,
    JobState,
    TaskKind,
    TaskState,
)
from fathom.core.exceptions import InteractionError, JobLeaseLostError
from fathom.infrastructure.interaction.orm.models import (
    ConversationRecord,
    EventRecord,
    ExecutionRecord,
    JobRecord,
    TaskRecord,
)
from fathom.schemas.interaction import (
    ClaimJob,
    FinishJob,
    Identity,
    Job,
    JobQuery,
    Metadata,
    Outcome,
    RecoverJob,
    RescheduleJob,
    ScheduleJob,
)


class TestJobRepository:
    """
    Verify durable job persistence through the persistent-store backed repository.
    """

    async def test_schedule_job_persists_pending_job_and_records_event(self) -> None:
        """
        Schedule one pending job and record the lifecycle event.
        """

        async with InteractionPostgresSchema(prefix="conversation_job_repository"):
            thread = await self.__thread()
            task = await self.__task(thread=thread)
            request = self.__schedule_request(thread=thread, task=task)

            result = await InteractionRepositoryFactory().jobs().schedule_job(request=request)

            assert result.identity == request.identity
            assert result.thread == thread
            assert result.task == task
            assert result.execution == task
            assert result.kind == JobKind.EXECUTION
            assert result.state == JobState.PENDING
            assert result.attempts == 0
            assert result.payload == Metadata(entries={"step": 1})
            event = await EventRecord.get(conversation_id=thread, sequence=1)
            assert event.kind == EventKind.JOB_SCHEDULED.value
            assert event.actor == SYSTEM_INTERACTION_ACTOR_ID
            assert event.task_id == task
            assert event.execution_id == task

    async def test_identical_schedule_replay_returns_existing_without_new_event(self) -> None:
        """
        Replay an identical schedule request without duplicating job or event rows.
        """

        async with InteractionPostgresSchema(prefix="conversation_job_repository"):
            thread = await self.__thread()
            request = self.__schedule_request(thread=thread)
            repository = InteractionRepositoryFactory().jobs()

            created = await repository.schedule_job(request=request)
            replayed = await repository.schedule_job(request=request)

            assert replayed == created
            assert await JobRecord.filter(conversation_id=thread).count() == 1
            assert await EventRecord.filter(conversation_id=thread).count() == 1

    async def test_conflicting_schedule_replay_raises_interaction_error(self) -> None:
        """
        Reject a reused job id with different schedule content.
        """

        async with InteractionPostgresSchema(prefix="conversation_job_repository"):
            thread = await self.__thread()
            request = self.__schedule_request(thread=thread)
            repository = InteractionRepositoryFactory().jobs()
            await repository.schedule_job(request=request)
            conflict = request.model_copy(update={"payload": Metadata(entries={"step": 2})})

            with pytest.raises(InteractionError, match="different content"):
                await repository.schedule_job(request=conflict)

    async def test_identical_schedule_replay_returns_existing_after_parent_archived(self) -> None:
        """
        Replay an existing job before validating the archived parent thread.
        """

        async with InteractionPostgresSchema(prefix="conversation_job_repository"):
            thread = await self.__thread()
            request = self.__schedule_request(thread=thread)
            repository = InteractionRepositoryFactory().jobs()
            created = await repository.schedule_job(request=request)
            await ConversationRecord.filter(id=thread).update(
                archived_at=self.__now(),
            )

            replayed = await repository.schedule_job(request=request)

            assert replayed == created
            assert await JobRecord.filter(conversation_id=thread).count() == 1
            assert await EventRecord.filter(conversation_id=thread).count() == 1

    async def test_schedule_job_validates_parent_thread_and_task(self) -> None:
        """
        Reject jobs whose parent thread or task is not active in the same thread.
        """

        async with InteractionPostgresSchema(prefix="conversation_job_repository"):
            repository = InteractionRepositoryFactory().jobs()
            missing_thread = self.__schedule_request(
                thread=str(uuid4()),
                execution=str(uuid4()),
            )

            with pytest.raises(InteractionError, match="Thread does not exist"):
                await repository.schedule_job(request=missing_thread)

            thread = await self.__thread()
            other_thread = await self.__thread()
            task = await self.__task(thread=other_thread)

            with pytest.raises(InteractionError, match="Job task belongs to a different thread"):
                await repository.schedule_job(
                    request=self.__schedule_request(thread=thread, task=task)
                )

    async def test_get_jobs_filters_by_execution(self) -> None:
        """
        Return only jobs attached to the requested execution.
        """

        async with InteractionPostgresSchema(prefix="conversation_job_repository"):
            thread = await self.__thread()
            first = await self.__task(thread=thread)
            second = await self.__task(thread=thread)
            repository = InteractionRepositoryFactory().jobs()
            await repository.schedule_job(
                request=self.__schedule_request(thread=thread, task=first)
            )
            await repository.schedule_job(
                request=self.__schedule_request(thread=thread, task=second)
            )

            jobs = await repository.get_jobs(
                query=JobQuery(tenant="tenant-a", thread=thread, execution=first)
            )

            assert len(jobs) == 1
            assert jobs[0].execution == first

    async def test_job_execution_must_match_task_execution(self) -> None:
        """
        Reject a job whose explicit execution disagrees with its task.
        """

        async with InteractionPostgresSchema(prefix="conversation_job_repository"):
            thread = await self.__thread()
            task = await self.__task(thread=thread)
            request = self.__schedule_request(thread=thread, task=task).model_copy(
                update={"execution": str(uuid4())}
            )

            with pytest.raises(InteractionError, match="execution does not match"):
                await InteractionRepositoryFactory().jobs().schedule_job(request=request)

    async def test_schedule_job_requires_execution_without_task(self) -> None:
        """
        Reject run-owned jobs that have neither task nor execution.
        """

        async with InteractionPostgresSchema(prefix="conversation_job_repository"):
            thread = await self.__thread()
            request = self.__schedule_request(thread=thread).model_copy(update={"execution": None})

            with pytest.raises(InteractionError, match="Job execution is required"):
                await InteractionRepositoryFactory().jobs().schedule_job(request=request)

    async def test_claim_job_uses_atomic_sql_and_filters_specific_job(self) -> None:
        """
        Claim only the requested pending job and increment its attempts.
        """

        async with InteractionPostgresSchema(prefix="conversation_job_repository"):
            thread = await self.__thread()
            now = self.__now()
            first = (
                await InteractionRepositoryFactory()
                .jobs()
                .schedule_job(
                    request=self.__schedule_request(
                        thread=thread,
                        available=now,
                    )
                )
            )
            second = (
                await InteractionRepositoryFactory()
                .jobs()
                .schedule_job(
                    request=self.__schedule_request(
                        thread=thread,
                        available=now,
                    )
                )
            )

            claimed = (
                await InteractionRepositoryFactory()
                .jobs()
                .claim_job(
                    request=ClaimJob(
                        tenant="tenant-a",
                        owner="worker-a",
                        claimed=now,
                        kind=JobKind.EXECUTION,
                        job=second.identity.id,
                    )
                )
            )

            assert claimed is not None
            assert claimed.identity.id == second.identity.id
            assert claimed.state == JobState.CLAIMED
            assert claimed.owner == "worker-a"
            assert claimed.attempts == 1
            untouched = (
                await InteractionRepositoryFactory()
                .jobs()
                .get_jobs(query=JobQuery(tenant="tenant-a", thread=thread, state=JobState.PENDING))
            )
            assert tuple(job.identity.id for job in untouched) == (first.identity.id,)

    async def test_concurrent_claim_allows_exactly_one_worker(self) -> None:
        """
        Race multiple workers against one pending job and allow one claim.
        """

        async with InteractionPostgresSchema(prefix="conversation_job_repository"):
            thread = await self.__thread()
            now = self.__now()
            scheduled = (
                await InteractionRepositoryFactory()
                .jobs()
                .schedule_job(request=self.__schedule_request(thread=thread, available=now))
            )

            async def claim(owner: str) -> Optional[Job]:
                return (
                    await InteractionRepositoryFactory()
                    .jobs()
                    .claim_job(
                        request=ClaimJob(
                            tenant="tenant-a",
                            owner=owner,
                            claimed=now,
                            job=scheduled.identity.id,
                        )
                    )
                )

            results = await asyncio.gather(*(claim(f"worker-{index}") for index in range(5)))

            claimed = [result for result in results if result is not None]
            assert len(claimed) == 1
            assert claimed[0].identity.id == scheduled.identity.id
            assert claimed[0].attempts == 1

    async def test_finish_job_records_terminal_outcome_and_replays(self) -> None:
        """
        Finish one claimed job and replay the identical terminal outcome.
        """

        async with InteractionPostgresSchema(prefix="conversation_job_repository"):
            thread = await self.__thread()
            now = self.__now()
            scheduled = (
                await InteractionRepositoryFactory()
                .jobs()
                .schedule_job(request=self.__schedule_request(thread=thread, available=now))
            )
            await (
                InteractionRepositoryFactory()
                .jobs()
                .claim_job(
                    request=ClaimJob(
                        tenant="tenant-a",
                        owner="worker-a",
                        claimed=now,
                        job=scheduled.identity.id,
                    )
                )
            )
            finish = FinishJob(
                tenant="tenant-a",
                job=scheduled.identity.id,
                owner="worker-a",
                state=JobState.COMPLETED,
                outcome=Outcome(code=JobCode.COMPLETED, detail="done"),
                finished=now + timedelta(seconds=1),
            )

            result = await InteractionRepositoryFactory().jobs().finish_job(request=finish)
            replayed = await InteractionRepositoryFactory().jobs().finish_job(request=finish)

            assert replayed == result
            assert result.state == JobState.COMPLETED
            assert result.outcome == finish.outcome
            event = await EventRecord.get(conversation_id=thread, sequence=2)
            assert event.kind == EventKind.JOB_COMPLETED.value

    async def test_finish_job_rejects_lost_or_invalid_lease(self) -> None:
        """
        Reject finish attempts that do not own a claimed lease.
        """

        async with InteractionPostgresSchema(prefix="conversation_job_repository"):
            thread = await self.__thread()
            now = self.__now()
            scheduled = (
                await InteractionRepositoryFactory()
                .jobs()
                .schedule_job(request=self.__schedule_request(thread=thread, available=now))
            )
            with pytest.raises(JobLeaseLostError):
                await (
                    InteractionRepositoryFactory()
                    .jobs()
                    .finish_job(
                        request=FinishJob(
                            tenant="tenant-a",
                            job=scheduled.identity.id,
                            owner="worker-a",
                            state=JobState.COMPLETED,
                            outcome=Outcome(code=JobCode.COMPLETED),
                            finished=now,
                        )
                    )
                )

    async def test_finish_job_rejects_soft_deleted_job(self) -> None:
        """
        Reject terminal updates for soft-deleted jobs.
        """

        async with InteractionPostgresSchema(prefix="conversation_job_repository"):
            thread = await self.__thread()
            now = self.__now()
            claimed = await self.__claimed_job(thread=thread, claimed=now, owner="worker-a")
            await JobRecord.filter(id=claimed.identity.id).update(deleted_at=now)

            with pytest.raises(InteractionError, match="Job does not exist"):
                await (
                    InteractionRepositoryFactory()
                    .jobs()
                    .finish_job(
                        request=FinishJob(
                            tenant="tenant-a",
                            job=claimed.identity.id,
                            owner="worker-a",
                            state=JobState.COMPLETED,
                            outcome=Outcome(code=JobCode.COMPLETED),
                            finished=now + timedelta(seconds=1),
                        )
                    )
                )

    async def test_recover_jobs_releases_stale_claims_and_records_events(self) -> None:
        """
        Recover stale claimed jobs up to the requested limit.
        """

        async with InteractionPostgresSchema(prefix="conversation_job_repository"):
            thread = await self.__thread()
            now = self.__now()
            first = await self.__claimed_job(thread=thread, claimed=now, owner="lost-a")
            second = await self.__claimed_job(thread=thread, claimed=now, owner="lost-b")
            expected = min(first.identity.id, second.identity.id)

            recovered = (
                await InteractionRepositoryFactory()
                .jobs()
                .recover_jobs(
                    request=RecoverJob(
                        tenant="tenant-a",
                        before=now + timedelta(seconds=1),
                        available_at=now + timedelta(minutes=1),
                        kind=JobKind.EXECUTION,
                        limit=1,
                    )
                )
            )

            assert tuple(job.identity.id for job in recovered) == (expected,)
            assert recovered[0].state == JobState.PENDING
            assert recovered[0].owner is None
            event = await EventRecord.get(conversation_id=thread, sequence=3)
            assert event.kind == EventKind.RECOVERY_LOST.value

    async def test_reschedule_job_releases_owned_claim(self) -> None:
        """
        Reschedule one owned claimed job back to pending.
        """

        async with InteractionPostgresSchema(prefix="conversation_job_repository"):
            thread = await self.__thread()
            now = self.__now()
            claimed = await self.__claimed_job(thread=thread, claimed=now, owner="worker-a")

            result = (
                await InteractionRepositoryFactory()
                .jobs()
                .reschedule_job(
                    request=RescheduleJob(
                        tenant="tenant-a",
                        job=claimed.identity.id,
                        owner="worker-a",
                        attempts=1,
                        available_at=now + timedelta(minutes=5),
                        rescheduled=now + timedelta(seconds=1),
                        detail="retry later",
                    )
                )
            )

            assert result.state == JobState.PENDING
            assert result.owner is None
            assert result.locked is None
            event = await EventRecord.get(conversation_id=thread, sequence=2)
            assert event.kind == EventKind.JOB_RESCHEDULED.value

    async def test_get_jobs_filters_and_hides_archived_threads(self) -> None:
        """
        Load jobs with filters while hiding archived parent conversations.
        """

        async with InteractionPostgresSchema(prefix="conversation_job_repository"):
            thread = await self.__thread()
            archived = await self.__thread()
            active_job = (
                await InteractionRepositoryFactory()
                .jobs()
                .schedule_job(
                    request=self.__schedule_request(thread=thread, kind=JobKind.EXECUTION)
                )
            )
            await (
                InteractionRepositoryFactory()
                .jobs()
                .schedule_job(request=self.__schedule_request(thread=thread, kind=JobKind.MEMORY))
            )
            await (
                InteractionRepositoryFactory()
                .jobs()
                .schedule_job(
                    request=self.__schedule_request(thread=archived, kind=JobKind.EXECUTION)
                )
            )
            await ConversationRecord.filter(id=archived).update(
                archived_at=self.__now(),
            )

            jobs = (
                await InteractionRepositoryFactory()
                .jobs()
                .get_jobs(query=JobQuery(tenant="tenant-a", kind=JobKind.EXECUTION))
            )

            assert tuple(job.identity.id for job in jobs) == (active_job.identity.id,)

    async def test_corrupt_job_row_raises_interaction_error(self) -> None:
        """
        Reject stored rows with unknown enum values.
        """

        async with InteractionPostgresSchema(prefix="conversation_job_repository"):
            thread = await self.__thread()
            with pytest.raises(IntegrityError):
                await JobRecord.create(
                    id=str(uuid4()),
                    tenant_id="tenant-a",
                    workspace_id=None,
                    conversation_id=thread,
                    execution_id=self.__executions()[thread],
                    task=None,
                    kind="alien",
                    state=JobState.PENDING.value,
                    attempts=0,
                    owner=None,
                    locked_at=None,
                    available_at=self.__now(),
                    payload={},
                    code=None,
                    detail=None,
                    created_at=self.__now(),
                    updated_at=self.__now(),
                    metadata={},
                )

    async def test_private_job_ids_are_plain_uuid_strings(self) -> None:
        """
        Preserve plain UUID job identifiers in storage.
        """

        async with InteractionPostgresSchema(prefix="conversation_job_repository"):
            thread = await self.__thread()
            request = self.__schedule_request(thread=thread)
            await InteractionRepositoryFactory().jobs().schedule_job(request=request)
            stored = await JobRecord.get(id=request.identity.id)

            assert str(UUID(stored.id)) == request.identity.id

    async def __claimed_job(self, *, thread: str, claimed: datetime, owner: str) -> Job:
        """
        Schedule and claim one job for lease-oriented tests.
        """

        scheduled = (
            await InteractionRepositoryFactory()
            .jobs()
            .schedule_job(request=self.__schedule_request(thread=thread, available=claimed))
        )
        claimed_job = (
            await InteractionRepositoryFactory()
            .jobs()
            .claim_job(
                request=ClaimJob(
                    tenant="tenant-a",
                    owner=owner,
                    claimed=claimed,
                    job=scheduled.identity.id,
                )
            )
        )
        if claimed_job is None:
            raise AssertionError("Expected job to be claimed.")

        return claimed_job

    async def __thread(
        self,
        *,
        archived: bool = False,
    ) -> str:
        """
        Insert one minimal thread row.
        """

        identifier = str(uuid4())
        now = self.__now()
        await ConversationRecord.create(
            id=identifier,
            tenant_id="tenant-a",
            workspace_id=None,
            title="Thread",
            digest=None,
            created_by=None,
            archived_at=now if archived else None,
            created_at=now,
            updated_at=now,
            metadata={},
        )
        self.__executions()[identifier] = await self.__execution(thread=identifier)
        return identifier

    def __executions(self) -> Dict[str, str]:
        """
        Return execution identifiers created for fixture conversations.
        """

        store = getattr(self, "__execution_by_thread", None)
        if store is None:
            store = {}
            setattr(self, "__execution_by_thread", store)

        return store

    async def __execution(self, *, thread: str) -> str:
        """
        Insert one execution row for conversation-scoped job tests.
        """

        execution = str(uuid4())
        now = self.__now()
        await ExecutionRecord.create(
            id=execution,
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=thread,
            intent="Do work",
            state=TaskState.RUNNING.value,
            outcome={},
            started_at=now,
            completed_at=None,
            created_at=now,
            created_by=None,
            updated_at=now,
            updated_by=None,
            metadata={},
        )
        return execution

    async def __task(self, *, thread: str) -> str:
        """
        Insert one minimal live task row.
        """

        identifier = str(uuid4())
        now = self.__now()
        await ExecutionRecord.create(
            id=identifier,
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=thread,
            intent="Do work",
            state=TaskState.RUNNING.value,
            outcome={},
            started_at=now,
            created_at=now,
            created_by=None,
            updated_at=now,
            updated_by=None,
            metadata={},
        )
        await TaskRecord.create(
            id=identifier,
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=thread,
            execution_id=identifier,
            created_by=None,
            assignee=None,
            origin=None,
            kind=TaskKind.AGENT.value,
            objective="Do work",
            reference=None,
            state=TaskState.RUNNING.value,
            code=None,
            detail=None,
            progress={},
            plan={},
            outcome={},
            summary=None,
            started_at=now,
            completed_at=None,
            elapsed=None,
            created_at=now,
            updated_at=now,
            metadata={},
        )
        return identifier

    def __schedule_request(
        self,
        *,
        thread: str,
        task: Optional[str] = None,
        execution: Optional[str] = None,
        kind: JobKind = JobKind.EXECUTION,
        available: Optional[datetime] = None,
    ) -> ScheduleJob:
        """
        Build one schedule request with a plain UUID identity.
        """

        now = available or self.__now()
        resolved_execution = execution
        if task is None and resolved_execution is None:
            resolved_execution = self.__executions()[thread]

        return ScheduleJob(
            identity=Identity(id=str(uuid4()), tenant="tenant-a", workspace=None),
            thread=thread,
            execution=resolved_execution,
            task=task,
            kind=kind,
            available_at=now,
            payload=Metadata(entries={"step": 1}),
            created_at=now,
            metadata=Metadata(entries={"source": "test"}),
        )

    def __now(self) -> datetime:
        """
        Return a stable timezone-aware timestamp for tests.
        """

        return datetime(2026, 1, 1, tzinfo=timezone.utc)
