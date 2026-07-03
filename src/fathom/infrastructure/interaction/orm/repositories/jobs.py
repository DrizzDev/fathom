from __future__ import annotations

import json
from datetime import datetime
from typing import List, Mapping, Optional, cast

from tortoise.exceptions import IntegrityError

from fathom.constants.collaboration import (
    EventKind,
    EventSource,
    JobCode,
    JobKind,
    JobState,
)
from fathom.core.exceptions import InteractionError, JobLeaseLostError
from fathom.infrastructure.interaction.orm.models import JobRecord
from fathom.infrastructure.interaction.orm.raw import RawSql
from fathom.infrastructure.interaction.orm.repositories.lifecycle import (
    DatabaseConnection,
    LifecycleRecorder,
    RawConnectionAdapter,
    TransactionScope,
)
from fathom.infrastructure.interaction.orm.repositories.reference import ReferenceGuard
from fathom.interaction.lifecycle import Lifecycle
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
    ThreadReference,
    ThreadScope,
    Timing,
    Visibility,
)


class JobRepository:
    """
    Repository for durable background job scheduling, leases, and recovery.
    """

    def __init__(
        self,
        *,
        raw: RawSql,
        validator: Lifecycle,
        references: ReferenceGuard,
        lifecycle: LifecycleRecorder,
        transaction: TransactionScope,
    ) -> None:
        """
        Initialize job persistence collaborators.
        """

        self.__raw = raw
        self.__guard = references
        self.__validator = validator
        self.__lifecycle = lifecycle
        self.__transaction = transaction

    async def schedule_job(self, *, request: ScheduleJob) -> Job:
        """
        Persist one pending background job or replay an identical job.
        """

        try:
            return await self.__schedule_job(request=request)
        except IntegrityError as exception:
            existing = await self.__load_job(
                connection=None,
                job=request.identity.id,
                tenant=request.identity.tenant,
            )
            if existing is not None and self.__same_schedule(job=existing, request=request):
                return existing

            raise InteractionError("Job insert conflicted with a different row.") from exception

    async def __schedule_job(self, *, request: ScheduleJob) -> Job:
        """
        Persist one scheduled job inside one transaction.
        """

        async with self.__transaction.transaction() as connection:
            existing = await self.__load_job(
                connection=connection,
                job=request.identity.id,
                tenant=request.identity.tenant,
            )
            if existing is not None:
                if not self.__same_schedule(job=existing, request=request):
                    raise InteractionError("Job identity already exists with different content.")

                return existing

            execution = await self.__require_references(request=request, connection=connection)

            await JobRecord.create(
                attempts=0,
                task_id=request.task,
                using_db=connection,
                execution_id=execution,
                id=request.identity.id,
                kind=request.kind.value,
                created_at=request.created,
                state=JobState.PENDING.value,
                available_at=request.available,
                conversation_id=request.thread,
                payload=request.payload.entries,
                metadata=request.metadata.entries,
                tenant_id=request.identity.tenant,
                workspace_id=request.identity.workspace,
            )

            await self.__lifecycle.record(
                task=request.task,
                execution=execution,
                connection=connection,
                thread=request.thread,
                created=request.created,
                kind=EventKind.JOB_SCHEDULED,
                tenant=request.identity.tenant,
                workspace=request.identity.workspace,
                payload=Metadata(entries={"kind": request.kind.value}),
            )
            job = await self.__load_job(
                connection=connection,
                job=request.identity.id,
                tenant=request.identity.tenant,
            )
            if job is None:
                raise InteractionError("Job was not persisted.")

            return job

    async def claim_job(self, *, request: ClaimJob) -> Optional[Job]:
        """
        Atomically claim the next available pending job.
        """

        async with self.__transaction.transaction() as connection:
            row = await self.__raw.fetchrow(
                job=request.job,
                owner=request.owner,
                name="jobs/claim.sql",
                tenant=request.tenant,
                locked_at=request.claimed,
                available_at=request.claimed,
                connection=RawConnectionAdapter(connection=connection),
                kind=request.kind.value if request.kind is not None else None,
            )

        if row is None:
            return None

        return self.__job_mapping(row=row)

    async def finish_job(self, *, request: FinishJob) -> Job:
        """
        Move one claimed job to a terminal state when the lease is still owned.
        """

        async with self.__transaction.transaction() as connection:
            job = await self.__load_job(
                job=request.job,
                tenant=request.tenant,
                connection=connection,
            )

            if job is None:
                raise InteractionError("Job does not exist.")

            if job.state == request.state and job.outcome is not None:
                if not self.__same_finish(job=job, request=request):
                    raise InteractionError("Job already finished with a different outcome.")

                return job

            if job.owner != request.owner:
                raise JobLeaseLostError(
                    job=request.job,
                    message="Job lease was lost; another worker now owns this claim.",
                )
            self.__validator.validate_job_finish(state=job.state, target=request.state)

            updated_count = await (
                JobRecord.filter(
                    id=request.job,
                    owner=request.owner,
                    tenant_id=request.tenant,
                    state=JobState.CLAIMED.value,
                )
                .using_db(connection)
                .update(
                    state=request.state.value,
                    updated_at=request.finished,
                    detail=request.outcome.detail,
                    code=request.outcome.code.value,
                )
            )
            if updated_count == 0:
                raise JobLeaseLostError(
                    job=request.job,
                    message="Job lease was lost between read and write.",
                )

            await self.__lifecycle.record(
                task=job.task,
                thread=job.thread,
                tenant=request.tenant,
                connection=connection,
                execution=job.execution,
                created=request.finished,
                source=EventSource.WORKER,
                workspace=job.identity.workspace,
                kind=self.__job_event_kind(state=request.state),
                payload=Metadata(entries={"code": request.outcome.code.value}),
            )
            finished = await self.__load_job(
                job=request.job,
                tenant=request.tenant,
                connection=connection,
            )
            if finished is None:
                raise InteractionError("Job was not updated.")

        return finished

    async def recover_jobs(self, *, request: RecoverJob) -> List[Job]:
        """
        Release stale claimed jobs back to pending state.
        """

        async with self.__transaction.transaction() as connection:
            recovered: List[Job] = []
            stale = await self.__stale_jobs(request=request, connection=connection)

            for job in stale:
                await (
                    JobRecord.filter(tenant_id=request.tenant, id=job.identity.id)
                    .using_db(connection)
                    .update(
                        owner=None,
                        locked_at=None,
                        state=JobState.PENDING.value,
                        updated_at=request.available,
                        available_at=request.available,
                    )
                )
                updated = await self.__load_job(
                    job=job.identity.id,
                    tenant=request.tenant,
                    connection=connection,
                )
                if updated is None:
                    raise InteractionError("Recovered job could not be loaded.")

                await self.__lifecycle.record(
                    task=job.task,
                    thread=job.thread,
                    connection=connection,
                    tenant=request.tenant,
                    execution=job.execution,
                    created=request.available,
                    source=EventSource.RECOVERY,
                    kind=EventKind.RECOVERY_LOST,
                    workspace=job.identity.workspace,
                    payload=Metadata(entries={"owner": job.owner, "kind": job.kind.value}),
                )
                recovered.append(updated)

        return recovered

    async def reschedule_job(self, *, request: RescheduleJob) -> Job:
        """
        Release one owned claimed job for retry after backoff.
        """

        async with self.__transaction.transaction() as connection:
            job = await self.__load_job(
                job=request.job,
                tenant=request.tenant,
                connection=connection,
            )

            if job is None:
                raise InteractionError("Job does not exist.")

            if job.state is not JobState.CLAIMED:
                raise InteractionError("Only claimed jobs can be rescheduled.")

            if job.owner != request.owner:
                raise JobLeaseLostError(
                    job=request.job,
                    message="Job lease was lost; cannot reschedule another worker's claim.",
                )

            if job.attempts != request.attempts:
                raise InteractionError("Job attempts changed before reschedule.")

            updated_count = await (
                JobRecord.filter(
                    id=request.job,
                    owner=request.owner,
                    tenant_id=request.tenant,
                    state=JobState.CLAIMED.value,
                )
                .using_db(connection)
                .update(
                    owner=None,
                    locked_at=None,
                    state=JobState.PENDING.value,
                    available_at=request.available,
                    updated_at=request.rescheduled,
                )
            )
            if updated_count == 0:
                raise JobLeaseLostError(
                    job=request.job,
                    message="Job lease was lost between read and write.",
                )

            await self.__lifecycle.record(
                task=job.task,
                thread=job.thread,
                connection=connection,
                tenant=request.tenant,
                execution=job.execution,
                source=EventSource.RECOVERY,
                kind=EventKind.JOB_RESCHEDULED,
                workspace=job.identity.workspace,
                payload=Metadata(
                    entries={
                        "kind": job.kind.value,
                        "detail": request.detail,
                        "attempts": request.attempts,
                    }
                ),
                created=request.rescheduled,
            )
            updated = await self.__load_job(
                job=request.job,
                tenant=request.tenant,
                connection=connection,
            )
            if updated is None:
                raise InteractionError("Rescheduled job could not be loaded.")

        return updated

    async def get_jobs(self, *, query: JobQuery) -> List[Job]:
        """
        Load visible tenant-scoped jobs with optional filters.
        """

        queryset = JobRecord.filter(
            tenant_id=query.tenant,
            **Visibility(deleted=query.include_deleted, archived=True).as_filters(),
        )

        if query.thread is not None:
            if not await self.__guard.thread_visible(
                scope=self.__scope(tenant=query.tenant, thread=query.thread)
            ):
                return []

            queryset = queryset.filter(conversation_id=query.thread)

        if query.execution is not None:
            queryset = queryset.filter(execution_id=query.execution)
        if query.kind is not None:
            queryset = queryset.filter(kind=query.kind.value)
        if query.state is not None:
            queryset = queryset.filter(state=query.state.value)

        rows = await queryset.order_by("created_at", "id")

        jobs = [self.__job(row=row) for row in rows]

        if query.thread is not None:
            return jobs

        return [
            job
            for job in jobs
            if await self.__guard.thread_visible(
                scope=self.__scope(tenant=job.identity.tenant, thread=job.thread)
            )
        ]

    async def __load_job(
        self,
        *,
        job: str,
        tenant: str,
        connection: Optional[DatabaseConnection],
    ) -> Optional[Job]:
        """
        Load one job row by identity.
        """

        queryset = JobRecord.filter(
            tenant_id=tenant,
            id=job,
            **Visibility(archived=True).as_filters(),
        )

        if connection is not None:
            queryset = queryset.using_db(connection)

        if row := await queryset.get_or_none():
            return self.__job(row=row)

        return None

    async def __stale_jobs(
        self,
        *,
        request: RecoverJob,
        connection: DatabaseConnection,
    ) -> List[Job]:
        """
        Load stale claimed jobs with a write lock.
        """

        queryset = JobRecord.filter(
            tenant_id=request.tenant,
            state=JobState.CLAIMED.value,
            locked_at__lte=request.before,
        )
        if request.kind is not None:
            queryset = queryset.filter(kind=request.kind.value)

        rows = await (
            queryset.using_db(connection)
            .select_for_update()
            .order_by("locked_at", "id")
            .limit(request.limit)
        )
        return [self.__job(row=row) for row in rows]

    async def __require_references(
        self,
        *,
        request: ScheduleJob,
        connection: DatabaseConnection,
    ) -> Optional[str]:
        """
        Validate parent thread and optional task references.
        """

        await self.__guard.active_thread(
            thread=request.thread,
            connection=connection,
            tenant=request.identity.tenant,
        )

        execution = request.execution

        if request.task is None:
            if execution is not None:
                await self.__guard.present_execution(
                    execution=execution,
                    connection=connection,
                    thread=request.thread,
                    tenant=request.identity.tenant,
                )

            else:
                raise InteractionError("Job execution is required.")

            return execution

        task = await self.__guard.present_task(
            label="Job",
            task=request.task,
            thread=request.thread,
            connection=connection,
            tenant=request.identity.tenant,
        )

        if execution is not None and task.execution_id != execution:
            raise InteractionError("Job execution does not match task execution.")

        return cast("Optional[str]", task.execution_id)

    def __scope(self, *, tenant: str, thread: str) -> ThreadScope:
        """
        Build a default thread scope hiding deleted and archived parents.
        """

        return ThreadScope(
            reference=ThreadReference(tenant=tenant, thread=thread),
            visibility=Visibility(),
        )

    def __job(self, *, row: JobRecord) -> Job:
        """
        Convert one persistent job row into the interaction schema.
        """

        return Job(
            owner=row.owner,
            task=row.task_id,
            attempts=row.attempts,
            locked_at=row.locked_at,
            thread=row.conversation_id,
            execution=row.execution_id,
            available_at=row.available_at,
            kind=self.__job_kind(value=row.kind),
            state=self.__job_state(value=row.state),
            outcome=self.__outcome(code=row.code, detail=row.detail),
            payload=self.__metadata(value=row.payload, field="payload"),
            metadata=self.__metadata(value=row.metadata, field="metadata"),
            timing=Timing(created_at=row.created_at, updated_at=row.updated_at),
            identity=Identity(id=row.id, tenant=row.tenant_id, workspace=row.workspace_id),
        )

    def __job_mapping(self, *, row: Mapping[str, object]) -> Job:
        """
        Convert one raw SQL job row into the interaction schema.
        """

        return Job(
            identity=Identity(
                id=self.__required_str(row=row, field="id"),
                tenant=self.__required_str(row=row, field="tenant"),
                workspace=self.__optional_str(row=row, field="workspace"),
            ),
            task=self.__optional_str(row=row, field="task_id"),
            owner=self.__optional_str(row=row, field="owner"),
            thread=self.__required_str(row=row, field="thread"),
            attempts=self.__required_int(row=row, field="attempts"),
            execution=self.__optional_str(row=row, field="execution"),
            locked_at=self.__optional_datetime(row=row, field="locked_at"),
            available_at=self.__required_datetime(row=row, field="available_at"),
            kind=self.__job_kind(value=self.__required_str(row=row, field="kind")),
            payload=self.__metadata(value=row.get("payload"), field="payload"),
            state=self.__job_state(value=self.__required_str(row=row, field="state")),
            outcome=self.__outcome(
                code=self.__optional_str(row=row, field="code"),
                detail=self.__optional_str(row=row, field="detail"),
            ),
            timing=Timing(
                created_at=self.__required_datetime(row=row, field="created_at"),
                updated_at=self.__required_datetime(row=row, field="updated_at"),
            ),
            metadata=self.__metadata(value=row.get("metadata"), field="metadata"),
        )

    def __metadata(self, *, value: object, field: str) -> Metadata:
        """
        Validate one stored JSON object as metadata.
        """

        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exception:
                raise InteractionError(f"Stored job {field} is invalid JSON.") from exception

        if not isinstance(value, dict):
            raise InteractionError(f"Stored job {field} is not an object.")

        if not all(isinstance(key, str) for key in value):
            raise InteractionError(f"Stored job {field} contains a non-string key.")

        return Metadata.model_validate({"entries": value})

    def __outcome(self, *, code: Optional[str], detail: Optional[str]) -> Optional[Outcome]:
        """
        Convert stored terminal outcome fields into a typed outcome.
        """

        if code is None:
            return None

        try:
            return Outcome(code=JobCode(code), detail=detail)
        except ValueError as exception:
            raise InteractionError(f"Unknown job outcome code in row: {code}.") from exception

    def __job_kind(self, *, value: str) -> JobKind:
        """
        Convert a stored job kind into an enum.
        """

        try:
            return JobKind(value)
        except ValueError as exception:
            raise InteractionError(f"Unknown job kind in row: {value}.") from exception

    def __job_state(self, *, value: str) -> JobState:
        """
        Convert a stored job state into an enum.
        """

        try:
            return JobState(value)
        except ValueError as exception:
            raise InteractionError(f"Unknown job state in row: {value}.") from exception

    def __required_str(self, *, row: Mapping[str, object], field: str) -> str:
        """
        Read one required string from a raw SQL row.
        """

        value = row.get(field)

        if not isinstance(value, str):
            raise InteractionError(f"Stored job {field} is not a string.")

        return value

    def __optional_str(self, *, row: Mapping[str, object], field: str) -> Optional[str]:
        """
        Read one optional string from a raw SQL row.
        """

        value = row.get(field)

        if value is None or isinstance(value, str):
            return value

        raise InteractionError(f"Stored job {field} is not a string.")

    def __required_int(self, *, row: Mapping[str, object], field: str) -> int:
        """
        Read one required integer from a raw SQL row.
        """

        value = row.get(field)

        if not isinstance(value, int) or isinstance(value, bool):
            raise InteractionError(f"Stored job {field} is not an integer.")

        return value

    def __required_datetime(self, *, row: Mapping[str, object], field: str) -> datetime:
        """
        Read one required timestamp from a raw SQL row.
        """

        value = row.get(field)

        if not isinstance(value, datetime):
            raise InteractionError(f"Stored job {field} is not a timestamp.")

        return value

    def __optional_datetime(
        self,
        *,
        field: str,
        row: Mapping[str, object],
    ) -> Optional[datetime]:
        """
        Read one optional timestamp from a raw SQL row.
        """

        value = row.get(field)

        if value is None:
            return None

        if isinstance(value, datetime):
            return value

        raise InteractionError(f"Stored job {field} is not a timestamp.")

    def __job_event_kind(self, *, state: JobState) -> EventKind:
        """
        Return the lifecycle event kind for a terminal job state.
        """

        if state is JobState.COMPLETED:
            return EventKind.JOB_COMPLETED

        if state is JobState.FAILED:
            return EventKind.JOB_FAILED

        if state is JobState.ABANDONED:
            return EventKind.JOB_ABANDONED

        raise InteractionError("Unsupported terminal job state.")

    def __same_schedule(self, *, job: Job, request: ScheduleJob) -> bool:
        """
        Check whether a schedule request replays an existing job.
        """

        return (
            job.task == request.task
            and job.kind == request.kind
            and job.thread == request.thread
            and job.payload == request.payload
            and job.identity == request.identity
            and job.metadata == request.metadata
            and job.available == request.available
            and job.timing.created == request.created
            and (request.execution is None or job.execution == request.execution)
        )

    def __same_finish(self, *, job: Job, request: FinishJob) -> bool:
        """
        Check whether a finish request replays an existing terminal outcome.
        """

        return job.state == request.state and job.outcome == request.outcome
