from __future__ import annotations

from typing import List, Optional

import aiosqlite
from pypika import SQLLiteQuery

from fathom.constants.collaboration import EventKind, EventSource, JobState
from fathom.constants.storage import SqlParameterStyle
from fathom.core.exceptions import InteractionError, JobLeaseLostError
from fathom.infrastructure.interaction.pypika.query import ParameterizedQuery
from fathom.infrastructure.interaction.pypika.sqlite import tables
from fathom.infrastructure.interaction.pypika.sqlite.repositories.context import StoreContext
from fathom.schemas.interaction import (
    ClaimJob,
    FinishJob,
    Job,
    JobQuery,
    Metadata,
    RecoverJob,
    RescheduleJob,
    ScheduleJob,
    Timing,
)


class JobRepository:
    """
    Job repository: persists, claims, and recovers background jobs.
    """

    def __init__(self, *, context: StoreContext) -> None:
        """
        Bind shared store context for job persistence.
        """

        self.__context = context

    async def schedule_job(self, *, request: ScheduleJob) -> Job:
        """
        Persist one pending background job and its lifecycle event.
        """

        timing = Timing(created_at=request.created, updated_at=request.created)
        async with self.__context.unit.session() as connection:
            await self.__context._require_thread(
                connection=connection,
                tenant=request.identity.tenant,
                thread=request.thread,
            )
            if request.task is not None:
                await self.__context._require_task_in_thread(
                    connection=connection,
                    tenant=request.identity.tenant,
                    thread=request.thread,
                    task=request.task,
                )
            existing = await self.__context._load_job(
                connection=connection,
                tenant=request.identity.tenant,
                job=request.identity.id,
            )
            if existing is not None:
                if not self.__same_schedule_job(job=existing, request=request):
                    raise InteractionError("Job identity already exists with different content.")

                return existing

            binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
            jobs = tables.JOBS
            statement = (
                SQLLiteQuery.into(jobs)
                .columns(
                    jobs.id,
                    jobs.tenant,
                    jobs.workspace,
                    jobs.thread,
                    jobs.task,
                    jobs.kind,
                    jobs.state,
                    jobs.attempts,
                    jobs.owner,
                    jobs.locked_at,
                    jobs.available_at,
                    jobs.payload,
                    jobs.code,
                    jobs.detail,
                    jobs.created_at,
                    jobs.updated_at,
                    jobs.metadata,
                )
                .insert(
                    binder.bind(value=request.identity.id),
                    binder.bind(value=request.identity.tenant),
                    binder.bind(value=request.identity.workspace),
                    binder.bind(value=request.thread),
                    binder.bind(value=request.task),
                    binder.bind(value=request.kind.value),
                    binder.bind(value=JobState.PENDING.value),
                    binder.bind(value=0),
                    binder.bind(value=None),
                    binder.bind(value=None),
                    binder.bind(value=self.__context._time(value=request.available)),
                    binder.bind(value=self.__context._json(value=request.payload.entries)),
                    binder.bind(value=None),
                    binder.bind(value=None),
                    binder.bind(value=self.__context._time(value=timing.created)),
                    binder.bind(value=self.__context._time(value=timing.updated)),
                    binder.bind(value=self.__context._json(value=request.metadata.entries)),
                )
            )
            sql, parameters = binder.render(query=statement)
            await connection.execute(sql, parameters)
            job = await self.__context._load_job(
                connection=connection,
                tenant=request.identity.tenant,
                job=request.identity.id,
            )
            await self.__context._record_event(
                connection=connection,
                subject=request.identity.id,
                tenant=request.identity.tenant,
                workspace=request.identity.workspace,
                thread=request.thread,
                task=request.task,
                actor=None,
                kind=EventKind.JOB_SCHEDULED,
                source=EventSource.INTERACTION,
                payload=Metadata(entries={"kind": request.kind.value}),
                created=request.created,
            )

        if job is None:
            raise InteractionError("Job was not persisted.")

        return job

    async def claim_job(self, *, request: ClaimJob) -> Optional[Job]:
        """
        Claim one available job for a worker.
        """

        async with self.__context.unit.session() as connection:
            job = await self.__claimable_job(connection=connection, request=request)
            if job is None:
                return None

            self.__context.lifecycle.validate_job_claim(state=job.state)
            cursor = await connection.execute(
                """
                UPDATE jobs
                SET state = ?, attempts = ?, owner = ?, locked_at = ?, updated_at = ?
                WHERE tenant = ? AND id = ? AND state = ? AND owner IS NULL
                """,
                (
                    JobState.CLAIMED.value,
                    job.attempts + 1,
                    request.owner,
                    self.__context._time(value=request.claimed),
                    self.__context._time(value=request.claimed),
                    request.tenant,
                    job.identity.id,
                    JobState.PENDING.value,
                ),
            )
            if cursor.rowcount == 0:
                return None

            return await self.__context._load_job(
                connection=connection,
                tenant=request.tenant,
                job=job.identity.id,
            )

    async def finish_job(self, *, request: FinishJob) -> Job:
        """
        Move one claimed job to a terminal state.
        """

        async with self.__context.unit.session() as connection:
            job = await self.__context._load_job(
                connection=connection, tenant=request.tenant, job=request.job
            )
            if job is None:
                raise InteractionError("Job does not exist.")

            if job.state == request.state and job.outcome is not None:
                if not self.__same_finish_job(job=job, request=request):
                    raise InteractionError("Job already finished with a different outcome.")

                return job

            if job.owner != request.owner:
                raise JobLeaseLostError(
                    job=request.job,
                    message="Job lease was lost; another worker now owns this claim.",
                )

            self.__context.lifecycle.validate_job_finish(state=job.state, target=request.state)
            update_cursor = await connection.execute(
                """
                UPDATE jobs
                SET state = ?, code = ?, detail = ?, updated_at = ?
                WHERE tenant = ? AND id = ? AND owner = ? AND state = ?
                """,
                (
                    request.state.value,
                    request.outcome.code.value,
                    request.outcome.detail,
                    self.__context._time(value=request.finished),
                    request.tenant,
                    request.job,
                    request.owner,
                    JobState.CLAIMED.value,
                ),
            )
            if update_cursor.rowcount == 0:
                raise JobLeaseLostError(
                    job=request.job,
                    message="Job lease was lost between read and write.",
                )
            finished = await self.__context._load_job(
                connection=connection,
                tenant=request.tenant,
                job=request.job,
            )
            await self.__context._record_event(
                connection=connection,
                subject=request.job,
                tenant=request.tenant,
                workspace=job.identity.workspace,
                thread=job.thread,
                task=job.task,
                actor=None,
                kind=self.__context._job_event_kind(state=request.state),
                source=EventSource.WORKER,
                payload=Metadata(entries={"code": request.outcome.code.value}),
                created=request.finished,
            )

        if finished is None:
            raise InteractionError("Job was not updated.")

        return finished

    async def recover_jobs(self, *, request: RecoverJob) -> List[Job]:
        """
        Release stale claimed jobs for retry.
        """

        async with self.__context.unit.session() as connection:
            stale = await self.__stale_jobs(connection=connection, request=request)
            recovered: List[Job] = []
            for job in stale:
                await connection.execute(
                    """
                    UPDATE jobs
                    SET state = ?, owner = ?, locked_at = ?, available_at = ?, updated_at = ?
                    WHERE tenant = ? AND id = ?
                    """,
                    (
                        JobState.PENDING.value,
                        None,
                        None,
                        self.__context._time(value=request.available),
                        self.__context._time(value=request.available),
                        request.tenant,
                        job.identity.id,
                    ),
                )
                updated = await self.__context._load_job(
                    connection=connection,
                    tenant=request.tenant,
                    job=job.identity.id,
                )
                if updated is None:
                    raise InteractionError("Recovered job could not be loaded.")

                await self.__context._record_event(
                    connection=connection,
                    subject=f"{job.identity.id}/{job.attempts}/{request.available.isoformat()}",
                    tenant=request.tenant,
                    workspace=job.identity.workspace,
                    thread=job.thread,
                    task=job.task,
                    actor=None,
                    kind=EventKind.RECOVERY_LOST,
                    source=EventSource.RECOVERY,
                    payload=Metadata(entries={"owner": job.owner, "kind": job.kind.value}),
                    created=request.available,
                )
                recovered.append(updated)

        return recovered

    async def reschedule_job(self, *, request: RescheduleJob) -> Job:
        """
        Release one claimed job for retry after backoff.
        """

        async with self.__context.unit.session() as connection:
            job = await self.__context._load_job(
                connection=connection, tenant=request.tenant, job=request.job
            )
            if job is None:
                raise InteractionError("Job does not exist.")
            if job.state != JobState.CLAIMED:
                raise InteractionError("Only claimed jobs can be rescheduled.")
            if job.owner != request.owner:
                raise JobLeaseLostError(
                    job=request.job,
                    message="Job lease was lost; cannot reschedule another worker's claim.",
                )
            if job.attempts != request.attempts:
                raise InteractionError("Job attempts changed before reschedule.")

            update_cursor = await connection.execute(
                """
                UPDATE jobs
                SET state = ?, owner = ?, locked_at = ?, available_at = ?, updated_at = ?
                WHERE tenant = ? AND id = ? AND owner = ? AND state = ?
                """,
                (
                    JobState.PENDING.value,
                    None,
                    None,
                    self.__context._time(value=request.available),
                    self.__context._time(value=request.rescheduled),
                    request.tenant,
                    request.job,
                    request.owner,
                    JobState.CLAIMED.value,
                ),
            )
            if update_cursor.rowcount == 0:
                raise JobLeaseLostError(
                    job=request.job,
                    message="Job lease was lost between read and write.",
                )
            updated = await self.__context._load_job(
                connection=connection,
                tenant=request.tenant,
                job=request.job,
            )
            if updated is None:
                raise InteractionError("Rescheduled job could not be loaded.")

            await self.__context._record_event(
                connection=connection,
                subject=f"{request.job}/retry/{request.attempts}",
                tenant=request.tenant,
                workspace=job.identity.workspace,
                thread=job.thread,
                task=job.task,
                actor=None,
                kind=EventKind.JOB_RESCHEDULED,
                source=EventSource.RECOVERY,
                payload=Metadata(
                    entries={
                        "kind": job.kind.value,
                        "attempts": request.attempts,
                        "detail": request.detail,
                    }
                ),
                created=request.rescheduled,
            )

        return updated

    async def get_jobs(self, *, query: JobQuery) -> List[Job]:
        """
        Load tenant-scoped jobs with any combination of optional filters.
        """

        async with (
            self.__context.unit.session() as connection,
            connection.execute(
                """
                SELECT *
                FROM jobs
                WHERE tenant = ?
                    AND (? IS NULL OR thread = ?)
                    AND (? IS NULL OR state = ?)
                    AND (? IS NULL OR kind = ?)
                ORDER BY created_at ASC, id ASC
                """,
                (
                    query.tenant,
                    query.thread,
                    query.thread,
                    query.state.value if query.state is not None else None,
                    query.state.value if query.state is not None else None,
                    query.kind.value if query.kind is not None else None,
                    query.kind.value if query.kind is not None else None,
                ),
            ) as cursor,
        ):
            rows = await cursor.fetchall()

        return [self.__context.rows.job(row=row) for row in rows]

    async def __claimable_job(
        self,
        *,
        connection: aiosqlite.Connection,
        request: ClaimJob,
    ) -> Optional[Job]:
        """
        Load the next job that can be claimed.
        """

        if request.job is not None:
            return await self.__claimable_job_identity(connection=connection, request=request)
        if request.kind is not None:
            return await self.__claimable_job_by_kind(connection=connection, request=request)

        async with connection.execute(
            """
            SELECT *
            FROM jobs
            WHERE tenant = ? AND state = ? AND available_at <= ?
            ORDER BY available_at ASC, id ASC
            LIMIT 1
            """,
            (request.tenant, JobState.PENDING.value, self.__context._time(value=request.claimed)),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        return self.__context.rows.job(row=row)

    async def __claimable_job_identity(
        self,
        *,
        connection: aiosqlite.Connection,
        request: ClaimJob,
    ) -> Optional[Job]:
        """
        Load one claimable job by identity.
        """

        async with connection.execute(
            """
            SELECT *
            FROM jobs
            WHERE tenant = ? AND id = ? AND state = ? AND available_at <= ?
            LIMIT 1
            """,
            (
                request.tenant,
                request.job,
                JobState.PENDING.value,
                self.__context._time(value=request.claimed),
            ),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        return self.__context.rows.job(row=row)

    async def __claimable_job_by_kind(
        self,
        *,
        connection: aiosqlite.Connection,
        request: ClaimJob,
    ) -> Optional[Job]:
        """
        Load one claimable job by kind.
        """

        kind = request.kind
        if kind is None:
            raise InteractionError("Job kind is required for kind-based claiming.")

        async with connection.execute(
            """
            SELECT *
            FROM jobs
            WHERE tenant = ? AND kind = ? AND state = ? AND available_at <= ?
            ORDER BY available_at ASC, id ASC
            LIMIT 1
            """,
            (
                request.tenant,
                kind.value,
                JobState.PENDING.value,
                self.__context._time(value=request.claimed),
            ),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        return self.__context.rows.job(row=row)

    async def __stale_jobs(
        self,
        *,
        connection: aiosqlite.Connection,
        request: RecoverJob,
    ) -> List[Job]:
        """
        Load claimed jobs whose locks are stale.
        """

        if request.kind is not None:
            return await self.__stale_jobs_by_kind(connection=connection, request=request)

        async with connection.execute(
            """
            SELECT *
            FROM jobs
            WHERE tenant = ? AND state = ? AND locked_at <= ?
            ORDER BY locked_at ASC, id ASC
            LIMIT ?
            """,
            (
                request.tenant,
                JobState.CLAIMED.value,
                self.__context._time(value=request.before),
                request.limit,
            ),
        ) as cursor:
            rows = await cursor.fetchall()

        return [self.__context.rows.job(row=row) for row in rows]

    async def __stale_jobs_by_kind(
        self,
        *,
        connection: aiosqlite.Connection,
        request: RecoverJob,
    ) -> List[Job]:
        """
        Load claimed jobs of one kind whose locks are stale.
        """

        kind = request.kind
        if kind is None:
            raise InteractionError("Job kind is required for kind-based recovery.")

        async with connection.execute(
            """
            SELECT *
            FROM jobs
            WHERE tenant = ? AND kind = ? AND state = ? AND locked_at <= ?
            ORDER BY locked_at ASC, id ASC
            LIMIT ?
            """,
            (
                request.tenant,
                kind.value,
                JobState.CLAIMED.value,
                self.__context._time(value=request.before),
                request.limit,
            ),
        ) as cursor:
            rows = await cursor.fetchall()

        return [self.__context.rows.job(row=row) for row in rows]

    def __same_schedule_job(self, *, job: Job, request: ScheduleJob) -> bool:
        """
        Check whether a job request replays an already scheduled job.
        """

        return (
            job.identity.tenant == request.identity.tenant
            and job.identity.workspace == request.identity.workspace
            and job.thread == request.thread
            and job.task == request.task
            and job.kind == request.kind
            and job.available == request.available
            and job.payload == request.payload
            and job.timing.created == request.created
            and job.metadata == request.metadata
        )

    def __same_finish_job(self, *, job: Job, request: FinishJob) -> bool:
        """
        Check whether a finish request replays an already stored job outcome.
        """

        if job.outcome is None:
            return False

        return job.state == request.state and job.outcome == request.outcome
