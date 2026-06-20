from __future__ import annotations

from typing import List, Optional

from pypika import PostgreSQLQuery

from fathom.constants.collaboration import EventKind, EventSource, JobState
from fathom.constants.storage import SqlParameterStyle
from fathom.core.exceptions import InteractionError, JobLeaseLostError
from fathom.infrastructure.interaction.pypika.postgres import tables
from fathom.infrastructure.interaction.pypika.postgres.repositories.context import (
    PostgresConnectionProtocol,
    PostgresContext,
)
from fathom.infrastructure.interaction.pypika.query import ParameterizedQuery
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


class PostgresJobRepository:
    """
    Postgres job repository: persists, claims, and recovers background jobs.
    """

    def __init__(self, *, context: PostgresContext) -> None:
        """
        Bind shared Postgres context for job persistence.
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

            binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
            jobs = tables.JOBS
            statement = (
                PostgreSQLQuery.into(jobs)
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
        Claim the next available job using native Postgres SKIP LOCKED semantics.
        """

        sql, parameters = self.__build_claim_query(request=request)

        async with (
            self.__context.unit.session() as connection,
            connection.execute(sql, parameters) as cursor,
        ):
            row = await cursor.fetchone()

        if row is None:
            return None

        return self.__context.rows.job(row=row)

    def __build_claim_query(
        self,
        *,
        request: ClaimJob,
    ) -> tuple[str, tuple[object, ...]]:
        """
        Render the atomic SKIP LOCKED claim CTE and its bound parameters.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        claimed = self.__context._time(value=request.claimed)
        kind_value = request.kind.value if request.kind is not None else None

        candidate_conditions: List[str] = [
            f"tenant = {binder.bind_placeholder(value=request.tenant)}",
            f"state = {binder.bind_placeholder(value=JobState.PENDING.value)}",
            f"available_at <= {binder.bind_placeholder(value=claimed)}",
        ]
        if request.job is not None:
            candidate_conditions.append(f"id = {binder.bind_placeholder(value=request.job)}")
        if kind_value is not None:
            candidate_conditions.append(f"kind = {binder.bind_placeholder(value=kind_value)}")

        set_state = binder.bind_placeholder(value=JobState.CLAIMED.value)
        set_owner = binder.bind_placeholder(value=request.owner)
        set_locked = binder.bind_placeholder(value=claimed)
        set_updated = binder.bind_placeholder(value=claimed)
        where_tenant = binder.bind_placeholder(value=request.tenant)

        sql = (
            "WITH candidate AS ("  # nosec B608 - fragments are fixed; values are bound.
            "SELECT id, attempts FROM jobs "
            f"WHERE {' AND '.join(candidate_conditions)} "
            "ORDER BY available_at ASC, id ASC "
            "FOR UPDATE SKIP LOCKED LIMIT 1"
            ") "
            "UPDATE jobs AS target "
            f"SET state = {set_state}, attempts = candidate.attempts + 1, "
            f"owner = {set_owner}, locked_at = {set_locked}, updated_at = {set_updated} "
            "FROM candidate "
            f"WHERE target.tenant = {where_tenant} AND target.id = candidate.id "
            "RETURNING target.*"
        )

        return sql, binder.parameters

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
            binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
            jobs = tables.JOBS
            statement = (
                PostgreSQLQuery.update(jobs)
                .set(jobs.state, binder.bind(value=request.state.value))
                .set(jobs.code, binder.bind(value=request.outcome.code.value))
                .set(jobs.detail, binder.bind(value=request.outcome.detail))
                .set(
                    jobs.updated_at, binder.bind(value=self.__context._time(value=request.finished))
                )
                .where(jobs.tenant == binder.bind(value=request.tenant))
                .where(jobs.id == binder.bind(value=request.job))
                .where(jobs.owner == binder.bind(value=request.owner))
                .where(jobs.state == binder.bind(value=JobState.CLAIMED.value))
            )
            sql, parameters = binder.render(query=statement)
            update_cursor = await connection.execute(sql, parameters)
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
                recover_binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
                jobs = tables.JOBS
                recover_statement = (
                    PostgreSQLQuery.update(jobs)
                    .set(jobs.state, recover_binder.bind(value=JobState.PENDING.value))
                    .set(jobs.owner, recover_binder.bind(value=None))
                    .set(jobs.locked_at, recover_binder.bind(value=None))
                    .set(
                        jobs.available_at,
                        recover_binder.bind(value=self.__context._time(value=request.available)),
                    )
                    .set(
                        jobs.updated_at,
                        recover_binder.bind(value=self.__context._time(value=request.available)),
                    )
                    .where(jobs.tenant == recover_binder.bind(value=request.tenant))
                    .where(jobs.id == recover_binder.bind(value=job.identity.id))
                )
                recover_sql, recover_parameters = recover_binder.render(query=recover_statement)
                await connection.execute(recover_sql, recover_parameters)
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

            binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
            jobs = tables.JOBS
            statement = (
                PostgreSQLQuery.update(jobs)
                .set(jobs.state, binder.bind(value=JobState.PENDING.value))
                .set(jobs.owner, binder.bind(value=None))
                .set(jobs.locked_at, binder.bind(value=None))
                .set(
                    jobs.available_at,
                    binder.bind(value=self.__context._time(value=request.available)),
                )
                .set(
                    jobs.updated_at,
                    binder.bind(value=self.__context._time(value=request.rescheduled)),
                )
                .where(jobs.tenant == binder.bind(value=request.tenant))
                .where(jobs.id == binder.bind(value=request.job))
                .where(jobs.owner == binder.bind(value=request.owner))
                .where(jobs.state == binder.bind(value=JobState.CLAIMED.value))
            )
            sql, parameters = binder.render(query=statement)
            update_cursor = await connection.execute(sql, parameters)
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

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        jobs = tables.JOBS
        statement = (
            PostgreSQLQuery.from_(jobs)
            .select(jobs.star)
            .where(jobs.tenant == binder.bind(value=query.tenant))
        )
        if query.thread is not None:
            statement = statement.where(jobs.thread == binder.bind(value=query.thread))
        if query.state is not None:
            statement = statement.where(jobs.state == binder.bind(value=query.state.value))
        if query.kind is not None:
            statement = statement.where(jobs.kind == binder.bind(value=query.kind.value))
        statement = statement.orderby(jobs.created_at).orderby(jobs.id)
        sql, parameters = binder.render(query=statement)
        async with (
            self.__context.unit.session() as connection,
            connection.execute(sql, parameters) as cursor,
        ):
            rows = await cursor.fetchall()

        return [self.__context.rows.job(row=row) for row in rows]

    async def __stale_jobs(
        self,
        *,
        connection: PostgresConnectionProtocol,
        request: RecoverJob,
    ) -> List[Job]:
        """
        Load claimed jobs whose locks are stale.
        """

        if request.kind is not None:
            return await self.__stale_jobs_by_kind(connection=connection, request=request)

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        jobs = tables.JOBS
        statement = (
            PostgreSQLQuery.from_(jobs)
            .select(jobs.star)
            .where(jobs.tenant == binder.bind(value=request.tenant))
            .where(jobs.state == binder.bind(value=JobState.CLAIMED.value))
            .where(jobs.locked_at <= binder.bind(value=self.__context._time(value=request.before)))
            .orderby(jobs.locked_at)
            .orderby(jobs.id)
            .limit(request.limit)
        )
        sql, parameters = binder.render(query=statement)
        async with connection.execute(sql, parameters) as cursor:
            rows = await cursor.fetchall()

        return [self.__context.rows.job(row=row) for row in rows]

    async def __stale_jobs_by_kind(
        self,
        *,
        connection: PostgresConnectionProtocol,
        request: RecoverJob,
    ) -> List[Job]:
        """
        Load claimed jobs of one kind whose locks are stale.
        """

        kind = request.kind
        if kind is None:
            raise InteractionError("Job kind is required for kind-based recovery.")

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        jobs = tables.JOBS
        statement = (
            PostgreSQLQuery.from_(jobs)
            .select(jobs.star)
            .where(jobs.tenant == binder.bind(value=request.tenant))
            .where(jobs.kind == binder.bind(value=kind.value))
            .where(jobs.state == binder.bind(value=JobState.CLAIMED.value))
            .where(jobs.locked_at <= binder.bind(value=self.__context._time(value=request.before)))
            .orderby(jobs.locked_at)
            .orderby(jobs.id)
            .limit(request.limit)
        )
        sql, parameters = binder.render(query=statement)
        async with connection.execute(sql, parameters) as cursor:
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
