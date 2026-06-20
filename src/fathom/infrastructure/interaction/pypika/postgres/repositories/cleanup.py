from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pypika import PostgreSQLQuery, Table
from pypika.terms import ExistsCriterion, Not
from pypika.terms import Tuple as PypikaTuple

from fathom.constants.collaboration import JobState
from fathom.constants.storage import SqlParameterStyle
from fathom.infrastructure.interaction.pypika.postgres import tables
from fathom.infrastructure.interaction.pypika.postgres.repositories.context import (
    PostgresConnectionProtocol,
    PostgresContext,
)
from fathom.infrastructure.interaction.pypika.query import ParameterizedQuery
from fathom.schemas.interaction import (
    CleanupCascadeResult,
    CleanupRequest,
    CleanupResult,
    SoftDeletedPurgeOutcome,
)

if TYPE_CHECKING:
    from datetime import datetime


class PostgresCleanupService:
    """
    Postgres cleanup service: retention sweeps for expired requests, jobs, events, and soft deletes.
    """

    def __init__(self, *, context: PostgresContext) -> None:
        """
        Bind shared Postgres context for retention sweeps.
        """

        self.__context = context

    async def cleanup(self, *, request: CleanupRequest) -> CleanupResult:
        """
        Delete expired requests rows, terminal jobs, old events, and
        physically purge soft-deleted entities according to the per-scope thresholds in the request.
        """

        async with self.__context.unit.session() as connection:
            idempotency_deleted = await self.__cleanup_requests(
                connection=connection, request=request
            )
            jobs_deleted = await self.__cleanup_terminal_jobs(
                connection=connection, request=request
            )
            events_deleted = await self.__cleanup_events(connection=connection, request=request)
            purge = await self.__cleanup_soft_deleted(connection=connection, request=request)

        return CleanupResult(
            tasks_purged=purge.tasks,
            jobs_deleted=jobs_deleted,
            events_deleted=events_deleted,
            messages_purged=purge.messages,
            artifacts_purged=purge.artifacts,
            scripts_purged=purge.cascade.scripts,
            script_versions_purged=purge.cascade.script_versions,
            threads_purged=purge.cascade.threads,
            contexts_purged=purge.cascade.contexts,
            jobs_cascade_purged=purge.cascade.jobs,
            idempotency_deleted=idempotency_deleted,
            sequences_purged=purge.cascade.sequences,
            events_cascade_purged=purge.cascade.events,
            memberships_purged=purge.cascade.memberships,
        )

    async def __cleanup_requests(
        self,
        *,
        request: CleanupRequest,
        connection: PostgresConnectionProtocol,
    ) -> int:
        """
        Delete requests rows whose stored expires_at < threshold (single statement).
        """

        if request.idempotency_before is None:
            return 0

        before = self.__context._time(value=request.idempotency_before)

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        requests = tables.REQUESTS
        victims = (
            PostgreSQLQuery.from_(requests)
            .select(requests.tenant, requests.key)
            .where(requests.expires_at < binder.bind(value=before))
        )
        if request.tenant is not None:
            victims = victims.where(requests.tenant == binder.bind(value=request.tenant))
        victims = (
            victims.orderby(requests.tenant)
            .orderby(requests.key)
            .limit(binder.bind(value=request.limit))
        )

        statement = (
            PostgreSQLQuery.from_(requests)
            .delete()
            .where(PypikaTuple(requests.tenant, requests.key).isin(victims))
        )
        sql, parameters = binder.render(query=statement)
        result = await connection.execute(sql, parameters)
        return max(result.rowcount or 0, 0)

    async def __cleanup_terminal_jobs(
        self,
        *,
        request: CleanupRequest,
        connection: PostgresConnectionProtocol,
    ) -> int:
        """
        Delete jobs in any terminal state with updated_at < threshold (single statement).
        """

        if request.terminal_jobs_before is None:
            return 0

        terminal_values = (
            JobState.COMPLETED.value,
            JobState.FAILED.value,
            JobState.ABANDONED.value,
        )
        before = self.__context._time(value=request.terminal_jobs_before)

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        jobs = tables.JOBS
        victims = (
            PostgreSQLQuery.from_(jobs)
            .select(jobs.tenant, jobs.id)
            .where(jobs.state.isin([binder.bind(value=value) for value in terminal_values]))
            .where(jobs.updated_at < binder.bind(value=before))
        )
        if request.tenant is not None:
            victims = victims.where(jobs.tenant == binder.bind(value=request.tenant))
        victims = (
            victims.orderby(jobs.tenant).orderby(jobs.id).limit(binder.bind(value=request.limit))
        )

        statement = (
            PostgreSQLQuery.from_(jobs)
            .delete()
            .where(PypikaTuple(jobs.tenant, jobs.id).isin(victims))
        )
        sql, parameters = binder.render(query=statement)
        result = await connection.execute(sql, parameters)
        return max(result.rowcount or 0, 0)

    async def __cleanup_events(
        self,
        *,
        request: CleanupRequest,
        connection: PostgresConnectionProtocol,
    ) -> int:
        """
        Delete lifecycle events with created_at < threshold (single statement).
        """

        if request.events_before is None:
            return 0

        before = self.__context._time(value=request.events_before)

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        events = tables.EVENTS
        victims = (
            PostgreSQLQuery.from_(events)
            .select(events.tenant, events.id)
            .where(events.created_at < binder.bind(value=before))
        )
        if request.tenant is not None:
            victims = victims.where(events.tenant == binder.bind(value=request.tenant))
        victims = (
            victims.orderby(events.tenant)
            .orderby(events.id)
            .limit(binder.bind(value=request.limit))
        )

        statement = (
            PostgreSQLQuery.from_(events)
            .delete()
            .where(PypikaTuple(events.tenant, events.id).isin(victims))
        )
        sql, parameters = binder.render(query=statement)
        result = await connection.execute(sql, parameters)
        return max(result.rowcount or 0, 0)

    async def __cleanup_soft_deleted(
        self,
        *,
        request: CleanupRequest,
        connection: PostgresConnectionProtocol,
    ) -> SoftDeletedPurgeOutcome:
        """
        Physically delete soft-deleted threads, tasks, messages, and artifacts
        older than the request threshold; return aggregated purge counts.
        """

        if request.soft_deleted_before is None:
            return SoftDeletedPurgeOutcome()

        before = self.__context._time(value=request.soft_deleted_before)
        messages = await self.__purge_soft_deleted_messages(
            before=before,
            limit=request.limit,
            tenant=request.tenant,
            connection=connection,
        )
        artifacts = await self.__purge_soft_deleted_artifacts(
            before=before,
            limit=request.limit,
            connection=connection,
            tenant=request.tenant,
        )
        tasks = await self.__purge_soft_deleted_tasks(
            before=before,
            limit=request.limit,
            connection=connection,
            tenant=request.tenant,
        )
        cascade = await self.__purge_soft_deleted_threads(
            before=before,
            limit=request.limit,
            tenant=request.tenant,
            connection=connection,
        )

        return SoftDeletedPurgeOutcome(
            cascade=cascade, tasks=tasks, messages=messages, artifacts=artifacts
        )

    async def __purge_soft_deleted_tasks(
        self,
        *,
        limit: int,
        before: "datetime",
        tenant: Optional[str],
        connection: PostgresConnectionProtocol,
    ) -> int:
        """
        Physically delete soft-deleted tasks no longer referenced by dependents (single statement).
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        target = Table("tasks").as_("target")
        child_tasks = Table("tasks").as_("child")
        messages = tables.MESSAGES
        events = tables.EVENTS
        artifacts = tables.ARTIFACTS
        scripts = tables.SCRIPTS
        script_versions = tables.SCRIPT_VERSIONS
        contexts = tables.CONTEXTS
        jobs = tables.JOBS

        victims = (
            PostgreSQLQuery.from_(target)
            .select(target.tenant, target.id)
            .where(target.deleted_at.notnull())
            .where(target.deleted_at < binder.bind(value=before))
            .where(
                Not(
                    ExistsCriterion(
                        PostgreSQLQuery.from_(child_tasks)
                        .select(1)
                        .where(child_tasks.tenant == target.tenant)
                        .where((child_tasks.parent == target.id) | (child_tasks.root == target.id))
                    )
                )
            )
            .where(
                Not(
                    ExistsCriterion(
                        PostgreSQLQuery.from_(messages)
                        .select(1)
                        .where(messages.tenant == target.tenant)
                        .where(messages.task == target.id)
                    )
                )
            )
            .where(
                Not(
                    ExistsCriterion(
                        PostgreSQLQuery.from_(events)
                        .select(1)
                        .where(events.tenant == target.tenant)
                        .where(events.task == target.id)
                    )
                )
            )
            .where(
                Not(
                    ExistsCriterion(
                        PostgreSQLQuery.from_(artifacts)
                        .select(1)
                        .where(artifacts.tenant == target.tenant)
                        .where(artifacts.task == target.id)
                    )
                )
            )
            .where(
                Not(
                    ExistsCriterion(
                        PostgreSQLQuery.from_(scripts)
                        .select(1)
                        .where(scripts.tenant == target.tenant)
                        .where(scripts.task == target.id)
                    )
                )
            )
            .where(
                Not(
                    ExistsCriterion(
                        PostgreSQLQuery.from_(script_versions)
                        .select(1)
                        .where(script_versions.tenant == target.tenant)
                        .where(script_versions.task == target.id)
                    )
                )
            )
            .where(
                Not(
                    ExistsCriterion(
                        PostgreSQLQuery.from_(contexts)
                        .select(1)
                        .where(contexts.tenant == target.tenant)
                        .where(contexts.task == target.id)
                    )
                )
            )
            .where(
                Not(
                    ExistsCriterion(
                        PostgreSQLQuery.from_(jobs)
                        .select(1)
                        .where(jobs.tenant == target.tenant)
                        .where(jobs.task == target.id)
                    )
                )
            )
        )
        if tenant is not None:
            victims = victims.where(target.tenant == binder.bind(value=tenant))
        victims = victims.orderby(target.tenant).orderby(target.id).limit(binder.bind(value=limit))

        tasks_table = tables.TASKS
        statement = (
            PostgreSQLQuery.from_(tasks_table)
            .delete()
            .where(PypikaTuple(tasks_table.tenant, tasks_table.id).isin(victims))
        )
        sql, parameters = binder.render(query=statement)
        result = await connection.execute(sql, parameters)
        return max(result.rowcount or 0, 0)

    async def __purge_soft_deleted_messages(
        self,
        *,
        limit: int,
        before: "datetime",
        tenant: Optional[str],
        connection: PostgresConnectionProtocol,
    ) -> int:
        """
        Physically delete soft-deleted messages no longer referenced (single statement).
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        target = Table("messages").as_("target")
        child_messages = Table("messages").as_("child")
        tasks_table = tables.TASKS

        victims = (
            PostgreSQLQuery.from_(target)
            .select(target.tenant, target.id)
            .where(target.deleted_at.notnull())
            .where(target.deleted_at < binder.bind(value=before))
            .where(
                Not(
                    ExistsCriterion(
                        PostgreSQLQuery.from_(tasks_table)
                        .select(1)
                        .where(tasks_table.tenant == target.tenant)
                        .where(tasks_table.origin == target.id)
                    )
                )
            )
            .where(
                Not(
                    ExistsCriterion(
                        PostgreSQLQuery.from_(child_messages)
                        .select(1)
                        .where(child_messages.tenant == target.tenant)
                        .where(child_messages.reply == target.id)
                    )
                )
            )
        )
        if tenant is not None:
            victims = victims.where(target.tenant == binder.bind(value=tenant))
        victims = victims.orderby(target.tenant).orderby(target.id).limit(binder.bind(value=limit))

        messages = tables.MESSAGES
        statement = (
            PostgreSQLQuery.from_(messages)
            .delete()
            .where(PypikaTuple(messages.tenant, messages.id).isin(victims))
        )
        sql, parameters = binder.render(query=statement)
        result = await connection.execute(sql, parameters)
        return max(result.rowcount or 0, 0)

    async def __purge_soft_deleted_threads(
        self,
        *,
        limit: int,
        before: "datetime",
        tenant: Optional[str],
        connection: PostgresConnectionProtocol,
    ) -> CleanupCascadeResult:
        """
        Physically purge soft-deleted threads and cascade-delete their FK-bound dependents.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        target = Table("threads").as_("target")
        tasks_table = tables.TASKS
        messages = tables.MESSAGES
        artifacts = tables.ARTIFACTS

        statement = (
            PostgreSQLQuery.from_(target)
            .select(target.tenant, target.id)
            .where(target.deleted_at.notnull())
            .where(target.deleted_at < binder.bind(value=before))
            .where(
                Not(
                    ExistsCriterion(
                        PostgreSQLQuery.from_(tasks_table)
                        .select(1)
                        .where(tasks_table.tenant == target.tenant)
                        .where(tasks_table.thread == target.id)
                    )
                )
            )
            .where(
                Not(
                    ExistsCriterion(
                        PostgreSQLQuery.from_(messages)
                        .select(1)
                        .where(messages.tenant == target.tenant)
                        .where(messages.thread == target.id)
                    )
                )
            )
            .where(
                Not(
                    ExistsCriterion(
                        PostgreSQLQuery.from_(artifacts)
                        .select(1)
                        .where(artifacts.tenant == target.tenant)
                        .where(artifacts.thread == target.id)
                    )
                )
            )
        )
        if tenant is not None:
            statement = statement.where(target.tenant == binder.bind(value=tenant))
        statement = statement.limit(binder.bind(value=limit))
        sql, parameters = binder.render(query=statement)
        async with connection.execute(sql, parameters) as cursor:
            rows = await cursor.fetchall()

        jobs_deleted = 0
        events_deleted = 0
        threads_deleted = 0
        contexts_deleted = 0
        sequences_deleted = 0
        memberships_deleted = 0
        scripts_deleted = 0
        script_versions_deleted = 0

        for row in rows:
            thread = str(row["id"])
            row_tenant = str(row["tenant"])

            version_result = await self.__delete_by_thread(
                connection=connection,
                table=tables.SCRIPT_VERSIONS,
                tenant=row_tenant,
                thread=thread,
            )
            script_versions_deleted += max(version_result or 0, 0)
            script_result = await self.__delete_by_thread(
                connection=connection,
                table=tables.SCRIPTS,
                tenant=row_tenant,
                thread=thread,
            )
            scripts_deleted += max(script_result or 0, 0)
            membership_result = await self.__delete_by_thread(
                connection=connection,
                table=tables.MEMBERSHIPS,
                tenant=row_tenant,
                thread=thread,
            )
            memberships_deleted += max(membership_result or 0, 0)
            context_result = await self.__delete_by_thread(
                connection=connection,
                table=tables.CONTEXTS,
                tenant=row_tenant,
                thread=thread,
            )
            contexts_deleted += max(context_result or 0, 0)
            job_result = await self.__delete_by_thread(
                connection=connection,
                table=tables.JOBS,
                tenant=row_tenant,
                thread=thread,
            )
            jobs_deleted += max(job_result or 0, 0)
            event_result = await self.__delete_by_thread(
                connection=connection,
                table=tables.EVENTS,
                tenant=row_tenant,
                thread=thread,
            )
            events_deleted += max(event_result or 0, 0)
            sequence_result = await self.__delete_by_thread(
                connection=connection,
                table=tables.SEQUENCES,
                tenant=row_tenant,
                thread=thread,
            )
            sequences_deleted += max(sequence_result or 0, 0)
            thread_result = await self.__delete_thread_row(
                connection=connection,
                tenant=row_tenant,
                thread=thread,
            )
            threads_deleted += max(thread_result or 0, 0)

        return CleanupCascadeResult(
            jobs=jobs_deleted,
            events=events_deleted,
            threads=threads_deleted,
            contexts=contexts_deleted,
            sequences=sequences_deleted,
            memberships=memberships_deleted,
            scripts=scripts_deleted,
            script_versions=script_versions_deleted,
        )

    async def __purge_soft_deleted_artifacts(
        self,
        *,
        limit: int,
        before: "datetime",
        tenant: Optional[str],
        connection: PostgresConnectionProtocol,
    ) -> int:
        """
        Delete soft-deleted artifacts no longer referenced by scripts (single statement).
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        target = Table("artifacts").as_("target")
        scripts = tables.SCRIPTS
        script_versions = tables.SCRIPT_VERSIONS

        victims = (
            PostgreSQLQuery.from_(target)
            .select(target.tenant, target.id)
            .where(target.deleted_at.notnull())
            .where(target.deleted_at < binder.bind(value=before))
            .where(
                Not(
                    ExistsCriterion(
                        PostgreSQLQuery.from_(scripts)
                        .select(1)
                        .where(scripts.tenant == target.tenant)
                        .where(scripts.artifact == target.id)
                    )
                )
            )
            .where(
                Not(
                    ExistsCriterion(
                        PostgreSQLQuery.from_(script_versions)
                        .select(1)
                        .where(script_versions.tenant == target.tenant)
                        .where(script_versions.artifact == target.id)
                    )
                )
            )
        )
        if tenant is not None:
            victims = victims.where(target.tenant == binder.bind(value=tenant))
        victims = victims.orderby(target.tenant).orderby(target.id).limit(binder.bind(value=limit))

        artifacts = tables.ARTIFACTS
        statement = (
            PostgreSQLQuery.from_(artifacts)
            .delete()
            .where(PypikaTuple(artifacts.tenant, artifacts.id).isin(victims))
        )
        sql, parameters = binder.render(query=statement)
        result = await connection.execute(sql, parameters)
        return max(result.rowcount or 0, 0)

    async def __delete_by_thread(
        self,
        *,
        connection: PostgresConnectionProtocol,
        table: Table,
        tenant: str,
        thread: str,
    ) -> int:
        """
        Delete all rows in one table matching tenant/thread.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        statement = (
            PostgreSQLQuery.from_(table)
            .delete()
            .where(table.tenant == binder.bind(value=tenant))
            .where(table.thread == binder.bind(value=thread))
        )
        sql, parameters = binder.render(query=statement)
        result = await connection.execute(sql, parameters)
        return result.rowcount or 0

    async def __delete_thread_row(
        self,
        *,
        connection: PostgresConnectionProtocol,
        tenant: str,
        thread: str,
    ) -> int:
        """
        Delete one thread row by primary key.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        threads = tables.THREADS
        statement = (
            PostgreSQLQuery.from_(threads)
            .delete()
            .where(threads.tenant == binder.bind(value=tenant))
            .where(threads.id == binder.bind(value=thread))
        )
        sql, parameters = binder.render(query=statement)
        result = await connection.execute(sql, parameters)
        return result.rowcount or 0
