from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Type

from tortoise.models import Model

from fathom.constants.collaboration import JobState
from fathom.infrastructure.interaction.orm.models import (
    ArtifactRecord,
    ContextRecord,
    ConversationRecord,
    EventRecord,
    ExecutionRecord,
    JobRecord,
    MembershipRecord,
    MessageRecord,
    RequestRecord,
    ScriptRecord,
    ScriptVersionRecord,
    SequenceRecord,
    TaskRecord,
)
from fathom.schemas.interaction import (
    CleanupCascadeResult,
    CleanupRequest,
    CleanupResult,
    SoftDeletedPurgeOutcome,
)

if TYPE_CHECKING:
    from datetime import datetime

    from fathom.infrastructure.interaction.orm.repositories.lifecycle import (
        DatabaseConnection,
        TransactionScope,
    )


class CleanupRepository:
    """
    Repository for bounded retention cleanup and physical purge sweeps.

    Existence probes intentionally use raw SQL for two reasons: Tortoise ORM
    cannot cleanly express the Postgres JSONB `?` operator used against
    context reference arrays, and multi-table reference checks combine into
    a single UNION ALL to keep per-row cleanup cost at one round-trip.
    """

    def __init__(self, *, transaction: "TransactionScope") -> None:
        """
        Initialize cleanup transaction boundary.
        """

        self.__transaction = transaction

    async def cleanup(self, *, request: CleanupRequest) -> CleanupResult:
        """
        Run one bounded cleanup sweep.
        """

        async with self.__transaction.transaction() as connection:
            idempotency_deleted = await self.__cleanup_requests(
                request=request,
                connection=connection,
            )
            jobs_deleted = await self.__cleanup_terminal_jobs(
                request=request,
                connection=connection,
            )
            events_deleted = await self.__cleanup_events(
                request=request,
                connection=connection,
            )
            purge = await self.__cleanup_soft_deleted(
                request=request,
                connection=connection,
            )

        return CleanupResult(
            tasks_purged=purge.tasks,
            jobs_deleted=jobs_deleted,
            events_deleted=events_deleted,
            messages_purged=purge.messages,
            artifacts_purged=purge.artifacts,
            threads_purged=purge.cascade.threads,
            scripts_purged=purge.cascade.scripts,
            contexts_purged=purge.cascade.contexts,
            jobs_cascade_purged=purge.cascade.jobs,
            idempotency_deleted=idempotency_deleted,
            sequences_purged=purge.cascade.sequences,
            events_cascade_purged=purge.cascade.events,
            executions_purged=purge.cascade.executions,
            memberships_purged=purge.cascade.memberships,
            script_versions_purged=purge.cascade.script_versions,
        )

    async def __cleanup_requests(
        self,
        *,
        request: CleanupRequest,
        connection: "DatabaseConnection",
    ) -> int:
        """
        Delete expired idempotency request rows.
        """

        if request.idempotency_before is None:
            return 0

        queryset = RequestRecord.filter(expires_at__lt=request.idempotency_before)

        if request.tenant is not None:
            queryset = queryset.filter(tenant_id=request.tenant)

        victims = (
            await queryset.using_db(connection).order_by("tenant_id", "key").limit(request.limit)
        )

        return await self.__delete_rows(
            rows=victims,
            record=RequestRecord,
            connection=connection,
        )

    async def __cleanup_terminal_jobs(
        self,
        *,
        request: CleanupRequest,
        connection: "DatabaseConnection",
    ) -> int:
        """
        Delete terminal jobs older than the retention threshold.
        """

        if request.terminal_jobs_before is None:
            return 0

        queryset = JobRecord.filter(
            state__in=(
                JobState.FAILED.value,
                JobState.COMPLETED.value,
                JobState.ABANDONED.value,
            ),
            updated_at__lt=request.terminal_jobs_before,
        )
        if request.tenant is not None:
            queryset = queryset.filter(tenant_id=request.tenant)

        victims = (
            await queryset.using_db(connection).order_by("tenant_id", "id").limit(request.limit)
        )

        return await self.__delete_rows(
            rows=victims,
            record=JobRecord,
            connection=connection,
        )

    async def __cleanup_events(
        self,
        *,
        request: CleanupRequest,
        connection: "DatabaseConnection",
    ) -> int:
        """
        Delete lifecycle events older than the retention threshold.
        """

        if request.events_before is None:
            return 0

        queryset = EventRecord.filter(created_at__lt=request.events_before)

        if request.tenant is not None:
            queryset = queryset.filter(tenant_id=request.tenant)

        victims = (
            await queryset.using_db(connection).order_by("tenant_id", "id").limit(request.limit)
        )

        deleted = 0

        for victim in victims:
            row = victim
            if await self.__event_is_referenced(
                event=row.id,
                tenant=row.tenant_id,
                connection=connection,
            ):
                continue

            deleted += await self.__delete_row(
                row=row,
                record=EventRecord,
                connection=connection,
            )

        return deleted

    async def __cleanup_soft_deleted(
        self,
        *,
        request: CleanupRequest,
        connection: "DatabaseConnection",
    ) -> SoftDeletedPurgeOutcome:
        """
        Physically purge eligible soft-deleted rows.
        """

        if request.soft_deleted_before is None:
            return SoftDeletedPurgeOutcome()

        messages = await self.__purge_soft_deleted_messages(
            limit=request.limit,
            tenant=request.tenant,
            connection=connection,
            before=request.soft_deleted_before,
        )
        artifacts = await self.__purge_soft_deleted_artifacts(
            limit=request.limit,
            tenant=request.tenant,
            connection=connection,
            before=request.soft_deleted_before,
        )
        tasks = await self.__purge_soft_deleted_tasks(
            limit=request.limit,
            tenant=request.tenant,
            connection=connection,
            before=request.soft_deleted_before,
        )
        cascade = await self.__purge_soft_deleted_threads(
            limit=request.limit,
            tenant=request.tenant,
            connection=connection,
            before=request.soft_deleted_before,
        )

        return SoftDeletedPurgeOutcome(
            tasks=tasks,
            cascade=cascade,
            messages=messages,
            artifacts=artifacts,
        )

    async def __purge_soft_deleted_messages(
        self,
        *,
        limit: int,
        before: "datetime",
        tenant: Optional[str],
        connection: "DatabaseConnection",
    ) -> int:
        """
        Purge soft-deleted messages that are no longer referenced.
        """

        queryset = MessageRecord.filter(deleted_at__isnull=False, deleted_at__lt=before)

        if tenant is not None:
            queryset = queryset.filter(tenant_id=tenant)

        candidates = await queryset.using_db(connection).order_by("tenant_id", "id").limit(limit)

        deleted = 0

        for candidate in candidates:
            row = candidate
            if await self.__message_is_referenced(
                message=row.id, tenant=row.tenant_id, connection=connection
            ):
                continue

            deleted += await self.__delete_row(
                row=row,
                record=MessageRecord,
                connection=connection,
            )

        return deleted

    async def __purge_soft_deleted_artifacts(
        self,
        *,
        limit: int,
        before: "datetime",
        tenant: Optional[str],
        connection: "DatabaseConnection",
    ) -> int:
        """
        Purge soft-deleted artifacts that are no longer referenced by scripts.
        """

        queryset = ArtifactRecord.filter(deleted_at__isnull=False, deleted_at__lt=before)

        if tenant is not None:
            queryset = queryset.filter(tenant_id=tenant)

        candidates = await queryset.using_db(connection).order_by("tenant_id", "id").limit(limit)

        deleted = 0

        for candidate in candidates:
            row = candidate
            if await self.__artifact_is_referenced(
                artifact=row.id,
                tenant=row.tenant_id,
                connection=connection,
            ):
                continue

            deleted += await self.__delete_row(
                row=row,
                record=ArtifactRecord,
                connection=connection,
            )

        return deleted

    async def __purge_soft_deleted_tasks(
        self,
        *,
        limit: int,
        before: "datetime",
        tenant: Optional[str],
        connection: "DatabaseConnection",
    ) -> int:
        """
        Purge soft-deleted tasks that are no longer referenced.
        """

        queryset = TaskRecord.filter(deleted_at__isnull=False, deleted_at__lt=before)

        if tenant is not None:
            queryset = queryset.filter(tenant_id=tenant)

        candidates = await queryset.using_db(connection).order_by("tenant_id", "id").limit(limit)

        deleted = 0

        for candidate in candidates:
            row = candidate
            if await self.__task_is_referenced(
                task=row.id,
                tenant=row.tenant_id,
                connection=connection,
            ):
                continue

            deleted += await self.__delete_row(
                row=row,
                record=TaskRecord,
                connection=connection,
            )

        return deleted

    async def __delete_rows(
        self,
        *,
        record: Type[Model],
        rows: Sequence[Model],
        connection: "DatabaseConnection",
    ) -> int:
        """
        Delete cleanup candidates by primary key.
        """

        deleted = 0
        for row in rows:
            deleted += await self.__delete_row(
                row=row,
                record=record,
                connection=connection,
            )

        return deleted

    async def __delete_row(
        self,
        *,
        row: Model,
        record: Type[Model],
        connection: "DatabaseConnection",
    ) -> int:
        """
        Delete one cleanup candidate by primary key.
        """

        return int(await record.filter(id=row.pk).using_db(connection).delete())

    async def __purge_soft_deleted_threads(
        self,
        *,
        limit: int,
        before: "datetime",
        tenant: Optional[str],
        connection: "DatabaseConnection",
    ) -> CleanupCascadeResult:
        """
        Purge soft-deleted threads and their thread-bound dependents.
        """

        queryset = ConversationRecord.filter(deleted_at__isnull=False, deleted_at__lt=before)

        if tenant is not None:
            queryset = queryset.filter(tenant_id=tenant)

        candidates = await queryset.using_db(connection).order_by("tenant_id", "id").limit(limit)

        result = CleanupCascadeResult()

        for candidate in candidates:
            row = candidate
            if await self.__thread_has_primary_children(
                thread=row.id,
                tenant=row.tenant_id,
                connection=connection,
            ):
                continue

            script_rows = await (
                ScriptRecord.filter(tenant_id=row.tenant_id, conversation_id=row.id)
                .using_db(connection)
                .values_list("id", flat=True)
            )
            script_versions = (
                await ScriptVersionRecord.filter(tenant_id=row.tenant_id, script_id__in=script_rows)
                .using_db(connection)
                .delete()
            )
            scripts = (
                await ScriptRecord.filter(tenant_id=row.tenant_id, conversation_id=row.id)
                .using_db(connection)
                .delete()
            )
            memberships = (
                await MembershipRecord.filter(
                    tenant_id=row.tenant_id,
                    conversation_id=row.id,
                )
                .using_db(connection)
                .delete()
            )
            contexts = (
                await ContextRecord.filter(tenant_id=row.tenant_id, conversation_id=row.id)
                .using_db(connection)
                .delete()
            )
            jobs = (
                await JobRecord.filter(tenant_id=row.tenant_id, conversation_id=row.id)
                .using_db(connection)
                .delete()
            )
            events = (
                await EventRecord.filter(tenant_id=row.tenant_id, conversation_id=row.id)
                .using_db(connection)
                .delete()
            )
            executions = (
                await ExecutionRecord.filter(tenant_id=row.tenant_id, conversation_id=row.id)
                .using_db(connection)
                .delete()
            )
            sequences = (
                await SequenceRecord.filter(tenant_id=row.tenant_id, conversation_id=row.id)
                .using_db(connection)
                .delete()
            )
            threads = (
                await ConversationRecord.filter(tenant_id=row.tenant_id, id=row.id)
                .using_db(connection)
                .delete()
            )

            result = CleanupCascadeResult(
                jobs=result.jobs + jobs,
                events=result.events + events,
                scripts=result.scripts + scripts,
                threads=result.threads + threads,
                contexts=result.contexts + contexts,
                sequences=result.sequences + sequences,
                executions=result.executions + executions,
                memberships=result.memberships + memberships,
                script_versions=result.script_versions + script_versions,
            )

        return result

    async def __message_is_referenced(
        self,
        *,
        tenant: str,
        message: str,
        connection: "DatabaseConnection",
    ) -> bool:
        """
        Check whether a message is referenced by another row.
        """

        return await self.__exists(
            connection=connection,
            parameters=[tenant, message],
            sql="""
            SELECT EXISTS (
                SELECT 1
                FROM tasks
                WHERE tenant_id = $1
                  AND origin_id = $2
                UNION ALL
                SELECT 1
                FROM messages
                WHERE tenant_id = $1
                  AND reply_id = $2
                UNION ALL
                SELECT 1
                FROM contexts
                WHERE tenant_id = $1
                  AND ("references" -> 'messages') ? $2
            ) AS referenced
            """,
        )

    async def __artifact_is_referenced(
        self,
        *,
        tenant: str,
        artifact: str,
        connection: "DatabaseConnection",
    ) -> bool:
        """
        Check whether an artifact is referenced by scripts or versions.
        """

        return await self.__exists(
            connection=connection,
            parameters=[tenant, artifact],
            sql="""
            SELECT EXISTS (
                SELECT 1
                FROM contexts
                WHERE tenant_id = $1
                  AND ("references" -> 'artifacts') ? $2
            ) AS referenced
            """,
        )

    async def __event_is_referenced(
        self,
        *,
        event: str,
        tenant: str,
        connection: "DatabaseConnection",
    ) -> bool:
        """
        Check whether a context JSON reference protects an event row.
        """

        return await self.__exists(
            connection=connection,
            parameters=[tenant, event],
            sql="""
            SELECT EXISTS (
                SELECT 1
                FROM contexts
                WHERE tenant_id = $1
                  AND ("references" -> 'events') ? $2
            ) AS referenced
            """,
        )

    async def __exists(
        self,
        *,
        sql: str,
        parameters: Sequence[object],
        connection: "DatabaseConnection",
    ) -> bool:
        """
        Execute one SELECT EXISTS probe against Postgres and return its result.
        """

        rows = await connection.execute_query_dict(sql, list(parameters))

        if not isinstance(rows, list) or not rows:
            return False

        first = rows[0]
        if not isinstance(first, dict):
            return False

        return bool(first.get("referenced"))

    async def __task_is_referenced(
        self,
        *,
        task: str,
        tenant: str,
        connection: "DatabaseConnection",
    ) -> bool:
        """
        Check whether a task is referenced by any dependent row.
        """

        return await self.__exists(
            connection=connection,
            parameters=[tenant, task],
            sql="""
            SELECT EXISTS (
                SELECT 1 FROM jobs WHERE tenant_id = $1 AND task_id = $2
                UNION ALL
                SELECT 1 FROM events WHERE tenant_id = $1 AND task_id = $2
                UNION ALL
                SELECT 1 FROM tasks WHERE tenant_id = $1 AND parent_id = $2
                UNION ALL
                SELECT 1 FROM scripts WHERE tenant_id = $1 AND task_id = $2
                UNION ALL
                SELECT 1 FROM messages WHERE tenant_id = $1 AND task_id = $2
                UNION ALL
                SELECT 1 FROM contexts WHERE tenant_id = $1 AND task_id = $2
                UNION ALL
                SELECT 1 FROM artifacts WHERE tenant_id = $1 AND task_id = $2
            ) AS referenced
            """,
        )

    async def __thread_has_primary_children(
        self,
        *,
        tenant: str,
        thread: str,
        connection: "DatabaseConnection",
    ) -> bool:
        """
        Check whether a soft-deleted thread still has primary child rows.
        """

        return await self.__exists(
            connection=connection,
            parameters=[tenant, thread],
            sql="""
            SELECT EXISTS (
                SELECT 1 FROM tasks WHERE tenant_id = $1 AND conversation_id = $2
                UNION ALL
                SELECT 1 FROM messages WHERE tenant_id = $1 AND conversation_id = $2
                UNION ALL
                SELECT 1 FROM artifacts WHERE tenant_id = $1 AND conversation_id = $2
            ) AS referenced
            """,
        )
