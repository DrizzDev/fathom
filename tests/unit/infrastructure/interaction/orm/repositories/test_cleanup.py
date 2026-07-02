from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

from tests.unit.infrastructure.interaction.orm.support import (
    InteractionPostgresSchema,
    InteractionRuntimeRegistry,
)

from fathom.constants.collaboration import (
    ActorKind,
    ArtifactBackend,
    ArtifactKind,
    ContextPurpose,
    EventKind,
    EventSource,
    IdempotencyState,
    JobCode,
    JobKind,
    JobState,
    MembershipRole,
    MembershipScope,
    TaskKind,
    TaskState,
)
from fathom.infrastructure.interaction.orm.models import (
    ActorRecord,
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
from fathom.infrastructure.interaction.orm.repositories import CleanupRepository
from fathom.schemas.interaction import CleanupRequest, References


class TestCleanupRepository:
    """
    Verify retention and physical purge behavior through the persistent-store backed repository.
    """

    async def test_cleanup_deletes_expired_requests_terminal_jobs_and_old_events(self) -> None:
        """
        Delete retention-scoped rows older than their thresholds.
        """

        async with InteractionPostgresSchema(prefix="conversation_cleanup_repository"):
            thread = await self.__thread()
            old = self.__now() - timedelta(days=10)
            new = self.__now()
            await self.__request(key="old", expires=old)
            await self.__request(key="new", expires=new)
            await self.__job(thread=thread, state=JobState.COMPLETED, updated=old)
            await self.__job(thread=thread, state=JobState.PENDING, updated=old)
            await self.__event(thread=thread, sequence=1, created=old)
            await self.__event(thread=thread, sequence=2, created=new)

            result = await CleanupRepository(
                transaction=InteractionRuntimeRegistry.require()
            ).cleanup(
                request=CleanupRequest(
                    tenant="tenant-a",
                    idempotency_before=self.__now() - timedelta(days=1),
                    terminal_jobs_before=self.__now() - timedelta(days=1),
                    events_before=self.__now() - timedelta(days=1),
                    limit=100,
                )
            )

            assert result.idempotency_deleted == 1
            assert result.jobs_deleted == 1
            assert result.events_deleted == 1
            assert await RequestRecord.filter(tenant_id="tenant-a").count() == 1
            assert await JobRecord.filter(tenant_id="tenant-a").count() == 1
            assert await EventRecord.filter(tenant_id="tenant-a").count() == 1

    async def test_soft_deleted_purge_removes_unreferenced_children_only(self) -> None:
        """
        Purge soft-deleted child rows only after dependency checks pass.
        """

        async with InteractionPostgresSchema(prefix="conversation_cleanup_repository"):
            thread = await self.__thread()
            old = self.__now() - timedelta(days=10)
            message = await self.__message(thread=thread, deleted=old)
            protected_message = await self.__message(thread=thread, deleted=old)
            await self.__message(thread=thread, reply=protected_message)
            artifact = await self.__artifact(thread=thread, deleted=old)
            protected_artifact = await self.__artifact(thread=thread, deleted=old)
            await self.__context(
                thread=thread,
                references=References(artifacts=(protected_artifact,)),
            )
            task = await self.__task(thread=thread, deleted=old)
            protected_task = await self.__task(thread=thread, deleted=old)
            await self.__job(thread=thread, task=protected_task)

            result = await CleanupRepository(
                transaction=InteractionRuntimeRegistry.require()
            ).cleanup(
                request=CleanupRequest(
                    tenant="tenant-a",
                    soft_deleted_before=self.__now() - timedelta(days=1),
                    limit=100,
                )
            )

            assert result.messages_purged == 1
            assert result.artifacts_purged == 1
            assert result.tasks_purged == 1
            assert await MessageRecord.get_or_none(id=message) is None
            assert await MessageRecord.get_or_none(id=protected_message) is not None
            assert await ArtifactRecord.get_or_none(id=artifact) is None
            assert await ArtifactRecord.get_or_none(id=protected_artifact) is not None
            assert await TaskRecord.get_or_none(id=task) is None
            assert await TaskRecord.get_or_none(id=protected_task) is not None

    async def test_cleanup_preserves_rows_referenced_by_contexts(self) -> None:
        """
        Preserve cleanup candidates that are referenced from context JSON.
        """

        async with InteractionPostgresSchema(prefix="conversation_cleanup_repository"):
            thread = await self.__thread()
            old = self.__now() - timedelta(days=10)
            message = await self.__message(thread=thread, deleted=old)
            artifact = await self.__artifact(thread=thread, deleted=old)
            event = await self.__event(thread=thread, sequence=1, created=old)
            await self.__context(
                thread=thread,
                references=References(
                    messages=(message,),
                    events=(event,),
                    artifacts=(artifact,),
                ),
            )

            result = await CleanupRepository(
                transaction=InteractionRuntimeRegistry.require()
            ).cleanup(
                request=CleanupRequest(
                    tenant="tenant-a",
                    soft_deleted_before=self.__now() - timedelta(days=1),
                    events_before=self.__now() - timedelta(days=1),
                    limit=100,
                )
            )

            assert result.messages_purged == 0
            assert result.artifacts_purged == 0
            assert result.events_deleted == 0
            assert await MessageRecord.get_or_none(id=message) is not None
            assert await ArtifactRecord.get_or_none(id=artifact) is not None
            assert await EventRecord.get_or_none(id=event) is not None

    async def test_soft_deleted_thread_purge_cascades_thread_bound_dependents(self) -> None:
        """
        Purge a soft-deleted thread and dependent rows after primary children are gone.
        """

        async with InteractionPostgresSchema(prefix="conversation_cleanup_repository"):
            thread = await self.__thread(deleted=self.__now() - timedelta(days=10))
            actor = await self.__actor()
            await MembershipRecord.create(
                id=str(uuid4()),
                tenant_id="tenant-a",
                workspace_id=None,
                conversation_id=thread,
                actor=actor,
                role=MembershipRole.OWNER.value,
                scope=MembershipScope.THREAD.value,
                joined_at=self.__now(),
                departed_at=None,
                metadata={},
                created_at=self.__now(),
                updated_at=self.__now(),
            )
            execution = await self.__execution(thread=thread)
            await self.__context(thread=thread, execution=execution)
            script = await self.__script(thread=thread, execution=execution)
            await self.__script_version(script=script)
            await self.__job(thread=thread, execution=execution)
            await self.__event(thread=thread, sequence=1, execution=execution)
            await SequenceRecord.create(
                id=str(uuid4()),
                tenant_id="tenant-a",
                conversation_id=thread,
                scope="event",
                value=1,
            )

            result = await CleanupRepository(
                transaction=InteractionRuntimeRegistry.require()
            ).cleanup(
                request=CleanupRequest(
                    tenant="tenant-a",
                    soft_deleted_before=self.__now() - timedelta(days=1),
                    limit=100,
                )
            )

            assert result.threads_purged == 1
            assert result.memberships_purged == 1
            assert result.contexts_purged == 1
            assert result.scripts_purged == 1
            assert result.script_versions_purged == 1
            assert result.executions_purged == 1
            assert result.jobs_cascade_purged == 1
            assert result.events_cascade_purged == 1
            assert result.sequences_purged == 1
            assert await ConversationRecord.get_or_none(id=thread) is None

    async def test_limit_bounds_each_cleanup_scope(self) -> None:
        """
        Limit the number of victims deleted per cleanup scope.
        """

        async with InteractionPostgresSchema(prefix="conversation_cleanup_repository"):
            old = self.__now() - timedelta(days=10)
            await self.__request(key="first", expires=old)
            await self.__request(key="second", expires=old)

            result = await CleanupRepository(
                transaction=InteractionRuntimeRegistry.require()
            ).cleanup(
                request=CleanupRequest(
                    tenant="tenant-a",
                    idempotency_before=self.__now() - timedelta(days=1),
                    limit=1,
                )
            )

            assert result.idempotency_deleted == 1
            assert await RequestRecord.filter(tenant_id="tenant-a").count() == 1

    async def __actor(self) -> str:
        """
        Insert one actor row.
        """

        actor = str(uuid4())
        await ActorRecord.create(
            id=actor,
            tenant_id="tenant-a",
            workspace_id=None,
            kind=ActorKind.HUMAN.value,
            name="Operator",
            external=None,
            runtime=None,
            provider=None,
            model=None,
            skills={},
            metadata={},
            created_at=self.__now(),
            updated_at=self.__now(),
        )
        return actor

    async def __thread(self, *, deleted: Optional[datetime] = None) -> str:
        """
        Insert one thread row.
        """

        thread = str(uuid4())
        await ConversationRecord.create(
            id=thread,
            tenant_id="tenant-a",
            workspace_id=None,
            title="Thread",
            digest=None,
            archived_at=None,
            created_by=None,
            created_at=self.__now(),
            updated_at=deleted or self.__now(),
            deleted_at=deleted,
            metadata={},
        )
        return thread

    async def __request(self, *, key: str, expires: datetime) -> None:
        """
        Insert one idempotency request row.
        """

        await RequestRecord.create(
            id=str(uuid4()),
            tenant_id="tenant-a",
            workspace_id=None,
            key=key,
            hash=f"hash-{key}",
            state=IdempotencyState.STARTED.value,
            response=None,
            expires_at=expires,
            created_at=self.__now(),
            metadata={},
        )

    async def __job(
        self,
        *,
        thread: str,
        state: JobState = JobState.PENDING,
        updated: Optional[datetime] = None,
        task: Optional[str] = None,
        execution: Optional[str] = None,
    ) -> None:
        """
        Insert one job row.
        """

        terminal = state in (JobState.COMPLETED, JobState.FAILED, JobState.ABANDONED)
        job = str(uuid4())
        resolved_execution = execution or task or await self.__execution(thread=thread)
        await JobRecord.create(
            id=job,
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=thread,
            execution_id=resolved_execution,
            task_id=task,
            kind=JobKind.EXECUTION.value,
            state=state.value,
            attempts=0,
            owner=None,
            locked_at=None,
            available_at=self.__now(),
            payload={},
            code=JobCode.COMPLETED.value if terminal else None,
            detail=None,
            created_at=self.__now(),
            updated_at=updated or self.__now(),
            metadata={},
        )
        if updated is not None:
            await JobRecord.filter(id=job).update(updated_at=updated)

    async def __event(
        self,
        *,
        thread: str,
        sequence: int,
        execution: Optional[str] = None,
        created: Optional[datetime] = None,
    ) -> str:
        """
        Insert one event row.
        """

        event = str(uuid4())
        await EventRecord.create(
            id=event,
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=thread,
            execution_id=execution,
            task_id=None,
            actor=None,
            sequence=sequence,
            kind=EventKind.THREAD_CREATED.value,
            source=EventSource.INTERACTION.value,
            payload={},
            metadata={},
            created_at=created or self.__now(),
        )
        return event

    async def __message(
        self,
        *,
        thread: str,
        deleted: Optional[datetime] = None,
        reply: Optional[str] = None,
    ) -> str:
        """
        Insert one message row.
        """

        message = str(uuid4())
        actor = await self.__actor()
        execution = await self.__execution(thread=thread)
        sequence = (
            await MessageRecord.filter(tenant_id="tenant-a", conversation_id=thread).count()
        ) + 1
        await MessageRecord.create(
            id=message,
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=thread,
            execution_id=execution,
            task_id=None,
            author=actor,
            reply_id=reply,
            sequence=sequence,
            kind="request",
            audience=["thread"],
            body={"text": "hello"},
            labels=[],
            metadata={},
            created_at=self.__now(),
            deleted_at=deleted,
        )
        return message

    async def __artifact(self, *, thread: str, deleted: Optional[datetime] = None) -> str:
        """
        Insert one artifact row.
        """

        artifact = str(uuid4())
        execution = await self.__execution(thread=thread)
        await ArtifactRecord.create(
            id=artifact,
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=thread,
            execution_id=execution,
            task_id=None,
            producer=None,
            kind=ArtifactKind.SCREENSHOT.value,
            uri="memory://artifact",
            backend=ArtifactBackend.LOCAL.value,
            mime="image/png",
            size=10,
            retention=None,
            labels=[],
            metadata={},
            created_at=self.__now(),
            deleted_at=deleted,
        )
        return artifact

    async def __task(
        self,
        *,
        thread: str,
        deleted: Optional[datetime],
    ) -> str:
        """
        Insert one task row.
        """

        task = str(uuid4())
        await ExecutionRecord.create(
            id=task,
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=thread,
            intent="Do work",
            state=TaskState.RUNNING.value,
            outcome={},
            started_at=self.__now(),
            created_at=self.__now(),
            created_by=None,
            updated_at=deleted or self.__now(),
            updated_by=None,
            deleted_at=deleted,
            metadata={},
        )
        await TaskRecord.create(
            id=task,
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=thread,
            execution_id=task,
            created_by=None,
            assignee=None,
            origin_id=None,
            kind=TaskKind.AGENT.value,
            objective="Do work",
            reference=None,
            state=TaskState.DELETED.value if deleted is not None else TaskState.RUNNING.value,
            code=None,
            detail=None,
            progress={},
            plan={},
            outcome={},
            summary=None,
            started_at=self.__now(),
            completed_at=deleted,
            elapsed=None,
            created_at=self.__now(),
            updated_at=deleted or self.__now(),
            deleted_at=deleted,
            metadata={},
        )
        return task

    async def __execution(self, *, thread: str) -> str:
        """
        Insert one execution row.
        """

        execution = str(uuid4())
        await ExecutionRecord.create(
            id=execution,
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=thread,
            intent="Script",
            state=TaskState.RUNNING.value,
            outcome={},
            started_at=self.__now(),
            created_at=self.__now(),
            created_by=None,
            updated_at=self.__now(),
            updated_by=None,
            metadata={},
        )
        return execution

    async def __script(self, *, thread: str, execution: Optional[str] = None) -> str:
        """
        Insert one script row.
        """

        script = str(uuid4())
        script_execution = execution or await self.__execution(thread=thread)
        await ScriptRecord.create(
            id=script,
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=thread,
            execution_id=script_execution,
            task_id=None,
            title=None,
            format="text/plain",
            status="active",
            content="script",
            revision=1,
            checksum="checksum",
            created_by=None,
            updated_by=None,
            created_at=self.__now(),
            updated_at=self.__now(),
            metadata={},
        )
        return script

    async def __script_version(self, *, script: str) -> None:
        """
        Insert one script version row.
        """

        await ScriptVersionRecord.create(
            id=str(uuid4()),
            tenant_id="tenant-a",
            workspace_id=None,
            script_id=script,
            version=1,
            source="generated",
            content="script",
            checksum="checksum",
            summary=None,
            actor=None,
            created_at=self.__now(),
            metadata={},
        )

    async def __context(
        self,
        *,
        thread: str,
        references: Optional[References] = None,
        execution: Optional[str] = None,
    ) -> None:
        """
        Insert one context row.
        """

        context_references = references or References()
        resolved_execution = execution or await self.__execution(thread=thread)
        await ContextRecord.create(
            id=str(uuid4()),
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=thread,
            execution_id=resolved_execution,
            task_id=None,
            consumer=None,
            purpose=ContextPurpose.EXECUTION.value,
            builder="builder@1",
            references=context_references.model_dump(mode="json"),
            budget={},
            filters={},
            hash=None,
            provider=None,
            model=None,
            created_at=self.__now(),
            updated_at=self.__now(),
            expires_at=None,
            metadata={},
        )

    def __now(self) -> datetime:
        """
        Return a stable timezone-aware timestamp for tests.
        """

        return datetime(2026, 1, 1, tzinfo=timezone.utc)
