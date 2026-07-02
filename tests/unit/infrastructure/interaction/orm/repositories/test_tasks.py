from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from uuid import uuid4

import pytest
from tests.unit.infrastructure.interaction.orm.repositories.factories import (
    InteractionRepositoryFactory,
)
from tests.unit.infrastructure.interaction.orm.support import InteractionPostgresSchema
from tortoise.exceptions import IntegrityError

from fathom.constants.collaboration import (
    ActorKind,
    Audience,
    EventKind,
    ExecutionState,
    MembershipRole,
    MembershipScope,
    MessageKind,
    TaskCode,
    TaskKind,
    TaskState,
)
from fathom.core.exceptions import InteractionError, TaskConflictError
from fathom.infrastructure.interaction.orm.models import (
    ActorRecord,
    ConversationRecord,
    EventRecord,
    ExecutionRecord,
    MembershipRecord,
    MessageRecord,
    TaskRecord,
)
from fathom.schemas.interaction import (
    Assignment,
    FinishTask,
    Identity,
    Lineage,
    Metadata,
    OpenTask,
    Plan,
    TaskOneQuery,
    TaskQuery,
    Terminal,
)


class TestTaskRepository:
    """
    Verify task persistence through the persistent-store backed repository.
    """

    async def test_open_task_persists_task_and_records_event(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_task_repository"):
            actor, thread = await self.__conversation()
            request = await self.__open_request(actor=actor, thread=thread, state=TaskState.RUNNING)

            result = await InteractionRepositoryFactory().tasks().open_task(request=request)

            assert result.identity == request.identity
            assert result.thread == thread
            assert result.assignment == request.assignment
            assert result.execution == request.execution
            assert result.lineage.root == request.identity.id
            assert result.kind == TaskKind.AGENT
            assert result.state == TaskState.RUNNING
            assert result.timing.started == request.created
            assert result.plan.progress == Metadata(entries={"step": 1})
            event = await EventRecord.get(conversation_id=thread, sequence=1)
            assert event.kind == EventKind.TASK_OPENED.value
            assert event.task_id == request.identity.id
            assert event.execution_id == request.execution
            started = await EventRecord.get(conversation_id=thread, sequence=2)
            assert started.kind == EventKind.TASK_STARTED.value
            assert started.task_id == request.identity.id
            assert started.created_by == actor
            stored = await ConversationRecord.get(id=thread)
            assert stored.digest is not None

    async def test_identical_open_replay_returns_existing_without_new_event(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_task_repository"):
            actor, thread = await self.__conversation()
            request = await self.__open_request(actor=actor, thread=thread)
            repository = InteractionRepositoryFactory().tasks()

            created = await repository.open_task(request=request)
            replayed = await repository.open_task(request=request)

            assert replayed == created
            assert await EventRecord.filter(conversation_id=thread).count() == 1
            assert await TaskRecord.filter(conversation_id=thread).count() == 1

    async def test_conflicting_open_replay_raises_interaction_error(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_task_repository"):
            actor, thread = await self.__conversation()
            request = await self.__open_request(actor=actor, thread=thread)
            repository = InteractionRepositoryFactory().tasks()
            await repository.open_task(request=request)
            conflict = request.model_copy(update={"plan": Plan(objective="Different objective")})

            with pytest.raises(InteractionError, match="different content"):
                await repository.open_task(request=conflict)

    async def test_open_task_requires_active_actor_membership(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_task_repository"):
            actor = await self.__actor()
            thread = await self.__thread(actor=actor)

            with pytest.raises(InteractionError, match="active member"):
                await (
                    InteractionRepositoryFactory()
                    .tasks()
                    .open_task(request=await self.__open_request(actor=actor, thread=thread))
                )

    async def test_open_task_validates_lineage_and_origin(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_task_repository"):
            actor, thread = await self.__conversation()
            repository = InteractionRepositoryFactory().tasks()
            parent = await repository.open_task(
                request=await self.__open_request(actor=actor, thread=thread)
            )
            origin = await self.__message(actor=actor, thread=thread)

            child = await self.__open_request(
                actor=actor,
                thread=thread,
                execution=parent.execution,
                lineage=Lineage(parent=parent.identity.id, root=parent.identity.id, origin=origin),
            )
            result = await repository.open_task(request=child)
            assert result.lineage.parent == parent.identity.id
            assert result.lineage.root == parent.identity.id
            assert result.lineage.origin == origin
            delegated = await EventRecord.get(
                conversation_id=thread,
                kind=EventKind.TASK_DELEGATED.value,
            )
            assert delegated.task_id == result.identity.id
            assert delegated.actor == actor
            assert delegated.payload == {"parent": parent.identity.id}

            invalid = await self.__open_request(
                actor=actor,
                thread=thread,
                execution=parent.execution,
                lineage=Lineage(root=parent.identity.id),
            )
            with pytest.raises(InteractionError, match="Root task must reference itself"):
                await repository.open_task(request=invalid)

    async def test_finish_task_updates_terminal_state_and_records_event(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_task_repository"):
            actor, thread = await self.__conversation()
            repository = InteractionRepositoryFactory().tasks()
            opened = await repository.open_task(
                request=await self.__open_request(
                    actor=actor, thread=thread, state=TaskState.RUNNING
                )
            )
            finish = self.__finish_request(task=opened.identity.id, state=TaskState.SUCCEEDED)

            result = await repository.finish_task(request=finish)
            replayed = await repository.finish_task(request=finish)

            assert replayed == result
            assert result.state == TaskState.SUCCEEDED
            assert result.terminal == finish.terminal
            assert result.summary == "done"
            assert result.timing.ended == finish.ended
            assert result.timing.elapsed == finish.elapsed
            assert result.plan.progress == Metadata(
                entries={
                    "code": finish.terminal.code.value,
                    "detail": finish.terminal.detail,
                    "elapsed": finish.elapsed,
                    "state": finish.state.value,
                    "summary": finish.summary,
                }
            )
            stored = await TaskRecord.get(id=opened.identity.id)
            assert stored.updated_by == actor
            assert stored.outcome == {
                "code": finish.terminal.code.value,
                "detail": finish.terminal.detail,
                "summary": finish.summary,
                "state": finish.state.value,
            }
            assert stored.progress == result.plan.progress.entries
            event = await EventRecord.get(
                conversation_id=thread,
                kind=EventKind.TASK_SUCCEEDED.value,
            )
            assert event.kind == EventKind.TASK_SUCCEEDED.value

    async def test_conflicting_finish_replay_raises_task_conflict(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_task_repository"):
            actor, thread = await self.__conversation()
            repository = InteractionRepositoryFactory().tasks()
            opened = await repository.open_task(
                request=await self.__open_request(
                    actor=actor, thread=thread, state=TaskState.RUNNING
                )
            )
            finish = self.__finish_request(task=opened.identity.id, state=TaskState.FAILED)
            await repository.finish_task(request=finish)
            conflict = finish.model_copy(update={"summary": "different"})

            with pytest.raises(TaskConflictError):
                await repository.finish_task(request=conflict)

    async def test_get_tasks_orders_and_hides_deleted_rows(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_task_repository"):
            actor, thread = await self.__conversation()
            repository = InteractionRepositoryFactory().tasks()
            base = datetime(2026, 1, 1, tzinfo=timezone.utc)
            first = await repository.open_task(
                request=await self.__open_request(actor=actor, thread=thread, created=base)
            )
            second = await repository.open_task(
                request=await self.__open_request(
                    actor=actor,
                    thread=thread,
                    created=base + timedelta(seconds=1),
                )
            )
            await TaskRecord.filter(id=second.identity.id).update(deleted_at=base)

            tasks = await repository.get_tasks(query=TaskQuery(tenant="tenant-a", thread=thread))
            hidden = await repository.get_task(
                query=TaskOneQuery(tenant="tenant-a", thread=thread, task=second.identity.id)
            )

            assert tuple(task.identity.id for task in tasks) == (first.identity.id,)
            assert hidden is None

    async def test_corrupt_task_row_raises_interaction_error(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_task_repository"):
            actor, thread = await self.__conversation()
            request = await self.__open_request(actor=actor, thread=thread)
            with pytest.raises(IntegrityError):
                await ExecutionRecord.create(
                    id=request.identity.id,
                    tenant_id=request.identity.tenant,
                    workspace_id=request.identity.workspace,
                    conversation_id=thread,
                    intent=request.plan.objective,
                    state=TaskState.QUEUED.value,
                    outcome={},
                    created_at=request.created,
                    created_by=actor,
                    updated_at=request.created,
                    updated_by=actor,
                    metadata={},
                )
                await TaskRecord.create(
                    id=request.identity.id,
                    tenant_id=request.identity.tenant,
                    workspace_id=request.identity.workspace,
                    conversation_id=thread,
                    execution_id=request.identity.id,
                    created_by=actor,
                    assignee=actor,
                    parent_id=None,
                    origin_id=None,
                    kind="alien",
                    objective=request.plan.objective,
                    reference=None,
                    state=TaskState.QUEUED.value,
                    code=None,
                    detail=None,
                    progress=request.plan.progress.entries,
                    plan=request.plan.plan.entries,
                    outcome={},
                    summary=None,
                    started_at=None,
                    elapsed=None,
                    created_at=request.created,
                    updated_at=request.created,
                    deleted_at=None,
                    metadata=request.metadata.entries,
                )

    async def __conversation(self) -> Tuple[str, str]:
        """
        Persist one actor, active thread, and membership.
        """

        actor = await self.__actor()
        thread = await self.__thread(actor=actor)
        await self.__membership(actor=actor, thread=thread)
        return actor, thread

    async def __actor(self) -> str:
        """
        Persist one actor row and return its id.
        """

        actor = str(uuid4())
        now = datetime.now(tz=timezone.utc)
        await ActorRecord.create(
            id=actor,
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            kind=ActorKind.HUMAN.value,
            name="operator",
            external=None,
            runtime=None,
            provider=None,
            model=None,
            skills={},
            metadata={},
            created_at=now,
            updated_at=now,
        )
        return actor

    async def __thread(self, *, actor: str) -> str:
        """
        Persist one active thread row and return its id.
        """

        thread = str(uuid4())
        now = datetime.now(tz=timezone.utc)
        await ConversationRecord.create(
            id=thread,
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            title="Plan",
            digest=None,
            created_by=actor,
            archived_at=None,
            metadata={},
            created_at=now,
            updated_at=now,
        )
        return thread

    async def __membership(self, *, actor: str, thread: str) -> None:
        """
        Persist one active membership row.
        """

        await MembershipRecord.create(
            id=str(uuid4()),
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            conversation_id=thread,
            actor=actor,
            role=MembershipRole.OWNER.value,
            scope=MembershipScope.THREAD.value,
            joined_at=datetime.now(tz=timezone.utc),
            departed_at=None,
            metadata={},
        )

    async def __message(self, *, actor: str, thread: str) -> str:
        """
        Persist one origin message row and return its id.
        """

        message = str(uuid4())
        execution = str(uuid4())
        now = datetime.now(tz=timezone.utc)
        await ExecutionRecord.create(
            id=execution,
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            conversation_id=thread,
            intent="Do work",
            state=TaskState.RUNNING.value,
            outcome={},
            started_at=now,
            created_at=now,
            updated_at=now,
            metadata={},
        )
        await MessageRecord.create(
            id=message,
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            conversation_id=thread,
            execution_id=execution,
            task_id=None,
            author=actor,
            reply_id=None,
            sequence=100,
            kind=MessageKind.REQUEST.value,
            audience=[Audience.THREAD.value],
            body={"text": "start"},
            labels=[],
            sanitized_at=None,
            sanitizer=None,
            metadata={},
            created_at=now,
            deleted_at=None,
        )
        return message

    async def __open_request(
        self,
        *,
        actor: str,
        thread: str,
        execution: Optional[str] = None,
        state: TaskState = TaskState.QUEUED,
        lineage: Optional[Lineage] = None,
        created: Optional[datetime] = None,
    ) -> OpenTask:
        """
        Build one task open request.
        """

        request = OpenTask(
            identity=Identity(id=str(uuid4()), tenant="tenant-a", workspace="workspace-a"),
            thread=thread,
            execution=execution or str(uuid4()),
            assignment=Assignment(creator=actor, assignee=actor),
            lineage=lineage or Lineage(),
            kind=TaskKind.AGENT,
            state=state,
            plan=Plan(
                objective="Do work",
                reference="ref",
                progress=Metadata(entries={"step": 1}),
                plan=Metadata(entries={"todo": ["open"]}),
            ),
            created_at=created or datetime.now(tz=timezone.utc),
            metadata=Metadata(entries={"source": "test"}),
        )
        await self.__execution(actor=actor, thread=thread, request=request)

        return request

    async def __execution(self, *, actor: str, thread: str, request: OpenTask) -> None:
        """
        Persist the execution row required before opening a task.
        """

        if await ExecutionRecord.filter(id=request.execution).exists():
            return

        await ExecutionRecord.create(
            id=request.execution,
            tenant_id=request.identity.tenant,
            workspace_id=request.identity.workspace,
            conversation_id=thread,
            intent=request.plan.objective,
            state=ExecutionState.RUNNING.value,
            created_at=request.created,
            created_by=actor,
            updated_at=request.created,
            updated_by=actor,
            started_at=request.created,
            metadata={},
        )

    def __finish_request(self, *, task: str, state: TaskState) -> FinishTask:
        """
        Build one task finish request.
        """

        return FinishTask(
            tenant="tenant-a",
            task=task,
            state=state,
            terminal=Terminal(code=TaskCode.COMPLETED, detail="ok"),
            summary="done",
            ended_at=datetime.now(tz=timezone.utc) + timedelta(seconds=3),
            elapsed=3000,
        )
