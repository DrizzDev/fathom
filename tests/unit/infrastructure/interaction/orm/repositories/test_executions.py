from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Tuple
from uuid import uuid4

import pytest
from tests.unit.infrastructure.interaction.orm.repositories.factories import (
    InteractionRepositoryFactory,
)
from tests.unit.infrastructure.interaction.orm.support import InteractionPostgresSchema

from fathom.constants.collaboration import ActorKind, ExecutionState, TaskCode
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.orm.models import (
    ActorRecord,
    ConversationRecord,
    ExecutionRecord,
    TaskRecord,
)
from fathom.schemas.interaction import (
    ExecutionQuery,
    FinishExecution,
    Identity,
    Metadata,
    StartExecution,
    Terminal,
)


class TestExecutionRepository:
    """
    Verify execution persistence owns run lifecycle rows.
    """

    async def test_start_execution_persists_execution_without_task(self) -> None:
        """
        Starting an execution creates only the execution row.
        """

        async with InteractionPostgresSchema(prefix="conversation_execution_repository"):
            actor, thread = await self.__thread()
            request = self.__start(actor=actor, thread=thread)

            result = (
                await InteractionRepositoryFactory().executions().start_execution(request=request)
            )

            assert result.identity == request.identity
            assert result.thread == thread
            assert result.intent == request.intent
            assert result.state == ExecutionState.RUNNING
            assert await ExecutionRecord.filter(id=request.identity.id).count() == 1
            assert await TaskRecord.filter(execution_id=request.identity.id).count() == 0

    async def test_identical_start_replay_returns_existing_execution(self) -> None:
        """
        Replaying the same start request returns the existing execution.
        """

        async with InteractionPostgresSchema(prefix="conversation_execution_repository"):
            actor, thread = await self.__thread()
            request = self.__start(actor=actor, thread=thread)
            repository = InteractionRepositoryFactory().executions()

            created = await repository.start_execution(request=request)
            replayed = await repository.start_execution(request=request)

            assert replayed == created
            assert await ExecutionRecord.filter(id=request.identity.id).count() == 1

    async def test_conflicting_start_replay_raises_interaction_error(self) -> None:
        """
        Replaying an execution identity with different content fails.
        """

        async with InteractionPostgresSchema(prefix="conversation_execution_repository"):
            actor, thread = await self.__thread()
            request = self.__start(actor=actor, thread=thread)
            repository = InteractionRepositoryFactory().executions()
            await repository.start_execution(request=request)

            with pytest.raises(InteractionError, match="different content"):
                await repository.start_execution(
                    request=request.model_copy(update={"intent": "Changed intent"})
                )

    async def test_finish_execution_updates_only_execution_row(self) -> None:
        """
        Finishing an execution updates the execution row without touching tasks.
        """

        async with InteractionPostgresSchema(prefix="conversation_execution_repository"):
            actor, thread = await self.__thread()
            start = self.__start(actor=actor, thread=thread)
            repository = InteractionRepositoryFactory().executions()
            await repository.start_execution(request=start)
            finish = FinishExecution(
                tenant=start.identity.tenant,
                execution=start.identity.id,
                actor=actor,
                state=ExecutionState.SUCCEEDED,
                terminal=Terminal(code=TaskCode.COMPLETED, detail="done"),
                summary="done",
                outcome=Metadata(entries={"steps": 3}),
                completed_at=start.started + timedelta(seconds=3),
            )

            result = await repository.finish_execution(request=finish)
            replayed = await repository.finish_execution(request=finish)

            assert replayed == result
            assert result.state == ExecutionState.SUCCEEDED
            assert result.terminal == finish.terminal
            assert result.outcome == finish.outcome
            assert await TaskRecord.filter(execution_id=start.identity.id).count() == 0

    async def test_get_execution_hides_deleted_rows(self) -> None:
        """
        Loading one execution excludes soft-deleted rows.
        """

        async with InteractionPostgresSchema(prefix="conversation_execution_repository"):
            actor, thread = await self.__thread()
            request = self.__start(actor=actor, thread=thread)
            repository = InteractionRepositoryFactory().executions()
            await repository.start_execution(request=request)
            await ExecutionRecord.filter(id=request.identity.id).update(deleted_at=request.started)

            result = await repository.get_execution(
                query=ExecutionQuery(
                    tenant=request.identity.tenant,
                    thread=thread,
                    execution=request.identity.id,
                )
            )

            assert result is None

    async def test_start_execution_requires_active_thread(self) -> None:
        """
        Starting an execution fails when the conversation does not exist.
        """

        async with InteractionPostgresSchema(prefix="conversation_execution_repository"):
            actor = await self.__actor()
            request = self.__start(actor=actor, thread=str(uuid4()))

            with pytest.raises(InteractionError, match="Thread does not exist"):
                await InteractionRepositoryFactory().executions().start_execution(request=request)

    async def __thread(self) -> Tuple[str, str]:
        """
        Persist one actor and active conversation.
        """

        actor = await self.__actor()
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
        return actor, thread

    async def __actor(self) -> str:
        """
        Persist one actor and return its id.
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

    def __start(self, *, actor: str, thread: str) -> StartExecution:
        """
        Build one execution start request.
        """

        return StartExecution(
            identity=Identity(id=str(uuid4()), tenant="tenant-a", workspace="workspace-a"),
            thread=thread,
            intent="Do work",
            actor=actor,
            started_at=datetime.now(tz=timezone.utc),
            metadata=Metadata(entries={"source": "test"}),
        )
