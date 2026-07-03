from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple
from uuid import uuid4

import pytest
from tests.unit.infrastructure.interaction.orm.repositories.factories import (
    InteractionRepositoryFactory,
)
from tests.unit.infrastructure.interaction.orm.support import InteractionPostgresSchema
from tortoise.exceptions import IntegrityError

from fathom.constants.collaboration import (
    ActorKind,
    ArtifactBackend,
    ArtifactKind,
    EventKind,
    Label,
    MembershipRole,
    MembershipScope,
    TaskKind,
    TaskState,
)
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.orm.models import (
    ActorRecord,
    ArtifactRecord,
    ConversationRecord,
    EventRecord,
    ExecutionRecord,
    MembershipRecord,
    TaskRecord,
)
from fathom.schemas.interaction import (
    ArtifactCursorQuery,
    ArtifactQuery,
    Identity,
    LinkArtifact,
    Metadata,
    SortOrder,
)


class TestArtifactRepository:
    """
    Verify artifact persistence through the persistent-store backed repository.
    """

    async def test_link_artifact_persists_artifact_and_records_event(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_artifact_repository"):
            actor, thread = await self.__conversation()
            task = await self.__task(thread=thread, actor=actor)
            request = self.__request(actor=actor, thread=thread, task=task)

            result = await InteractionRepositoryFactory().artifacts().link_artifact(request=request)

            assert result.identity == request.identity
            assert result.thread == thread
            assert result.task == task
            assert result.producer == actor
            assert result.kind == ArtifactKind.SCREENSHOT
            assert result.backend == ArtifactBackend.LOCAL
            assert result.labels == (Label.DISPLAY_AUDIT, Label.MEMORY_SKIP)
            event = await EventRecord.get(conversation_id=thread, sequence=1)
            artifact = await ArtifactRecord.get(id=result.identity.id)
            assert artifact.created_by == actor
            assert artifact.updated_by == actor
            assert event.kind == EventKind.ARTIFACT_LINKED.value
            assert event.source == "artifact"
            assert event.actor == actor
            assert event.created_by == actor
            assert event.execution_id == artifact.execution_id

    async def test_identical_replay_returns_existing_without_new_event(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_artifact_repository"):
            actor, thread = await self.__conversation()
            request = self.__request(actor=actor, thread=thread)
            repository = InteractionRepositoryFactory().artifacts()

            created = await repository.link_artifact(request=request)
            replay = request.model_copy(update={"labels": tuple(reversed(request.labels))})
            replayed = await repository.link_artifact(request=replay)

            assert replayed == created
            assert await ArtifactRecord.filter(conversation_id=thread).count() == 1
            assert await EventRecord.filter(conversation_id=thread).count() == 1

    async def test_conflicting_replay_raises_interaction_error(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_artifact_repository"):
            actor, thread = await self.__conversation()
            request = self.__request(actor=actor, thread=thread)
            repository = InteractionRepositoryFactory().artifacts()
            await repository.link_artifact(request=request)
            conflict = request.model_copy(update={"uri": "local://changed"})

            with pytest.raises(InteractionError, match="different content"):
                await repository.link_artifact(request=conflict)

    async def test_identical_replay_returns_existing_after_parent_archived(self) -> None:
        """
        Replay an existing artifact before validating the archived parent thread.
        """

        async with InteractionPostgresSchema(prefix="conversation_artifact_repository"):
            actor, thread = await self.__conversation()
            request = self.__request(actor=actor, thread=thread)
            repository = InteractionRepositoryFactory().artifacts()
            created = await repository.link_artifact(request=request)
            await ConversationRecord.filter(id=thread).update(
                archived_at=datetime.now(tz=timezone.utc),
            )

            replayed = await repository.link_artifact(request=request)

            assert replayed == created
            assert await ArtifactRecord.filter(conversation_id=thread).count() == 1
            assert await EventRecord.filter(conversation_id=thread).count() == 1

    async def test_link_artifact_validates_parent_rows(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_artifact_repository"):
            actor, thread = await self.__conversation()
            missing_task = str(uuid4())

            with pytest.raises(InteractionError, match="Artifact task does not exist"):
                await (
                    InteractionRepositoryFactory()
                    .artifacts()
                    .link_artifact(
                        request=self.__request(actor=actor, thread=thread, task=missing_task)
                    )
                )

    async def test_link_artifact_requires_active_producer_membership(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_artifact_repository"):
            actor, thread = await self.__conversation(joined=False)

            with pytest.raises(InteractionError, match="active member"):
                await (
                    InteractionRepositoryFactory()
                    .artifacts()
                    .link_artifact(request=self.__request(actor=actor, thread=thread))
                )

    async def test_link_artifact_requires_execution_without_task(self) -> None:
        """
        Reject run-owned artifacts that have neither task nor execution.
        """

        async with InteractionPostgresSchema(prefix="conversation_artifact_repository"):
            actor, thread = await self.__conversation()
            request = self.__request(actor=actor, thread=thread).model_copy(
                update={"execution": None}
            )

            with pytest.raises(InteractionError, match="Artifact execution is required"):
                await InteractionRepositoryFactory().artifacts().link_artifact(request=request)

    async def test_get_artifacts_filters_task_orders_and_hides_deleted_rows(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_artifact_repository"):
            actor, thread = await self.__conversation()
            task = await self.__task(thread=thread, actor=actor)
            repository = InteractionRepositoryFactory().artifacts()
            first = await repository.link_artifact(
                request=self.__request(actor=actor, thread=thread, task=task)
            )
            second = await repository.link_artifact(
                request=self.__request(actor=actor, thread=thread, task=task)
            )
            await ArtifactRecord.filter(id=second.identity.id).update(
                deleted_at=datetime.now(tz=timezone.utc)
            )

            artifacts = await repository.get_artifacts(
                query=ArtifactQuery(tenant="tenant-a", thread=thread, task=task)
            )

            assert tuple(artifact.identity.id for artifact in artifacts) == (first.identity.id,)

    async def test_list_artifacts_filters_and_paginates(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_artifact_repository"):
            actor, thread = await self.__conversation()
            repository = InteractionRepositoryFactory().artifacts()
            base = datetime(2026, 1, 1, tzinfo=timezone.utc)
            first = await repository.link_artifact(
                request=self.__request(
                    actor=actor,
                    thread=thread,
                    kind=ArtifactKind.SCREENSHOT,
                    created=base,
                )
            )
            second = await repository.link_artifact(
                request=self.__request(
                    actor=actor,
                    thread=thread,
                    kind=ArtifactKind.SCREENSHOT,
                    created=base + timedelta(seconds=1),
                )
            )
            await repository.link_artifact(
                request=self.__request(
                    actor=actor,
                    thread=thread,
                    kind=ArtifactKind.REPORT,
                    created=base + timedelta(seconds=2),
                )
            )

            page = await repository.list_artifacts(
                query=ArtifactCursorQuery(
                    tenant="tenant-a",
                    thread=thread,
                    producer=actor,
                    kinds=(ArtifactKind.SCREENSHOT,),
                    since=base,
                    until=base + timedelta(seconds=2),
                    order=SortOrder.ASC,
                    limit=1,
                )
            )
            next_page = await repository.list_artifacts(
                query=ArtifactCursorQuery(
                    tenant="tenant-a",
                    thread=thread,
                    producer=actor,
                    kinds=(ArtifactKind.SCREENSHOT,),
                    since=base,
                    until=base + timedelta(seconds=2),
                    order=SortOrder.ASC,
                    limit=1,
                    cursor=page.next,
                )
            )

            assert tuple(artifact.identity.id for artifact in page.items) == (first.identity.id,)
            assert tuple(artifact.identity.id for artifact in next_page.items) == (
                second.identity.id,
            )
            assert page.total == 2
            assert next_page.next is None

    async def test_archived_and_deleted_threads_hide_artifacts(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_artifact_repository"):
            actor, archived_thread = await self.__conversation()
            deleted_actor, deleted_thread = await self.__conversation()
            repository = InteractionRepositoryFactory().artifacts()
            await repository.link_artifact(
                request=self.__request(actor=actor, thread=archived_thread)
            )
            await repository.link_artifact(
                request=self.__request(actor=deleted_actor, thread=deleted_thread)
            )
            now = datetime.now(tz=timezone.utc)
            await ConversationRecord.filter(id=archived_thread).update(archived_at=now)
            await ConversationRecord.filter(id=deleted_thread).update(deleted_at=now)

            archived_page = await repository.list_artifacts(
                query=ArtifactCursorQuery(tenant="tenant-a", thread=archived_thread)
            )
            deleted_artifacts = await repository.get_artifacts(
                query=ArtifactQuery(tenant="tenant-a", thread=deleted_thread)
            )

            assert archived_page.items == ()
            assert archived_page.total == 0
            assert deleted_artifacts == []

    async def test_corrupt_artifact_row_raises_interaction_error(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_artifact_repository"):
            actor, thread = await self.__conversation()
            artifact = (
                await InteractionRepositoryFactory()
                .artifacts()
                .link_artifact(request=self.__request(actor=actor, thread=thread))
            )
            with pytest.raises(IntegrityError):
                await ArtifactRecord.filter(id=artifact.identity.id).update(kind="unknown")

    async def test_invalid_labels_shape_raises_interaction_error(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_artifact_repository"):
            actor, thread = await self.__conversation()
            artifact = (
                await InteractionRepositoryFactory()
                .artifacts()
                .link_artifact(request=self.__request(actor=actor, thread=thread))
            )
            await ArtifactRecord.filter(id=artifact.identity.id).update(labels={})

            with pytest.raises(InteractionError, match="Invalid artifact labels"):
                await (
                    InteractionRepositoryFactory()
                    .artifacts()
                    .get_artifacts(query=ArtifactQuery(tenant="tenant-a", thread=thread))
                )

    async def __conversation(self, *, joined: bool = True) -> Tuple[str, str]:
        """
        Insert one actor and thread pair with optional membership.
        """

        now = datetime.now(tz=timezone.utc)
        actor = str(uuid4())
        thread = str(uuid4())
        await ActorRecord.create(
            id=actor,
            tenant_id="tenant-a",
            workspace_id=None,
            kind=ActorKind.HUMAN.value,
            name="Operator",
            skills={},
            metadata={},
            created_at=now,
            updated_at=now,
        )
        await ConversationRecord.create(
            id=thread,
            tenant_id="tenant-a",
            workspace_id=None,
            created_by=actor,
            metadata={},
            created_at=now,
            updated_at=now,
        )
        if joined:
            await MembershipRecord.create(
                id=str(uuid4()),
                tenant_id="tenant-a",
                workspace_id=None,
                conversation_id=thread,
                actor=actor,
                role=MembershipRole.OWNER.value,
                scope=MembershipScope.THREAD.value,
                joined_at=now,
                metadata={},
            )
        self.__executions()[thread] = await self.__execution(actor=actor, thread=thread)
        return actor, thread

    def __executions(self) -> Dict[str, str]:
        """
        Return execution identifiers created for fixture conversations.
        """

        store = getattr(self, "__execution_by_thread", None)
        if store is None:
            store = {}
            setattr(self, "__execution_by_thread", store)

        return store

    async def __execution(self, *, actor: str, thread: str) -> str:
        """
        Insert one execution row for conversation-scoped artifact tests.
        """

        execution = str(uuid4())
        now = datetime.now(tz=timezone.utc)
        await ExecutionRecord.create(
            id=execution,
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=thread,
            intent="Do it",
            state=TaskState.RUNNING.value,
            outcome={},
            started_at=now,
            completed_at=None,
            created_at=now,
            created_by=actor,
            updated_at=now,
            updated_by=actor,
            metadata={},
        )
        return execution

    async def __task(self, *, thread: str, actor: str) -> str:
        """
        Insert one task row for task-scoped artifact tests.
        """

        identifier = str(uuid4())
        execution = str(uuid4())
        now = datetime.now(tz=timezone.utc)
        await ExecutionRecord.create(
            id=execution,
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=thread,
            intent="Do it",
            state=TaskState.RUNNING.value,
            outcome={},
            started_at=now,
            created_at=now,
            created_by=actor,
            updated_at=now,
            updated_by=actor,
            metadata={},
        )
        await TaskRecord.create(
            id=identifier,
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=thread,
            execution_id=execution,
            created_by=actor,
            assignee=actor,
            kind=TaskKind.FATHOM.value,
            objective="Do it",
            state=TaskState.RUNNING.value,
            progress={},
            plan={},
            outcome={},
            metadata={},
            created_at=now,
            updated_at=now,
        )
        return identifier

    def __request(
        self,
        *,
        actor: str,
        thread: str,
        task: Optional[str] = None,
        execution: Optional[str] = None,
        kind: ArtifactKind = ArtifactKind.SCREENSHOT,
        created: Optional[datetime] = None,
    ) -> LinkArtifact:
        """
        Build one artifact link request.
        """

        resolved_execution = execution
        if task is None and resolved_execution is None:
            resolved_execution = self.__executions()[thread]

        return LinkArtifact(
            identity=Identity(id=str(uuid4()), tenant="tenant-a", workspace=None),
            thread=thread,
            execution=resolved_execution,
            task=task,
            producer=actor,
            kind=kind,
            uri=f"local://{uuid4()}",
            backend=ArtifactBackend.LOCAL,
            mime="image/png",
            size=100,
            retention="ephemeral",
            labels=(Label.DISPLAY_AUDIT, Label.MEMORY_SKIP),
            created_at=created or datetime.now(tz=timezone.utc),
            metadata=Metadata(entries={"source": "test"}),
        )
