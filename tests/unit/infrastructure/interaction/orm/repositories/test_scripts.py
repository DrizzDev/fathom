from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Optional, Tuple
from uuid import UUID, uuid4

import pytest
from asyncpg.exceptions import RaiseError
from tests.unit.infrastructure.interaction.orm.repositories.factories import (
    InteractionRepositoryFactory,
)
from tests.unit.infrastructure.interaction.orm.support import InteractionPostgresSchema
from tortoise.exceptions import IntegrityError

from fathom.constants.collaboration import (
    ActorKind,
    ScriptFormat,
    ScriptStatus,
    ScriptVersionSource,
    TaskKind,
    TaskState,
)
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.orm.models import (
    ActorRecord,
    ConversationRecord,
    ExecutionRecord,
    ScriptRecord,
    ScriptVersionRecord,
    TaskRecord,
)
from fathom.schemas.interaction import (
    Identity,
    Metadata,
    SaveScript,
    ScriptListQuery,
    ScriptQuery,
    ScriptVersionQuery,
    SortOrder,
)


class TestScriptRepository:
    """
    Verify script persistence through the persistent-store backed repository.
    """

    async def test_save_script_creates_live_row_and_first_version(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_script_repository"):
            actor, thread = await self.__conversation()
            task = await self.__task(thread=thread, actor=actor)
            request = self.__request(
                actor=actor,
                thread=thread,
                task=task,
                content="open browser",
            )

            result = await InteractionRepositoryFactory().scripts().save_script(request=request)
            versions = (
                await InteractionRepositoryFactory()
                .scripts()
                .get_script_versions(
                    query=ScriptVersionQuery(
                        tenant="tenant-a",
                        script=request.identity.id,
                    )
                )
            )

            assert result.identity == request.identity
            assert result.thread == thread
            assert result.task == task
            assert result.artifact is None
            assert result.revision == 1
            assert result.format == ScriptFormat.TEXT_PLAIN
            assert result.status == ScriptStatus.ACTIVE
            assert result.created_by == actor
            assert result.updated_by == actor
            assert len(versions) == 1
            assert UUID(versions[0].identity.id)
            assert versions[0].thread == thread
            assert versions[0].task == task
            assert versions[0].artifact is None
            assert versions[0].version == 1
            assert versions[0].checksum == sha256(b"open browser").hexdigest()
            row = await ScriptVersionRecord.get(script_id=request.identity.id, version=1)
            assert row.created_by == actor
            assert row.updated_by == actor

    async def test_same_content_update_changes_mutable_fields_without_new_version(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_script_repository"):
            actor, thread = await self.__conversation()
            task = await self.__task(thread=thread, actor=actor)
            repository = InteractionRepositoryFactory().scripts()
            created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
            request = self.__request(
                actor=actor,
                thread=thread,
                task=task,
                title="Original",
                content="same",
                created=created_at,
            )
            await repository.save_script(request=request)
            updated = request.model_copy(
                update={
                    "title": "Updated",
                    "created": created_at + timedelta(seconds=1),
                    "metadata": Metadata(entries={"updated": True}),
                }
            )

            result = await repository.save_script(request=updated)

            assert result.title == "Updated"
            assert result.revision == 1
            assert result.metadata == Metadata(entries={"updated": True})
            assert await ScriptVersionRecord.filter(script_id=request.identity.id).count() == 1

    async def test_concurrent_identical_save_creates_one_script_version(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_script_repository"):
            actor, thread = await self.__conversation()
            task = await self.__task(thread=thread, actor=actor)
            request = self.__request(actor=actor, thread=thread, task=task)

            async def save() -> int:
                """
                Save the same script from one concurrent worker.
                """

                result = await InteractionRepositoryFactory().scripts().save_script(request=request)
                return result.revision

            revisions = await asyncio.gather(*(save() for _ in range(5)))

            assert revisions == [1, 1, 1, 1, 1]
            assert await ScriptRecord.filter(id=request.identity.id).count() == 1
            assert await ScriptVersionRecord.filter(script_id=request.identity.id).count() == 1

    async def test_changed_content_bumps_revision_and_appends_version(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_script_repository"):
            actor, thread = await self.__conversation()
            task = await self.__task(thread=thread, actor=actor)
            repository = InteractionRepositoryFactory().scripts()
            request = self.__request(actor=actor, thread=thread, task=task, content="v1")
            await repository.save_script(request=request)
            changed = request.model_copy(
                update={
                    "content": "v2",
                    "source": ScriptVersionSource.EDITED,
                    "summary": "Edited",
                    "created": request.created + timedelta(seconds=1),
                }
            )

            result = await repository.save_script(request=changed)
            versions = await repository.get_script_versions(
                query=ScriptVersionQuery(tenant="tenant-a", script=request.identity.id)
            )

            assert result.revision == 2
            assert tuple(version.version for version in versions) == (1, 2)
            assert versions[1].source == ScriptVersionSource.EDITED
            assert versions[1].summary == "Edited"
            assert versions[1].checksum == sha256(b"v2").hexdigest()
            row = await ScriptVersionRecord.get(script_id=request.identity.id, version=2)
            assert row.created_by == actor
            assert row.updated_by == actor

    async def test_save_script_validates_references(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_script_repository"):
            actor, thread = await self.__conversation()

            with pytest.raises(InteractionError, match="Script task does not exist"):
                await (
                    InteractionRepositoryFactory()
                    .scripts()
                    .save_script(
                        request=self.__request(actor=actor, thread=thread, task=str(uuid4()))
                    )
                )

    async def test_save_script_rejects_different_thread_replay(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_script_repository"):
            actor, thread = await self.__conversation()
            other_actor, other_thread = await self.__conversation()
            task = await self.__task(thread=thread, actor=actor)
            other_task = await self.__task(thread=other_thread, actor=other_actor)
            request = self.__request(actor=actor, thread=thread, task=task)
            await InteractionRepositoryFactory().scripts().save_script(request=request)
            conflict = request.model_copy(
                update={"thread": other_thread, "task": other_task, "actor": other_actor}
            )

            with pytest.raises(InteractionError, match="different thread"):
                await InteractionRepositoryFactory().scripts().save_script(request=conflict)

    async def test_identical_save_replay_returns_existing_after_parent_archived(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_script_repository"):
            actor, thread = await self.__conversation()
            task = await self.__task(thread=thread, actor=actor)
            request = self.__request(actor=actor, thread=thread, task=task)
            repository = InteractionRepositoryFactory().scripts()
            created = await repository.save_script(request=request)
            await ConversationRecord.filter(id=thread).update(
                archived_at=datetime.now(tz=timezone.utc),
            )

            replayed = await repository.save_script(request=request)

            assert replayed == created
            assert await ScriptRecord.filter(conversation_id=thread).count() == 1
            assert await ScriptVersionRecord.filter(script_id=request.identity.id).count() == 1

    async def test_get_scripts_and_versions_hide_archived_thread(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_script_repository"):
            actor, thread = await self.__conversation()
            task = await self.__task(thread=thread, actor=actor)
            request = self.__request(actor=actor, thread=thread, task=task)
            repository = InteractionRepositoryFactory().scripts()
            await repository.save_script(request=request)
            await ConversationRecord.filter(id=thread).update(
                archived_at=datetime.now(tz=timezone.utc)
            )

            scripts = await repository.get_scripts(
                query=ScriptQuery(tenant="tenant-a", script=request.identity.id)
            )
            versions = await repository.get_script_versions(
                query=ScriptVersionQuery(tenant="tenant-a", script=request.identity.id)
            )

            assert scripts == []
            assert versions == []

    async def test_script_versions_include_archived_thread_when_requested(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_script_repository"):
            actor, thread = await self.__conversation()
            task = await self.__task(thread=thread, actor=actor)
            request = self.__request(actor=actor, thread=thread, task=task)
            repository = InteractionRepositoryFactory().scripts()
            await repository.save_script(request=request)
            await ConversationRecord.filter(id=thread).update(
                archived_at=datetime.now(tz=timezone.utc)
            )

            versions = await repository.get_script_versions(
                query=ScriptVersionQuery(
                    tenant="tenant-a",
                    script=request.identity.id,
                    include_archived=True,
                )
            )

            assert tuple(version.version for version in versions) == (1,)

    async def test_list_scripts_filters_and_paginates(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_script_repository"):
            actor, thread = await self.__conversation()
            task = await self.__task(thread=thread, actor=actor)
            repository = InteractionRepositoryFactory().scripts()
            base = datetime(2026, 1, 1, tzinfo=timezone.utc)
            first = await repository.save_script(
                request=self.__request(
                    actor=actor,
                    thread=thread,
                    task=task,
                    content="one",
                    created=base,
                )
            )
            second = await repository.save_script(
                request=self.__request(
                    actor=actor,
                    thread=thread,
                    task=task,
                    content="two",
                    created=base + timedelta(seconds=1),
                )
            )

            page = await repository.list_scripts(
                query=ScriptListQuery(
                    tenant="tenant-a",
                    thread=thread,
                    order=SortOrder.ASC,
                    limit=1,
                )
            )
            next_page = await repository.list_scripts(
                query=ScriptListQuery(
                    tenant="tenant-a",
                    thread=thread,
                    order=SortOrder.ASC,
                    limit=1,
                    cursor=page.next,
                )
            )

            assert tuple(script.identity.id for script in page.items) == (first.identity.id,)
            assert tuple(script.identity.id for script in next_page.items) == (second.identity.id,)
            assert page.total == 2
            assert next_page.next is None

    async def test_deleted_scripts_are_hidden_unless_requested(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_script_repository"):
            actor, thread = await self.__conversation()
            task = await self.__task(thread=thread, actor=actor)
            script = (
                await InteractionRepositoryFactory()
                .scripts()
                .save_script(request=self.__request(actor=actor, thread=thread, task=task))
            )
            await ScriptRecord.filter(id=script.identity.id).update(
                deleted_at=datetime.now(tz=timezone.utc)
            )

            hidden = (
                await InteractionRepositoryFactory()
                .scripts()
                .get_scripts(query=ScriptQuery(tenant="tenant-a", thread=thread))
            )
            visible = (
                await InteractionRepositoryFactory()
                .scripts()
                .get_scripts(
                    query=ScriptQuery(tenant="tenant-a", thread=thread, include_deleted=True)
                )
            )

            assert hidden == []
            assert tuple(item.identity.id for item in visible) == (script.identity.id,)

    async def test_corrupt_script_rows_raise_interaction_error(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_script_repository"):
            actor, thread = await self.__conversation()
            task = await self.__task(thread=thread, actor=actor)
            script = (
                await InteractionRepositoryFactory()
                .scripts()
                .save_script(request=self.__request(actor=actor, thread=thread, task=task))
            )
            with pytest.raises(IntegrityError):
                await ScriptRecord.filter(id=script.identity.id).update(format="unknown")

    async def test_script_version_rows_reject_updates(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_script_repository"):
            actor, thread = await self.__conversation()
            task = await self.__task(thread=thread, actor=actor)
            script = (
                await InteractionRepositoryFactory()
                .scripts()
                .save_script(request=self.__request(actor=actor, thread=thread, task=task))
            )
            with pytest.raises(RaiseError, match="append-only table script_versions"):
                await ScriptVersionRecord.filter(script_id=script.identity.id).update(
                    source="unknown"
                )

    async def __conversation(self) -> Tuple[str, str]:
        """
        Insert one active actor and thread pair.
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
        return actor, thread

    async def __task(self, *, thread: str, actor: str) -> str:
        """
        Insert one task row for script reference tests.
        """

        identifier = str(uuid4())
        now = datetime.now(tz=timezone.utc)
        await ExecutionRecord.create(
            id=identifier,
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
            execution_id=identifier,
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
        task: str,
        title: Optional[str] = "Script",
        content: str = "content",
        created: Optional[datetime] = None,
    ) -> SaveScript:
        """
        Build one script save request.
        """

        return SaveScript(
            identity=Identity(id=str(uuid4()), tenant="tenant-a", workspace=None),
            thread=thread,
            task=task,
            title=title,
            format=ScriptFormat.TEXT_PLAIN,
            status=ScriptStatus.ACTIVE,
            content=content,
            source=ScriptVersionSource.GENERATED,
            summary=None,
            actor=actor,
            created_at=created or datetime.now(tz=timezone.utc),
            metadata=Metadata(entries={"source": "test"}),
        )
