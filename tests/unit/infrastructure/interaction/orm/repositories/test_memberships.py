from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from tests.unit.infrastructure.interaction.orm.repositories.factories import (
    InteractionRepositoryFactory,
)
from tests.unit.infrastructure.interaction.orm.support import InteractionPostgresSchema
from tortoise.exceptions import IntegrityError

from fathom.constants.collaboration import (
    ActorKind,
    EventKind,
    MembershipRole,
    MembershipScope,
)
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.orm.models import (
    ActorRecord,
    ConversationRecord,
    EventRecord,
    MembershipRecord,
)
from fathom.schemas.interaction import Identity, JoinThread, Metadata


class TestMembershipRepository:
    """
    Verify membership persistence through the persistent-store backed repository.
    """

    async def test_join_thread_persists_membership_and_records_event(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_membership_repository"):
            actor = await self.__actor()
            thread = await self.__thread(actor=actor)
            request = self.__request(actor=actor, thread=thread)

            result = await InteractionRepositoryFactory().memberships().join_thread(request=request)

            assert result.identity == request.identity
            assert result.thread == thread
            assert result.actor == actor
            assert result.role == MembershipRole.OWNER
            assert result.scope == MembershipScope.THREAD
            assert result.joined == request.joined
            assert result.departed_at is None
            assert result.metadata == Metadata(entries={"source": "test"})
            event = await EventRecord.get(conversation_id=thread, sequence=1)
            assert event.kind == EventKind.ACTOR_JOINED.value
            assert event.actor == actor
            stored = await ConversationRecord.get(id=thread)
            assert stored.digest is not None

    async def test_identical_replay_returns_existing_without_new_event(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_membership_repository"):
            actor = await self.__actor()
            thread = await self.__thread(actor=actor)
            request = self.__request(actor=actor, thread=thread)
            repository = InteractionRepositoryFactory().memberships()

            created = await repository.join_thread(request=request)
            replayed = await repository.join_thread(request=request)

            assert replayed == created
            assert await EventRecord.filter(conversation_id=thread).count() == 1
            assert await MembershipRecord.filter(conversation_id=thread).count() == 1

    async def test_conflicting_identity_replay_raises_interaction_error(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_membership_repository"):
            actor = await self.__actor()
            thread = await self.__thread(actor=actor)
            request = self.__request(actor=actor, thread=thread)
            repository = InteractionRepositoryFactory().memberships()
            await repository.join_thread(request=request)
            conflict = request.model_copy(update={"role": MembershipRole.OBSERVER})

            with pytest.raises(InteractionError, match="different content"):
                await repository.join_thread(request=conflict)

    async def test_active_actor_membership_reuses_existing_row(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_membership_repository"):
            actor = await self.__actor()
            thread = await self.__thread(actor=actor)
            first = self.__request(actor=actor, thread=thread)
            second = self.__request(actor=actor, thread=thread)
            repository = InteractionRepositoryFactory().memberships()

            created = await repository.join_thread(request=first)
            reused = await repository.join_thread(request=second)

            assert reused == created
            assert reused.identity.id == first.identity.id
            assert await EventRecord.filter(conversation_id=thread).count() == 1
            assert await MembershipRecord.filter(conversation_id=thread).count() == 1

    async def test_active_actor_membership_reuses_existing_role(self) -> None:
        """
        Reuse one active actor membership even when a caller requests another role.
        """

        async with InteractionPostgresSchema(prefix="conversation_membership_repository"):
            actor = await self.__actor()
            thread = await self.__thread(actor=actor)
            first = self.__request(actor=actor, thread=thread)
            second = self.__request(
                actor=actor,
                thread=thread,
                role=MembershipRole.OBSERVER,
            )
            repository = InteractionRepositoryFactory().memberships()
            created = await repository.join_thread(request=first)
            reused = await repository.join_thread(request=second)

            assert reused == created
            assert reused.role == first.role
            assert await EventRecord.filter(conversation_id=thread).count() == 1
            assert await MembershipRecord.filter(conversation_id=thread).count() == 1

    async def test_departed_actor_can_rejoin_thread(self) -> None:
        """
        Allow a new active membership after the previous one departed.
        """

        async with InteractionPostgresSchema(prefix="conversation_membership_repository"):
            actor = await self.__actor()
            thread = await self.__thread(actor=actor)
            first = self.__request(actor=actor, thread=thread)
            repository = InteractionRepositoryFactory().memberships()
            created = await repository.join_thread(request=first)
            await MembershipRecord.filter(id=created.identity.id).update(
                departed_at=datetime.now(tz=timezone.utc)
            )
            second = self.__request(actor=actor, thread=thread)

            rejoined = await repository.join_thread(request=second)

            assert rejoined.identity.id == second.identity.id
            assert await MembershipRecord.filter(conversation_id=thread).count() == 2
            assert await EventRecord.filter(conversation_id=thread).count() == 2

    async def test_soft_deleted_membership_does_not_count_as_active_for_rejoin(self) -> None:
        """
        Regression: __find_active_membership must skip soft-deleted rows via .present().
        """

        async with InteractionPostgresSchema(prefix="conversation_membership_repository"):
            actor = await self.__actor()
            thread = await self.__thread(actor=actor)
            first = self.__request(actor=actor, thread=thread)
            repository = InteractionRepositoryFactory().memberships()
            created = await repository.join_thread(request=first)
            await MembershipRecord.filter(id=created.identity.id).update(
                deleted_at=datetime.now(tz=timezone.utc)
            )
            second = self.__request(actor=actor, thread=thread)

            rejoined = await repository.join_thread(request=second)

            assert rejoined.identity.id == second.identity.id
            assert rejoined.identity.id != created.identity.id
            assert await MembershipRecord.filter(conversation_id=thread).count() == 2

    async def test_departed_membership_identity_is_not_replayed(self) -> None:
        """
        Reject replay against a departed membership row.
        """

        async with InteractionPostgresSchema(prefix="conversation_membership_repository"):
            actor = await self.__actor()
            thread = await self.__thread(actor=actor)
            request = self.__request(actor=actor, thread=thread)
            repository = InteractionRepositoryFactory().memberships()
            created = await repository.join_thread(request=request)
            await MembershipRecord.filter(id=created.identity.id).update(
                departed_at=datetime.now(tz=timezone.utc)
            )

            with pytest.raises(InteractionError, match="insert conflicted"):
                await repository.join_thread(request=request)

    async def test_join_requires_existing_thread_and_actor(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_membership_repository"):
            actor = await self.__actor()
            thread = str(uuid4())
            repository = InteractionRepositoryFactory().memberships()

            with pytest.raises(InteractionError, match="Thread does not exist"):
                await repository.join_thread(request=self.__request(actor=actor, thread=thread))

            existing_thread = await self.__thread(actor=actor)
            with pytest.raises(InteractionError, match="Actor does not exist"):
                await repository.join_thread(
                    request=self.__request(actor=str(uuid4()), thread=existing_thread)
                )

    async def test_deleted_thread_cannot_be_joined(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_membership_repository"):
            actor = await self.__actor()
            thread = await self.__thread(actor=actor, deleted=True)

            with pytest.raises(InteractionError, match="Thread does not exist"):
                await (
                    InteractionRepositoryFactory()
                    .memberships()
                    .join_thread(request=self.__request(actor=actor, thread=thread))
                )

    async def test_unknown_role_or_scope_raises_interaction_error(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_membership_repository"):
            actor = await self.__actor()
            thread = await self.__thread(actor=actor)
            request = self.__request(actor=actor, thread=thread)
            with pytest.raises(IntegrityError):
                await MembershipRecord.create(
                    id=request.identity.id,
                    tenant_id=request.identity.tenant,
                    workspace_id=request.identity.workspace,
                    conversation_id=thread,
                    actor=actor,
                    role="alien",
                    scope=request.scope.value,
                    joined_at=request.joined,
                    departed_at=None,
                    metadata=request.metadata.entries,
                )

            with pytest.raises(IntegrityError):
                await MembershipRecord.create(
                    id=str(uuid4()),
                    tenant_id=request.identity.tenant,
                    workspace_id=request.identity.workspace,
                    conversation_id=thread,
                    actor=actor,
                    role=request.role.value,
                    scope="galaxy",
                    joined_at=request.joined,
                    departed_at=None,
                    metadata=request.metadata.entries,
                )

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

    async def __thread(self, *, actor: str, deleted: bool = False) -> str:
        """
        Persist one thread row and return its id.
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
            deleted_at=now if deleted else None,
        )
        return thread

    def __request(
        self,
        *,
        actor: str,
        thread: str,
        role: MembershipRole = MembershipRole.OWNER,
        scope: MembershipScope = MembershipScope.THREAD,
    ) -> JoinThread:
        """
        Build one membership join request.
        """

        return JoinThread(
            identity=Identity(id=str(uuid4()), tenant="tenant-a", workspace="workspace-a"),
            thread=thread,
            actor=actor,
            role=role,
            scope=scope,
            joined_at=datetime.now(tz=timezone.utc) + timedelta(seconds=1),
            metadata=Metadata(entries={"source": "test"}),
        )
