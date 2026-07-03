from __future__ import annotations

import unittest
from datetime import datetime, timezone
from typing import Tuple
from uuid import uuid4

from fathom.adapters.interaction.orm.postgres import PostgresInteraction
from fathom.adapters.signing.noop import NoopSigner
from fathom.constants.collaboration import ActorKind, MembershipRole
from fathom.constants.storage import PostgresMigrationMode
from fathom.core.exceptions import InteractionError
from fathom.core.services.conversation import ConversationService, Ports
from fathom.infrastructure.interaction.orm.models import MembershipRecord
from fathom.schemas.configuration import PostgresInteractionConfiguration
from fathom.schemas.conversation import (
    ActorInput,
    AddActor,
    ConversationListQuery,
    ConversationThreadQuery,
    JoinMember,
    ThreadCreate,
)
from tests.unit.infrastructure.interaction.orm.support import PostgresSchema


class TestConversationAuthorization(unittest.IsolatedAsyncioTestCase):
    """
    Verify tenant-and-membership authorization through the conversation service.
    """

    async def build_service(
        self, *, prefix: str
    ) -> Tuple[PostgresInteraction, ConversationService]:
        """
        Create a migrated disposable schema and a conversation service bound to it.
        """

        schema = PostgresSchema(prefix=prefix)
        await schema.__aenter__()
        interaction = PostgresInteraction(
            configuration=PostgresInteractionConfiguration(
                database="postgres",
                host="localhost",
                migration_mode=PostgresMigrationMode.VALIDATE,
                password="postgres",
                pool_max_size=2,
                schema_name=schema.name,
                user="postgres",
            )
        )
        await interaction.initialize()
        self.addAsyncCleanup(interaction.aclose)
        self.addAsyncCleanup(schema.__aexit__, None, None, None)
        service = ConversationService(
            signer=NoopSigner(),
            ports=Ports(interaction=interaction),
        )
        return interaction, service

    async def test_fresh_actor_with_no_memberships_sees_no_threads(self) -> None:
        """
        A first-time actor in an existing tenant must not enumerate other actors' threads.
        """

        _, service = await self.build_service(prefix="auth_fresh")

        aman = await self.__actor(service=service, actor_id="aman@example.com")
        tanisha = await self.__actor(service=service, actor_id="tanisha@example.com")
        fresh = "aditya@example.com"

        await self.__thread(service=service, creator=aman, title="private-a")
        await self.__thread(service=service, creator=tanisha, title="private-b")

        page = await service.list(
            query=ConversationListQuery(
                tenant="tenant-a",
                operator=fresh,
                limit=100,
            ),
        )

        self.assertEqual(0, page.total)
        self.assertEqual(0, len(page.items))
        self.assertFalse(any(item.title == "private-a" for item in page.items))
        self.assertFalse(any(item.title == "private-b" for item in page.items))

    async def test_member_sees_only_own_threads(self) -> None:
        """
        Each active member sees only threads with active memberships.
        """

        _, service = await self.build_service(prefix="auth_own")

        aman = await self.__actor(service=service, actor_id="aman@example.com")
        tanisha = await self.__actor(service=service, actor_id="tanisha@example.com")

        aman_thread = await self.__thread(service=service, creator=aman, title="aman-thread")
        tanisha_thread = await self.__thread(
            service=service, creator=tanisha, title="tanisha-thread"
        )

        aman_page = await service.list(
            query=ConversationListQuery(
                tenant="tenant-a",
                operator=aman,
                limit=100,
            ),
        )
        self.assertEqual(1, len(aman_page.items))
        self.assertEqual(aman_thread, aman_page.items[0].id)

        tanisha_page = await service.list(
            query=ConversationListQuery(
                tenant="tenant-a",
                operator=tanisha,
                limit=100,
            ),
        )
        self.assertEqual(1, len(tanisha_page.items))
        self.assertEqual(tanisha_thread, tanisha_page.items[0].id)

    async def test_non_member_get_thread_raises_not_found(self) -> None:
        """
        A non-member get surfaces as thread-does-not-exist to avoid existence disclosure.
        """

        _, service = await self.build_service(prefix="auth_get")

        aman = await self.__actor(service=service, actor_id="aman@example.com")
        outsider = await self.__actor(service=service, actor_id="outsider@example.com")
        aman_thread = await self.__thread(service=service, creator=aman, title="private")

        with self.assertRaises(InteractionError):
            await service.get(
                query=ConversationThreadQuery(
                    tenant="tenant-a",
                    operator=outsider,
                    thread=aman_thread,
                ),
            )

    async def test_soft_deleted_membership_hides_thread_from_former_member(self) -> None:
        """
        A soft-deleted membership must not let a former member re-enumerate the thread.
        """

        _, service = await self.build_service(prefix="auth_soft_delete")

        aman = await self.__actor(service=service, actor_id="aman@example.com")
        observer = await self.__actor(service=service, actor_id="observer@example.com")
        aman_thread = await self.__thread(service=service, creator=aman, title="shared")

        observer_membership = str(uuid4())
        await service.join(
            request=JoinMember(
                id=observer_membership,
                tenant="tenant-a",
                workspace=None,
                actor=observer,
                thread=aman_thread,
                role=MembershipRole.OBSERVER,
                joined=datetime.now(tz=timezone.utc),
                metadata={},
            ),
        )

        page = await service.list(
            query=ConversationListQuery(
                tenant="tenant-a",
                operator=observer,
                limit=100,
            ),
        )
        self.assertEqual(1, len(page.items))

        await MembershipRecord.filter(id=observer_membership).update(
            deleted_at=datetime.now(tz=timezone.utc)
        )

        page = await service.list(
            query=ConversationListQuery(
                tenant="tenant-a",
                operator=observer,
                limit=100,
            ),
        )
        self.assertEqual(0, page.total)
        self.assertEqual(0, len(page.items))

    async def __actor(self, *, service: ConversationService, actor_id: str) -> str:
        """
        Register one actor via the service and return the stable identifier.
        """

        await service.actor(
            request=AddActor(
                id=actor_id,
                tenant="tenant-a",
                workspace=None,
                kind=ActorKind.HUMAN,
                name=actor_id,
                external=None,
                provider=None,
                model=None,
                created=datetime.now(tz=timezone.utc),
                metadata={},
            ),
        )
        return actor_id

    async def __thread(self, *, service: ConversationService, creator: str, title: str) -> str:
        """
        Create one thread with the given creator and return its identifier.
        """

        thread_id = str(uuid4())
        membership_id = str(uuid4())
        await service.create(
            request=ThreadCreate(
                id=thread_id,
                tenant="tenant-a",
                workspace=None,
                title=title,
                created=datetime.now(tz=timezone.utc),
                creator=ActorInput(
                    id=creator,
                    workspace=None,
                    kind=ActorKind.HUMAN,
                    name=creator,
                    provider=None,
                    model=None,
                ),
                member=membership_id,
                role=MembershipRole.OWNER,
                metadata={},
            ),
        )
        return thread_id
