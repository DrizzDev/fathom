from __future__ import annotations

from fathom.core.exceptions import ThreadNotFoundError
from fathom.core.services.conversation.ports import MemberStore, ThreadStore
from fathom.schemas import interaction as InteractionSchemas


class ConversationAccessGuard:
    """
    Enforces active membership before client-facing conversation reads.
    """

    def __init__(self, *, threads: ThreadStore, members: MemberStore) -> None:
        """
        Initialize guard dependencies.
        """

        self.__threads = threads
        self.__members = members

    async def require(
        self,
        *,
        tenant: str,
        thread: str,
        operator: str,
        include_archived: bool = False,
    ) -> InteractionSchemas.Thread:
        """
        Require that an actor is an active member of a conversation.
        """

        stored = await self.__threads.get(
            query=InteractionSchemas.ThreadQuery(
                tenant=tenant,
                thread=thread,
                include_archived=include_archived,
            )
        )
        if stored is None:
            raise ThreadNotFoundError(
                thread=thread,
                message="Conversation thread does not exist.",
            )

        membership = await self.__members.find(
            query=InteractionSchemas.MembershipQuery(
                tenant=tenant,
                thread=thread,
                actor=operator,
            )
        )
        if membership is None:
            raise ThreadNotFoundError(
                thread=thread,
                message="Conversation thread does not exist.",
            )

        return stored
