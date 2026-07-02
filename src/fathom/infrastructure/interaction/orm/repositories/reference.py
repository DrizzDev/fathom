from __future__ import annotations

from typing import cast

from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.orm.models import (
    ActorRecord,
    ArtifactRecord,
    ConversationRecord,
    EventRecord,
    ExecutionRecord,
    MembershipRecord,
    MessageRecord,
    TaskRecord,
)
from fathom.infrastructure.interaction.orm.repositories.lifecycle import DatabaseConnection
from fathom.schemas.interaction import MembershipVisibility, ThreadScope, Visibility


class ReferenceGuard:
    """
    Validates tenant-scoped references shared by ORM repositories.
    """

    async def thread_visible(self, *, scope: ThreadScope) -> bool:
        """
        Return true when the parent conversation row is visible under the scope.
        """

        return cast(
            "bool",
            await ConversationRecord.filter(
                tenant_id=scope.reference.tenant,
                id=scope.reference.thread,
                **Visibility(
                    deleted=scope.visibility.deleted,
                    archived=scope.visibility.archived,
                ).as_filters(),
            ).exists(),
        )

    async def active_thread(
        self,
        *,
        tenant: str,
        thread: str,
        connection: DatabaseConnection,
    ) -> None:
        """
        Require an active conversation row.
        """

        row = (
            await ConversationRecord.filter(
                tenant_id=tenant,
                id=thread,
                **Visibility().as_filters(),
            )
            .using_db(connection)
            .get_or_none()
        )
        if row is None:
            raise InteractionError("Thread does not exist.")

    async def present_execution(
        self,
        *,
        tenant: str,
        thread: str,
        execution: str,
        connection: DatabaseConnection,
    ) -> None:
        """
        Require a non-deleted execution in the expected conversation.
        """

        row = (
            await ExecutionRecord.filter(
                tenant_id=tenant,
                id=execution,
                conversation_id=thread,
                **Visibility(archived=True).as_filters(),
            )
            .using_db(connection)
            .get_or_none()
        )
        if row is None:
            raise InteractionError("Execution does not exist in thread.")

    async def present_task(
        self,
        *,
        task: str,
        label: str,
        thread: str,
        tenant: str,
        connection: DatabaseConnection,
    ) -> TaskRecord:
        """
        Return a non-deleted task from the expected conversation.
        """

        row = (
            await TaskRecord.filter(
                tenant_id=tenant,
                id=task,
                **Visibility(archived=True).as_filters(),
            )
            .using_db(connection)
            .get_or_none()
        )
        if row is None:
            raise InteractionError(f"{label} task does not exist.")

        if row.conversation_id != thread:
            raise InteractionError(f"{label} task belongs to a different thread.")

        return cast("TaskRecord", row)

    async def active_membership(
        self,
        *,
        actor: str,
        tenant: str,
        thread: str,
        connection: DatabaseConnection,
    ) -> None:
        """
        Require an existing actor with active membership in a conversation.
        """

        row = await ActorRecord.get_or_none(id=actor, tenant_id=tenant, using_db=connection)

        if row is None:
            raise InteractionError("Actor does not exist.")

        membership_exists = (
            await MembershipRecord.filter(
                tenant_id=tenant,
                conversation_id=thread,
                actor=actor,
                **MembershipVisibility().as_filters(),
            )
            .using_db(connection)
            .exists()
        )
        if not membership_exists:
            raise InteractionError("Actor is not an active member of the thread.")

    async def present_message(
        self,
        *,
        tenant: str,
        thread: str,
        message: str,
        connection: DatabaseConnection,
    ) -> None:
        """
        Require a non-deleted message in the expected conversation.
        """

        row = (
            await MessageRecord.filter(
                tenant_id=tenant,
                id=message,
                conversation_id=thread,
                **Visibility(archived=True).as_filters(),
            )
            .using_db(connection)
            .get_or_none()
        )
        if row is None:
            raise InteractionError("Message does not exist.")

    async def present_event(
        self,
        *,
        event: str,
        thread: str,
        tenant: str,
        connection: DatabaseConnection,
    ) -> None:
        """
        Require an event in the expected conversation.
        """

        row = await EventRecord.get_or_none(tenant_id=tenant, id=event, using_db=connection)

        if row is None:
            raise InteractionError("Event does not exist.")

        if row.conversation_id != thread:
            raise InteractionError("Event belongs to a different thread.")

    async def present_artifact(
        self,
        *,
        tenant: str,
        thread: str,
        artifact: str,
        connection: DatabaseConnection,
    ) -> None:
        """
        Require a non-deleted artifact in the expected conversation.
        """

        row = (
            await ArtifactRecord.filter(
                tenant_id=tenant,
                id=artifact,
                conversation_id=thread,
                **Visibility(archived=True).as_filters(),
            )
            .using_db(connection)
            .get_or_none()
        )
        if row is None:
            raise InteractionError("Artifact does not exist.")
