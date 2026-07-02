from __future__ import annotations

from typing import Dict, Mapping, Optional

from pydantic import JsonValue
from tortoise.exceptions import IntegrityError

from fathom.constants.collaboration import ExecutionState, TaskCode
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.orm.models import ConversationRecord, ExecutionRecord
from fathom.infrastructure.interaction.orm.repositories.lifecycle import (
    DatabaseConnection,
    TransactionScope,
)
from fathom.schemas.interaction import (
    Execution,
    ExecutionQuery,
    FinishExecution,
    Identity,
    Metadata,
    StartExecution,
    Terminal,
    Timing,
    Visibility,
)


class ExecutionRepository:
    """
    Persistent-store backed repository for user intent executions.
    """

    def __init__(self, *, transaction: TransactionScope) -> None:
        """
        Initialize execution persistence collaborators.
        """

        self.__transaction = transaction

    async def start_execution(self, *, request: StartExecution) -> Execution:
        """
        Persist one execution or replay an identical existing execution.
        """

        try:
            return await self.__start_execution(request=request)
        except IntegrityError as exception:
            existing = await self.__load_execution(
                connection=None,
                execution=request.identity.id,
                tenant=request.identity.tenant,
            )
            if existing is not None and self.__same_start(request=request, execution=existing):
                return existing

            raise InteractionError(
                "Execution insert conflicted with a different row."
            ) from exception

    async def finish_execution(self, *, request: FinishExecution) -> Execution:
        """
        Persist the terminal outcome for one execution.
        """

        async with self.__transaction.transaction() as connection:
            execution = await self.__load_execution(
                connection=connection,
                tenant=request.tenant,
                execution=request.execution,
            )
            if execution is None:
                raise InteractionError("Execution does not exist.")

            if execution.state == request.state and execution.terminal is not None:
                if not self.__same_finish(execution=execution, request=request):
                    raise InteractionError("Execution already finished with a different outcome.")

                return execution

            await (
                ExecutionRecord.filter(
                    tenant_id=request.tenant,
                    id=request.execution,
                    **Visibility(archived=True).as_filters(),
                )
                .using_db(connection)
                .update(
                    summary=request.summary,
                    updated_by=request.actor,
                    state=request.state.value,
                    updated_at=request.completed,
                    detail=request.terminal.detail,
                    completed_at=request.completed,
                    outcome=request.outcome.entries,
                    code=request.terminal.code.value,
                )
            )

            updated = await self.__load_execution(
                connection=connection,
                tenant=request.tenant,
                execution=request.execution,
            )
            if updated is None:
                raise InteractionError("Execution was not updated.")

            return updated

    async def get_execution(self, *, query: ExecutionQuery) -> Optional[Execution]:
        """
        Load one active tenant-scoped execution by identifier.
        """

        return await self.__load_execution(
            connection=None,
            tenant=query.tenant,
            thread=query.thread,
            execution=query.execution,
        )

    async def __start_execution(self, *, request: StartExecution) -> Execution:
        """
        Persist one execution inside one transaction.
        """

        async with self.__transaction.transaction() as connection:
            existing = await self.__load_execution(
                connection=connection,
                tenant=request.identity.tenant,
                execution=request.identity.id,
            )
            if existing is not None:
                if not self.__same_start(execution=existing, request=request):
                    raise InteractionError(
                        "Execution identity already exists with different content."
                    )

                return existing

            await self.__require_thread(
                thread=request.thread,
                connection=connection,
                tenant=request.identity.tenant,
            )
            await ExecutionRecord.create(
                using_db=connection,
                intent=request.intent,
                id=request.identity.id,
                created_by=request.actor,
                updated_by=request.actor,
                state=request.state.value,
                created_at=request.started,
                updated_at=request.started,
                started_at=request.started,
                conversation_id=request.thread,
                workflow_id=request.workflow_id,
                metadata=request.metadata.entries,
                tenant_id=request.identity.tenant,
                workspace_id=request.identity.workspace,
            )

            execution = await self.__load_execution(
                connection=connection,
                execution=request.identity.id,
                tenant=request.identity.tenant,
            )
            if execution is None:
                raise InteractionError("Execution was not persisted.")

            return execution

    async def __require_thread(
        self,
        *,
        tenant: str,
        thread: str,
        connection: DatabaseConnection,
    ) -> None:
        """
        Require an active thread before an execution references it.
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

    async def __load_execution(
        self,
        *,
        tenant: str,
        execution: str,
        connection: Optional[DatabaseConnection],
        thread: Optional[str] = None,
    ) -> Optional[Execution]:
        """
        Load one execution row, optionally scoped to a conversation.
        """

        filters: Dict[str, object] = {
            "tenant_id": tenant,
            "id": execution,
            "deleted_at__isnull": True,
        }
        if thread is not None:
            filters["conversation_id"] = thread

        queryset = ExecutionRecord.filter(**filters)
        if connection is not None:
            queryset = queryset.using_db(connection)

        row = await queryset.get_or_none()
        if row is None:
            return None

        return self.__execution(row=row)

    def __same_start(self, *, execution: Execution, request: StartExecution) -> bool:
        """
        Check whether a start request replays an already opened execution.
        """

        return (
            execution.state == request.state
            and execution.thread == request.thread
            and execution.intent == request.intent
            and execution.metadata == request.metadata
            and execution.identity == request.identity
            and execution.timing.started == request.started
        )

    def __same_finish(self, *, execution: Execution, request: FinishExecution) -> bool:
        """
        Check whether a finish request replays an already stored terminal outcome.
        """

        if execution.terminal is None:
            return False

        return (
            execution.state == request.state
            and execution.summary == request.summary
            and execution.outcome == request.outcome
            and execution.timing.ended == request.completed
            and execution.terminal.code == request.terminal.code
            and execution.terminal.detail == request.terminal.detail
        )

    def __execution(self, *, row: ExecutionRecord) -> Execution:
        """
        Convert one persistent execution model into the interaction schema.
        """

        return Execution(
            intent=row.intent,
            summary=row.summary,
            thread=row.conversation_id,
            workflow_id=row.workflow_id,
            state=self.__state(value=row.state),
            terminal=self.__terminal(code=row.code, detail=row.detail),
            outcome=self.__metadata(value=row.outcome, field="outcome"),
            metadata=self.__metadata(value=row.metadata, field="metadata"),
            timing=Timing(
                created_at=row.created_at,
                updated_at=row.updated_at,
                started_at=row.started_at,
                ended_at=row.completed_at,
            ),
            deleted_at=row.deleted_at,
            identity=Identity(
                id=row.id,
                tenant=row.tenant_id,
                workspace=row.workspace_id,
            ),
        )

    def __state(self, *, value: str) -> ExecutionState:
        """
        Convert stored execution state text into the public enum.
        """

        try:
            return ExecutionState(value)
        except ValueError as exception:
            raise InteractionError(f"Unknown execution state in row: {value}.") from exception

    def __metadata(self, *, value: JsonValue, field: str) -> Metadata:
        """
        Convert stored JSON object fields into metadata.
        """

        if not isinstance(value, Mapping):
            raise InteractionError(f"Execution {field} is not an object.")

        return Metadata(entries=value)

    def __terminal(self, *, code: Optional[str], detail: Optional[str]) -> Optional[Terminal]:
        """
        Convert stored terminal fields into a terminal outcome.
        """

        if code is None:
            return None

        try:
            task_code = TaskCode(code)
        except ValueError as exception:
            raise InteractionError(
                f"Unknown execution terminal code in row: {code}."
            ) from exception

        return Terminal(code=task_code, detail=detail)
