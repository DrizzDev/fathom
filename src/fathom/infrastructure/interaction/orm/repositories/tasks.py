from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import JsonValue
from tortoise.exceptions import IntegrityError
from tortoise.expressions import Q

from fathom.constants.collaboration import EventKind, TaskCode, TaskKind, TaskState
from fathom.core.exceptions import InteractionError, TaskConflictError
from fathom.infrastructure.interaction.orm.models import (
    ActorRecord,
    ConversationRecord,
    ExecutionRecord,
    MembershipRecord,
    MessageRecord,
    TaskRecord,
)
from fathom.infrastructure.interaction.orm.repositories.lifecycle import (
    DatabaseConnection,
    LifecycleRecorder,
    TransactionScope,
)
from fathom.interaction.lifecycle import Lifecycle
from fathom.schemas.interaction import (
    Assignment,
    FinishTask,
    Identity,
    Lineage,
    MembershipVisibility,
    Metadata,
    OpenTask,
    Plan,
    Task,
    TaskOneQuery,
    TaskQuery,
    Terminal,
    Timing,
    Visibility,
)


class TaskRepository:
    """
    persistent-store backed repository for durable conversation tasks.
    """

    __FINISH_EVENT_KINDS: Dict[TaskState, EventKind] = {
        TaskState.FAILED: EventKind.TASK_FAILED,
        TaskState.EXPIRED: EventKind.TASK_EXPIRED,
        TaskState.DELETED: EventKind.TASK_DELETED,
        TaskState.SUCCEEDED: EventKind.TASK_SUCCEEDED,
        TaskState.CANCELLED: EventKind.TASK_CANCELLED,
    }

    def __init__(
        self,
        *,
        lifecycle: Lifecycle,
        recorder: LifecycleRecorder,
        transaction: TransactionScope,
    ) -> None:
        """
        Initialize task persistence collaborators.
        """

        self.__recorder = recorder
        self.__lifecycle = lifecycle
        self.__transaction = transaction

    async def open_task(self, *, request: OpenTask) -> Task:
        """
        Persist one task or replay an identical existing task.
        """

        root = request.lineage.root or request.identity.id

        self.__lifecycle.validate_task_lineage(
            root=root,
            task=request.identity.id,
            parent=request.lineage.parent,
        )
        if request.state in self.__FINISH_EVENT_KINDS:
            raise InteractionError("Task cannot be opened in a terminal state.")

        try:
            return await self.__open_task(request=request, root=root)
        except IntegrityError as exception:
            existing = await self.__load_task(
                connection=None, task=request.identity.id, tenant=request.identity.tenant
            )
            if existing is not None and self.__same_open_task(
                root=root, task=existing, request=request
            ):
                return existing

            raise InteractionError("Task insert conflicted with a different row.") from exception

    async def __open_task(self, *, request: OpenTask, root: str) -> Task:
        """
        Persist one task inside one transaction.
        """

        async with self.__transaction.transaction() as connection:
            if existing := await self.__load_task(
                connection=connection, task=request.identity.id, tenant=request.identity.tenant
            ):
                if not self.__same_open_task(task=existing, request=request, root=root):
                    raise InteractionError("Task identity already exists with different content.")

                return existing

            await self.__require_thread(
                thread=request.thread,
                connection=connection,
                tenant=request.identity.tenant,
            )
            await self.__require_actor_membership(
                connection=connection,
                thread=request.thread,
                tenant=request.identity.tenant,
                actor=request.assignment.creator,
            )
            await self.__require_actor_membership(
                connection=connection,
                thread=request.thread,
                tenant=request.identity.tenant,
                actor=request.assignment.assignee,
            )

            execution = await self.__require_execution(
                request=request,
                connection=connection,
            )

            if request.lineage.parent is not None:
                await self.__require_task_in_thread(
                    thread=request.thread,
                    connection=connection,
                    task=request.lineage.parent,
                    tenant=request.identity.tenant,
                )
            if root != request.identity.id:
                root_task = await self.__require_task_in_thread(
                    task=root,
                    thread=request.thread,
                    connection=connection,
                    tenant=request.identity.tenant,
                )

                if root_task.execution != execution:
                    raise InteractionError("Root task belongs to a different execution.")

            if request.lineage.origin is not None:
                await self.__require_message_in_thread(
                    thread=request.thread,
                    connection=connection,
                    tenant=request.identity.tenant,
                    message=request.lineage.origin,
                )

            started_at = request.created if request.state == TaskState.RUNNING else None

            await TaskRecord.create(
                using_db=connection,
                started_at=started_at,
                id=request.identity.id,
                execution_id=execution,
                kind=request.kind.value,
                state=request.state.value,
                created_at=request.created,
                conversation_id=request.thread,
                plan=request.plan.plan.entries,
                parent_id=request.lineage.parent,
                origin_id=request.lineage.origin,
                objective=request.plan.objective,
                reference=request.plan.reference,
                metadata=request.metadata.entries,
                tenant_id=request.identity.tenant,
                assignee=request.assignment.assignee,
                created_by=request.assignment.creator,
                progress=request.plan.progress.entries,
                workspace_id=request.identity.workspace,
                root_id=root if root != request.identity.id else None,
            )

            task = await self.__load_task(
                connection=connection, task=request.identity.id, tenant=request.identity.tenant
            )
            if task is None:
                raise InteractionError("Task was not persisted.")

            await self.__recorder.record(
                connection=connection,
                thread=request.thread,
                created=request.created,
                task=request.identity.id,
                kind=EventKind.TASK_OPENED,
                execution=request.execution,
                tenant=request.identity.tenant,
                workspace=request.identity.workspace,
                actor=request.assignment.assignee or request.assignment.creator,
                payload=Metadata(
                    entries={"kind": request.kind.value, "state": request.state.value}
                ),
            )
            if request.state is TaskState.RUNNING:
                await self.__recorder.record(
                    connection=connection,
                    thread=request.thread,
                    created=request.created,
                    task=request.identity.id,
                    kind=EventKind.TASK_STARTED,
                    execution=request.execution,
                    tenant=request.identity.tenant,
                    workspace=request.identity.workspace,
                    payload=Metadata(entries={"kind": request.kind.value}),
                    actor=request.assignment.assignee or request.assignment.creator,
                )

            if request.lineage.parent is not None:
                await self.__recorder.record(
                    connection=connection,
                    thread=request.thread,
                    created=request.created,
                    task=request.identity.id,
                    execution=request.execution,
                    kind=EventKind.TASK_DELEGATED,
                    tenant=request.identity.tenant,
                    actor=request.assignment.creator,
                    workspace=request.identity.workspace,
                    payload=Metadata(entries={"parent": request.lineage.parent}),
                )

            return task

    async def finish_task(self, *, request: FinishTask) -> Task:
        """
        Move one task to a terminal state.
        """

        async with self.__transaction.transaction() as connection:
            task = await self.__load_task(
                task=request.task, connection=connection, tenant=request.tenant
            )
            if task is None:
                raise InteractionError("Task does not exist.")

            if task.state == request.state and task.terminal is not None:
                if not self.__same_finish(task=task, request=request):
                    raise TaskConflictError(
                        task=request.task,
                        message="Task already finished with a different outcome.",
                    )

                return task

            self.__lifecycle.validate_task_transition(current=task.state, target=request.state)

            await (
                TaskRecord.filter(
                    id=request.task,
                    deleted_at__isnull=True,
                    tenant_id=request.tenant,
                )
                .using_db(connection)
                .update(
                    outcome={
                        "summary": request.summary,
                        "state": request.state.value,
                        "detail": request.terminal.detail,
                        "code": request.terminal.code.value,
                    },
                    progress={
                        "summary": request.summary,
                        "state": request.state.value,
                        "elapsed": request.elapsed,
                        "detail": request.terminal.detail,
                        "code": request.terminal.code.value,
                    },
                    summary=request.summary,
                    elapsed=request.elapsed,
                    updated_at=request.ended,
                    state=request.state.value,
                    completed_at=request.ended,
                    detail=request.terminal.detail,
                    code=request.terminal.code.value,
                    updated_by=task.assignment.assignee or task.assignment.creator,
                )
            )
            finished = await self.__load_task(
                task=request.task, tenant=request.tenant, connection=connection
            )
            if finished is None:
                raise InteractionError("Task was not updated.")

            await self.__recorder.record(
                task=request.task,
                thread=task.thread,
                created=request.ended,
                connection=connection,
                tenant=request.tenant,
                execution=task.execution,
                workspace=task.identity.workspace,
                kind=self.__finish_event_kind(state=request.state),
                actor=task.assignment.assignee or task.assignment.creator,
                payload=Metadata(entries={"code": request.terminal.code.value}),
            )

        return finished

    async def get_tasks(self, *, query: TaskQuery) -> List[Task]:
        """
        Load tenant-scoped tasks for one thread, honoring the caller's deletion filter.
        """

        rows = await TaskRecord.filter(
            tenant_id=query.tenant,
            conversation_id=query.thread,
            **self.__deletion_filters(query=query),
        ).order_by("created_at", "id")

        return [self.__task(row=row) for row in rows]

    async def get_task(self, *, query: TaskOneQuery) -> Optional[Task]:
        """
        Load one active tenant-scoped task by identifier.
        """

        return await self.__load_task(
            connection=None,
            task=query.task,
            tenant=query.tenant,
            thread=query.thread,
        )

    async def recent_task(self, *, query: TaskQuery) -> Optional[Task]:
        """
        Load the most-recently-created task in the thread using a bounded single-row query, honoring the caller's deletion filter.
        """

        row = (
            await TaskRecord.filter(
                tenant_id=query.tenant,
                conversation_id=query.thread,
                **self.__deletion_filters(query=query),
            )
            .order_by("-created_at", "-id")
            .first()
        )

        if row is None:
            return None

        return self.__task(row=row)

    async def top_roots(self, *, query: TaskQuery, limit: int) -> List[Task]:
        """
        Load the top-N root tasks in the thread, ordered by created_at DESC, LIMIT at SQL.
        """

        rows = (
            await TaskRecord.filter(
                tenant_id=query.tenant,
                parent_id__isnull=True,
                conversation_id=query.thread,
                **self.__deletion_filters(query=query),
            )
            .order_by("-created_at", "-id")
            .limit(limit)
        )

        return [self.__task(row=row) for row in rows]

    async def descendants(self, *, query: TaskQuery, roots: List[str]) -> List[Task]:
        """
        Load every task whose root points to one of the supplied root ids using a single indexed IN scan.
        """

        if not roots:
            return []

        rows = await TaskRecord.filter(
            root_id__in=roots,
            tenant_id=query.tenant,
            conversation_id=query.thread,
            **self.__deletion_filters(query=query),
        ).order_by("created_at", "id")

        return [self.__task(row=row) for row in rows]

    async def subtree(self, *, query: TaskQuery, root: str) -> List[Task]:
        """
        Load one subtree rooted at the supplied task via a single OR-filtered SQL scan.
        """

        rows = (
            await TaskRecord.filter(
                tenant_id=query.tenant,
                conversation_id=query.thread,
                **self.__deletion_filters(query=query),
            )
            .filter(Q(id=root) | Q(root_id=root))
            .order_by("created_at", "id")
        )

        return [self.__task(row=row) for row in rows]

    @staticmethod
    def __deletion_filters(*, query: TaskQuery) -> Dict[str, bool]:
        """
        Return the tortoise filter kwargs implementing the query's deletion flag.
        """

        return {} if query.deleted else {"deleted_at__isnull": True}

    async def __load_task(
        self,
        *,
        task: str,
        tenant: str,
        thread: Optional[str] = None,
        connection: Optional[DatabaseConnection],
    ) -> Optional[Task]:
        """
        Load one task row, optionally scoped to a conversation.
        """

        filters: Dict[str, object] = {
            "id": task,
            "tenant_id": tenant,
            "deleted_at__isnull": True,
        }
        if thread is not None:
            filters["conversation_id"] = thread

        queryset = TaskRecord.filter(**filters)

        if connection is not None:
            queryset = queryset.using_db(connection)

        row = await queryset.get_or_none()

        if row is None:
            return None

        return self.__task(row=row)

    async def __require_thread(
        self,
        *,
        tenant: str,
        thread: str,
        connection: DatabaseConnection,
    ) -> None:
        """
        Require an active thread before a task references it.
        """

        row = (
            await ConversationRecord.filter(
                id=thread,
                tenant_id=tenant,
                **Visibility().as_filters(),
            )
            .using_db(connection)
            .get_or_none()
        )
        if row is None:
            raise InteractionError("Thread does not exist.")

    async def __require_execution(
        self,
        *,
        request: OpenTask,
        connection: DatabaseConnection,
    ) -> str:
        """
        Require an active execution before a task references it.
        """

        row = (
            await ExecutionRecord.filter(
                id=request.execution,
                conversation_id=request.thread,
                tenant_id=request.identity.tenant,
                **Visibility(archived=True).as_filters(),
            )
            .using_db(connection)
            .get_or_none()
        )
        if row is None:
            raise InteractionError("Execution does not exist.")

        if row.workspace_id != request.identity.workspace:
            raise InteractionError("Execution belongs to a different workspace.")

        if request.lineage.parent is not None:
            parent = await self.__load_task(
                connection=connection,
                task=request.lineage.parent,
                tenant=request.identity.tenant,
            )
            if parent is None:
                raise InteractionError("Task does not exist.")

            if parent.execution != request.execution:
                raise InteractionError("Parent task belongs to a different execution.")

        return request.execution

    async def __require_actor_membership(
        self,
        *,
        tenant: str,
        thread: str,
        actor: Optional[str],
        connection: DatabaseConnection,
    ) -> None:
        """
        Require an actor and active membership when an actor reference is present.
        """

        if actor is None:
            return

        row = await ActorRecord.get_or_none(id=actor, tenant_id=tenant, using_db=connection)
        if row is None:
            raise InteractionError("Actor does not exist.")

        membership_exists = (
            await MembershipRecord.filter(
                actor=actor,
                tenant_id=tenant,
                conversation_id=thread,
                **MembershipVisibility().as_filters(),
            )
            .using_db(connection)
            .exists()
        )
        if not membership_exists:
            raise InteractionError("Actor is not an active member of the thread.")

    async def __require_task_in_thread(
        self,
        *,
        task: str,
        tenant: str,
        thread: str,
        connection: DatabaseConnection,
    ) -> Task:
        """
        Require that a task exists in the expected thread.
        """

        existing = await self.__load_task(tenant=tenant, task=task, connection=connection)

        if existing is None:
            raise InteractionError("Task does not exist.")

        if existing.thread != thread:
            raise InteractionError("Task belongs to a different thread.")

        return existing

    async def __require_message_in_thread(
        self,
        *,
        tenant: str,
        thread: str,
        message: str,
        connection: DatabaseConnection,
    ) -> None:
        """
        Require that a message exists in the expected thread.
        """

        row = (
            await MessageRecord.filter(
                id=message,
                tenant_id=tenant,
                conversation_id=thread,
                **Visibility(archived=True).as_filters(),
            )
            .using_db(connection)
            .get_or_none()
        )
        if row is None:
            raise InteractionError("Message does not exist.")

    def __same_open_task(self, *, task: Task, request: OpenTask, root: str) -> bool:
        """
        Check whether a task request replays an already opened task.
        """

        return (
            task.lineage.root == root
            and task.kind == request.kind
            and task.plan == request.plan
            and task.state == request.state
            and task.thread == request.thread
            and task.metadata == request.metadata
            and task.execution == request.execution
            and task.assignment == request.assignment
            and task.timing.created == request.created
            and task.lineage.parent == request.lineage.parent
            and task.lineage.origin == request.lineage.origin
            and task.identity.tenant == request.identity.tenant
            and task.identity.workspace == request.identity.workspace
        )

    def __same_finish(self, *, task: Task, request: FinishTask) -> bool:
        """
        Check whether a finish request replays an already stored terminal outcome.
        """

        if task.terminal is None:
            return False

        return (
            task.state == request.state
            and task.summary == request.summary
            and task.timing.ended == request.ended
            and task.timing.elapsed == request.elapsed
            and task.terminal.code == request.terminal.code
            and task.terminal.detail == request.terminal.detail
        )

    def __finish_event_kind(self, *, state: TaskState) -> EventKind:
        """
        Convert a terminal task state into its event kind.
        """

        kind = self.__FINISH_EVENT_KINDS.get(state)
        if kind is None:
            raise InteractionError(f"Task state '{state.value}' has no terminal event kind.")

        return kind

    def __task(self, *, row: TaskRecord) -> Task:
        """
        Convert one persistent task model into the interaction schema.
        """

        return Task(
            summary=row.summary,
            deleted_at=row.deleted_at,
            thread=row.conversation_id,
            execution=row.execution_id,
            kind=self.__kind(value=row.kind),
            state=self.__state(value=row.state),
            terminal=self.__terminal(code=row.code, detail=row.detail),
            assignment=Assignment(creator=row.created_by, assignee=row.assignee),
            lineage=Lineage(root=row.root_id or row.id, parent=row.parent_id, origin=row.origin_id),
            timing=Timing(
                elapsed=row.elapsed,
                created_at=row.created_at,
                updated_at=row.updated_at,
                started_at=row.started_at,
                ended_at=row.completed_at,
            ),
            plan=Plan(
                objective=row.objective,
                reference=row.reference,
                plan=self.__metadata(value=row.plan, field="plan"),
                progress=self.__metadata(value=row.progress, field="progress"),
            ),
            metadata=self.__metadata(value=row.metadata, field="metadata"),
            identity=Identity(id=row.id, tenant=row.tenant_id, workspace=row.workspace_id),
        )

    def __kind(self, *, value: str) -> TaskKind:
        """
        Convert stored task kind text into the public enum.
        """

        try:
            return TaskKind(value)
        except ValueError as exception:
            raise InteractionError(f"Unknown task kind in row: {value}.") from exception

    def __state(self, *, value: str) -> TaskState:
        """
        Convert stored task state text into the public enum.
        """

        try:
            return TaskState(value)
        except ValueError as exception:
            raise InteractionError(f"Unknown task state in row: {value}.") from exception

    def __terminal(self, *, code: Optional[str], detail: Optional[str]) -> Optional[Terminal]:
        """
        Convert stored terminal fields into a terminal outcome.
        """

        if code is None:
            return None

        try:
            task_code = TaskCode(code)
        except ValueError as exception:
            raise InteractionError(f"Unknown task terminal code in row: {code}.") from exception

        return Terminal(code=task_code, detail=detail)

    def __metadata(self, *, value: JsonValue, field: str) -> Metadata:
        """
        Convert stored JSON object into metadata.
        """

        if isinstance(value, dict):
            return Metadata(entries=value)

        raise InteractionError(f"Invalid task {field} in row.")
