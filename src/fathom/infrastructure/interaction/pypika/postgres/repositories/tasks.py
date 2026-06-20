from __future__ import annotations

from typing import List, Optional

from pypika import PostgreSQLQuery

from fathom.constants.collaboration import EventKind, EventSource, TaskState
from fathom.constants.storage import SqlParameterStyle
from fathom.core.exceptions import InteractionError, TaskConflictError
from fathom.infrastructure.interaction.pypika.postgres import tables
from fathom.infrastructure.interaction.pypika.postgres.repositories.context import PostgresContext
from fathom.infrastructure.interaction.pypika.query import ParameterizedQuery
from fathom.schemas.interaction import (
    FinishTask,
    Metadata,
    OpenTask,
    Task,
    TaskOneQuery,
    TaskQuery,
    Timing,
)


class PostgresTaskRepository:
    """
    Postgres task repository: persists and transitions tasks within threads.
    """

    def __init__(self, *, context: PostgresContext) -> None:
        """
        Bind shared Postgres context for task persistence.
        """

        self.__context = context

    async def open_task(self, *, request: OpenTask) -> Task:
        """
        Persist one task in a thread.
        """

        root = request.lineage.root or request.identity.id

        self.__context.lifecycle.validate_task_lineage(
            root=root,
            task=request.identity.id,
            parent=request.lineage.parent,
        )
        timing = Timing(created_at=request.created, updated_at=request.created)

        async with self.__context.unit.session() as connection:
            existing = await self.__context._load_task(
                connection=connection,
                task=request.identity.id,
                tenant=request.identity.tenant,
            )
            if existing is not None:
                if not self.__same_open_task(task=existing, request=request, root=root):
                    raise InteractionError("Task identity already exists with different content.")

                return existing

            await self.__context._require_thread(
                connection=connection,
                thread=request.thread,
                tenant=request.identity.tenant,
            )

            if request.assignment.creator is not None:
                await self.__context._require_actor(
                    connection=connection,
                    tenant=request.identity.tenant,
                    actor=request.assignment.creator,
                )
                await self.__context._require_active_membership(
                    connection=connection,
                    thread=request.thread,
                    tenant=request.identity.tenant,
                    actor=request.assignment.creator,
                )

            if request.assignment.assignee is not None:
                await self.__context._require_actor(
                    connection=connection,
                    tenant=request.identity.tenant,
                    actor=request.assignment.assignee,
                )
                await self.__context._require_active_membership(
                    connection=connection,
                    thread=request.thread,
                    tenant=request.identity.tenant,
                    actor=request.assignment.assignee,
                )

            if request.lineage.parent is not None:
                await self.__context._require_task_in_thread(
                    connection=connection,
                    thread=request.thread,
                    task=request.lineage.parent,
                    tenant=request.identity.tenant,
                )

            if root != request.identity.id:
                await self.__context._require_task_in_thread(
                    task=root,
                    connection=connection,
                    thread=request.thread,
                    tenant=request.identity.tenant,
                )

            if request.lineage.origin is not None:
                await self.__context._require_message_in_thread(
                    connection=connection,
                    thread=request.thread,
                    tenant=request.identity.tenant,
                    message=request.lineage.origin,
                )

            binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
            tasks_table = tables.TASKS
            started_at = (
                self.__context._time(value=request.created)
                if request.state == TaskState.RUNNING
                else None
            )
            statement = (
                PostgreSQLQuery.into(tasks_table)
                .columns(
                    tasks_table.id,
                    tasks_table.tenant,
                    tasks_table.workspace,
                    tasks_table.thread,
                    tasks_table.creator,
                    tasks_table.assignee,
                    tasks_table.parent,
                    tasks_table.root,
                    tasks_table.origin,
                    tasks_table.kind,
                    tasks_table.objective,
                    tasks_table.reference,
                    tasks_table.state,
                    tasks_table.code,
                    tasks_table.detail,
                    tasks_table.progress,
                    tasks_table.plan,
                    tasks_table.summary,
                    tasks_table.started_at,
                    tasks_table.ended_at,
                    tasks_table.elapsed,
                    tasks_table.created_at,
                    tasks_table.updated_at,
                    tasks_table.deleted_at,
                    tasks_table.metadata,
                )
                .insert(
                    binder.bind(value=request.identity.id),
                    binder.bind(value=request.identity.tenant),
                    binder.bind(value=request.identity.workspace),
                    binder.bind(value=request.thread),
                    binder.bind(value=request.assignment.creator),
                    binder.bind(value=request.assignment.assignee),
                    binder.bind(value=request.lineage.parent),
                    binder.bind(value=root),
                    binder.bind(value=request.lineage.origin),
                    binder.bind(value=request.kind.value),
                    binder.bind(value=request.plan.objective),
                    binder.bind(value=request.plan.reference),
                    binder.bind(value=request.state.value),
                    binder.bind(value=None),
                    binder.bind(value=None),
                    binder.bind(value=self.__context._json(value=request.plan.progress.entries)),
                    binder.bind(value=self.__context._json(value=request.plan.plan.entries)),
                    binder.bind(value=None),
                    binder.bind(value=started_at),
                    binder.bind(value=None),
                    binder.bind(value=None),
                    binder.bind(value=self.__context._time(value=timing.created)),
                    binder.bind(value=self.__context._time(value=timing.updated)),
                    binder.bind(value=None),
                    binder.bind(value=self.__context._json(value=request.metadata.entries)),
                )
            )
            sql, parameters = binder.render(query=statement)
            await connection.execute(sql, parameters)

            task = await self.__context._load_task(
                connection=connection,
                task=request.identity.id,
                tenant=request.identity.tenant,
            )

            await self.__context._record_event(
                connection=connection,
                thread=request.thread,
                created=request.created,
                task=request.identity.id,
                kind=EventKind.TASK_OPENED,
                subject=request.identity.id,
                tenant=request.identity.tenant,
                source=EventSource.INTERACTION,
                workspace=request.identity.workspace,
                actor=request.assignment.assignee or request.assignment.creator,
                payload=Metadata(
                    entries={"kind": request.kind.value, "state": request.state.value}
                ),
            )

        if task is None:
            raise InteractionError("Task was not persisted.")

        return task

    async def finish_task(self, *, request: FinishTask) -> Task:
        """
        Move one task to a terminal state.
        """

        async with self.__context.unit.session() as connection:
            task = await self.__context._load_task(
                task=request.task,
                connection=connection,
                tenant=request.tenant,
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

            self.__context.lifecycle.validate_task_transition(
                current=task.state,
                target=request.state,
            )
            binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
            tasks_table = tables.TASKS
            statement = (
                PostgreSQLQuery.update(tasks_table)
                .set(tasks_table.state, binder.bind(value=request.state.value))
                .set(tasks_table.code, binder.bind(value=request.terminal.code.value))
                .set(tasks_table.detail, binder.bind(value=request.terminal.detail))
                .set(tasks_table.summary, binder.bind(value=request.summary))
                .set(
                    tasks_table.ended_at,
                    binder.bind(value=self.__context._time(value=request.ended)),
                )
                .set(tasks_table.elapsed, binder.bind(value=request.elapsed))
                .set(
                    tasks_table.updated_at,
                    binder.bind(value=self.__context._time(value=request.ended)),
                )
                .where(tasks_table.tenant == binder.bind(value=request.tenant))
                .where(tasks_table.id == binder.bind(value=request.task))
            )
            sql, parameters = binder.render(query=statement)
            await connection.execute(sql, parameters)

            finished = await self.__context._load_task(
                task=request.task,
                connection=connection,
                tenant=request.tenant,
            )
            await self.__context._record_event(
                task=request.task,
                thread=task.thread,
                connection=connection,
                subject=request.task,
                tenant=request.tenant,
                created=request.ended,
                source=EventSource.INTERACTION,
                workspace=task.identity.workspace,
                actor=task.assignment.assignee or task.assignment.creator,
                kind=self.__context._task_event_kind(state=request.state),
                payload=Metadata(entries={"code": request.terminal.code.value}),
            )

        if finished is None:
            raise InteractionError("Task was not updated.")

        return finished

    async def get_tasks(self, *, query: TaskQuery) -> List[Task]:
        """
        Load tenant-scoped tasks for one thread.
        """

        async with (
            self.__context.unit.session() as connection,
            connection.execute(
                """
                SELECT *
                FROM tasks
                WHERE tenant = $1 AND thread = $2 AND deleted_at IS NULL
                ORDER BY created_at ASC, id ASC
                """,
                (query.tenant, query.thread),
            ) as cursor,
        ):
            rows = await cursor.fetchall()

        return [self.__context.rows.task(row=row) for row in rows]

    async def get_task(self, *, query: TaskOneQuery) -> Optional[Task]:
        """
        Load one tenant-scoped task by identifier.
        """

        async with self.__context.unit.session() as connection:
            return await self.__context._load_task(
                task=query.task,
                tenant=query.tenant,
                connection=connection,
            )

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
