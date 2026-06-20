from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from pypika import SQLLiteQuery

from fathom.constants.collaboration import EventKind, EventSource
from fathom.constants.storage import SqlParameterStyle
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.pypika.query import (
    CursorPaginatedQuery,
    ParameterizedQuery,
    SortDirection,
)
from fathom.infrastructure.interaction.pypika.query import (
    SortOrder as KeysetSortOrder,
)
from fathom.infrastructure.interaction.pypika.sqlite import tables
from fathom.infrastructure.interaction.pypika.sqlite.repositories.context import StoreContext
from fathom.schemas.interaction import (
    Artifact,
    ArtifactCursorQuery,
    ArtifactPage,
    ArtifactQuery,
    LinkArtifact,
    Metadata,
    SortOrder,
)


class ArtifactRepository:
    """
    Artifact repository: persists and lists artifact references.
    """

    def __init__(self, *, context: StoreContext) -> None:
        """
        Bind shared store context for artifact persistence.
        """

        self.__context = context

    async def link_artifact(self, *, request: LinkArtifact) -> Artifact:
        """
        Persist one artifact reference and its lifecycle event.
        """

        async with self.__context.unit.session() as connection:
            await self.__context._require_thread(
                connection=connection,
                tenant=request.identity.tenant,
                thread=request.thread,
            )
            if request.task is not None:
                await self.__context._require_task_in_thread(
                    connection=connection,
                    tenant=request.identity.tenant,
                    thread=request.thread,
                    task=request.task,
                )
            if request.producer is not None:
                await self.__context._require_actor(
                    connection=connection,
                    tenant=request.identity.tenant,
                    actor=request.producer,
                )
                await self.__context._require_active_membership(
                    connection=connection,
                    tenant=request.identity.tenant,
                    thread=request.thread,
                    actor=request.producer,
                )
            existing = await self.__context._load_artifact(
                connection=connection,
                tenant=request.identity.tenant,
                artifact=request.identity.id,
            )
            if existing is not None:
                if not self.__same_artifact(artifact=existing, request=request):
                    raise InteractionError(
                        "Artifact identity already exists with different content."
                    )

                return existing

            binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
            artifacts = tables.ARTIFACTS
            statement = (
                SQLLiteQuery.into(artifacts)
                .columns(
                    artifacts.id,
                    artifacts.tenant,
                    artifacts.workspace,
                    artifacts.thread,
                    artifacts.task,
                    artifacts.producer,
                    artifacts.kind,
                    artifacts.uri,
                    artifacts.backend,
                    artifacts.mime,
                    artifacts.size,
                    artifacts.retention,
                    artifacts.labels,
                    artifacts.created_at,
                    artifacts.deleted_at,
                    artifacts.metadata,
                )
                .insert(
                    binder.bind(value=request.identity.id),
                    binder.bind(value=request.identity.tenant),
                    binder.bind(value=request.identity.workspace),
                    binder.bind(value=request.thread),
                    binder.bind(value=request.task),
                    binder.bind(value=request.producer),
                    binder.bind(value=request.kind.value),
                    binder.bind(value=request.uri),
                    binder.bind(value=request.backend.value),
                    binder.bind(value=request.mime),
                    binder.bind(value=request.size),
                    binder.bind(value=request.retention),
                    binder.bind(
                        value=self.__context._json(value=[label.value for label in request.labels])
                    ),
                    binder.bind(value=self.__context._time(value=request.created)),
                    binder.bind(value=None),
                    binder.bind(value=self.__context._json(value=request.metadata.entries)),
                )
            )
            sql, parameters = binder.render(query=statement)
            await connection.execute(sql, parameters)
            artifact = await self.__context._load_artifact(
                connection=connection,
                tenant=request.identity.tenant,
                artifact=request.identity.id,
            )
            await self.__context._record_event(
                connection=connection,
                subject=request.identity.id,
                tenant=request.identity.tenant,
                workspace=request.identity.workspace,
                thread=request.thread,
                task=request.task,
                actor=request.producer,
                kind=EventKind.ARTIFACT_LINKED,
                source=EventSource.ARTIFACT,
                payload=Metadata(
                    entries={"kind": request.kind.value, "backend": request.backend.value}
                ),
                created=datetime.now(tz=timezone.utc),
            )

        if artifact is None:
            raise InteractionError("Artifact was not persisted.")

        return artifact

    async def get_artifacts(self, *, query: ArtifactQuery) -> List[Artifact]:
        """
        Load tenant-scoped artifacts for one thread and optional task.
        """

        if query.task is not None:
            return await self.__task_artifacts(query=query)

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
        artifacts = tables.ARTIFACTS
        statement = (
            SQLLiteQuery.from_(artifacts)
            .select(artifacts.star)
            .where(artifacts.tenant == binder.bind(value=query.tenant))
            .where(artifacts.thread == binder.bind(value=query.thread))
            .where(artifacts.deleted_at.isnull())
            .orderby(artifacts.created_at)
            .orderby(artifacts.id)
        )
        sql, parameters = binder.render(query=statement)
        async with (
            self.__context.unit.session() as connection,
            connection.execute(sql, parameters) as cursor,
        ):
            rows = await cursor.fetchall()

        return [self.__context.rows.artifact(row=row) for row in rows]

    async def list_artifacts(self, *, query: ArtifactCursorQuery) -> ArtifactPage:
        """
        Load artifacts with SQL-side cursor pagination.
        """

        direction = (
            SortDirection.DESCENDING if query.order == SortOrder.DESC else SortDirection.ASCENDING
        )
        artifacts = tables.ARTIFACTS
        helper = CursorPaginatedQuery(
            table=artifacts,
            ordering=KeysetSortOrder(
                column="created_at",
                tiebreaker="id",
                direction=direction,
            ),
            parameter_style=SqlParameterStyle.QUESTION_MARK,
        )
        helper.where(artifacts.tenant == helper.bind(value=query.tenant))
        helper.where(artifacts.thread == helper.bind(value=query.thread))
        helper.where(artifacts.deleted_at.isnull())
        if query.task is not None:
            helper.where(artifacts.task == helper.bind(value=query.task))
        if query.producer is not None:
            helper.where(artifacts.producer == helper.bind(value=query.producer))
        if query.kinds:
            helper.where(
                artifacts.kind.isin([helper.bind(value=kind.value) for kind in query.kinds])
            )
        if query.since is not None:
            helper.where(
                artifacts.created_at >= helper.bind(value=self.__context._time(value=query.since))
            )
        if query.until is not None:
            helper.where(
                artifacts.created_at < helper.bind(value=self.__context._time(value=query.until))
            )

        count_sql, count_parameters = helper.count_sql_and_parameters()
        page_sql, page_parameters = helper.page_sql_and_parameters(
            cursor=self.__context._decode_keyset_cursor(value=query.cursor),
            limit=query.limit + 1,
        )

        async with self.__context.unit.session() as connection:
            total = await self.__context._optional_count(
                connection=connection,
                sql=count_sql,
                parameters=count_parameters,
                requested=query.count_total,
            )
            async with connection.execute(page_sql, page_parameters) as cursor_rows:
                rows = await cursor_rows.fetchall()

        items, next_cursor = self.__context._paginate(
            rows=[self.__context.rows.artifact(row=row) for row in rows],
            limit=query.limit,
            timestamp=lambda artifact: artifact.created,
            identifier=lambda artifact: artifact.identity.id,
        )
        return ArtifactPage(items=tuple(items), next=next_cursor, total=total)

    async def __task_artifacts(self, *, query: ArtifactQuery) -> List[Artifact]:
        """
        Load tenant-scoped artifacts for one task.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
        artifacts = tables.ARTIFACTS
        statement = (
            SQLLiteQuery.from_(artifacts)
            .select(artifacts.star)
            .where(artifacts.tenant == binder.bind(value=query.tenant))
            .where(artifacts.thread == binder.bind(value=query.thread))
            .where(artifacts.task == binder.bind(value=query.task))
            .where(artifacts.deleted_at.isnull())
            .orderby(artifacts.created_at)
            .orderby(artifacts.id)
        )
        sql, parameters = binder.render(query=statement)
        async with (
            self.__context.unit.session() as connection,
            connection.execute(sql, parameters) as cursor,
        ):
            rows = await cursor.fetchall()

        return [self.__context.rows.artifact(row=row) for row in rows]

    def __same_artifact(self, *, artifact: Artifact, request: LinkArtifact) -> bool:
        """
        Check whether an artifact request replays an already linked artifact.

        Labels compare as a set; tuple ordering is not significant.
        """

        return (
            artifact.identity.tenant == request.identity.tenant
            and artifact.identity.workspace == request.identity.workspace
            and artifact.thread == request.thread
            and artifact.task == request.task
            and artifact.producer == request.producer
            and artifact.kind == request.kind
            and artifact.uri == request.uri
            and artifact.backend == request.backend
            and artifact.mime == request.mime
            and artifact.size == request.size
            and artifact.retention == request.retention
            and frozenset(artifact.labels) == frozenset(request.labels)
            and artifact.created == request.created
            and artifact.metadata == request.metadata
        )
