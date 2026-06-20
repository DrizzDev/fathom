from __future__ import annotations

from hashlib import sha256
from typing import List, Optional

import aiosqlite
from pypika import Order
from pypika.dialects import SQLLiteQuery

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
    SaveScript,
    Script,
    ScriptListQuery,
    ScriptPage,
    ScriptQuery,
    ScriptVersion,
    ScriptVersionQuery,
    SortOrder,
)


class ScriptRepository:
    """
    Script repository: persists live scripts and immutable versions.
    """

    def __init__(self, *, context: StoreContext) -> None:
        """
        Bind shared store context for script persistence.
        """

        self.__context = context

    async def save_script(self, *, request: SaveScript) -> Script:
        """
        Persist a script atomically and apply replay semantics on conflict.
        """

        async with self.__context.unit.session() as connection:
            await self.__require_references(connection=connection, request=request)

            inserted = await self.__try_insert(connection=connection, request=request)
            if inserted:
                await self.__insert_version(
                    version=1,
                    request=request,
                    script=request.identity.id,
                    connection=connection,
                )
            else:
                await self.__apply_replay(connection=connection, request=request)

        script = await self.__script(
            script=request.identity.id,
            tenant=request.identity.tenant,
        )
        if script is None:
            raise InteractionError("Script was not persisted.")

        return script

    async def get_scripts(self, *, query: ScriptQuery) -> List[Script]:
        """
        Load scripts by identity or conversation filters.
        """

        scripts = tables.SCRIPTS
        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)

        statement = (
            SQLLiteQuery.from_(scripts)
            .select(scripts.star)
            .where(scripts.tenant == binder.bind(value=query.tenant))
        )

        if query.script is not None:
            statement = statement.where(scripts.id == binder.bind(value=query.script))

        if query.task is not None:
            statement = statement.where(scripts.task == binder.bind(value=query.task))

        if query.thread is not None:
            statement = statement.where(scripts.thread == binder.bind(value=query.thread))

        if query.artifact is not None:
            statement = statement.where(scripts.artifact == binder.bind(value=query.artifact))

        if not query.include_deleted:
            statement = statement.where(scripts.deleted_at.isnull())

        statement = statement.orderby(scripts.updated_at, order=Order.desc).orderby(scripts.id)
        sql, parameters = binder.render(query=statement)

        async with (
            self.__context.unit.session() as connection,
            connection.execute(sql, parameters) as cursor,
        ):
            rows = await cursor.fetchall()

        return [self.__context.rows.script(row=row) for row in rows]

    async def get_script_versions(self, *, query: ScriptVersionQuery) -> List[ScriptVersion]:
        """
        Load immutable versions for one script.
        """

        script_versions = tables.SCRIPT_VERSIONS
        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)

        statement = (
            SQLLiteQuery.from_(script_versions)
            .select(script_versions.star)
            .where(script_versions.tenant == binder.bind(value=query.tenant))
            .where(script_versions.script == binder.bind(value=query.script))
        )

        if query.version is not None:
            statement = statement.where(script_versions.version == binder.bind(value=query.version))

        statement = statement.orderby(script_versions.version)
        sql, parameters = binder.render(query=statement)

        async with (
            self.__context.unit.session() as connection,
            connection.execute(sql, parameters) as cursor,
        ):
            rows = await cursor.fetchall()

        return [self.__context.rows.script_version(row=row) for row in rows]

    async def list_scripts(self, *, query: ScriptListQuery) -> ScriptPage:
        """
        Load scripts with SQL-side cursor pagination ordered by updated timestamp.
        """

        direction = (
            SortDirection.DESCENDING if query.order == SortOrder.DESC else SortDirection.ASCENDING
        )

        scripts = tables.SCRIPTS
        helper = CursorPaginatedQuery(
            table=scripts,
            ordering=KeysetSortOrder(
                tiebreaker="id",
                column="updated_at",
                direction=direction,
            ),
            parameter_style=SqlParameterStyle.QUESTION_MARK,
        )

        helper.where(scripts.tenant == helper.bind(value=query.tenant))
        helper.where(scripts.thread == helper.bind(value=query.thread))

        if not query.include_deleted:
            helper.where(scripts.deleted_at.isnull())

        if query.task is not None:
            helper.where(scripts.task == helper.bind(value=query.task))

        if query.since is not None:
            helper.where(
                scripts.updated_at >= helper.bind(value=self.__context._time(value=query.since))
            )

        if query.until is not None:
            helper.where(
                scripts.updated_at < helper.bind(value=self.__context._time(value=query.until))
            )

        count_sql, count_parameters = helper.count_sql_and_parameters()
        page_sql, page_parameters = helper.page_sql_and_parameters(
            limit=query.limit + 1,
            cursor=self.__context._decode_keyset_cursor(value=query.cursor),
        )

        async with self.__context.unit.session() as connection:
            total = await self.__context._optional_count(
                sql=count_sql,
                connection=connection,
                requested=query.count,
                parameters=count_parameters,
            )
            async with connection.execute(page_sql, page_parameters) as cursor_rows:
                rows = await cursor_rows.fetchall()

        items, next_cursor = self.__context._paginate(
            limit=query.limit,
            identifier=lambda script: script.identity.id,
            timestamp=lambda script: script.timing.updated,
            rows=[self.__context.rows.script(row=row) for row in rows],
        )
        return ScriptPage(items=tuple(items), next=next_cursor, total=total)

    async def __require_references(
        self,
        *,
        connection: aiosqlite.Connection,
        request: SaveScript,
    ) -> None:
        """
        Validate every foreign-key referenced by the save request.
        """

        await self.__context._require_thread(
            connection=connection,
            thread=request.thread,
            tenant=request.identity.tenant,
        )

        if request.task is not None:
            await self.__context._require_task_in_thread(
                task=request.task,
                thread=request.thread,
                connection=connection,
                tenant=request.identity.tenant,
            )

        if request.artifact is not None:
            artifact = await self.__context._load_artifact(
                connection=connection,
                artifact=request.artifact,
                tenant=request.identity.tenant,
            )

            if artifact is None:
                raise InteractionError("Script artifact does not exist.")

            if artifact.thread != request.thread:
                raise InteractionError("Script artifact belongs to a different thread.")

        if request.actor is not None:
            await self.__context._require_actor(
                actor=request.actor,
                connection=connection,
                tenant=request.identity.tenant,
            )

    async def __try_insert(
        self,
        *,
        connection: aiosqlite.Connection,
        request: SaveScript,
    ) -> bool:
        """
        Insert atomically using ON CONFLICT DO NOTHING; return True when a row was inserted.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
        scripts = tables.SCRIPTS
        statement = (
            SQLLiteQuery.into(scripts)
            .columns(
                scripts.id,
                scripts.tenant,
                scripts.workspace,
                scripts.thread,
                scripts.task,
                scripts.artifact,
                scripts.title,
                scripts.format,
                scripts.status,
                scripts.content,
                scripts.revision,
                scripts.created_by,
                scripts.updated_by,
                scripts.created_at,
                scripts.updated_at,
                scripts.deleted_at,
                scripts.metadata,
            )
            .insert(
                binder.bind(value=request.identity.id),
                binder.bind(value=request.identity.tenant),
                binder.bind(value=request.identity.workspace),
                binder.bind(value=request.thread),
                binder.bind(value=request.task),
                binder.bind(value=request.artifact),
                binder.bind(value=request.title),
                binder.bind(value=request.format),
                binder.bind(value=request.status.value),
                binder.bind(value=request.content),
                binder.bind(value=1),
                binder.bind(value=request.actor),
                binder.bind(value=request.actor),
                binder.bind(value=self.__context._time(value=request.created)),
                binder.bind(value=self.__context._time(value=request.created)),
                binder.bind(value=None),
                binder.bind(value=self.__context._json(value=request.metadata.entries)),
            )
        )

        sql, parameters = binder.render(query=statement)
        sql = f"{sql} ON CONFLICT (tenant, id) DO NOTHING RETURNING id"

        async with connection.execute(sql, parameters) as cursor:
            row = await cursor.fetchone()

        return row is not None

    async def __apply_replay(
        self,
        *,
        connection: aiosqlite.Connection,
        request: SaveScript,
    ) -> None:
        """
        Bring an existing row to the supplied state, bumping revision only when content changes.
        """

        existing = await self.__script(
            script=request.identity.id,
            tenant=request.identity.tenant,
        )

        if existing is None:
            raise InteractionError("Script vanished during save.")

        if existing.thread != request.thread:
            raise InteractionError("Script identity already exists in a different thread.")

        content_changed = existing.content != request.content
        revision = existing.revision + 1 if content_changed else existing.revision

        if content_changed:
            await self.__insert_version(
                version=revision,
                request=request,
                script=request.identity.id,
                connection=connection,
            )

        await self.__apply_update(
            connection=connection,
            request=request,
            revision=revision,
        )

    async def __apply_update(
        self,
        *,
        connection: aiosqlite.Connection,
        request: SaveScript,
        revision: int,
    ) -> None:
        """
        Update mutable script fields in place.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
        scripts = tables.SCRIPTS
        statement = (
            SQLLiteQuery.update(scripts)
            .set(scripts.revision, binder.bind(value=revision))
            .set(scripts.task, binder.bind(value=request.task))
            .set(scripts.title, binder.bind(value=request.title))
            .set(scripts.format, binder.bind(value=request.format))
            .set(scripts.content, binder.bind(value=request.content))
            .set(scripts.updated_by, binder.bind(value=request.actor))
            .set(scripts.artifact, binder.bind(value=request.artifact))
            .set(scripts.status, binder.bind(value=request.status.value))
            .set(
                scripts.updated_at,
                binder.bind(value=self.__context._time(value=request.created)),
            )
            .set(
                scripts.metadata,
                binder.bind(value=self.__context._json(value=request.metadata.entries)),
            )
            .where(scripts.tenant == binder.bind(value=request.identity.tenant))
            .where(scripts.id == binder.bind(value=request.identity.id))
        )

        sql, parameters = binder.render(query=statement)
        await connection.execute(sql, parameters)

    async def __insert_version(
        self,
        *,
        script: str,
        version: int,
        request: SaveScript,
        connection: aiosqlite.Connection,
    ) -> None:
        """
        Insert one immutable script version.
        """

        script_versions = tables.SCRIPT_VERSIONS
        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)

        statement = (
            SQLLiteQuery.into(script_versions)
            .columns(
                script_versions.id,
                script_versions.tenant,
                script_versions.workspace,
                script_versions.script,
                script_versions.thread,
                script_versions.task,
                script_versions.artifact,
                script_versions.version,
                script_versions.source,
                script_versions.content,
                script_versions.checksum,
                script_versions.summary,
                script_versions.actor,
                script_versions.created_at,
                script_versions.metadata,
            )
            .insert(
                binder.bind(value=f"{script}:v{version}"),
                binder.bind(value=request.identity.tenant),
                binder.bind(value=request.identity.workspace),
                binder.bind(value=script),
                binder.bind(value=request.thread),
                binder.bind(value=request.task),
                binder.bind(value=request.artifact),
                binder.bind(value=version),
                binder.bind(value=request.source.value),
                binder.bind(value=request.content),
                binder.bind(value=self.__checksum(content=request.content)),
                binder.bind(value=request.summary),
                binder.bind(value=request.actor),
                binder.bind(value=self.__context._time(value=request.created)),
                binder.bind(value=self.__context._json(value=request.metadata.entries)),
            )
        )
        sql, parameters = binder.render(query=statement)
        await connection.execute(sql, parameters)

    async def __script(self, *, tenant: str, script: str) -> Optional[Script]:
        """
        Load one script by primary key.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
        scripts = tables.SCRIPTS
        statement = (
            SQLLiteQuery.from_(scripts)
            .select(scripts.star)
            .where(scripts.id == binder.bind(value=script))
            .where(scripts.tenant == binder.bind(value=tenant))
            .where(scripts.deleted_at.isnull())
        )

        sql, parameters = binder.render(query=statement)
        async with (
            self.__context.unit.session() as connection,
            connection.execute(sql, parameters) as cursor,
        ):
            row = await cursor.fetchone()

        if row is None:
            return None

        return self.__context.rows.script(row=row)

    def __checksum(self, *, content: str) -> str:
        """
        Return a stable SHA-256 checksum for script content.
        """

        return sha256(content.encode("utf-8")).hexdigest()
