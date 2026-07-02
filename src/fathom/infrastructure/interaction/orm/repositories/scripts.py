from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING, Dict, List, Optional

from pydantic import JsonValue
from tortoise.exceptions import IntegrityError

from fathom.constants.collaboration import (
    ScriptFormat,
    ScriptStatus,
    ScriptVersionSource,
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
from fathom.infrastructure.interaction.orm.repositories.lifecycle import (
    DatabaseConnection,
    IdentifierSource,
    TransactionScope,
)
from fathom.infrastructure.interaction.orm.repositories.paginator import (
    KeysetPaginator,
    TimestampColumn,
)
from fathom.infrastructure.interaction.orm.repositories.reference import ReferenceGuard
from fathom.schemas.interaction import (
    Identity,
    Metadata,
    SaveScript,
    Script,
    ScriptListQuery,
    ScriptPage,
    ScriptQuery,
    ScriptVersion,
    ScriptVersionQuery,
    ThreadReference,
    ThreadScope,
    Timing,
    Visibility,
)

if TYPE_CHECKING:
    from datetime import datetime


class ScriptRepository:
    """
    Repository for live scripts and immutable script versions.
    """

    def __init__(
        self,
        *,
        references: ReferenceGuard,
        transaction: TransactionScope,
        identifier_source: IdentifierSource,
    ) -> None:
        """
        Initialize script persistence collaborators.
        """

        self.__guard = references
        self.__transaction = transaction
        self.__identifier_source = identifier_source

    async def save_script(self, *, request: SaveScript) -> Script:
        """
        Persist a script or update an existing script with replay semantics.
        """

        try:
            return await self.__save_script(request=request)
        except IntegrityError:
            try:
                return await self.__save_script(request=request)
            except IntegrityError as replay_exception:
                raise InteractionError(
                    "Script save conflicted with another writer."
                ) from replay_exception

    async def __save_script(self, *, request: SaveScript) -> Script:
        """
        Persist or update a script inside one transaction.
        """

        async with self.__transaction.transaction() as connection:
            existing = await self.__locked_script(
                connection=connection,
                script=request.identity.id,
                tenant=request.identity.tenant,
            )

            if existing is not None and existing.deleted is not None:
                raise InteractionError("Script identity already belongs to a deleted script.")

            if existing is not None and self.__same_live_script(script=existing, request=request):
                return existing

            execution_id = await self.__require_references(request=request, connection=connection)

            if existing is None:
                await self.__insert_script(
                    request=request, connection=connection, execution_id=execution_id
                )
                await self.__insert_version(
                    version=1, request=request, connection=connection, script=request.identity.id
                )
            else:
                await self.__apply_replay(request=request, existing=existing, connection=connection)

            saved = await self.__load_script(
                include_deleted=False,
                connection=connection,
                script=request.identity.id,
                tenant=request.identity.tenant,
            )
            if saved is None:
                raise InteractionError("Script was not persisted.")

        return saved

    async def get_scripts(self, *, query: ScriptQuery) -> List[Script]:
        """
        Load scripts by identity or conversation filters.
        """

        if query.thread is not None and not await self.__guard.thread_visible(
            scope=self.__scope(
                tenant=query.tenant,
                thread=query.thread,
                deleted=query.include_deleted,
                archived=query.include_archived,
            )
        ):
            return []

        if query.artifact is not None:
            return []

        queryset = ScriptRecord.filter(
            tenant_id=query.tenant,
            **Visibility(deleted=query.include_deleted, archived=True).as_filters(),
        )
        if query.task is not None:
            queryset = queryset.filter(task_id=query.task)
        if query.script is not None:
            queryset = queryset.filter(id=query.script)
        if query.thread is not None:
            queryset = queryset.filter(conversation_id=query.thread)

        rows = await queryset.order_by("-updated_at", "id")
        scripts = [self.__script(row=row) for row in rows]

        if query.thread is not None:
            return scripts

        return [
            script
            for script in scripts
            if await self.__guard.thread_visible(
                scope=self.__scope(
                    tenant=script.identity.tenant,
                    thread=script.thread,
                    deleted=query.include_deleted,
                    archived=query.include_archived,
                )
            )
        ]

    async def get_script_versions(self, *, query: ScriptVersionQuery) -> List[ScriptVersion]:
        """
        Load immutable versions for one visible script.
        """

        script = await self.__load_script(
            connection=None,
            tenant=query.tenant,
            script=query.script,
            include_deleted=query.include_deleted,
        )
        if script is None or not await self.__guard.thread_visible(
            scope=self.__scope(
                tenant=query.tenant,
                thread=script.thread,
                deleted=query.include_deleted,
                archived=query.include_archived,
            )
        ):
            return []

        queryset = ScriptVersionRecord.filter(
            tenant_id=query.tenant,
            script_id=query.script,
        )
        if query.version is not None:
            queryset = queryset.filter(version=query.version)

        rows = await queryset.order_by("version")
        return [self.__version(row=row, script=script) for row in rows]

    async def list_scripts(self, *, query: ScriptListQuery) -> ScriptPage:
        """
        Load visible scripts with keyset pagination ordered by updated timestamp.
        """

        if not await self.__guard.thread_visible(
            scope=self.__scope(
                tenant=query.tenant,
                thread=query.thread,
                deleted=query.include_deleted,
                archived=query.include_archived,
            )
        ):
            return ScriptPage(items=(), next=None, total=0)

        queryset = ScriptRecord.filter(
            tenant_id=query.tenant,
            conversation_id=query.thread,
            **Visibility(deleted=query.include_deleted, archived=True).as_filters(),
        )
        if query.task is not None:
            queryset = queryset.filter(task_id=query.task)
        if query.since is not None:
            queryset = queryset.filter(updated_at__gte=query.since)
        if query.until is not None:
            queryset = queryset.filter(updated_at__lt=query.until)

        total = await queryset.count() if query.count else 0

        page = await KeysetPaginator[ScriptRecord, Script](
            column=TimestampColumn.UPDATED,
        ).paginate(
            queryset=queryset,
            limit=query.limit,
            order=query.order,
            cursor=query.cursor,
            project=self.__page_script,
            stamp=self.__script_updated,
            identity=self.__script_identity,
        )

        return ScriptPage(items=page.items, next=page.next, total=total)

    async def __insert_script(
        self,
        *,
        execution_id: str,
        request: SaveScript,
        connection: DatabaseConnection,
    ) -> None:
        """
        Insert the live script row.
        """

        await ScriptRecord.create(
            revision=1,
            title=request.title,
            using_db=connection,
            task_id=request.task,
            id=request.identity.id,
            content=request.content,
            created_by=request.actor,
            updated_by=request.actor,
            execution_id=execution_id,
            created_at=request.created,
            format=request.format.value,
            status=request.status.value,
            conversation_id=request.thread,
            tenant_id=request.identity.tenant,
            metadata=request.metadata.entries,
            workspace_id=request.identity.workspace,
            checksum=self.__checksum(content=request.content),
        )

    async def __apply_replay(
        self,
        *,
        existing: Script,
        request: SaveScript,
        connection: DatabaseConnection,
    ) -> None:
        """
        Apply replay/update semantics to an existing live script.
        """

        if existing.thread != request.thread:
            raise InteractionError("Script identity already exists in a different thread.")

        if existing.task != request.task:
            raise InteractionError("Script identity already exists for a different task.")

        content_changed = existing.content != request.content
        revision = existing.revision + 1 if content_changed else existing.revision

        if content_changed:
            await self.__insert_version(
                request=request,
                version=revision,
                connection=connection,
                script=request.identity.id,
            )

        await (
            ScriptRecord.filter(
                id=request.identity.id,
                tenant_id=request.identity.tenant,
            )
            .using_db(connection)
            .update(
                revision=revision,
                title=request.title,
                content=request.content,
                updated_by=request.actor,
                updated_at=request.created,
                format=request.format.value,
                status=request.status.value,
                metadata=request.metadata.entries,
                checksum=self.__checksum(content=request.content),
            )
        )

    async def __insert_version(
        self,
        *,
        script: str,
        version: int,
        request: SaveScript,
        connection: DatabaseConnection,
    ) -> None:
        """
        Insert one immutable script version.
        """

        await ScriptVersionRecord.create(
            version=version,
            script_id=script,
            using_db=connection,
            actor=request.actor,
            content=request.content,
            summary=request.summary,
            updated_by=request.actor,
            created_by=request.actor,
            created_at=request.created,
            source=request.source.value,
            metadata=request.metadata.entries,
            tenant_id=request.identity.tenant,
            id=self.__identifier_source.next(),
            workspace_id=request.identity.workspace,
            checksum=self.__checksum(content=request.content),
        )

    async def __locked_script(
        self,
        *,
        tenant: str,
        script: str,
        connection: DatabaseConnection,
    ) -> Optional[Script]:
        """
        Load one script row with a write lock for replay/update.
        """

        row = await (
            ScriptRecord.filter(id=script, tenant_id=tenant)
            .using_db(connection)
            .select_for_update()
            .get_or_none()
        )
        if row is None:
            return None

        return self.__script(row=row)

    def __same_live_script(self, *, script: Script, request: SaveScript) -> bool:
        """
        Check whether a save request is an exact replay of the live script row.
        """

        return (
            script.task == request.task
            and script.title == request.title
            and script.format == request.format
            and script.status == request.status
            and script.thread == request.thread
            and script.content == request.content
            and script.updated_by == request.actor
            and script.identity == request.identity
            and script.metadata == request.metadata
        )

    async def __load_script(
        self,
        *,
        tenant: str,
        script: str,
        include_deleted: bool,
        connection: Optional[DatabaseConnection],
    ) -> Optional[Script]:
        """
        Load one script row by identity.
        """

        filters: Dict[str, object] = {"tenant_id": tenant, "id": script}
        if not include_deleted:
            filters["deleted_at__isnull"] = True

        queryset = ScriptRecord.filter(**filters)

        if connection is not None:
            queryset = queryset.using_db(connection)

        row = await queryset.get_or_none()

        if row is None:
            return None

        return self.__script(row=row)

    async def __require_references(
        self,
        *,
        request: SaveScript,
        connection: DatabaseConnection,
    ) -> str:
        """
        Validate every foreign-key referenced by the save request.
        """

        await self.__require_thread(
            thread=request.thread,
            connection=connection,
            tenant=request.identity.tenant,
        )
        execution_id = await self.__require_execution(
            request=request,
            connection=connection,
        )

        if request.actor is not None:
            await self.__require_actor(
                actor=request.actor,
                connection=connection,
                tenant=request.identity.tenant,
            )

        return execution_id

    async def __require_execution(
        self,
        *,
        request: SaveScript,
        connection: DatabaseConnection,
    ) -> str:
        """
        Resolve and validate the execution that owns the script.
        """

        if request.task is not None:
            task_execution = await self.__require_task_in_thread(
                task=request.task,
                thread=request.thread,
                connection=connection,
                tenant=request.identity.tenant,
            )
            if request.execution is not None and request.execution != task_execution:
                raise InteractionError("Script execution does not match the task execution.")

            return task_execution

        if request.execution is None:
            raise InteractionError("Script execution is required when task is absent.")

        await self.__require_execution_in_thread(
            thread=request.thread,
            connection=connection,
            execution=request.execution,
            tenant=request.identity.tenant,
        )

        return request.execution

    async def __require_execution_in_thread(
        self,
        *,
        tenant: str,
        thread: str,
        execution: str,
        connection: DatabaseConnection,
    ) -> None:
        """
        Require a live execution in the target thread before saving a script.
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
            raise InteractionError("Script execution does not exist.")

    def __scope(self, *, tenant: str, thread: str, deleted: bool, archived: bool) -> ThreadScope:
        """
        Build a thread scope from raw script-read arguments.
        """

        return ThreadScope(
            reference=ThreadReference(tenant=tenant, thread=thread),
            visibility=Visibility(deleted=deleted, archived=archived),
        )

    async def __require_thread(
        self,
        *,
        tenant: str,
        thread: str,
        connection: DatabaseConnection,
    ) -> None:
        """
        Require an active thread before saving a script.
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

    async def __require_task_in_thread(
        self,
        *,
        task: str,
        tenant: str,
        thread: str,
        connection: DatabaseConnection,
    ) -> str:
        """
        Require a live task in the target thread before saving a script.
        """

        row = (
            await TaskRecord.filter(
                tenant_id=tenant,
                id=task,
                conversation_id=thread,
                **Visibility(archived=True).as_filters(),
            )
            .using_db(connection)
            .get_or_none()
        )
        if row is None:
            raise InteractionError("Script task does not exist.")

        execution = row.execution_id

        if not isinstance(execution, str):
            raise InteractionError("Script task execution id is invalid.")

        return execution

    async def __require_actor(
        self,
        *,
        actor: str,
        tenant: str,
        connection: DatabaseConnection,
    ) -> None:
        """
        Require an actor before saving it as script editor.
        """

        row = await ActorRecord.get_or_none(id=actor, tenant_id=tenant, using_db=connection)

        if row is None:
            raise InteractionError("Actor does not exist.")

    def __script(self, *, row: ScriptRecord) -> Script:
        """
        Convert one persistent script model into the interaction schema.
        """

        return Script(
            artifact=None,
            title=row.title,
            task=row.task_id,
            content=row.content,
            revision=row.revision,
            deleted_at=row.deleted_at,
            created_by=row.created_by,
            updated_by=row.updated_by,
            thread=row.conversation_id,
            format=self.__format(value=row.format),
            status=self.__status(value=row.status),
            metadata=self.__metadata(value=row.metadata, field="metadata"),
            timing=Timing(created_at=row.created_at, updated_at=row.updated_at),
            identity=Identity(id=row.id, tenant=row.tenant_id, workspace=row.workspace_id),
        )

    def __page_script(self, row: ScriptRecord) -> Script:
        """
        Convert one script row for pagination.
        """

        return self.__script(row=row)

    def __script_updated(self, script: Script) -> datetime:
        """
        Return the script update timestamp used by pagination.
        """

        return script.timing.updated

    def __script_identity(self, script: Script) -> str:
        """
        Return the script identifier used by pagination.
        """

        return script.identity.id

    def __version(self, *, row: ScriptVersionRecord, script: Script) -> ScriptVersion:
        """
        Convert one persistent script version model into the interaction schema.
        """

        return ScriptVersion(
            artifact=None,
            actor=row.actor,
            task=script.task,
            summary=row.summary,
            version=row.version,
            content=row.content,
            script=row.script_id,
            thread=script.thread,
            checksum=row.checksum,
            created_at=row.created_at,
            source=self.__version_source(value=row.source),
            metadata=self.__metadata(value=row.metadata, field="metadata"),
            identity=Identity(id=row.id, tenant=row.tenant_id, workspace=row.workspace_id),
        )

    def __format(self, *, value: str) -> ScriptFormat:
        """
        Convert stored script format text into the public enum.
        """

        try:
            return ScriptFormat(value)
        except ValueError as exception:
            raise InteractionError(f"Unknown script format in row: {value}.") from exception

    def __status(self, *, value: str) -> ScriptStatus:
        """
        Convert stored script status text into the public enum.
        """

        try:
            return ScriptStatus(value)
        except ValueError as exception:
            raise InteractionError(f"Unknown script status in row: {value}.") from exception

    def __version_source(self, *, value: str) -> ScriptVersionSource:
        """
        Convert stored script version source text into the public enum.
        """

        try:
            return ScriptVersionSource(value)
        except ValueError as exception:
            raise InteractionError(f"Unknown script version source in row: {value}.") from exception

    def __metadata(self, *, value: JsonValue, field: str) -> Metadata:
        """
        Convert stored JSON object into metadata.
        """

        if isinstance(value, dict):
            return Metadata(entries=value)

        raise InteractionError(f"Invalid script {field} metadata in row.")

    def __checksum(self, *, content: str) -> str:
        """
        Return a stable SHA-256 checksum for script content.
        """

        return sha256(content.encode("utf-8")).hexdigest()
