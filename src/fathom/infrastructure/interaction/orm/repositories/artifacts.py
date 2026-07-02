from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from pydantic import JsonValue
from tortoise.exceptions import IntegrityError

from fathom.constants.collaboration import (
    ArtifactBackend,
    ArtifactKind,
    EventKind,
    EventSource,
    Label,
)
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.orm.models import (
    ArtifactRecord,
    TaskRecord,
)
from fathom.infrastructure.interaction.orm.repositories.lifecycle import (
    DatabaseConnection,
    LifecycleRecorder,
    TransactionScope,
)
from fathom.infrastructure.interaction.orm.repositories.paginator import (
    KeysetPaginator,
    TimestampColumn,
)
from fathom.infrastructure.interaction.orm.repositories.reference import ReferenceGuard
from fathom.schemas.interaction import (
    Artifact,
    ArtifactCursorQuery,
    ArtifactPage,
    ArtifactQuery,
    Identity,
    LinkArtifact,
    Metadata,
    ThreadReference,
    ThreadScope,
    Visibility,
)

if TYPE_CHECKING:
    from datetime import datetime


class ArtifactRepository:
    """
    Repository for durable artifact references.
    """

    def __init__(
        self,
        *,
        references: ReferenceGuard,
        lifecycle: LifecycleRecorder,
        transaction: TransactionScope,
    ) -> None:
        """
        Initialize artifact persistence collaborators.
        """

        self.__guard = references
        self.__lifecycle = lifecycle
        self.__transaction = transaction

    async def link_artifact(self, *, request: LinkArtifact) -> Artifact:
        """
        Persist one artifact reference or replay an identical existing artifact.
        """

        try:
            return await self.__link_artifact(request=request)
        except IntegrityError as exception:
            existing = await self.__load_artifact(
                connection=None,
                tenant=request.identity.tenant,
                artifact_id=request.identity.id,
            )
            if existing is not None and self.__same_artifact(request=request, artifact=existing):
                return existing

            raise InteractionError(
                "Artifact insert conflicted with a different row."
            ) from exception

    async def __link_artifact(self, *, request: LinkArtifact) -> Artifact:
        """
        Persist one artifact reference inside one transaction.
        """

        async with self.__transaction.transaction() as connection:
            if existing := await self.__load_artifact(
                connection=connection,
                tenant=request.identity.tenant,
                artifact_id=request.identity.id,
            ):
                if self.__same_artifact(artifact=existing, request=request):
                    return existing

                raise InteractionError("Artifact identity already exists with different content.")

            await self.__guard.active_thread(
                thread=request.thread,
                connection=connection,
                tenant=request.identity.tenant,
            )

            execution = request.execution

            if request.task is not None:
                task_execution = await self.__task_execution(
                    task=request.task,
                    thread=request.thread,
                    connection=connection,
                    tenant=request.identity.tenant,
                )
                if execution is not None and task_execution != execution:
                    raise InteractionError("Artifact execution does not match task execution.")

                execution = task_execution

            elif execution is not None:
                await self.__guard.present_execution(
                    execution=execution,
                    thread=request.thread,
                    connection=connection,
                    tenant=request.identity.tenant,
                )

            else:
                raise InteractionError("Artifact execution is required.")

            if request.producer is not None:
                await self.__guard.active_membership(
                    thread=request.thread,
                    connection=connection,
                    actor=request.producer,
                    tenant=request.identity.tenant,
                )

            await ArtifactRecord.create(
                uri=request.uri,
                mime=request.mime,
                size=request.size,
                using_db=connection,
                task_id=request.task,
                id=request.identity.id,
                execution_id=execution,
                kind=request.kind.value,
                producer=request.producer,
                created_at=request.created,
                updated_by=request.producer,
                created_by=request.producer,
                retention=request.retention,
                backend=request.backend.value,
                conversation_id=request.thread,
                metadata=request.metadata.entries,
                tenant_id=request.identity.tenant,
                workspace_id=request.identity.workspace,
                labels=[label.value for label in request.labels],
            )

            artifact = await self.__load_artifact(
                connection=connection,
                tenant=request.identity.tenant,
                artifact_id=request.identity.id,
            )
            if artifact is None:
                raise InteractionError("Artifact was not persisted.")

            await self.__lifecycle.record(
                task=request.task,
                execution=execution,
                connection=connection,
                thread=request.thread,
                actor=request.producer,
                created=request.created,
                source=EventSource.ARTIFACT,
                tenant=request.identity.tenant,
                kind=EventKind.ARTIFACT_LINKED,
                workspace=request.identity.workspace,
                payload=Metadata(
                    entries={"kind": request.kind.value, "backend": request.backend.value}
                ),
            )

            return artifact

    async def get_artifacts(self, *, query: ArtifactQuery) -> List[Artifact]:
        """
        Load visible tenant-scoped artifacts for one thread and optional task.
        """

        if not await self.__guard.thread_visible(scope=self.__scope(query=query)):
            return []

        queryset = ArtifactRecord.filter(
            tenant_id=query.tenant,
            conversation_id=query.thread,
            **Visibility(deleted=query.include_deleted, archived=True).as_filters(),
        )
        if query.task is not None:
            queryset = queryset.filter(task_id=query.task)

        rows = await queryset.order_by("created_at", "id")
        return [self.__artifact(row=row) for row in rows]

    async def list_artifacts(self, *, query: ArtifactCursorQuery) -> ArtifactPage:
        """
        Load visible artifacts with keyset pagination.
        """

        if not await self.__guard.thread_visible(scope=self.__scope(query=query)):
            return ArtifactPage(items=(), next=None, total=0)

        queryset = ArtifactRecord.filter(
            tenant_id=query.tenant,
            conversation_id=query.thread,
            **Visibility(deleted=query.include_deleted, archived=True).as_filters(),
        )
        if query.task is not None:
            queryset = queryset.filter(task_id=query.task)
        if query.producer is not None:
            queryset = queryset.filter(producer=query.producer)
        if query.since is not None:
            queryset = queryset.filter(created_at__gte=query.since)
        if query.until is not None:
            queryset = queryset.filter(created_at__lt=query.until)
        if query.kinds:
            queryset = queryset.filter(kind__in=tuple(kind.value for kind in query.kinds))

        total = await queryset.count() if query.count_total else 0

        page = await KeysetPaginator[ArtifactRecord, Artifact](
            column=TimestampColumn.CREATED,
        ).paginate(
            queryset=queryset,
            limit=query.limit,
            order=query.order,
            cursor=query.cursor,
            project=self.__page_artifact,
            stamp=self.__artifact_created,
            identity=self.__artifact_identity,
        )

        return ArtifactPage(items=page.items, next=page.next, total=total)

    async def __load_artifact(
        self,
        *,
        tenant: str,
        artifact_id: str,
        connection: Optional[DatabaseConnection],
    ) -> Optional[Artifact]:
        """
        Load one active artifact row by identity.
        """

        queryset = ArtifactRecord.filter(
            tenant_id=tenant,
            id=artifact_id,
            **Visibility(archived=True).as_filters(),
        )

        if connection is not None:
            queryset = queryset.using_db(connection)

        row = await queryset.get_or_none()
        if row is None:
            return None

        return self.__artifact(row=row)

    def __scope(self, *, query: ArtifactQuery | ArtifactCursorQuery) -> ThreadScope:
        """
        Build a thread scope from an artifact read query.
        """

        return ThreadScope(
            reference=ThreadReference(tenant=query.tenant, thread=query.thread),
            visibility=Visibility(
                deleted=query.include_deleted,
                archived=query.include_archived,
            ),
        )

    async def __task_execution(
        self,
        *,
        task: str,
        tenant: str,
        thread: str,
        connection: DatabaseConnection,
    ) -> str:
        """
        Load the execution id for the task producing an artifact.
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
            raise InteractionError("Artifact task does not exist.")

        execution = row.execution_id
        if not isinstance(execution, str):
            raise InteractionError("Artifact task execution id is invalid.")

        return execution

    def __same_artifact(self, *, artifact: Artifact, request: LinkArtifact) -> bool:
        """
        Check whether an artifact request matches an already stored artifact.
        """

        return (
            artifact.uri == request.uri
            and artifact.task == request.task
            and artifact.kind == request.kind
            and artifact.mime == request.mime
            and artifact.size == request.size
            and artifact.thread == request.thread
            and artifact.backend == request.backend
            and artifact.created == request.created
            and artifact.producer == request.producer
            and artifact.metadata == request.metadata
            and artifact.retention == request.retention
            and artifact.identity.tenant == request.identity.tenant
            and artifact.identity.workspace == request.identity.workspace
            and frozenset(artifact.labels) == frozenset(request.labels)
        )

    def __artifact(self, *, row: ArtifactRecord) -> Artifact:
        """
        Convert one persistent artifact model into the interaction schema.
        """

        return Artifact(
            uri=row.uri,
            mime=row.mime,
            size=row.size,
            task=row.task_id,
            producer=row.producer,
            retention=row.retention,
            created_at=row.created_at,
            deleted_at=row.deleted_at,
            thread=row.conversation_id,
            kind=self.__kind(value=row.kind),
            labels=self.__labels(value=row.labels),
            backend=self.__backend(value=row.backend),
            metadata=self.__metadata(value=row.metadata, field="metadata"),
            identity=Identity(id=row.id, tenant=row.tenant_id, workspace=row.workspace_id),
        )

    def __page_artifact(self, row: ArtifactRecord) -> Artifact:
        """
        Convert one artifact row for pagination.
        """

        return self.__artifact(row=row)

    def __artifact_created(self, artifact: Artifact) -> datetime:
        """
        Return the artifact creation timestamp used by pagination.
        """

        return artifact.created

    def __artifact_identity(self, artifact: Artifact) -> str:
        """
        Return the artifact identifier used by pagination.
        """

        return artifact.identity.id

    def __kind(self, *, value: str) -> ArtifactKind:
        """
        Convert stored artifact kind text into the public enum.
        """

        try:
            return ArtifactKind(value)
        except ValueError as exception:
            raise InteractionError(f"Unknown artifact kind in row: {value}.") from exception

    def __backend(self, *, value: str) -> ArtifactBackend:
        """
        Convert stored artifact backend text into the public enum.
        """

        try:
            return ArtifactBackend(value)
        except ValueError as exception:
            raise InteractionError(f"Unknown artifact backend in row: {value}.") from exception

    def __labels(self, *, value: JsonValue) -> Tuple[Label, ...]:
        """
        Convert stored label strings into label enums.
        """

        if not isinstance(value, list):
            raise InteractionError("Invalid artifact labels in row.")

        labels: List[str] = []

        for label in value:
            if not isinstance(label, str):
                raise InteractionError("Invalid artifact labels in row.")

            labels.append(label)

        try:
            return tuple(Label(label) for label in labels)
        except ValueError as exception:
            raise InteractionError("Unknown artifact label in row.") from exception

    def __metadata(self, *, value: JsonValue, field: str) -> Metadata:
        """
        Convert stored JSON object into metadata.
        """

        if isinstance(value, dict):
            return Metadata(entries=value)

        raise InteractionError(f"Invalid artifact {field} metadata in row.")
