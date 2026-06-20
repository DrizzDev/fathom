from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Protocol, Tuple, Type, TypeVar, cast

from pydantic import JsonValue

from fathom.constants.collaboration import (
    ActorKind,
    ArtifactBackend,
    ArtifactKind,
    Audience,
    ContextPurpose,
    EventKind,
    EventSource,
    IdempotencyState,
    JobCode,
    JobKind,
    JobState,
    Label,
    MembershipRole,
    MembershipScope,
    MessageKind,
    PolicyScope,
    ScriptFormat,
    ScriptStatus,
    ScriptVersionSource,
    TaskCode,
    TaskKind,
    TaskState,
    ThreadState,
)
from fathom.core.exceptions import InteractionError
from fathom.schemas.interaction import (
    Actor,
    Artifact,
    Assignment,
    Content,
    Context,
    Event,
    Governance,
    Idempotency,
    Identity,
    Job,
    Lineage,
    Membership,
    MemoryReference,
    Message,
    Metadata,
    Outcome,
    Plan,
    Policy,
    References,
    Runtime,
    Script,
    ScriptVersion,
    Task,
    Terminal,
    Thread,
    Timing,
)

if TYPE_CHECKING:

    class AsyncpgRecord(Protocol):
        """
        Minimal asyncpg record shape consumed by row mapping.
        """

        def __getitem__(self, key: str | int) -> object:
            """
            Return one column value by name.
            """

            ...
else:
    AsyncpgRecord = object


EnumT = TypeVar("EnumT", bound=Enum)


class PostgresRowMapper:
    """
    Maps native asyncpg.Record rows into interaction Pydantic entities.

    Postgres returns native dict/list values for JSONB columns and native
    timezone-aware datetimes for TIMESTAMPTZ columns (the asyncpg pool
    registers a JSON codec at acquire time so dict/list passes through).
    The mapper therefore avoids json.loads / fromisoformat conversions and only normalizes datetimes to UTC at the boundary.
    """

    def thread(self, *, row: AsyncpgRecord) -> Thread:
        """
        Convert a Postgres threads row to a Thread model.
        """

        return Thread(
            identity=Identity(
                id=str(row["id"]),
                tenant=str(row["tenant"]),
                workspace=self.__text(value=row["workspace"]),
            ),
            state=self.__enum(enum_class=ThreadState, value=row["state"], field="state"),
            title=self.__text(value=row["title"]),
            digest=self.__text(value=row["digest"]),
            creator=self.__text(value=row["creator"]),
            cursor=self.__optional_int(value=row["cursor"]),
            timing=Timing(
                created_at=self.__time(value=row["created_at"]),
                updated_at=self.__time(value=row["updated_at"]),
            ),
            metadata=self.__metadata(value=row["metadata"]),
            deleted_at=self.__optional_time(value=row["deleted_at"]),
            archived_at=self.__optional_time(value=row["archived_at"]),
        )

    def actor(self, *, row: AsyncpgRecord) -> Actor:
        """
        Convert a Postgres actors row to an Actor model.
        """

        return Actor(
            identity=Identity(
                id=str(row["id"]),
                tenant=str(row["tenant"]),
                workspace=self.__text(value=row["workspace"]),
            ),
            name=str(row["name"]),
            kind=self.__enum(enum_class=ActorKind, value=row["kind"], field="kind"),
            external=self.__text(value=row["external"]),
            runtime=Runtime(
                model=self.__text(value=row["model"]),
                kind=self.__text(value=row["runtime"]),
                provider=self.__text(value=row["provider"]),
            ),
            skills=self.__metadata(value=row["skills"]),
            timing=Timing(
                created_at=self.__time(value=row["created_at"]),
                updated_at=self.__time(value=row["updated_at"]),
            ),
            metadata=self.__metadata(value=row["metadata"]),
        )

    def membership(self, *, row: AsyncpgRecord) -> Membership:
        """
        Convert a Postgres memberships row to a Membership model.
        """

        return Membership(
            identity=Identity(
                id=str(row["id"]),
                tenant=str(row["tenant"]),
                workspace=self.__text(value=row["workspace"]),
            ),
            actor=str(row["actor"]),
            thread=str(row["thread"]),
            role=self.__enum(enum_class=MembershipRole, value=row["role"], field="role"),
            scope=self.__enum(enum_class=MembershipScope, value=row["scope"], field="scope"),
            joined_at=self.__time(value=row["joined_at"]),
            metadata=self.__metadata(value=row["metadata"]),
            departed_at=self.__optional_time(value=row["departed_at"]),
        )

    def task(self, *, row: AsyncpgRecord) -> Task:
        """
        Convert a Postgres tasks row to a Task model.
        """

        terminal = None
        code = self.__text(value=row["code"])

        if code is not None:
            terminal = Terminal(
                code=self.__enum(enum_class=TaskCode, value=code, field="code"),
                detail=self.__text(value=row["detail"]),
            )

        return Task(
            identity=Identity(
                id=str(row["id"]),
                tenant=str(row["tenant"]),
                workspace=self.__text(value=row["workspace"]),
            ),
            thread=str(row["thread"]),
            assignment=Assignment(
                creator=self.__text(value=row["creator"]),
                assignee=self.__text(value=row["assignee"]),
            ),
            lineage=Lineage(
                root=self.__text(value=row["root"]),
                parent=self.__text(value=row["parent"]),
                origin=self.__text(value=row["origin"]),
            ),
            kind=self.__enum(enum_class=TaskKind, value=row["kind"], field="kind"),
            state=self.__enum(enum_class=TaskState, value=row["state"], field="state"),
            plan=Plan(
                objective=str(row["objective"]),
                plan=self.__metadata(value=row["plan"]),
                reference=self.__text(value=row["reference"]),
                progress=self.__metadata(value=row["progress"]),
            ),
            terminal=terminal,
            summary=self.__text(value=row["summary"]),
            timing=Timing(
                created_at=self.__time(value=row["created_at"]),
                updated_at=self.__time(value=row["updated_at"]),
                elapsed=self.__optional_int(value=row["elapsed"]),
                ended_at=self.__optional_time(value=row["ended_at"]),
                started_at=self.__optional_time(value=row["started_at"]),
            ),
            metadata=self.__metadata(value=row["metadata"]),
            deleted_at=self.__optional_time(value=row["deleted_at"]),
        )

    def message(self, *, row: AsyncpgRecord) -> Message:
        """
        Convert a Postgres messages row to a Message model.
        """

        return Message(
            identity=Identity(
                id=str(row["id"]),
                tenant=str(row["tenant"]),
                workspace=self.__text(value=row["workspace"]),
            ),
            author=str(row["author"]),
            thread=str(row["thread"]),
            kind=self.__enum(enum_class=MessageKind, value=row["kind"], field="kind"),
            task=self.__text(value=row["task"]),
            reply=self.__text(value=row["reply"]),
            audience=self.__enum(enum_class=Audience, value=row["audience"], field="audience"),
            sequence=self.__integer(value=row["sequence"]),
            content=Content(
                body=self.__json(value=row["body"]),
                labels=self.__labels(value=row["labels"]),
                sanitizer=self.__text(value=row["sanitizer"]),
                sanitized_at=self.__optional_time(value=row["sanitized_at"]),
            ),
            created_at=self.__time(value=row["created_at"]),
            metadata=self.__metadata(value=row["metadata"]),
            deleted_at=self.__optional_time(value=row["deleted_at"]),
        )

    def event(self, *, row: AsyncpgRecord) -> Event:
        """
        Convert a Postgres events row to an Event model.
        """

        return Event(
            identity=Identity(
                id=str(row["id"]),
                tenant=str(row["tenant"]),
                workspace=self.__text(value=row["workspace"]),
            ),
            thread=str(row["thread"]),
            kind=self.__enum(enum_class=EventKind, value=row["kind"], field="kind"),
            task=self.__text(value=row["task"]),
            actor=self.__text(value=row["actor"]),
            source=self.__enum(enum_class=EventSource, value=row["source"], field="source"),
            sequence=self.__integer(value=row["sequence"]),
            payload=self.__metadata(value=row["payload"]),
            created_at=self.__time(value=row["created_at"]),
            metadata=self.__metadata(value=row["metadata"]),
        )

    def artifact(self, *, row: AsyncpgRecord) -> Artifact:
        """
        Convert a Postgres artifacts row to an Artifact model.
        """

        return Artifact(
            identity=Identity(
                id=str(row["id"]),
                tenant=str(row["tenant"]),
                workspace=self.__text(value=row["workspace"]),
            ),
            uri=str(row["uri"]),
            thread=str(row["thread"]),
            task=self.__text(value=row["task"]),
            kind=self.__enum(enum_class=ArtifactKind, value=row["kind"], field="kind"),
            mime=self.__text(value=row["mime"]),
            labels=self.__labels(value=row["labels"]),
            producer=self.__text(value=row["producer"]),
            size=self.__optional_int(value=row["size"]),
            backend=self.__enum(enum_class=ArtifactBackend, value=row["backend"], field="backend"),
            retention=self.__text(value=row["retention"]),
            metadata=self.__metadata(value=row["metadata"]),
            created_at=self.__time(value=row["created_at"]),
            deleted_at=self.__optional_time(value=row["deleted_at"]),
        )

    def script(self, *, row: AsyncpgRecord) -> Script:
        """
        Convert a Postgres scripts row to a Script model.
        """

        return Script(
            identity=Identity(
                id=str(row["id"]),
                tenant=str(row["tenant"]),
                workspace=self.__text(value=row["workspace"]),
            ),
            format=self.__enum(enum_class=ScriptFormat, value=row["format"], field="format"),
            thread=str(row["thread"]),
            content=str(row["content"]),
            task=self.__text(value=row["task"]),
            title=self.__text(value=row["title"]),
            status=self.__enum(enum_class=ScriptStatus, value=row["status"], field="status"),
            artifact=self.__text(value=row["artifact"]),
            revision=self.__integer(value=row["revision"]),
            created_by=self.__text(value=row["created_by"]),
            updated_by=self.__text(value=row["updated_by"]),
            timing=Timing(
                created_at=self.__time(value=row["created_at"]),
                updated_at=self.__time(value=row["updated_at"]),
            ),
            metadata=self.__metadata(value=row["metadata"]),
            deleted_at=self.__optional_time(value=row["deleted_at"]),
        )

    def script_version(self, *, row: AsyncpgRecord) -> ScriptVersion:
        """
        Convert a Postgres script_versions row to a ScriptVersion model.
        """

        return ScriptVersion(
            identity=Identity(
                id=str(row["id"]),
                tenant=str(row["tenant"]),
                workspace=self.__text(value=row["workspace"]),
            ),
            script=str(row["script"]),
            thread=str(row["thread"]),
            content=str(row["content"]),
            checksum=str(row["checksum"]),
            task=self.__text(value=row["task"]),
            actor=self.__text(value=row["actor"]),
            summary=self.__text(value=row["summary"]),
            artifact=self.__text(value=row["artifact"]),
            version=self.__integer(value=row["version"]),
            source=self.__enum(enum_class=ScriptVersionSource, value=row["source"], field="source"),
            created_at=self.__time(value=row["created_at"]),
            metadata=self.__metadata(value=row["metadata"]),
        )

    def policy(self, *, row: AsyncpgRecord) -> Policy:
        """
        Convert a Postgres policies row to a Policy model.
        """

        return Policy(
            identity=Identity(
                id=str(row["id"]),
                tenant=str(row["tenant"]),
                workspace=self.__text(value=row["workspace"]),
            ),
            name=str(row["name"]),
            scope=self.__enum(enum_class=PolicyScope, value=row["scope"], field="scope"),
            region=self.__text(value=row["region"]),
            governance=Governance(
                labels=self.__metadata(value=row["labels"]),
                memories=self.__metadata(value=row["memories"]),
                artifacts=self.__metadata(value=row["artifacts"]),
                retention=self.__metadata(value=row["retention"]),
                sanitizers=self.__metadata(value=row["sanitizers"]),
            ),
            timing=Timing(
                created_at=self.__time(value=row["created_at"]),
                updated_at=self.__time(value=row["updated_at"]),
            ),
            metadata=self.__metadata(value=row["metadata"]),
        )

    def job(self, *, row: AsyncpgRecord) -> Job:
        """
        Convert a Postgres jobs row to a Job model.
        """

        outcome = None
        code = self.__text(value=row["code"])

        if code is not None:
            outcome = Outcome(
                code=self.__enum(enum_class=JobCode, value=code, field="code"),
                detail=self.__text(value=row["detail"]),
            )

        return Job(
            identity=Identity(
                id=str(row["id"]),
                tenant=str(row["tenant"]),
                workspace=self.__text(value=row["workspace"]),
            ),
            outcome=outcome,
            thread=str(row["thread"]),
            kind=self.__enum(enum_class=JobKind, value=row["kind"], field="kind"),
            state=self.__enum(enum_class=JobState, value=row["state"], field="state"),
            task=self.__text(value=row["task"]),
            owner=self.__text(value=row["owner"]),
            payload=self.__metadata(value=row["payload"]),
            attempts=self.__integer(value=row["attempts"]),
            available_at=self.__time(value=row["available_at"]),
            locked_at=self.__optional_time(value=row["locked_at"]),
            timing=Timing(
                created_at=self.__time(value=row["created_at"]),
                updated_at=self.__time(value=row["updated_at"]),
            ),
            metadata=self.__metadata(value=row["metadata"]),
        )

    def context(self, *, row: AsyncpgRecord) -> Context:
        """
        Convert a Postgres contexts row to a Context model.
        """

        return Context(
            identity=Identity(
                id=str(row["id"]),
                tenant=str(row["tenant"]),
                workspace=self.__text(value=row["workspace"]),
            ),
            thread=str(row["thread"]),
            builder=str(row["builder"]),
            task=self.__text(value=row["task"]),
            hash=self.__text(value=row["hash"]),
            model=self.__text(value=row["model"]),
            consumer=self.__text(value=row["consumer"]),
            purpose=self.__enum(enum_class=ContextPurpose, value=row["purpose"], field="purpose"),
            provider=self.__text(value=row["provider"]),
            budget=self.__metadata(value=row["budget"]),
            filters=self.__metadata(value=row["filters"]),
            metadata=self.__metadata(value=row["metadata"]),
            created_at=self.__time(value=row["created_at"]),
            references=self.__references(value=row["references"]),
            expires_at=self.__optional_time(value=row["expires_at"]),
        )

    def idempotency(self, *, row: AsyncpgRecord) -> Idempotency:
        """
        Convert a Postgres requests row to an Idempotency model.
        """

        raw = row["response"]
        response: Optional[JsonValue] = None

        if raw is not None:
            response = self.__json(value=raw)

        return Idempotency(
            response=response,
            key=str(row["key"]),
            hash=str(row["hash"]),
            tenant=str(row["tenant"]),
            state=self.__enum(enum_class=IdempotencyState, value=row["state"], field="state"),
            created_at=self.__time(value=row["created_at"]),
            expires_at=self.__time(value=row["expires_at"]),
            metadata=self.__metadata(value=row["metadata"]),
        )

    def __references(self, *, value: object) -> References:
        """
        Convert a JSONB references value into a References model.
        """

        if not isinstance(value, dict):
            return References()

        memories: List[MemoryReference] = []
        raw = value.get("memories")

        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and "system" in item and "reference" in item:
                    memories.append(
                        MemoryReference(
                            system=str(item["system"]),
                            reference=str(item["reference"]),
                        )
                    )

        def __strings(key: str) -> Tuple[str, ...]:
            """
            Return one string-reference list from the references payload.
            """

            entries = value.get(key)

            if not isinstance(entries, list):
                return ()

            return tuple(str(item) for item in entries)

        return References(
            memories=tuple(memories),
            events=__strings("events"),
            messages=__strings("messages"),
            artifacts=__strings("artifacts"),
        )

    def __metadata(self, *, value: object) -> Metadata:
        """
        Wrap a JSONB object value as a Metadata model.
        """

        if not isinstance(value, dict):
            return Metadata()

        return Metadata(entries=value)

    def __labels(self, *, value: object) -> Tuple[Label, ...]:
        """
        Wrap a JSONB array value as a tuple of labels.
        """

        if not isinstance(value, list):
            return ()

        return tuple(Label(str(item)) for item in value)

    def __json(self, *, value: object) -> JsonValue:
        """
        Pass a native JSONB value through as a JsonValue.
        """

        if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
            return cast("JsonValue", value)

        return cast("JsonValue", str(value))

    def __time(self, *, value: object) -> datetime:
        """
        Return a timezone-aware datetime from a TIMESTAMPTZ column.
        """

        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)

            return value

        return datetime.fromisoformat(str(value))

    def __optional_time(self, *, value: object) -> Optional[datetime]:
        """
        Return an optional timezone-aware datetime from a TIMESTAMPTZ column.
        """

        if value is None:
            return None

        return self.__time(value=value)

    def __text(self, *, value: object) -> Optional[str]:
        """
        Convert an optional database value to text.
        """

        if value is None:
            return None

        return str(value)

    def __optional_int(self, *, value: object) -> Optional[int]:
        """
        Pass an optional integer through unchanged.
        """

        if value is None:
            return None

        return self.__integer(value=value)

    def __integer(self, *, value: object) -> int:
        """
        Convert a required numeric database value to an integer.
        """

        return int(cast("int", value))

    def __enum(
        self,
        *,
        enum_class: Type[EnumT],
        value: object,
        field: str,
    ) -> EnumT:
        """
        Convert a stored value to its enum form or raise InteractionError.
        """

        try:
            return enum_class(str(value))
        except ValueError as exception:
            raise InteractionError(
                f"Invalid stored {field}={value!r} for {enum_class.__name__}"
            ) from exception
