from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple, Type, TypeVar, cast

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
    from sqlite3 import Row


EnumT = TypeVar("EnumT", bound=Enum)


class RowMapper:
    """
    Maps SQLite rows into interaction Pydantic entities.
    """

    def thread(self, *, row: Row) -> Thread:
        """
        Convert a SQLite thread row to a Thread model.
        """

        return Thread(
            identity=Identity(
                id=str(row["id"]),
                tenant=str(row["tenant"]),
                workspace=self.__text(value=row["workspace"]),
            ),
            title=self.__text(value=row["title"]),
            state=self.__enum(enum_class=ThreadState, value=row["state"], field="state"),
            digest=self.__text(value=row["digest"]),
            cursor=self.__integer(value=row["cursor"]),
            creator=self.__text(value=row["creator"]),
            timing=Timing(
                created_at=self.__time(value=row["created_at"]),
                updated_at=self.__time(value=row["updated_at"]),
            ),
            archived_at=self.__optional_time(value=row["archived_at"]),
            deleted_at=self.__optional_time(value=row["deleted_at"]),
            metadata=self.__metadata(value=row["metadata"]),
        )

    def actor(self, *, row: Row) -> Actor:
        """
        Convert a SQLite actor row to an Actor model.
        """

        return Actor(
            identity=Identity(
                id=str(row["id"]),
                tenant=str(row["tenant"]),
                workspace=self.__text(value=row["workspace"]),
            ),
            kind=self.__enum(enum_class=ActorKind, value=row["kind"], field="kind"),
            name=str(row["name"]),
            external=self.__text(value=row["external"]),
            runtime=Runtime(
                kind=self.__text(value=row["runtime"]),
                provider=self.__text(value=row["provider"]),
                model=self.__text(value=row["model"]),
            ),
            skills=self.__metadata(value=row["skills"]),
            timing=Timing(
                created_at=self.__time(value=row["created_at"]),
                updated_at=self.__time(value=row["updated_at"]),
            ),
            metadata=self.__metadata(value=row["metadata"]),
        )

    def membership(self, *, row: Row) -> Membership:
        """
        Convert a SQLite membership row to a Membership model.
        """

        return Membership(
            identity=Identity(
                id=str(row["id"]),
                tenant=str(row["tenant"]),
                workspace=self.__text(value=row["workspace"]),
            ),
            thread=str(row["thread"]),
            actor=str(row["actor"]),
            role=self.__enum(enum_class=MembershipRole, value=row["role"], field="role"),
            scope=self.__enum(enum_class=MembershipScope, value=row["scope"], field="scope"),
            joined_at=self.__time(value=row["joined_at"]),
            departed_at=self.__optional_time(value=row["departed_at"]),
            metadata=self.__metadata(value=row["metadata"]),
        )

    def task(self, *, row: Row) -> Task:
        """
        Convert a SQLite task row to a Task model.
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
                parent=self.__text(value=row["parent"]),
                root=self.__text(value=row["root"]),
                origin=self.__text(value=row["origin"]),
            ),
            kind=self.__enum(enum_class=TaskKind, value=row["kind"], field="kind"),
            state=self.__enum(enum_class=TaskState, value=row["state"], field="state"),
            plan=Plan(
                objective=str(row["objective"]),
                reference=self.__text(value=row["reference"]),
                plan=self.__metadata(value=row["plan"]),
                progress=self.__metadata(value=row["progress"]),
            ),
            terminal=terminal,
            summary=self.__text(value=row["summary"]),
            timing=Timing(
                created_at=self.__time(value=row["created_at"]),
                updated_at=self.__time(value=row["updated_at"]),
                started_at=self.__optional_time(value=row["started_at"]),
                ended_at=self.__optional_time(value=row["ended_at"]),
                elapsed=self.__integer(value=row["elapsed"]),
            ),
            deleted_at=self.__optional_time(value=row["deleted_at"]),
            metadata=self.__metadata(value=row["metadata"]),
        )

    def message(self, *, row: Row) -> Message:
        """
        Convert a SQLite message row to a Message model.
        """

        return Message(
            identity=Identity(
                id=str(row["id"]),
                tenant=str(row["tenant"]),
                workspace=self.__text(value=row["workspace"]),
            ),
            thread=str(row["thread"]),
            task=self.__text(value=row["task"]),
            author=str(row["author"]),
            reply=self.__text(value=row["reply"]),
            sequence=int(row["sequence"]),
            kind=self.__enum(enum_class=MessageKind, value=row["kind"], field="kind"),
            audience=self.__enum(enum_class=Audience, value=row["audience"], field="audience"),
            content=Content(
                body=self.__json(value=row["body"]),
                labels=self.__labels(value=row["labels"]),
                sanitizer=self.__text(value=row["sanitizer"]),
                sanitized_at=self.__optional_time(value=row["sanitized_at"]),
            ),
            created_at=self.__time(value=row["created_at"]),
            deleted_at=self.__optional_time(value=row["deleted_at"]),
            metadata=self.__metadata(value=row["metadata"]),
        )

    def event(self, *, row: Row) -> Event:
        """
        Convert a SQLite event row to an Event model.
        """

        return Event(
            identity=Identity(
                id=str(row["id"]),
                tenant=str(row["tenant"]),
                workspace=self.__text(value=row["workspace"]),
            ),
            thread=str(row["thread"]),
            task=self.__text(value=row["task"]),
            actor=self.__text(value=row["actor"]),
            sequence=int(row["sequence"]),
            kind=self.__enum(enum_class=EventKind, value=row["kind"], field="kind"),
            source=self.__enum(enum_class=EventSource, value=row["source"], field="source"),
            payload=self.__metadata(value=row["payload"]),
            created_at=self.__time(value=row["created_at"]),
            metadata=self.__metadata(value=row["metadata"]),
        )

    def artifact(self, *, row: Row) -> Artifact:
        """
        Convert a SQLite artifact row to an Artifact model.
        """

        return Artifact(
            identity=Identity(
                id=str(row["id"]),
                tenant=str(row["tenant"]),
                workspace=self.__text(value=row["workspace"]),
            ),
            thread=str(row["thread"]),
            task=self.__text(value=row["task"]),
            producer=self.__text(value=row["producer"]),
            kind=self.__enum(enum_class=ArtifactKind, value=row["kind"], field="kind"),
            uri=str(row["uri"]),
            backend=self.__enum(enum_class=ArtifactBackend, value=row["backend"], field="backend"),
            mime=self.__text(value=row["mime"]),
            size=self.__integer(value=row["size"]),
            retention=self.__text(value=row["retention"]),
            labels=self.__labels(value=row["labels"]),
            created_at=self.__time(value=row["created_at"]),
            deleted_at=self.__optional_time(value=row["deleted_at"]),
            metadata=self.__metadata(value=row["metadata"]),
        )

    def script(self, *, row: Row) -> Script:
        """
        Convert a SQLite script row to a Script model.
        """

        return Script(
            identity=Identity(
                id=str(row["id"]),
                tenant=str(row["tenant"]),
                workspace=self.__text(value=row["workspace"]),
            ),
            thread=str(row["thread"]),
            format=self.__enum(enum_class=ScriptFormat, value=row["format"], field="format"),
            content=str(row["content"]),
            revision=int(row["revision"]),
            task=self.__text(value=row["task"]),
            title=self.__text(value=row["title"]),
            status=self.__enum(enum_class=ScriptStatus, value=row["status"], field="status"),
            artifact=self.__text(value=row["artifact"]),
            created_by=self.__text(value=row["created_by"]),
            updated_by=self.__text(value=row["updated_by"]),
            timing=Timing(
                created_at=self.__time(value=row["created_at"]),
                updated_at=self.__time(value=row["updated_at"]),
            ),
            metadata=self.__metadata(value=row["metadata"]),
            deleted_at=self.__optional_time(value=row["deleted_at"]),
        )

    def script_version(self, *, row: Row) -> ScriptVersion:
        """
        Convert a SQLite script version row to a ScriptVersion model.
        """

        return ScriptVersion(
            identity=Identity(
                id=str(row["id"]),
                tenant=str(row["tenant"]),
                workspace=self.__text(value=row["workspace"]),
            ),
            script=str(row["script"]),
            thread=str(row["thread"]),
            version=int(row["version"]),
            content=str(row["content"]),
            checksum=str(row["checksum"]),
            task=self.__text(value=row["task"]),
            actor=self.__text(value=row["actor"]),
            summary=self.__text(value=row["summary"]),
            artifact=self.__text(value=row["artifact"]),
            source=self.__enum(enum_class=ScriptVersionSource, value=row["source"], field="source"),
            created_at=self.__time(value=row["created_at"]),
            metadata=self.__metadata(value=row["metadata"]),
        )

    def policy(self, *, row: Row) -> Policy:
        """
        Convert a SQLite policy row to a Policy model.
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

    def job(self, *, row: Row) -> Job:
        """
        Convert a SQLite job row to a Job model.
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
            attempts=int(row["attempts"]),
            kind=self.__enum(enum_class=JobKind, value=row["kind"], field="kind"),
            state=self.__enum(enum_class=JobState, value=row["state"], field="state"),
            task=self.__text(value=row["task"]),
            owner=self.__text(value=row["owner"]),
            payload=self.__metadata(value=row["payload"]),
            available_at=self.__time(value=row["available_at"]),
            locked_at=self.__optional_time(value=row["locked_at"]),
            timing=Timing(
                created_at=self.__time(value=row["created_at"]),
                updated_at=self.__time(value=row["updated_at"]),
            ),
            metadata=self.__metadata(value=row["metadata"]),
        )

    def context(self, *, row: Row) -> Context:
        """
        Convert a SQLite context row to a Context model.
        """

        return Context(
            identity=Identity(
                id=str(row["id"]),
                tenant=str(row["tenant"]),
                workspace=self.__text(value=row["workspace"]),
            ),
            thread=str(row["thread"]),
            builder=str(row["builder"]),
            hash=self.__text(value=row["hash"]),
            task=self.__text(value=row["task"]),
            model=self.__text(value=row["model"]),
            consumer=self.__text(value=row["consumer"]),
            purpose=self.__enum(enum_class=ContextPurpose, value=row["purpose"], field="purpose"),
            budget=self.__metadata(value=row["budget"]),
            provider=self.__text(value=row["provider"]),
            filters=self.__metadata(value=row["filters"]),
            created_at=self.__time(value=row["created_at"]),
            metadata=self.__metadata(value=row["metadata"]),
            references=self.__references(value=row["references"]),
            expires_at=self.__optional_time(value=row["expires_at"]),
        )

    def __references(self, *, value: str) -> References:
        """
        Parse a JSON object column into References.
        """

        parsed = cast("JsonValue", json.loads(value))
        if not isinstance(parsed, dict):
            return References()

        memories: List[MemoryReference] = []
        raw = parsed.get("memories")

        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and "system" in item and "reference" in item:
                    memories.append(
                        MemoryReference(
                            system=str(item["system"]), reference=str(item["reference"])
                        )
                    )

        def __strings(key: str) -> Tuple[str, ...]:
            """ """

            entries = parsed.get(key)
            if not isinstance(entries, list):
                return ()

            return tuple(str(item) for item in entries)

        return References(
            memories=tuple(memories),
            events=__strings("events"),
            messages=__strings("messages"),
            artifacts=__strings("artifacts"),
        )

    def idempotency(self, *, row: Row) -> Idempotency:
        """
        Convert a SQLite requests row to an Idempotency model.
        """

        response: Optional[JsonValue] = None
        raw = self.__text(value=row["response"])
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

    def __metadata(self, *, value: object) -> Metadata:
        """
        Parse a JSON object column into Metadata.
        """

        parsed = value if isinstance(value, dict) else cast("JsonValue", json.loads(str(value)))

        if not isinstance(parsed, dict):
            return Metadata()

        return Metadata(entries=parsed)

    def __labels(self, *, value: object) -> Tuple[Label, ...]:
        """
        Parse a JSON array column into labels.
        """

        parsed = value if isinstance(value, list) else cast("JsonValue", json.loads(str(value)))
        if not isinstance(parsed, list):
            return ()

        return tuple(self.__enum(enum_class=Label, value=item, field="label") for item in parsed)

    def __json(self, *, value: object) -> JsonValue:
        """
        Parse a JSON value column.
        """

        if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
            if isinstance(value, str):
                return cast("JsonValue", json.loads(value))

            return cast("JsonValue", value)

        return cast("JsonValue", json.loads(str(value)))

    def __time(self, *, value: object) -> datetime:
        """
        Parse a required ISO-8601 datetime value.
        """

        if isinstance(value, datetime):
            return value

        return datetime.fromisoformat(str(value))

    def __optional_time(self, *, value: object | None) -> Optional[datetime]:
        """
        Parse an optional ISO-8601 datetime value.
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

    def __integer(self, *, value: Optional[int]) -> Optional[int]:
        """
        Convert an optional database value to an integer.
        """

        return value

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
