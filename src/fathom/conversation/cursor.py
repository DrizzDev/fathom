from __future__ import annotations

import base64
import json
from datetime import datetime
from typing import Dict, Optional

from fathom.constants.conversation import CURSOR_VERSION
from fathom.core.exceptions import InteractionError


class CompositeTimelineCursor:
    """
    Composite cursor that pins per-kind pagination positions for the timeline.
    """

    __KEYS = ("messages", "events", "artifacts", "contexts")

    def __init__(
        self,
        *,
        messages: Optional[str] = None,
        events: Optional[str] = None,
        artifacts: Optional[str] = None,
        contexts: Optional[str] = None,
    ) -> None:
        """
        Bind one composite pagination position.
        """

        self.__positions: Dict[str, Optional[str]] = {
            "messages": messages,
            "events": events,
            "artifacts": artifacts,
            "contexts": contexts,
        }

    @property
    def messages(self) -> Optional[str]:
        """
        Return the message-page cursor.
        """

        return self.__positions["messages"]

    @property
    def events(self) -> Optional[str]:
        """
        Return the event-page cursor.
        """

        return self.__positions["events"]

    @property
    def artifacts(self) -> Optional[str]:
        """
        Return the artifact-page cursor.
        """

        return self.__positions["artifacts"]

    @property
    def contexts(self) -> Optional[str]:
        """
        Return the context-page cursor.
        """

        return self.__positions["contexts"]

    def is_empty(self) -> bool:
        """
        Return True when no per-kind position is set.
        """

        return all(self.__positions[key] is None for key in self.__KEYS)

    def encode(self) -> str:
        """
        Encode the composite into an opaque URL-safe token.
        """

        payload = json.dumps(
            {"v": CURSOR_VERSION, **{key: self.__positions[key] for key in self.__KEYS}},
            separators=(",", ":"),
            sort_keys=True,
        )
        return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")

    @classmethod
    def decode(cls, *, value: str) -> "CompositeTimelineCursor":
        """
        Decode a composite token, raising InteractionError on malformed input.
        """

        try:
            decoded = base64.urlsafe_b64decode(value.encode("ascii")).decode("utf-8")
            payload = json.loads(decoded)
            if not isinstance(payload, dict):
                raise ValueError("Cursor payload must be an object.")
            if payload.get("v") != CURSOR_VERSION:
                raise ValueError("Unsupported cursor version.")
        except (TypeError, ValueError, json.JSONDecodeError) as exception:
            raise InteractionError("Invalid timeline cursor.") from exception

        return cls(
            messages=cls.__optional(payload=payload, key="messages"),
            events=cls.__optional(payload=payload, key="events"),
            artifacts=cls.__optional(payload=payload, key="artifacts"),
            contexts=cls.__optional(payload=payload, key="contexts"),
        )

    @staticmethod
    def __optional(*, payload: Dict[str, object], key: str) -> Optional[str]:
        """
        Return one optional sub-cursor string from the decoded payload.
        """

        value = payload.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise InteractionError("Invalid timeline cursor.")
        return value


class OpaqueCursor:
    """
    Stable, opaque cursor encoding a timestamp and identifier boundary.
    """

    def __init__(self, *, created: datetime, identifier: str) -> None:
        """
        Bind one pagination boundary.
        """

        self.__created = created
        self.__identifier = identifier

    @property
    def created(self) -> datetime:
        """
        Return the timestamp boundary.
        """

        return self.__created

    @property
    def identifier(self) -> str:
        """
        Return the identifier boundary.
        """

        return self.__identifier

    def encode(self) -> str:
        """
        Encode the boundary into an opaque URL-safe token.
        """

        payload = json.dumps(
            {
                "v": CURSOR_VERSION,
                "c": self.__created.isoformat(),
                "i": self.__identifier,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")

    @classmethod
    def decode(cls, *, value: str) -> OpaqueCursor:
        """
        Decode an opaque token into a cursor boundary.
        """

        try:
            decoded = base64.urlsafe_b64decode(value.encode("ascii")).decode("utf-8")
            payload = json.loads(decoded)
            if not isinstance(payload, dict):
                raise ValueError("Cursor payload must be an object.")
            if payload.get("v") != CURSOR_VERSION:
                raise ValueError("Unsupported cursor version.")
            created = datetime.fromisoformat(str(payload["c"]))
            identifier = str(payload["i"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exception:
            raise InteractionError("Invalid pagination cursor.") from exception

        return cls(created=created, identifier=identifier)
