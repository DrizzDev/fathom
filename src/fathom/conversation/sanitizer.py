from __future__ import annotations

from datetime import datetime, timezone
from typing import Final, Tuple

from pydantic import JsonValue

from fathom.constants.collaboration import Label


class ContentSanitizer:
    """
    Deterministic content sanitizer applied to every persisted message.

    Pairs with `PrivacyClassifier`: the classifier attaches labels, the
    sanitizer records that the content was processed and returns a body fit for storage.
    The default profile is a no-op that preserves the original body verbatim; future profiles can redact or substitute based on the attached labels without changing the call sites.

    Determinism is required so idempotent message replay produces an identical `(body, sanitizer_id, sanitized_at)` triple.
    The `sanitized_at` value is supplied by the caller (typically the message-creation moment) and never read from the wall clock here.
    """

    __SANITIZER_ID: Final[str] = "noop@1"

    def sanitize(
        self,
        *,
        at: datetime,
        body: JsonValue,
        labels: Tuple[Label, ...],
    ) -> "SanitizedContent":
        """
        Return the sanitized body, sanitizer profile id, and timestamp.

        The no-op profile preserves `body` verbatim regardless of which
        labels are attached. Replacing this method with a redacting
        profile is enough to make every message-write path scrub PII
        without touching the recorder, conversation service, or store.
        """

        del labels

        return SanitizedContent(
            body=body,
            sanitized=at,
            sanitizer=self.__SANITIZER_ID,
        )


class SanitizedContent:
    """
    Result of one sanitization pass: the resulting body plus stamping.
    """

    def __init__(self, *, body: JsonValue, sanitizer: str, sanitized: datetime) -> None:
        """
        Capture the sanitized body, sanitizer profile id, and timestamp.
        """

        self.body = body
        self.sanitizer = sanitizer
        self.sanitized = sanitized

    @classmethod
    def now(cls, *, body: JsonValue, sanitizer: str) -> "SanitizedContent":
        """
        Build a sanitized-content stamp anchored at the current wall clock.
        """

        return cls(body=body, sanitizer=sanitizer, sanitized=datetime.now(tz=timezone.utc))
