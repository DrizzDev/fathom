from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256

from fathom.constants.collaboration import EventKind, EventSource
from fathom.core.exceptions import InteractionError
from fathom.schemas.interaction import Metadata


class EventDigest:
    """
    Domain collaborator that turns a thread event payload into a canonical digest.
    """

    def compute(
        self,
        *,
        sequence: int,
        kind: EventKind,
        payload: Metadata,
        created: datetime,
        source: EventSource,
    ) -> str:
        """
        Return a stable SHA-256 digest for one event payload.
        """

        if created.tzinfo is None:
            raise InteractionError(
                "EventDigest requires a timezone-aware timestamp; naive datetime received."
            )

        body = json.dumps(
            {
                "kind": kind.value,
                "sequence": sequence,
                "source": source.value,
                "payload": payload.entries,
                "created": created.astimezone(timezone.utc).isoformat(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(body.encode("utf-8"), usedforsecurity=False).hexdigest()
