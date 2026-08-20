from __future__ import annotations

import asyncio
import time
import uuid
from logging import getLogger
from typing import Any, Dict, Optional

from pydantic import JsonValue

from fathom.interfaces.journal import RuntimeJournalPort
from fathom.schemas.events import RuntimeEvent, RuntimeEventKind

logger = getLogger(__name__)


class RuntimeEventEmitter:
    """
    Builds typed runtime events and dispatches them to the journal port.
    """

    def __init__(
        self,
        *,
        workflow_id: str,
        journal: RuntimeJournalPort,
    ) -> None:
        """
        Initialize the emitter with a journal port and run identifier.
        """

        self.__journal = journal
        self.__workflow_id = workflow_id

    async def emit(
        self,
        *,
        step: int,
        payload: JsonValue,
        kind: RuntimeEventKind,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Build a :class:`RuntimeEvent` and dispatch it to the journal.

        ``correlation_id`` is the idempotency key for retried emissions: a stable value yields the
        same event identifier so the journal can deduplicate; ``None`` generates a fresh uuid4.
        """

        identifier = correlation_id if correlation_id is not None else str(uuid.uuid4())
        event = RuntimeEvent(
            step=step,
            kind=kind,
            payload=payload,
            identifier=identifier,
            workflow=self.__workflow_id,
            created=int(time.time() * 1000),
        )

        try:
            await self.__journal.record(event=event)
        except asyncio.CancelledError:
            # Journal write is best-effort; cancellation must propagate
            # so the caller's task tree unwinds cleanly.
            raise
        except Exception as exception:
            logger.warning(
                "Runtime event emission failed",
                extra={
                    **self.context(),
                    "event": "runtime.event.emit.failed",
                    "event.kind": kind.value,
                    "error.message": str(exception),
                },
            )

    def context(self) -> Dict[str, Any]:
        """
        Return shared structured-logging context for emitter callers.
        """

        return {
            "component": "core.runtime.emitter",
            "workflow.id": self.__workflow_id,
        }
