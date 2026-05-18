from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from fathom.core.recovery.strategy import RecoveryStrategy
from fathom.core.recovery.types import (
    BoundedFailureOutcome,
    RecoveryOutcome,
    RecoveryRequest,
    RecoveryTrigger,
)

if TYPE_CHECKING:
    from fathom.core.recovery.factory import RecoveryContext


class BoundedFailureRecovery(RecoveryStrategy):
    """
    Terminates the run with a structured diagnostic when no other strategy applies.
    """

    @classmethod
    def build(cls, context: "RecoveryContext") -> "BoundedFailureRecovery":
        """
        Construct a :class:`BoundedFailureRecovery` from the factory context.
        """

        _ = context
        return cls()

    @property
    def name(self) -> str:
        """
        Stable identifier used in configuration and logs.
        """

        return "failure"

    def supports(self, *, trigger: RecoveryTrigger) -> bool:
        """
        Return whether this strategy handles the given trigger.
        """

        _ = trigger

        return True

    async def recover(self, *, request: RecoveryRequest) -> Optional[RecoveryOutcome]:
        """
        Emit a bounded-failure outcome carrying the supplied diagnostic.
        """

        return BoundedFailureOutcome(
            diagnostic=self.__diagnostic_for(request=request),
            summary=f"BoundedFailureRecovery: terminating run for trigger {request.trigger.value}.",
        )

    @staticmethod
    def __diagnostic_for(*, request: RecoveryRequest) -> str:
        """
        Build the structured diagnostic surfaced to telemetry and the audit log.
        """

        block = request.block_reason.value if request.block_reason is not None else "unspecified"

        return (
            f"Recovery exhausted: trigger={request.trigger.value} "
            f"block_reason={block} stuck_sub_goal={request.stuck_sub_goal!r} reason={request.reason}"
        )
