from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from fathom.core.recovery.strategy import RecoveryStrategy
from fathom.core.recovery.types import (
    EscalateOutcome,
    NoopOutcome,
    RecoveryOutcome,
    RecoveryRequest,
    RecoveryTrigger,
)
from fathom.schemas.supervision import BlockReason

if TYPE_CHECKING:
    from fathom.core.recovery.factory import RecoveryContext


class HumanEscalationRecovery(RecoveryStrategy):
    """
    Routes unsafe or ambiguous stuck states to the human operator.
    """

    @classmethod
    def build(cls, context: "RecoveryContext") -> "HumanEscalationRecovery":
        """
        Construct a :class:`HumanEscalationRecovery` from the factory context.
        """

        _ = context
        return cls()

    @property
    def name(self) -> str:
        """
        Stable identifier used in configuration and logs.
        """

        return "escalation"

    def supports(self, *, trigger: RecoveryTrigger) -> bool:
        """
        Return whether this strategy handles the given trigger.
        """

        return trigger in (
            RecoveryTrigger.LOOP_DETECTED,
            RecoveryTrigger.REQUEST_REPLAN,
            RecoveryTrigger.ACTION_BLOCKED,
            RecoveryTrigger.VERIFY_REJECTED,
        )

    async def recover(self, *, request: RecoveryRequest) -> Optional[RecoveryOutcome]:
        """
        Surface a question for the human when the situation requires intervention.
        """

        if not self.__should_escalate(request=request):
            return NoopOutcome(
                summary="HumanEscalationRecovery: situation does not require human escalation."
            )

        question = self.__question_for(request=request)

        return EscalateOutcome(
            question=question,
            summary=f"HumanEscalationRecovery: asking the human about {request.stuck_sub_goal[:60]!r}.",
        )

    @staticmethod
    def __should_escalate(*, request: RecoveryRequest) -> bool:
        """
        Return whether the request matches a human-escalation block reason.
        """

        if request.block_reason in (BlockReason.UNSAFE_ACTION, BlockReason.TARGET_AMBIGUOUS):
            return True

        return request.escape_report is not None and request.escape_report.routes_to_human()

    @staticmethod
    def __question_for(*, request: RecoveryRequest) -> str:
        """
        Build the question surfaced to the human operator.
        """

        if request.escape_report is not None:
            return request.escape_report.detail

        return f"I'm stuck on {request.stuck_sub_goal!r}. {request.reason} How should I proceed?"
