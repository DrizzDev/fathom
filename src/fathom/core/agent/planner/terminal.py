from __future__ import annotations

from fathom.constants.state import CompletionReason
from fathom.core.agent.state import AgentState


class TerminalPolicy:
    """
    Picks the most specific completion reason for a state that can no longer continue.
    """

    @staticmethod
    def reason(*, state: AgentState) -> str:
        """
        Return the terminal completion reason for the non-continuing state.
        """

        if state.is_complete:
            return state.completion_reason or CompletionReason.SUCCESS.value

        if state.retries.planner.exhausted:
            return CompletionReason.RETRY_BUDGET_EXHAUSTED.value

        if state.is_stuck:
            return CompletionReason.STUCK.value

        return CompletionReason.FAILED.value
