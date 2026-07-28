from __future__ import annotations

from typing import Optional

from fathom.constants.state import TERMINAL_COMPLETION_REASONS, CompletionReason, RunOutcome
from fathom.constants.turn.termination import TerminationStatus


class TerminationResolver:
    """
    Resolves the honest terminal status from the run outcome and completion reason, in one place.
    """

    def resolve(self, *, outcome: RunOutcome, reason: Optional[str]) -> TerminationStatus:
        """
        Return the terminal status for one run.
        """

        if outcome is RunOutcome.CANCELLED:
            return TerminationStatus.CANCELLED

        if outcome is RunOutcome.FAILED:
            return TerminationStatus.FAILED

        if reason == CompletionReason.INTERVENTION_REQUIRED.value:
            return TerminationStatus.NEEDS_INPUT

        if reason == CompletionReason.UNSATISFIABLE.value:
            return TerminationStatus.UNSATISFIABLE

        if reason in TERMINAL_COMPLETION_REASONS:
            return TerminationStatus.FAILED

        return TerminationStatus.COMPLETED
