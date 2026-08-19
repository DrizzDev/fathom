from __future__ import annotations

import unittest

from fathom.constants.state import CompletionReason, RunOutcome
from fathom.constants.turn.termination import TerminationStatus
from fathom.core.agent.termination import TerminationResolver


class TerminationResolverTest(unittest.TestCase):
    """
    Cover the honest-status resolution for every terminal family.
    """

    def setUp(self) -> None:
        """
        Build the resolver under test.
        """

        self.resolver = TerminationResolver()

    def test_cancellation_wins_over_any_reason(self) -> None:
        """
        A cancelled executor resolves CANCELLED regardless of the recorded reason.
        """

        status = self.resolver.resolve(
            outcome=RunOutcome.CANCELLED,
            reason=CompletionReason.STUCK.value,
        )

        self.assertEqual(status, TerminationStatus.CANCELLED)

    def test_unanswered_ask_resolves_needs_input(self) -> None:
        """
        The bounded-HITL soft-fail surfaces as NEEDS_INPUT, not a generic failure.
        """

        status = self.resolver.resolve(
            outcome=RunOutcome.COMPLETED,
            reason=CompletionReason.INTERVENTION_REQUIRED.value,
        )

        self.assertEqual(status, TerminationStatus.NEEDS_INPUT)

    def test_refuted_criterion_resolves_unsatisfiable(self) -> None:
        """
        An observed-refuted criterion surfaces its own honest status.
        """

        status = self.resolver.resolve(
            outcome=RunOutcome.COMPLETED,
            reason=CompletionReason.UNSATISFIABLE.value,
        )

        self.assertEqual(status, TerminationStatus.UNSATISFIABLE)

    def test_terminal_reasons_resolve_failed(self) -> None:
        """
        Legacy terminal reasons keep resolving to FAILED.
        """

        status = self.resolver.resolve(
            outcome=RunOutcome.COMPLETED,
            reason=CompletionReason.STUCK.value,
        )

        self.assertEqual(status, TerminationStatus.FAILED)

    def test_clean_run_resolves_completed(self) -> None:
        """
        A completed run with a success reason resolves COMPLETED.
        """

        status = self.resolver.resolve(
            outcome=RunOutcome.COMPLETED,
            reason=CompletionReason.SUCCESS.value,
        )

        self.assertEqual(status, TerminationStatus.COMPLETED)
