from __future__ import annotations

import unittest

from fathom.core.recovery.strategies.escalation import HumanEscalationRecovery
from fathom.core.recovery.types import EscalateOutcome, NoopOutcome, RecoveryTrigger
from fathom.schemas.escape import EscapeCategory
from fathom.schemas.supervision import BlockReason

from ._fixtures import escape, request


class HumanEscalationRecoveryTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins for the human-escalation strategy.
    """

    async def test_escalates_on_unsafe_block_reason(self) -> None:
        """
        Strategy must escalate when the supervisor flags an unsafe action.
        """

        recovery_request = request(block_reason=BlockReason.UNSAFE_ACTION)

        outcome = await HumanEscalationRecovery().recover(request=recovery_request)

        assert isinstance(outcome, EscalateOutcome)
        self.assertIsInstance(outcome, EscalateOutcome)

        self.assertTrue(outcome.question)

    async def test_escalates_on_ambiguous_target(self) -> None:
        """
        Strategy must escalate when the supervisor flags ambiguous candidates.
        """

        recovery_request = request(block_reason=BlockReason.TARGET_AMBIGUOUS)

        outcome = await HumanEscalationRecovery().recover(request=recovery_request)

        self.assertIsInstance(outcome, EscalateOutcome)

    async def test_escalates_on_human_route_escape(self) -> None:
        """
        Strategy must escalate when the escape report's category routes to the human.
        """

        recovery_request = request(
            escape_report=escape(EscapeCategory.UNSAFE_ACTION, detail="delete is irreversible"),
        )

        outcome = await HumanEscalationRecovery().recover(request=recovery_request)

        assert isinstance(outcome, EscalateOutcome)
        self.assertIsInstance(outcome, EscalateOutcome)
        self.assertEqual(outcome.question, "delete is irreversible")

    async def test_declines_when_no_escalation_signal(self) -> None:
        """
        Strategy must defer when neither block reason nor escape report routes to human.
        """

        recovery_request = request(block_reason=BlockReason.REPEATED_NO_EFFECT)

        outcome = await HumanEscalationRecovery().recover(request=recovery_request)

        self.assertIsInstance(outcome, NoopOutcome)

    def test_supports_expected_triggers(self) -> None:
        """
        Strategy must opt-in to the documented trigger set.
        """

        strategy = HumanEscalationRecovery()
        self.assertFalse(strategy.supports(trigger=RecoveryTrigger.NO_PROGRESS))
        self.assertTrue(strategy.supports(trigger=RecoveryTrigger.LOOP_DETECTED))
        self.assertTrue(strategy.supports(trigger=RecoveryTrigger.VERIFY_REJECTED))
