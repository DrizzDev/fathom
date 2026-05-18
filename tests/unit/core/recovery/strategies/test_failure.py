from __future__ import annotations

import unittest

from fathom.core.recovery.strategies.failure import BoundedFailureRecovery
from fathom.core.recovery.types import BoundedFailureOutcome, RecoveryTrigger
from fathom.schemas.supervision import BlockReason

from ._fixtures import request


class BoundedFailureRecoveryTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins for the bounded-failure terminal strategy.
    """

    async def test_emits_diagnostic_for_every_trigger(self) -> None:
        """
        Strategy must emit a bounded-failure outcome regardless of the trigger.
        """

        recovery_request = request(
            reason="exhausted retries",
            stuck_sub_goal="Tap on Continue",
            trigger=RecoveryTrigger.NO_PROGRESS,
            block_reason=BlockReason.REPEATED_NO_EFFECT,
        )

        outcome = await BoundedFailureRecovery().recover(request=recovery_request)

        assert isinstance(outcome, BoundedFailureOutcome)
        self.assertIsInstance(outcome, BoundedFailureOutcome)

        self.assertIn("NO_PROGRESS", outcome.diagnostic)
        self.assertIn("Tap on Continue", outcome.diagnostic)
        self.assertIn("repeated_no_effect", outcome.diagnostic)

    def test_supports_every_trigger(self) -> None:
        """
        Strategy must be the terminal fallback for every supported trigger.
        """

        strategy = BoundedFailureRecovery()

        for trigger in RecoveryTrigger:
            self.assertTrue(strategy.supports(trigger=trigger))
