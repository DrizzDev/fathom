from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.core.recovery.strategies.scroll import ScrollBoundaryRecovery
from fathom.core.recovery.types import NoopOutcome, RecoveryTrigger, TryActionOutcome
from fathom.schemas.supervision import BlockReason

from ._fixtures import bounds, element, observation, request


class ScrollBoundaryRecoveryTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins for the scroll-boundary surfacing strategy.
    """

    async def test_surfaces_visible_call_to_action(self) -> None:
        """
        Strategy must tap the first visible CTA in lieu of continuing scroll.
        """

        cta = element(
            identifier="cta_show",
            text="Show results",
            rect=bounds(x=120, y=900, width=600, height=120),
        )
        recovery_request = request(
            block_reason=BlockReason.NON_SCROLLABLE_SURFACE,
            screen=observation(calls_to_action=(cta,)),
        )

        outcome = await ScrollBoundaryRecovery().recover(request=recovery_request)

        self.assertIsInstance(outcome, TryActionOutcome)
        assert isinstance(outcome, TryActionOutcome)
        self.assertEqual(outcome.action.action_type, ActionType.TAP)
        self.assertEqual(outcome.action.label_id, "cta_show")

    async def test_declines_when_no_cta_present(self) -> None:
        """
        Strategy must defer when no visible CTA is observed.
        """

        recovery_request = request(
            block_reason=BlockReason.NON_SCROLLABLE_SURFACE,
            screen=observation(),
        )

        outcome = await ScrollBoundaryRecovery().recover(request=recovery_request)

        self.assertIsInstance(outcome, NoopOutcome)

    def test_supports_expected_triggers(self) -> None:
        """
        Strategy must opt-in to the documented trigger set.
        """

        strategy = ScrollBoundaryRecovery()
        self.assertTrue(strategy.supports(trigger=RecoveryTrigger.LOOP_DETECTED))
        self.assertTrue(strategy.supports(trigger=RecoveryTrigger.NO_PROGRESS))
        self.assertTrue(strategy.supports(trigger=RecoveryTrigger.ACTION_BLOCKED))
        self.assertTrue(strategy.supports(trigger=RecoveryTrigger.REQUEST_REPLAN))
        self.assertFalse(strategy.supports(trigger=RecoveryTrigger.VERIFY_REJECTED))
