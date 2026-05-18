from __future__ import annotations

import unittest

from fathom.core.recovery.strategies.overlay import OverlayRecovery
from fathom.core.recovery.types import (
    NoopOutcome,
    RecoveryTrigger,
    TryActionOutcome,
)
from fathom.schemas.observation import OverlayObservation
from fathom.schemas.supervision import BlockReason

from ._fixtures import (
    bounds,
    element,
    observation,
    request,
)


class OverlayRecoveryTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins for the deterministic overlay dismissal strategy.
    """

    async def test_proposes_first_unused_dismiss_candidate(self) -> None:
        """
        OverlayRecovery must tap the first dismiss candidate not tried recently.
        """

        dismiss = element(identifier="close", text="Close")
        overlay = OverlayObservation(
            visible=True,
            candidates=(dismiss,),
            bounds=bounds(width=600, height=400),
        )
        recovery_request = request(
            screen=observation(overlays=(overlay,)),
            block_reason=BlockReason.OVERLAY_STILL_PRESENT,
        )

        outcome = await OverlayRecovery().recover(request=recovery_request)

        assert isinstance(outcome, TryActionOutcome)
        self.assertIsInstance(outcome, TryActionOutcome)
        self.assertEqual(outcome.action.label_id, "close")

    async def test_declines_when_block_reason_is_unrelated(self) -> None:
        """
        Strategy must defer when the supervisor block reason is not overlay-related.
        """

        recovery_request = request(
            screen=observation(),
            block_reason=BlockReason.KEYBOARD_OCCLUDING,
        )

        outcome = await OverlayRecovery().recover(request=recovery_request)

        self.assertIsInstance(outcome, NoopOutcome)

    async def test_declines_when_all_candidates_were_recent(self) -> None:
        """
        Strategy must defer once every dismiss candidate is already in recent_actions.
        """

        dismiss = element(identifier="close", text="Close")
        overlay = OverlayObservation(
            visible=True,
            candidates=(dismiss,),
            bounds=bounds(width=600, height=400),
        )
        recovery_request = request(
            recent=["Close"],
            screen=observation(overlays=(overlay,)),
            block_reason=BlockReason.OVERLAY_STILL_PRESENT,
        )

        outcome = await OverlayRecovery().recover(request=recovery_request)

        self.assertIsInstance(outcome, NoopOutcome)

    def test_supports_expected_triggers(self) -> None:
        """
        Strategy must opt-in to the documented trigger set.
        """

        strategy = OverlayRecovery()

        self.assertTrue(strategy.supports(trigger=RecoveryTrigger.NO_PROGRESS))
        self.assertTrue(strategy.supports(trigger=RecoveryTrigger.LOOP_DETECTED))
        self.assertTrue(strategy.supports(trigger=RecoveryTrigger.REQUEST_REPLAN))
        self.assertTrue(strategy.supports(trigger=RecoveryTrigger.ACTION_BLOCKED))
        self.assertFalse(strategy.supports(trigger=RecoveryTrigger.VERIFY_REJECTED))
