from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.core.recovery.strategies.keyboard import KeyboardRecovery
from fathom.core.recovery.types import NoopOutcome, RecoveryTrigger, TryActionOutcome
from fathom.schemas.observation import KeyboardObservation
from fathom.schemas.supervision import BlockReason

from ._fixtures import bounds, element, observation, request


class KeyboardRecoveryTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins for the keyboard dismissal strategy.
    """

    async def test_uses_known_dismiss_candidate(self) -> None:
        """
        Strategy must tap a known dismiss control when one is exposed.
        """

        dismiss = element(identifier="kb_done", text="Done")

        keyboard = KeyboardObservation(
            visible=True,
            dismiss=(dismiss,),
            bounds=bounds(y=2000, width=1206, height=400),
        )
        recovery_request = request(
            screen=observation(keyboard=keyboard),
            block_reason=BlockReason.KEYBOARD_OCCLUDING,
        )

        outcome = await KeyboardRecovery().recover(request=recovery_request)

        assert isinstance(outcome, TryActionOutcome)
        self.assertIsInstance(outcome, TryActionOutcome)

        self.assertEqual(outcome.action.label_id, "kb_done")
        self.assertEqual(outcome.action.action_type, ActionType.TAP)

    async def test_falls_back_to_hide_keyboard_action(self) -> None:
        """
        Strategy must fall back to HIDE_KEYBOARD when no dismiss candidate is known.
        """

        keyboard = KeyboardObservation(
            visible=True,
            bounds=bounds(y=2000, width=1206, height=400),
        )
        recovery_request = request(
            screen=observation(keyboard=keyboard),
            block_reason=BlockReason.KEYBOARD_OCCLUDING,
        )

        outcome = await KeyboardRecovery().recover(request=recovery_request)

        assert isinstance(outcome, TryActionOutcome)
        self.assertIsInstance(outcome, TryActionOutcome)
        self.assertEqual(outcome.action.action_type, ActionType.HIDE_KEYBOARD)

    async def test_declines_when_keyboard_not_visible(self) -> None:
        """
        Strategy must defer when the keyboard is not visible.
        """

        recovery_request = request(
            screen=observation(),
            block_reason=BlockReason.KEYBOARD_OCCLUDING,
        )

        outcome = await KeyboardRecovery().recover(request=recovery_request)

        self.assertIsInstance(outcome, NoopOutcome)

    def test_supports_expected_triggers(self) -> None:
        """
        Strategy must opt-in to the documented trigger set.
        """

        strategy = KeyboardRecovery()

        self.assertTrue(strategy.supports(trigger=RecoveryTrigger.NO_PROGRESS))
        self.assertTrue(strategy.supports(trigger=RecoveryTrigger.REQUEST_REPLAN))
        self.assertTrue(strategy.supports(trigger=RecoveryTrigger.ACTION_BLOCKED))
        self.assertFalse(strategy.supports(trigger=RecoveryTrigger.VERIFY_REJECTED))
