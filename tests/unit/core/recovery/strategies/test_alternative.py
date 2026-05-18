from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.core.recovery.strategies.alternative import AlternativeTargetRecovery
from fathom.core.recovery.types import NoopOutcome, RecoveryTrigger, TryActionOutcome
from fathom.schemas.supervision import BlockReason

from ._fixtures import candidate, element, observation, request


class AlternativeTargetRecoveryTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins for the localization-candidate retry strategy.
    """

    async def test_retries_with_first_unused_candidate(self) -> None:
        """
        Strategy must propose the first localization candidate not in recent_actions.
        """

        matched = element(identifier="cta_continue", text="Continue")
        candidates = [
            candidate(reason="exact text match", matched=matched, score=0.85),
        ]
        recovery_request = request(
            screen=observation(),
            candidates=candidates,
            trigger=RecoveryTrigger.TARGET_UNRESOLVED,
            block_reason=BlockReason.TARGET_UNRESOLVED,
        )

        outcome = await AlternativeTargetRecovery().recover(request=recovery_request)

        assert isinstance(outcome, TryActionOutcome)
        self.assertIsInstance(outcome, TryActionOutcome)

        self.assertEqual(outcome.action.target, "Continue")
        self.assertEqual(outcome.action.action_type, ActionType.TAP)

    async def test_declines_when_no_candidates_present(self) -> None:
        """
        Strategy must defer when no localization candidates are available.
        """

        recovery_request = request(
            candidates=[],
            trigger=RecoveryTrigger.TARGET_UNRESOLVED,
            block_reason=BlockReason.TARGET_UNRESOLVED,
        )

        outcome = await AlternativeTargetRecovery().recover(request=recovery_request)

        self.assertIsInstance(outcome, NoopOutcome)

    async def test_declines_when_block_reason_is_unrelated(self) -> None:
        """
        Strategy must defer when the block reason is not target-related.
        """

        recovery_request = request(
            trigger=RecoveryTrigger.NO_PROGRESS,
            block_reason=BlockReason.KEYBOARD_OCCLUDING,
            candidates=[candidate(reason="fixture", matched=element(identifier="x"))],
        )

        outcome = await AlternativeTargetRecovery().recover(request=recovery_request)

        self.assertIsInstance(outcome, NoopOutcome)
