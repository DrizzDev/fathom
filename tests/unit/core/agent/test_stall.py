from __future__ import annotations

import unittest
from typing import List

from fathom.constants.turn.stall import StallState
from fathom.core.agent.stall import StallPolicy
from fathom.schemas.effect import ActionEffect, ActionEffectStatus


class StallPolicyTest(unittest.TestCase):
    """
    Cover momentum classification, including the UNCERTAIN under-detection fix.
    """

    def setUp(self) -> None:
        """
        Build the policy under test.
        """

        self.policy = StallPolicy()

    def test_progressing_stream_flows(self) -> None:
        """
        A stream ending in progress carries no stall streak.
        """

        signal = self.policy.assess(
            effects=self.__effects(
                ActionEffectStatus.NO_PROGRESS,
                ActionEffectStatus.PROGRESS,
            )
        )

        self.assertEqual(signal.state, StallState.FLOWING)
        self.assertEqual(signal.streak, 0)

    def test_uncertain_effects_count_toward_the_stall(self) -> None:
        """
        The under-detection fix: mixed UNCERTAIN and NO_PROGRESS effects stall together.
        """

        signal = self.policy.assess(
            effects=self.__effects(
                ActionEffectStatus.PROGRESS,
                ActionEffectStatus.UNCERTAIN,
                ActionEffectStatus.NO_PROGRESS,
                ActionEffectStatus.UNCERTAIN,
            )
        )

        self.assertEqual(signal.state, StallState.STALLED)
        self.assertEqual(signal.streak, 3)

    def test_short_streak_reads_uncertain(self) -> None:
        """
        One flat effect is ambiguity, not a stall.
        """

        signal = self.policy.assess(
            effects=self.__effects(
                ActionEffectStatus.PROGRESS,
                ActionEffectStatus.NO_PROGRESS,
            )
        )

        self.assertEqual(signal.state, StallState.UNCERTAIN)
        self.assertEqual(signal.streak, 1)

    @staticmethod
    def __effects(*statuses: ActionEffectStatus) -> List[ActionEffect]:
        """
        Build an effect stream with the given trailing statuses.
        """

        return [
            ActionEffect(status=status, phash_distance=0, visual_progress=0.0)
            for status in statuses
        ]
