from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.schemas.subgoal import GoalState, Progress, SubGoal
from tests.builders import SuccessFixtures


class ProgressDeferralFieldTest(unittest.TestCase):
    """
    Pins the escalation-deferral counter, now owned by mutable Progress and surfaced by GoalState.
    """

    @staticmethod
    def __state() -> GoalState:
        return GoalState(
            goal=SubGoal(index=0, objective="test", success=SuccessFixtures.observed())
        )

    def test_default_is_zero(self) -> None:
        """
        A fresh goal starts with zero deferrals.
        """

        self.assertEqual(self.__state().deferral_count, 0)

    def test_record_and_clear_mutate_in_place(self) -> None:
        """
        record_deferral / clear_deferrals mutate the recovery counter in place.
        """

        state = self.__state()
        state.record_deferral()
        state.record_deferral()
        self.assertEqual(state.deferral_count, 2)
        state.clear_deferrals()
        self.assertEqual(state.deferral_count, 0)

    def test_negative_recovery_is_rejected(self) -> None:
        """
        The recovery counter cannot be assigned a negative value.
        """

        progress = Progress()
        with self.assertRaises(ValidationError):
            progress.recovery = -1

    def test_round_trip_preserves_value(self) -> None:
        """
        Serialization preserves the recovery counter through a checkpoint round trip.
        """

        state = self.__state()
        state.record_deferral()
        state.record_deferral()
        state.record_deferral()
        restored = GoalState.model_validate(state.model_dump())
        self.assertEqual(restored.deferral_count, 3)
