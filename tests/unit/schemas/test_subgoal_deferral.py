from __future__ import annotations

import unittest

from fathom.schemas.subgoal import SubGoal


class SubGoalDeferralFieldTest(unittest.TestCase):
    """
    Pins the escalation-deferral counter field on :class:`SubGoal`.
    """

    def test_default_is_zero(self) -> None:
        """
        Newly constructed sub-goals start with zero deferrals.
        """

        sub_goal = SubGoal(description="test", index=0)
        self.assertEqual(sub_goal.deferral_count, 0)

    def test_field_is_mutable(self) -> None:
        """
        ``deferral_count`` is mutated in-place by :class:`AgentState` helpers.
        """

        sub_goal = SubGoal(description="test", index=0)
        sub_goal.deferral_count += 1
        sub_goal.deferral_count += 1
        self.assertEqual(sub_goal.deferral_count, 2)

    def test_negative_values_rejected(self) -> None:
        """
        ``deferral_count`` cannot go below zero.
        """

        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            SubGoal(description="test", index=0, deferral_count=-1)

    def test_round_trip_preserves_value(self) -> None:
        """
        Checkpoint serialization preserves the counter.
        """

        sub_goal = SubGoal(description="test", index=0, deferral_count=3)
        restored = SubGoal.model_validate(sub_goal.model_dump())
        self.assertEqual(restored.deferral_count, 3)

    def test_legacy_checkpoint_without_field_defaults_to_zero(self) -> None:
        """
        Old checkpoints written before this field existed must rehydrate cleanly.
        """

        legacy_payload = {"description": "legacy", "index": 0}
        sub_goal = SubGoal.model_validate(legacy_payload)
        self.assertEqual(sub_goal.deferral_count, 0)
