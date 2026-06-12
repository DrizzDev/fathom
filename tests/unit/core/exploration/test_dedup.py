from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.core.exploration.config import DedupConfig, SamplingConfig
from fathom.core.exploration.dedup import ActionKey, DedupPolicy
from fathom.schemas.actions import Action, Bounds


class TestDedupPolicy(unittest.TestCase):
    """Dedup keys survive label drift and sampling caps bound list enumeration."""

    def setUp(self) -> None:
        self.__policy = DedupPolicy(dedup=DedupConfig(), sampling=SamplingConfig())

    @staticmethod
    def __tap(*, x: int, y: int, name: str = "Element") -> Action:
        """Build a tap action with bounds at the given normalized coordinate."""

        return Action(
            action_type=ActionType.TAP,
            rationale="unit-test action",
            natural_language_target=name,
            bounds=Bounds(x=x, y=y, width=10, height=10),
        )

    def test_key_is_stable_across_label_drift(self) -> None:
        first = self.__tap(x=100, y=200, name="Home")
        drifted = self.__tap(x=108, y=204, name="Homepage tab")

        self.assertEqual(DedupPolicy.key_for(first), DedupPolicy.key_for(drifted))

    def test_key_distinguishes_distant_taps(self) -> None:
        near = self.__tap(x=100, y=200)
        far = self.__tap(x=500, y=800)

        self.assertNotEqual(DedupPolicy.key_for(near), DedupPolicy.key_for(far))

    def test_key_falls_back_to_label_without_bounds(self) -> None:
        action = Action(
            action_type=ActionType.TAP,
            rationale="unit-test action",
            natural_language_target="Search",
        )

        self.assertEqual(DedupPolicy.key_for(action), ActionKey(kind="tap", label="search"))

    def test_repeatable_action_types(self) -> None:
        scroll = Action(action_type=ActionType.SCROLL, rationale="scroll for more")
        back = Action(action_type=ActionType.BACK, rationale="go back")
        tap = self.__tap(x=10, y=10)

        self.assertTrue(DedupPolicy.is_repeatable(scroll))
        self.assertTrue(DedupPolicy.is_repeatable(back))
        self.assertFalse(DedupPolicy.is_repeatable(tap))

    def test_is_novel_tracks_tried_keys(self) -> None:
        action = self.__tap(x=300, y=400)
        tried = {DedupPolicy.key_for(action)}

        self.assertTrue(self.__policy.is_novel(action=self.__tap(x=10, y=10), tried=tried))
        self.assertFalse(self.__policy.is_novel(action=action, tried=tried))

    def test_is_over_sampled_respects_category_caps(self) -> None:
        self.assertTrue(self.__policy.is_over_sampled(category="content_item", sampled=3))
        self.assertFalse(self.__policy.is_over_sampled(category="content_item", sampled=2))

    def test_uncapped_categories_are_never_over_sampled(self) -> None:
        self.assertFalse(self.__policy.is_over_sampled(category="primary_action", sampled=99))
        self.assertFalse(self.__policy.is_over_sampled(category=None, sampled=99))

    def test_action_key_is_hashable_value_object(self) -> None:
        first = ActionKey(kind="tap", label="2_4")
        same = ActionKey(kind="tap", label="2_4")

        self.assertEqual(first, same)
        self.assertEqual(hash(first), hash(same))
        self.assertIn(same, {first})
