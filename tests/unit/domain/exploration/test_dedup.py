"""
Unit tests for the action deduplication and sampling policy.
"""

from __future__ import annotations

from fathom.constants import ActionType
from fathom.domain.exploration.config import DedupConfig, SamplingConfig
from fathom.domain.exploration.dedup import ActionKey, DedupPolicy
from fathom.schemas.actions import Action, Bounds


class TestActionKey:
    """
    Key construction and its resilience to free-form label drift.
    """

    def test_key_is_stable_across_label_drift(self) -> None:
        bounds = Bounds(x=100, y=200, width=50, height=50)
        first = Action(
            action_type=ActionType.TAP,
            rationale="r",
            bounds=bounds,
            natural_language_target="Login button",
        )
        second = Action(
            action_type=ActionType.TAP,
            rationale="r",
            bounds=bounds,
            natural_language_target="Sign in",
        )
        assert DedupPolicy.key_for(first) == DedupPolicy.key_for(second)

    def test_key_falls_back_to_natural_language_target(self) -> None:
        action = Action(action_type=ActionType.TAP, rationale="r", natural_language_target="Login")
        assert DedupPolicy.key_for(action) == ActionKey(kind="tap", label="login")

    def test_key_falls_back_to_target_when_unnamed(self) -> None:
        action = Action(action_type=ActionType.TAP, rationale="r", target="submit_btn")
        assert DedupPolicy.key_for(action).label == "submit_btn"

    def test_key_is_hashable_for_set_membership(self) -> None:
        action = Action(action_type=ActionType.TAP, rationale="r", target="x")
        assert DedupPolicy.key_for(action) in {DedupPolicy.key_for(action)}


class TestDedupPolicy:
    """
    Novelty, repeatability, and sampling decisions.
    """

    @staticmethod
    def __policy() -> DedupPolicy:
        return DedupPolicy(dedup=DedupConfig(), sampling=SamplingConfig())

    def test_navigation_actions_are_repeatable(self) -> None:
        for action_type in (ActionType.BACK, ActionType.SCROLL, ActionType.SWIPE_UP):
            action = Action(action_type=action_type, rationale="r")
            assert DedupPolicy.is_repeatable(action) is True

    def test_tap_and_type_are_not_repeatable(self) -> None:
        for action_type in (ActionType.TAP, ActionType.TYPE):
            action = Action(action_type=action_type, rationale="r")
            assert DedupPolicy.is_repeatable(action) is False

    def test_action_is_novel_when_key_absent(self) -> None:
        action = Action(action_type=ActionType.TAP, rationale="r", natural_language_target="X")
        assert self.__policy().is_novel(action=action, tried=set()) is True

    def test_action_is_not_novel_when_key_present(self) -> None:
        action = Action(action_type=ActionType.TAP, rationale="r", natural_language_target="X")
        tried = {DedupPolicy.key_for(action)}
        assert self.__policy().is_novel(action=action, tried=tried) is False

    def test_limit_for_known_and_unknown_categories(self) -> None:
        policy = self.__policy()
        assert policy.limit_for("content_item") == 3
        assert policy.limit_for("filter_or_category") == 4
        assert policy.limit_for("primary_action") is None
        assert policy.limit_for(None) is None

    def test_over_sampled_at_or_above_limit(self) -> None:
        policy = self.__policy()
        assert policy.is_over_sampled(category="content_item", sampled=3) is True
        assert policy.is_over_sampled(category="content_item", sampled=4) is True

    def test_not_over_sampled_below_limit(self) -> None:
        assert self.__policy().is_over_sampled(category="content_item", sampled=2) is False

    def test_uncapped_categories_are_never_over_sampled(self) -> None:
        policy = self.__policy()
        assert policy.is_over_sampled(category="primary_action", sampled=99) is False
        assert policy.is_over_sampled(category=None, sampled=99) is False

    def test_retries_reflects_config(self) -> None:
        policy = DedupPolicy(dedup=DedupConfig(retries=5), sampling=SamplingConfig())
        assert policy.retries == 5
