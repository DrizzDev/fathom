"""
Unit tests for the DFS depth-floor policy.
"""

from __future__ import annotations

from fathom.domain.exploration.config import DepthFloorConfig
from fathom.domain.exploration.depth import DepthFloorPolicy


class TestDepthFloorPolicy:
    """
    Behaviour of the depth-floor veto and activation rules.
    """

    @staticmethod
    def __policy(minimum: int = 4) -> DepthFloorPolicy:
        return DepthFloorPolicy(config=DepthFloorConfig(minimum=minimum))

    def test_minimum_reflects_config(self) -> None:
        assert self.__policy(minimum=6).minimum == 6

    def test_vetoes_first_exhaustion_below_floor(self) -> None:
        assert self.__policy().should_veto(depth=2, retries=0) is True

    def test_does_not_veto_after_a_retry(self) -> None:
        assert self.__policy().should_veto(depth=2, retries=1) is False

    def test_does_not_veto_at_or_above_floor(self) -> None:
        policy = self.__policy()
        assert policy.should_veto(depth=4, retries=0) is False
        assert policy.should_veto(depth=9, retries=0) is False

    def test_active_only_below_floor_after_a_retry(self) -> None:
        policy = self.__policy()
        assert policy.is_active(depth=2, retries=1) is True
        assert policy.is_active(depth=2, retries=0) is False
        assert policy.is_active(depth=4, retries=1) is False

    def test_veto_and_active_are_mutually_exclusive(self) -> None:
        policy = self.__policy()
        for depth in range(0, 8):
            for retries in range(0, 3):
                assert not (
                    policy.should_veto(depth=depth, retries=retries)
                    and policy.is_active(depth=depth, retries=retries)
                )
