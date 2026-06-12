from __future__ import annotations

import unittest

from fathom.core.exploration.config import DepthFloorConfig
from fathom.core.exploration.depth import DepthFloorPolicy


class TestDepthFloorPolicy(unittest.TestCase):
    """The depth floor vetoes premature exhaustion only on shallow first attempts."""

    def setUp(self) -> None:
        self.__policy = DepthFloorPolicy(config=DepthFloorConfig(minimum=4))

    def test_vetoes_shallow_first_exhaustion(self) -> None:
        self.assertTrue(self.__policy.should_veto(depth=2, retries=0))

    def test_no_veto_at_or_past_floor(self) -> None:
        self.assertFalse(self.__policy.should_veto(depth=4, retries=0))
        self.assertFalse(self.__policy.should_veto(depth=7, retries=0))

    def test_no_veto_once_retried(self) -> None:
        self.assertFalse(self.__policy.should_veto(depth=2, retries=1))

    def test_directive_active_only_after_a_vetoed_attempt(self) -> None:
        self.assertTrue(self.__policy.is_active(depth=2, retries=1))
        self.assertFalse(self.__policy.is_active(depth=2, retries=0))
        self.assertFalse(self.__policy.is_active(depth=4, retries=1))

    def test_minimum_property(self) -> None:
        self.assertEqual(self.__policy.minimum, 4)
