from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.core.agent.opener import OpenerSignalPolicy


class OpenerSignalPolicyTest(unittest.TestCase):
    """
    Covers the opener next-phase signal used to infer an opener sub-goal has advanced.
    """

    def test_interactive_command_indicates_advance(self) -> None:
        """
        A tap planned during an opener sub-goal signals the agent has moved past opening.
        """

        self.assertTrue(OpenerSignalPolicy().advanced(action_type=ActionType.TAP))

    def test_long_press_does_not_indicate_advance(self) -> None:
        """
        A long-press is not a next-phase signal even though it shares tap's capability profile.
        """

        self.assertFalse(OpenerSignalPolicy().advanced(action_type=ActionType.LONG_PRESS))

    def test_navigation_command_does_not_indicate_advance(self) -> None:
        """
        A back press during an opener sub-goal is not treated as next-phase progress.
        """

        self.assertFalse(OpenerSignalPolicy().advanced(action_type=ActionType.BACK))
