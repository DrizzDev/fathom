from __future__ import annotations

import unittest

from fathom.constants.tools import ToolName, TurnMode
from fathom.core.agent.tools.policies.mode import TurnModeToolPolicy
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.tools import ToolPolicyContext


class TurnModeToolPolicyTest(unittest.TestCase):
    """
    Pins :class:`TurnModeToolPolicy` as a set-membership gate on per-turn mode flags.
    """

    @staticmethod
    def __context(*, modes: frozenset[TurnMode]) -> ToolPolicyContext:
        """
        Build a :class:`ToolPolicyContext` with a specific set of mode flags active.
        """

        return ToolPolicyContext(
            modes=modes,
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )

    def test_applies_when_required_mode_in_set(self) -> None:
        """
        Required mode present → policy exposes its bound tool.
        """

        policy = TurnModeToolPolicy(
            tool=ToolName.VERIFY_GOAL,
            required_mode=TurnMode.VERIFY,
        )

        self.assertTrue(
            policy.applies(context=self.__context(modes=frozenset({TurnMode.VERIFY}))),
        )

    def test_does_not_apply_when_required_mode_absent(self) -> None:
        """
        Required mode absent → policy keeps its bound tool hidden.
        """

        policy = TurnModeToolPolicy(
            tool=ToolName.VERIFY_GOAL,
            required_mode=TurnMode.VERIFY,
        )

        self.assertFalse(policy.applies(context=self.__context(modes=frozenset())))

    def test_applies_alongside_other_modes(self) -> None:
        """
        A turn carrying multiple mode flags still exposes the gated tool if its mode is present.
        """

        policy = TurnModeToolPolicy(
            tool=ToolName.VERIFY_GOAL,
            required_mode=TurnMode.VERIFY,
        )

        self.assertTrue(
            policy.applies(context=self.__context(modes=frozenset({TurnMode.VERIFY}))),
        )

    def test_bound_tool_is_returned_by_property(self) -> None:
        """
        The tool the policy gates must be reachable via :attr:`tool`.
        """

        policy = TurnModeToolPolicy(
            tool=ToolName.VERIFY_GOAL,
            required_mode=TurnMode.VERIFY,
        )

        self.assertIs(policy.tool, ToolName.VERIFY_GOAL)


if __name__ == "__main__":
    unittest.main()
