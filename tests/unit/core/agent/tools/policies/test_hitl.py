from __future__ import annotations

import unittest

from fathom.constants.tools import ToolName
from fathom.core.agent.tools.policies.hitl import HitlToolPolicy
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.tools import ToolPolicyContext


class HitlToolPolicyTest(unittest.TestCase):
    """
    Pins :class:`HitlToolPolicy` as a single-signal gate on the HITL capability flag.
    """

    @staticmethod
    def __context(*, hitl: bool) -> ToolPolicyContext:
        """
        Build a :class:`ToolPolicyContext` with the HITL capability flag toggled.
        """

        return ToolPolicyContext(
            modes=frozenset(),
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=hitl)),
        )

    def test_applies_when_hitl_capability_enabled(self) -> None:
        """
        HITL on → policy exposes its bound tool.
        """

        policy = HitlToolPolicy(tool=ToolName.ASK_USER)

        self.assertTrue(policy.applies(context=self.__context(hitl=True)))

    def test_does_not_apply_when_hitl_capability_disabled(self) -> None:
        """
        HITL off → policy does not expose its bound tool.
        """

        policy = HitlToolPolicy(tool=ToolName.ASK_USER)

        self.assertFalse(policy.applies(context=self.__context(hitl=False)))

    def test_bound_tool_is_returned_by_property(self) -> None:
        """
        The tool the policy gates must be reachable via :attr:`tool`.
        """

        policy = HitlToolPolicy(tool=ToolName.ASK_USER)

        self.assertIs(policy.tool, ToolName.ASK_USER)


if __name__ == "__main__":
    unittest.main()
