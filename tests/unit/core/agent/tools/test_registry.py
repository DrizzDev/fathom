from __future__ import annotations

import unittest

from fathom.constants.tools import ToolName, TurnMode
from fathom.core.agent.tools.policies.hitl import HitlToolPolicy
from fathom.core.agent.tools.policies.mode import TurnModeToolPolicy
from fathom.core.agent.tools.registry import DEFAULT_TOOL_POLICIES
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.tools import ToolPolicyContext


class DefaultToolPoliciesTest(unittest.TestCase):
    """
    Pins the composition of :data:`DEFAULT_TOOL_POLICIES` so an accidental edit fails CI.
    """

    def test_ask_user_is_gated_on_hitl_capability(self) -> None:
        """
        ASK_USER must be gated by a :class:`HitlToolPolicy`.
        """

        hitl_policies = [
            policy for policy in DEFAULT_TOOL_POLICIES if isinstance(policy, HitlToolPolicy)
        ]

        self.assertEqual(len(hitl_policies), 1)
        self.assertIs(hitl_policies[0].tool, ToolName.ASK_USER)

    def test_verify_goal_is_gated_on_verify_mode(self) -> None:
        """
        VERIFY_GOAL must be gated by a :class:`TurnModeToolPolicy` requiring :attr:`TurnMode.VERIFY`.
        """

        gated = [
            policy
            for policy in DEFAULT_TOOL_POLICIES
            if isinstance(policy, TurnModeToolPolicy) and policy.tool is ToolName.VERIFY_GOAL
        ]

        self.assertEqual(len(gated), 1)

    def test_validate_state_is_gated_on_verify_mode(self) -> None:
        """
        VALIDATE_STATE must be gated by a :class:`TurnModeToolPolicy` requiring :attr:`TurnMode.VERIFY`.
        """

        gated = [
            policy
            for policy in DEFAULT_TOOL_POLICIES
            if isinstance(policy, TurnModeToolPolicy) and policy.tool is ToolName.VALIDATE_STATE
        ]

        self.assertEqual(len(gated), 1)

    def test_no_other_tools_are_gated_by_default(self) -> None:
        """
        Exactly three tools are gated by default: ASK_USER, VERIFY_GOAL, VALIDATE_STATE.
        """

        gated_tools = {policy.tool for policy in DEFAULT_TOOL_POLICIES}

        self.assertEqual(
            gated_tools,
            {ToolName.ASK_USER, ToolName.VERIFY_GOAL, ToolName.VALIDATE_STATE},
        )

    def test_verify_mode_policies_target_verify_flag(self) -> None:
        """
        Every :class:`TurnModeToolPolicy` in defaults must require :attr:`TurnMode.VERIFY`.
        """

        verify_context = ToolPolicyContext(
            modes=frozenset({TurnMode.VERIFY}),
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        for policy in DEFAULT_TOOL_POLICIES:
            if isinstance(policy, TurnModeToolPolicy):
                with self.subTest(tool=policy.tool):
                    self.assertTrue(policy.applies(context=verify_context))


if __name__ == "__main__":
    unittest.main()
