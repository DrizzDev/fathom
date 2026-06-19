from __future__ import annotations

import itertools
import unittest

from fathom.constants.tools import ToolName, TurnMode
from fathom.core.agent.tools.registry import DEFAULT_TOOL_POLICIES
from fathom.core.agent.tools.scope import ToolScope
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.tools import ToolPolicyContext


class ToolScopeMatrixTest(unittest.TestCase):
    """
    Exhaustive matrix: every (mode subset × HITL on/off) combination must satisfy
    the liveness invariant and the verification gating contract.
    """

    @staticmethod
    def __mode_subsets() -> list[frozenset[TurnMode]]:
        """
        Enumerate every subset of :class:`TurnMode` values.
        """

        modes = list(TurnMode)
        subsets: list[frozenset[TurnMode]] = []

        for size in range(len(modes) + 1):
            for combo in itertools.combinations(modes, size):
                subsets.append(frozenset(combo))

        return subsets

    def test_base_tools_present_in_every_combination(self) -> None:
        """
        EXECUTE_UI, STORE_MEMORY, RECALL_MEMORY must appear in every (modes, hitl) combination.
        """

        scope = ToolScope(policies=DEFAULT_TOOL_POLICIES)

        for modes in self.__mode_subsets():
            for hitl in (False, True):
                with self.subTest(modes=sorted(mode.value for mode in modes), hitl=hitl):
                    result = scope.compute(
                        context=ToolPolicyContext(
                            modes=modes,
                            capabilities=RuntimeCapabilities(
                                hitl=HITLCapability(enabled=hitl),
                            ),
                        ),
                    )
                    self.assertIn(ToolName.EXECUTE_UI, result.names)
                    self.assertIn(ToolName.STORE_MEMORY, result.names)
                    self.assertIn(ToolName.RECALL_MEMORY, result.names)

    def test_verify_tools_present_iff_verify_mode_active(self) -> None:
        """
        VERIFY_GOAL and VALIDATE_STATE appear if and only if :attr:`TurnMode.VERIFY` is in the mode set.
        """

        scope = ToolScope(policies=DEFAULT_TOOL_POLICIES)

        for modes in self.__mode_subsets():
            for hitl in (False, True):
                with self.subTest(modes=sorted(mode.value for mode in modes), hitl=hitl):
                    result = scope.compute(
                        context=ToolPolicyContext(
                            modes=modes,
                            capabilities=RuntimeCapabilities(
                                hitl=HITLCapability(enabled=hitl),
                            ),
                        ),
                    )
                    expected = TurnMode.VERIFY in modes
                    self.assertEqual(ToolName.VERIFY_GOAL in result.names, expected)
                    self.assertEqual(ToolName.VALIDATE_STATE in result.names, expected)

    def test_ask_user_present_iff_hitl_enabled(self) -> None:
        """
        ASK_USER appears if and only if the HITL capability is enabled.
        """

        scope = ToolScope(policies=DEFAULT_TOOL_POLICIES)

        for modes in self.__mode_subsets():
            for hitl in (False, True):
                with self.subTest(modes=sorted(mode.value for mode in modes), hitl=hitl):
                    result = scope.compute(
                        context=ToolPolicyContext(
                            modes=modes,
                            capabilities=RuntimeCapabilities(
                                hitl=HITLCapability(enabled=hitl),
                            ),
                        ),
                    )
                    self.assertEqual(ToolName.ASK_USER in result.names, hitl)


if __name__ == "__main__":
    unittest.main()
