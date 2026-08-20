from __future__ import annotations

import unittest
from unittest.mock import patch

from pydantic import ValidationError

from fathom.constants.tools import BASE_TOOLS, ToolName, TurnMode
from fathom.core.agent.tools.policy import ToolPolicy
from fathom.core.agent.tools.registry import DEFAULT_TOOL_POLICIES
from fathom.core.agent.tools.scope import ToolScope
from fathom.core.exceptions import InvariantViolation
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.tools import ToolPolicyContext

VERIFY_TOOLS = frozenset({ToolName.VERIFY_GOAL, ToolName.VALIDATE_STATE})


class _AlwaysAllowsToolPolicy(ToolPolicy):
    """
    Stub policy that always exposes its bound tool — for composition tests.
    """

    def __init__(self, *, tool: ToolName) -> None:
        self.__tool = tool

    @property
    def tool(self) -> ToolName:
        """ """

        return self.__tool

    def applies(self, *, context: ToolPolicyContext) -> bool:
        """ """

        _ = context
        return True


class _NeverAllowsToolPolicy(ToolPolicy):
    """
    Stub policy that never exposes its bound tool — for composition tests.
    """

    def __init__(self, *, tool: ToolName) -> None:
        """ """

        self.__tool = tool

    @property
    def tool(self) -> ToolName:
        """ """

        return self.__tool

    def applies(self, *, context: ToolPolicyContext) -> bool:
        """ """

        _ = context
        return False


class _DropExecuteUiPolicy(ToolPolicy):
    """
    Diagnostic stub that pretends to gate EXECUTE_UI off — for invariant tests only.
    """

    @property
    def tool(self) -> ToolName:
        """ """

        return ToolName.STORE_MEMORY

    def applies(self, *, context: ToolPolicyContext) -> bool:
        """ """

        _ = context
        return False


class ToolScopeCompositionTest(unittest.TestCase):
    """
    Pins :class:`ToolScope` as a pure policy composer that knows no concrete tool rules.
    """

    @staticmethod
    def __context(
        *,
        hitl: bool = False,
        modes: frozenset[TurnMode] = frozenset(),
    ) -> ToolPolicyContext:
        """
        Build a frozen :class:`ToolPolicyContext` for the test scenarios.
        """

        return ToolPolicyContext(
            modes=modes,
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=hitl)),
        )

    def test_empty_policy_set_returns_base_tools_only(self) -> None:
        """
        With no policies bound, the result is exactly the unconditional BASE_TOOLS set.
        """

        scope = ToolScope(policies=())
        result = scope.compute(context=self.__context())

        self.assertEqual(result.names, BASE_TOOLS)

    def test_always_applies_policy_adds_its_tool(self) -> None:
        """
        A policy whose ``applies`` returns True contributes its bound tool to the result.
        """

        scope = ToolScope(
            policies=(_AlwaysAllowsToolPolicy(tool=ToolName.ASK_USER),),
        )
        result = scope.compute(context=self.__context())

        self.assertIn(ToolName.ASK_USER, result.names)

    def test_never_applies_policy_does_not_add_its_tool(self) -> None:
        """
        A policy whose ``applies`` returns False contributes nothing to the result.
        """

        scope = ToolScope(
            policies=(_NeverAllowsToolPolicy(tool=ToolName.ASK_USER),),
        )
        result = scope.compute(context=self.__context())

        self.assertNotIn(ToolName.ASK_USER, result.names)

    def test_duplicate_policies_for_same_tool_are_idempotent(self) -> None:
        """
        Two policies that both gate the same tool produce the same single tool entry.
        """

        scope = ToolScope(
            policies=(
                _AlwaysAllowsToolPolicy(tool=ToolName.VERIFY_GOAL),
                _AlwaysAllowsToolPolicy(tool=ToolName.VERIFY_GOAL),
            ),
        )
        result = scope.compute(context=self.__context())

        self.assertIn(ToolName.VERIFY_GOAL, result.names)
        self.assertEqual(
            sum(1 for name in result.names if name == ToolName.VERIFY_GOAL),
            1,
        )

    def test_base_tools_always_present_regardless_of_policies(self) -> None:
        """
        EXECUTE_UI, STORE_MEMORY, RECALL_MEMORY appear in every result, with or without policies.
        """

        for policies in (
            (),
            (_NeverAllowsToolPolicy(tool=ToolName.ASK_USER),),
            (_AlwaysAllowsToolPolicy(tool=ToolName.VERIFY_GOAL),),
        ):
            with self.subTest(policy_count=len(policies)):
                scope = ToolScope(policies=policies)
                result = scope.compute(context=self.__context())

                self.assertIn(ToolName.EXECUTE_UI, result.names)
                self.assertIn(ToolName.STORE_MEMORY, result.names)
                self.assertIn(ToolName.RECALL_MEMORY, result.names)

    def test_returns_frozen_allowed_tools(self) -> None:
        """
        :class:`AllowedTools` returned by :meth:`compute` must be frozen.
        """

        scope = ToolScope(policies=())
        result = scope.compute(context=self.__context())

        with self.assertRaises(ValidationError):
            result.names = frozenset()  # type: ignore[misc]

    def test_invariant_raises_when_execute_ui_missing_from_base(self) -> None:
        """
        ToolScope must raise :class:`InvariantViolation` if EXECUTE_UI ever leaves the result.
        Simulated by stubbing :data:`BASE_TOOLS` via a guarded patch.
        """

        with patch(
            "fathom.core.agent.tools.scope.BASE_TOOLS",
            frozenset({ToolName.STORE_MEMORY}),
        ):
            scope = ToolScope(policies=())
            with self.assertRaises(InvariantViolation):
                scope.compute(context=self.__context())


class ToolScopeModeGateTest(unittest.TestCase):
    """
    Pins the mode-driven VERIFY gate: an active goal keeps base UI tactics regardless of its
    success kind, and VERIFY-only tools appear solely in the no-active-goal / final-verification phase.
    """

    __VERIFY_TOOLS = VERIFY_TOOLS

    @staticmethod
    def __tools(*, modes: frozenset[TurnMode], hitl: bool = False) -> frozenset[ToolName]:
        """
        Compute the allowed tool set for a given mode set.
        """

        return (
            ToolScope(policies=DEFAULT_TOOL_POLICIES)
            .compute(
                context=ToolPolicyContext(
                    modes=modes,
                    capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=hitl)),
                ),
            )
            .names
        )

    def test_active_goal_retains_base_ui_without_verify(self) -> None:
        """
        An active goal (empty mode set) keeps the base UI tactics and exposes no VERIFY-only tools.
        """

        tools = self.__tools(modes=frozenset())

        self.assertIn(ToolName.EXECUTE_UI, tools)
        self.assertFalse(tools & self.__VERIFY_TOOLS)

    def test_final_verification_phase_exposes_verify_tools(self) -> None:
        """
        The no-active-goal / final-verification phase (VERIFY mode) exposes the VERIFY-only tools.
        """

        tools = self.__tools(modes=frozenset({TurnMode.VERIFY}))

        self.assertTrue(tools & self.__VERIFY_TOOLS)
        self.assertIn(ToolName.EXECUTE_UI, tools)

    def test_ask_user_tracks_hitl_capability(self) -> None:
        """
        ASK_USER is exposed exactly when HITL is enabled, independent of mode.
        """

        self.assertIn(ToolName.ASK_USER, self.__tools(modes=frozenset(), hitl=True))
        self.assertNotIn(ToolName.ASK_USER, self.__tools(modes=frozenset(), hitl=False))


if __name__ == "__main__":
    unittest.main()
