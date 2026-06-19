from __future__ import annotations

import unittest
from unittest.mock import patch

from pydantic import ValidationError
from tests.fixtures.intents import VERIFY_TOOLS
from tests.unit.core.agent.tools._legacy import _LegacyToolScope

from fathom.constants.tools import BASE_TOOLS, ToolName, TurnMode
from fathom.core.agent.tools.policy import ToolPolicy
from fathom.core.agent.tools.registry import DEFAULT_TOOL_POLICIES
from fathom.core.agent.tools.scope import ToolScope
from fathom.core.exceptions import InvariantViolation
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.subgoal import SubGoalKind
from fathom.schemas.tools import ToolPolicyContext


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


class ToolScopeMigrationParityTest(unittest.TestCase):
    """
    Pins the per-cell behavioral diff between the legacy intent-keyword gate
    and the new TurnMode-set gate. Delete one release after rollout.
    """

    __VERIFY_TOOLS = VERIFY_TOOLS

    @staticmethod
    def __new_tool_set(*, hitl: bool, sub_goal_kind: SubGoalKind) -> frozenset[ToolName]:
        """
        Compute the allowed tool set under the new framework for a sub-goal kind.
        """

        modes = (
            frozenset({TurnMode.VERIFY}) if sub_goal_kind == SubGoalKind.VALIDATION else frozenset()
        )
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

    @staticmethod
    def __legacy_tool_set(*, intent: str, hitl: bool) -> frozenset[ToolName]:
        """
        Compute the allowed tool set under the legacy intent-keyword framework.
        """

        return (
            _LegacyToolScope()
            .compute(
                intent=intent,
                capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=hitl)),
            )
            .names
        )

    def test_truth_table_pins_eight_cells(self) -> None:
        """
        Exhaustive truth table — intent keyword x sub-goal kind x HITL.
        """

        cases = (
            ("verify the offerwall", SubGoalKind.VALIDATION, False, True, True),
            ("verify the offerwall", SubGoalKind.VALIDATION, True, True, True),
            ("verify the offerwall", SubGoalKind.ACTION, False, True, False),
            ("verify the offerwall", SubGoalKind.ACTION, True, True, False),
            ("open the app", SubGoalKind.VALIDATION, False, False, True),
            ("open the app", SubGoalKind.VALIDATION, True, False, True),
            ("open the app", SubGoalKind.ACTION, True, False, False),
            ("open the app", SubGoalKind.ACTION, False, False, False),
        )

        for intent, kind, hitl, legacy_has_verify, new_has_verify in cases:
            with self.subTest(intent=intent, kind=kind, hitl=hitl):
                new_tools = self.__new_tool_set(sub_goal_kind=kind, hitl=hitl)
                legacy_tools = self.__legacy_tool_set(intent=intent, hitl=hitl)

                self.assertIn(ToolName.EXECUTE_UI, new_tools)
                self.assertIn(ToolName.EXECUTE_UI, legacy_tools)
                self.assertEqual(ToolName.ASK_USER in new_tools, hitl)
                self.assertEqual(ToolName.ASK_USER in legacy_tools, hitl)
                self.assertEqual(bool(new_tools & self.__VERIFY_TOOLS), new_has_verify)
                self.assertEqual(bool(legacy_tools & self.__VERIFY_TOOLS), legacy_has_verify)


if __name__ == "__main__":
    unittest.main()
