from __future__ import annotations

from typing import Final, FrozenSet

import pytest

from fathom.constants.tools import ToolName, TurnMode
from fathom.core.agent.tools.registry import DEFAULT_TOOL_POLICIES
from fathom.core.agent.tools.scope import ToolScope
from fathom.core.capability.catalog import CommandCatalogProvider
from fathom.core.services.decomposer import IntentDecomposer
from fathom.core.services.directive import DirectivePolicy
from fathom.interfaces.llm import LLMPort
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.subgoal import SubGoal, SubGoalKind
from fathom.schemas.tools import ToolPolicyContext
from tests.fixtures.intents import VERIFY_TOOLS, IntentCorpus


def _tool_set_for(*, sub_goal: SubGoal, hitl: bool) -> FrozenSet[ToolName]:
    """
    Reproduce the planner mapping and compute the per-turn tool set.
    """

    modes = frozenset({TurnMode.VERIFY}) if sub_goal.kind == SubGoalKind.VALIDATION else frozenset()
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


def _assert_contract(*, sub_goal: SubGoal, intent: str, hitl: bool) -> None:
    """
    Assert the per-turn tool set obeys the framework contract for one sub-goal.
    """

    tools = _tool_set_for(sub_goal=sub_goal, hitl=hitl)
    assert ToolName.EXECUTE_UI in tools
    assert ToolName.STORE_MEMORY in tools
    assert ToolName.RECALL_MEMORY in tools
    assert (ToolName.ASK_USER in tools) == hitl

    if sub_goal.kind == SubGoalKind.VALIDATION:
        assert VERIFY_TOOLS.issubset(tools), (
            f"VALIDATION sub-goal missed verification tools "
            f"for intent={intent!r} sub_goal={sub_goal.description!r}"
        )
    else:
        assert not (tools & VERIFY_TOOLS), (
            f"ACTION sub-goal leaked verification tools "
            f"for intent={intent!r} sub_goal={sub_goal.description!r}"
        )


class TestLiveToolScope:
    """
    Live decomposer x :class:`ToolScope` replay — guards production correctness end-to-end.
    """

    __VERIFY_TOOLS: Final[FrozenSet[ToolName]] = VERIFY_TOOLS

    @pytest.mark.asyncio
    @pytest.mark.parametrize("intent", IntentCorpus.representative())
    async def test_representative_intent_obeys_tool_scope_contract(
        self,
        *,
        intent: str,
        llm: LLMPort,
    ) -> None:
        """
        Quick-sample subset of the corpus runs end-to-end against the live decomposer.
        """

        sub_goals = await IntentDecomposer(
            llm=llm, directive_policy=DirectivePolicy(catalog=CommandCatalogProvider().build())
        ).decompose(intent=intent)
        assert sub_goals, f"decomposer returned no sub-goals for intent: {intent!r}"

        for sub_goal in sub_goals:
            for hitl in (False, True):
                _assert_contract(sub_goal=sub_goal, intent=intent, hitl=hitl)


@pytest.mark.slow
class TestLiveToolScopeFullCorpus:
    """
    Exhaustive corpus replay through the live decomposer.

    Every intent from ``tests/fixtures/intents.txt`` runs against real Gemini.
    Costly — gated by both FATHOM_RUN_LIVE_TESTS=1 and the ``slow`` pytest marker.
    Invoke explicitly with ``pytest -m slow tests/live/core/agent/tools/``.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("intent", IntentCorpus.all_intents())
    async def test_every_corpus_intent_obeys_tool_scope_contract(
        self,
        *,
        intent: str,
        llm: LLMPort,
    ) -> None:
        """
        Every corpus intent's decomposition must obey the framework contract end-to-end.
        """

        sub_goals = await IntentDecomposer(
            llm=llm, directive_policy=DirectivePolicy(catalog=CommandCatalogProvider().build())
        ).decompose(intent=intent)
        assert sub_goals, f"decomposer returned no sub-goals for intent: {intent!r}"

        for sub_goal in sub_goals:
            for hitl in (False, True):
                _assert_contract(sub_goal=sub_goal, intent=intent, hitl=hitl)
