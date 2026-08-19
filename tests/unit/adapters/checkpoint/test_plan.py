from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from langgraph.constants import START

from fathom.adapters.checkpoint.plan import LangGraphPlanStore
from fathom.constants.state import IntentStateKey
from fathom.schemas.plan import Plan
from fathom.schemas.subgoal import GoalState, SubGoal
from fathom.schemas.success import CommandSuccess
from tests.builders import SuccessFixtures


class _FakeGraph:
    """
    Compiled-graph double exposing a fixed snapshot and recording pre-graph seeds.
    """

    def __init__(self, *, values: Dict[str, Any]) -> None:
        self.__values = values
        self.updates: List[Tuple[Dict[str, Any], Optional[str]]] = []

    async def aget_state(self, config: Any) -> Any:
        _ = config
        return SimpleNamespace(next=(), values=self.__values)

    async def aupdate_state(
        self, config: Any, values: Dict[str, Any], as_node: Optional[str] = None
    ) -> Any:
        self.updates.append((values, as_node))
        return config


class LangGraphPlanStoreTest(unittest.IsolatedAsyncioTestCase):
    """
    Covers plan read/seed translation across the LangGraph checkpoint surface.
    """

    def setUp(self) -> None:
        """
        Build the stateless plan store adapter under test.
        """

        self.__store = LangGraphPlanStore()

    @staticmethod
    def __goal() -> SubGoal:
        """
        Build one accepted sub-goal.
        """

        return SubGoal(
            index=0,
            objective="Tap the login button",
            success=SuccessFixtures.command(quote="Tap", intent="Tap the login button"),
        )

    def __checkpoint(self) -> Dict[str, Any]:
        """
        Build a persisted checkpoint blob carrying one accepted sub-goal.
        """

        return {
            "intent": "Tap the login button",
            "current_sub_goal_index": 0,
            "sub_goals": [GoalState(goal=self.__goal()).model_dump(mode="json")],
        }

    async def test_read_returns_none_on_fresh_run(self) -> None:
        """
        A checkpoint with no persisted plan resolves to a fresh run.
        """

        self.assertIsNone(await self.__store.read(run=_FakeGraph(values={}), workflow="run-1"))

    async def test_read_rebuilds_plan_from_checkpoint(self) -> None:
        """
        A checkpoint holding an accepted plan is rebuilt into a typed Plan.
        """

        graph = _FakeGraph(
            values={IntentStateKey.AGENT_STATE_CHECKPOINT.value: self.__checkpoint()}
        )
        plan = await self.__store.read(run=graph, workflow="run-1")

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.intent, "Tap the login button")
        self.assertEqual(len(plan.goals), 1)
        self.assertIsInstance(plan.goals[0].success, CommandSuccess)

    async def test_read_returns_none_without_sub_goals(self) -> None:
        """
        A checkpoint present but carrying no sub-goals is not a reusable plan.
        """

        graph = _FakeGraph(
            values={IntentStateKey.AGENT_STATE_CHECKPOINT.value: {"intent": "x", "sub_goals": []}}
        )

        self.assertIsNone(await self.__store.read(run=graph, workflow="run-1"))

    async def test_seed_writes_checkpoint_attributed_to_start(self) -> None:
        """
        Seeding commits the plan and index attributed to START, before any node runs.
        """

        graph = _FakeGraph(values={})
        plan = Plan(intent="Tap the login button", cursor=0, goals=(self.__goal(),))

        await self.__store.seed(run=graph, workflow="run-1", plan=plan)

        self.assertEqual(len(graph.updates), 1)
        values, as_node = graph.updates[0]
        self.assertEqual(as_node, START)
        self.assertIn(IntentStateKey.AGENT_STATE_CHECKPOINT.value, values)
        self.assertEqual(values[IntentStateKey.CURRENT_SUB_GOAL_INDEX.value], 0)
