from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from tests.builders import SuccessFixtures

from fathom.adapters.checkpoint import LangGraphPlanStore, SqliteCheckpointStore
from fathom.constants.graph import NodeName
from fathom.runtime.checkpoint_serde import CheckpointSerdeFactory
from fathom.schemas.checkpoint import SqliteCheckpointPolicy
from fathom.schemas.plan import Plan
from fathom.schemas.subgoal import SubGoal
from fathom.strategies.graph.state import IntentGraphState


class _GroundSpy:
    """
    Entry-node double counting executions so a test can prove no node ran before persistence.
    """

    def __init__(self) -> None:
        self.runs = 0

    def __call__(self, state: IntentGraphState) -> Dict[str, Any]:
        _ = state
        self.runs += 1
        return {}


class LangGraphPlanStoreDurabilityTest(unittest.IsolatedAsyncioTestCase):
    """
    Proves the plan store seeds durably through the real SQLite checkpointer before graph entry.
    """

    @staticmethod
    def __store(*, directory: Path) -> SqliteCheckpointStore:
        """
        Build a real SQLite checkpoint store with the production serde.
        """

        return SqliteCheckpointStore(
            directory=directory,
            policy=SqliteCheckpointPolicy(),
            serde=CheckpointSerdeFactory.build(),
        )

    def __graph(self, *, checkpointer: Any, ground: _GroundSpy) -> CompiledStateGraph:
        """
        Compile a minimal intent graph whose entry node is the counting spy.
        """

        workflow = StateGraph(IntentGraphState)
        workflow.add_node(NodeName.GROUND.value, ground)
        workflow.set_entry_point(NodeName.GROUND.value)
        workflow.add_edge(NodeName.GROUND.value, END)
        return workflow.compile(checkpointer=checkpointer)

    @staticmethod
    def __plan() -> Plan:
        """
        Build a plan carrying one accepted sub-goal.
        """

        return Plan(
            intent="Tap the login button",
            cursor=0,
            goals=(
                SubGoal(
                    index=0,
                    objective="Tap the login button",
                    success=SuccessFixtures.command(quote="Tap", intent="Tap the login button"),
                ),
            ),
        )

    async def test_seeded_plan_is_durable_before_graph_entry_and_survives_crash(self) -> None:
        """
        The seeded plan persists before any node runs and is byte-identical after a crash-restart.
        """

        workflow_id = "plan-durability"
        plan = self.__plan()

        with tempfile.TemporaryDirectory() as directory:
            store = self.__store(directory=Path(directory))

            store_adapter = LangGraphPlanStore()

            ground_first = _GroundSpy()
            async with store.open(workflow_id=workflow_id) as saver:
                graph = self.__graph(checkpointer=saver, ground=ground_first)
                await store_adapter.seed(run=graph, workflow=workflow_id, plan=plan)

                snapshot = await graph.aget_state({"configurable": {"thread_id": workflow_id}})
                self.assertEqual(ground_first.runs, 0)
                self.assertEqual(snapshot.next, (NodeName.GROUND.value,))

            ground_second = _GroundSpy()
            async with store.open(workflow_id=workflow_id) as saver:
                graph = self.__graph(checkpointer=saver, ground=ground_second)
                restored = await store_adapter.read(run=graph, workflow=workflow_id)

                self.assertEqual(restored, plan)
                self.assertEqual(ground_second.runs, 0)

    async def test_fresh_run_has_no_persisted_plan(self) -> None:
        """
        A run whose checkpoint was never seeded reports no plan.
        """

        workflow_id = "plan-fresh"

        with tempfile.TemporaryDirectory() as directory:
            store = self.__store(directory=Path(directory))
            async with store.open(workflow_id=workflow_id) as saver:
                graph = self.__graph(checkpointer=saver, ground=_GroundSpy())
                restored = await LangGraphPlanStore().read(run=graph, workflow=workflow_id)

                self.assertIsNone(restored)
