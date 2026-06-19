from __future__ import annotations

import unittest
from typing import Any, Dict

from fathom.constants.state import IntentStateKey
from fathom.core.agent.state import AgentState
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.subgoal import SubGoal
from fathom.strategies.graph.intent.nodes.persistence import GraphStatePersistence


class _StubContext:
    """
    :class:`GraphContext` test double exposing only the surface area
    :class:`GraphStatePersistence` actually reads.

    The real :class:`GraphContext` has 30+ ports and services. The
    persistence helper consumes none of them — it reads
    ``workflow_id``, ``agent_state``, and ``set_agent_state`` only.
    Constructing a thin stub here keeps the tests focused on the
    round-trip semantics rather than on context assembly.
    """

    def __init__(
        self,
        *,
        agent_state: AgentState,
        workflow_id: str = "run-test",
        capabilities: RuntimeCapabilities = RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
    ) -> None:
        """
        Hold the live :class:`AgentState` and record any replacement
        via :meth:`set_agent_state` so the test can assert the swap.
        """

        self.agent_state = agent_state
        self.workflow_id = workflow_id
        self.capabilities = capabilities
        self.replaced_state: AgentState | None = None

    def set_agent_state(self, state: AgentState) -> None:
        """
        Replace the live AgentState with the restored checkpoint.

        The real context exposes the same setter signature; the stub
        also records the replacement so the test can assert that
        :meth:`GraphStatePersistence.restore` reached this seam.
        """

        self.agent_state = state
        self.replaced_state = state


class GraphStatePersistenceRoundTripTest(unittest.TestCase):
    """
    Pins the :class:`GraphStatePersistence` round-trip contract.

    The persist/restore pair must round-trip :class:`AgentState`
    through the graph dict so a LangGraph checkpoint can resume a run
    after restart. The tests cover: persist writes both keys with the
    right shape, restore replaces the live state when a checkpoint is
    present, restore advances the sub-goal index when only the index
    survived, and restore is a no-op for empty state dicts.
    """

    @staticmethod
    def __sub_goals() -> list[SubGoal]:
        """
        Three deterministic sub-goals so the sub-goal-index-only
        restore path can move the index without re-loading the goals.
        """

        return [
            SubGoal(index=0, description="Open the app"),
            SubGoal(index=1, description="Tap on search"),
            SubGoal(index=2, description="Type the query"),
        ]

    def __build_state(self, *, intent: str = "search masala dosa") -> AgentState:
        """
        Live :class:`AgentState` seeded with the deterministic
        sub-goal list and the supplied intent.
        """

        state = AgentState(
            intent=intent, capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False))
        )
        state.set_sub_goals(self.__sub_goals())
        return state

    def test_persist_writes_checkpoint_and_index_keys(self) -> None:
        """
        :meth:`persist` must write both the checkpoint payload and the
        sub-goal index into the supplied result dict using the canonical
        :class:`IntentStateKey` string values.
        """

        state = self.__build_state()
        state.record_sub_goal_action()
        helper = GraphStatePersistence(context=_StubContext(agent_state=state))  # type: ignore[arg-type]
        result: Dict[Any, Any] = {}

        helper.persist(result=result)

        self.assertIn(IntentStateKey.AGENT_STATE_CHECKPOINT.value, result)
        self.assertIn(IntentStateKey.CURRENT_SUB_GOAL_INDEX.value, result)
        self.assertIsInstance(result[IntentStateKey.AGENT_STATE_CHECKPOINT.value], dict)
        self.assertEqual(result[IntentStateKey.CURRENT_SUB_GOAL_INDEX.value], 0)

    def test_round_trip_preserves_intent_and_progress(self) -> None:
        """
        A full persist→restore cycle must rebuild an :class:`AgentState`
        with the same intent and sub-goal progress. The context's
        :meth:`set_agent_state` must be called with the restored state.
        """

        original = self.__build_state(intent="search masala dosa")
        original.record_sub_goal_action()
        original.record_sub_goal_action()
        source_context = _StubContext(agent_state=original)
        helper = GraphStatePersistence(context=source_context)  # type: ignore[arg-type]
        graph_state: Dict[Any, Any] = {}
        helper.persist(result=graph_state)

        restore_target = _StubContext(
            agent_state=AgentState(
                intent="placeholder",
                capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
            ),
        )
        restore_helper = GraphStatePersistence(context=restore_target)  # type: ignore[arg-type]
        restore_helper.restore(state=graph_state)  # type: ignore[arg-type]

        self.assertIsNotNone(restore_target.replaced_state)
        assert restore_target.replaced_state is not None
        self.assertEqual(restore_target.replaced_state.intent, "search masala dosa")
        self.assertEqual(restore_target.replaced_state.current_sub_goal_action_count, 2)

    def test_restore_advances_subgoal_index_when_only_index_survives(self) -> None:
        """
        Some graph-state payloads carry only the sub-goal index (e.g.
        partial recovery patches). :meth:`restore` must accept that
        path and move the index without requiring a full checkpoint.
        """

        live = self.__build_state()
        context = _StubContext(agent_state=live)
        helper = GraphStatePersistence(context=context)  # type: ignore[arg-type]

        helper.restore(state={IntentStateKey.CURRENT_SUB_GOAL_INDEX: 2})  # type: ignore[arg-type]

        self.assertEqual(context.agent_state.current_sub_goal_index, 2)

    def test_restore_with_empty_state_is_a_noop(self) -> None:
        """
        An empty graph-state dict must not mutate the live AgentState
        and must not call :meth:`set_agent_state`.
        """

        live = self.__build_state()
        context = _StubContext(agent_state=live)
        helper = GraphStatePersistence(context=context)  # type: ignore[arg-type]

        helper.restore(state={})  # type: ignore[arg-type]

        self.assertIsNone(context.replaced_state)
        self.assertEqual(context.agent_state.current_sub_goal_index, 0)
