from __future__ import annotations

import logging
from typing import List, Optional, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from fathom.constants.graph import NodeName
from fathom.constants.state import CommonStateKey, IntentStateKey
from fathom.interfaces.graph import GraphBuilder
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.intent.nodes import IntentGraphFactory
from fathom.strategies.graph.state import IntentGraphState

logger = logging.getLogger(__name__)


class IntentGraphBuilder(GraphBuilder):
    """
    Constructs the LangGraph workflow for intent execution.
    """

    def __init__(self, context: GraphContext) -> None:
        self.__context = context

    def build(
        self,
        interrupt_before: Optional[List[str]] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> CompiledStateGraph:
        """
        Builds and compiles the graph.
        """

        workflow = StateGraph(IntentGraphState)
        nodes = IntentGraphFactory.build(context=self.__context)

        workflow.add_node(NodeName.GROUND, nodes[NodeName.GROUND])
        workflow.add_node(NodeName.ANALYZE, nodes[NodeName.ANALYZE])
        workflow.add_node(NodeName.EXECUTE, nodes[NodeName.EXECUTE])
        workflow.add_node(NodeName.RECORD, nodes[NodeName.RECORD])

        workflow.set_entry_point(NodeName.GROUND)
        workflow.add_edge(NodeName.GROUND, NodeName.ANALYZE)

        workflow.add_conditional_edges(
            NodeName.ANALYZE,
            self.__route_after_analyze,
            {
                NodeName.END: NodeName.END,
                NodeName.GROUND: NodeName.GROUND,
                NodeName.EXECUTE: NodeName.EXECUTE,
            },
        )

        workflow.add_edge(NodeName.EXECUTE, NodeName.RECORD)

        workflow.add_conditional_edges(
            NodeName.RECORD,
            self.__route_after_record,
            {
                NodeName.END: NodeName.END,
                NodeName.GROUND: NodeName.GROUND,
            },
        )

        return workflow.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)

    def __route_after_analyze(self, state: IntentGraphState) -> str:
        """
        Route after analyze based on completion and retry status.
        """

        is_complete = state.get(cast("str", CommonStateKey.IS_COMPLETE))
        should_retry = state.get(cast("str", IntentStateKey.SHOULD_RETRY))
        planned_step = state.get(cast("str", IntentStateKey.PLANNED_STEP))

        logger.info(
            f"[ROUTING] After ANALYZE: is_complete={is_complete}, "
            f"should_retry={should_retry}, has_planned_step={planned_step is not None}, "
            f"planned_step_type={type(planned_step).__name__}"
        )

        if is_complete:
            logger.info("[ROUTING] -> END (is_complete=True)")
            return NodeName.END

        if should_retry:
            logger.info("[ROUTING] -> GROUND (should_retry=True)")
            return NodeName.GROUND

        if not planned_step:
            logger.info(f"[ROUTING] -> GROUND (no planned_step, value={planned_step})")
            return NodeName.GROUND

        logger.info("[ROUTING] -> EXECUTE")
        return NodeName.EXECUTE

    def __route_after_record(self, state: IntentGraphState) -> str:
        """
        Route after record based on completion status.
        """

        if state.get(cast("str", CommonStateKey.IS_COMPLETE)):
            return NodeName.END

        return NodeName.GROUND
