from __future__ import annotations

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

        if state.get(cast("str", CommonStateKey.IS_COMPLETE)):
            return NodeName.END

        if state.get(cast("str", IntentStateKey.SHOULD_RETRY)):
            return NodeName.GROUND

        if not state.get(cast("str", IntentStateKey.PLANNED_STEP)):
            return NodeName.GROUND

        return NodeName.EXECUTE

    def __route_after_record(self, state: IntentGraphState) -> str:
        """
        Route after record based on completion status.
        """

        if state.get(cast("str", CommonStateKey.IS_COMPLETE)):
            return NodeName.END

        return NodeName.GROUND
