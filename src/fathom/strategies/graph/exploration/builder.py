from __future__ import annotations

from typing import Any, List, Optional, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from fathom.constants.graph import NodeName
from fathom.constants.state import CommonStateKey, ExplorationStateKey
from fathom.interfaces.graph import GraphBuilder
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.exploration.nodes import ExplorationGraphFactory
from fathom.strategies.graph.exploration.state import ExplorationGraphState


class ExplorationGraphBuilder(GraphBuilder):
    """
    Constructs the LangGraph workflow for autonomous application exploration.
    """

    def __init__(self, context: GraphContext) -> None:
        self.__context = context

    def build(
        self,
        interrupt_before: Optional[List[str]] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> CompiledStateGraph:
        """
        Builds and compiles the exploration graph.
        """

        workflow = StateGraph(cast("Any", ExplorationGraphState))
        nodes = ExplorationGraphFactory.build(self.__context)

        # Nodes
        workflow.add_node(NodeName.GROUND, nodes[NodeName.GROUND])

        workflow.add_node(NodeName.SCAN, nodes[NodeName.SCAN])
        workflow.add_node(NodeName.EXECUTE, nodes[NodeName.EXECUTE])

        workflow.add_node(NodeName.RECORD, nodes[NodeName.RECORD])
        workflow.add_node(NodeName.NAVIGATE, nodes[NodeName.NAVIGATE])
        workflow.add_node(NodeName.BFS_ROUTE, nodes[NodeName.BFS_ROUTE])

        # Entry point and edges
        workflow.set_entry_point(NodeName.GROUND)
        workflow.add_edge(NodeName.GROUND, NodeName.SCAN)

        workflow.add_conditional_edges(
            NodeName.SCAN,
            self.__route_after_scan,
            {
                NodeName.EXECUTE: NodeName.EXECUTE,
                NodeName.BFS_ROUTE: NodeName.BFS_ROUTE,
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

        # BFS specific navigation edges
        workflow.add_edge(NodeName.BFS_ROUTE, NodeName.NAVIGATE)
        workflow.add_edge(NodeName.NAVIGATE, NodeName.GROUND)

        return workflow.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)

    def __route_after_scan(self, state: ExplorationGraphState) -> str:
        """
        Route after scan based on content exhaustion.
        """

        if state.get(ExplorationStateKey.CONTENT_EXHAUSTED):
            return NodeName.BFS_ROUTE

        return NodeName.EXECUTE

    def __route_after_record(self, state: ExplorationGraphState) -> str:
        """
        Route after record based on completion status.
        """

        if state.get(CommonStateKey.IS_COMPLETE):
            return NodeName.END

        return NodeName.GROUND
