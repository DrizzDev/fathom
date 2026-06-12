from __future__ import annotations

from typing import List, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from fathom.constants.graph import NodeName
from fathom.interfaces.graph import GraphBuilder
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.exploration.nodes import ExplorationGraphFactory
from fathom.strategies.graph.exploration.state import ExplorationGraphState


class ExplorationGraphBuilder(GraphBuilder):
    """
    Constructs the LangGraph workflow for autonomous DFS application exploration.

    Topology::

        ground ──> bfs_route ──SCAN──────> scan ──> execute ─┐
              (capture?)      ├─BACKTRACK─> navigate ─────────┤
                              └─ADVANCE ──> navigate ─────────┤
                                                              v
                              ground <── record <─────────────┘
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

        workflow = StateGraph(ExplorationGraphState)
        provider = ExplorationGraphFactory.build(self.__context)
        nodes = provider.node_callables()

        # 1. Add nodes
        workflow.add_node(NodeName.GROUND, nodes[NodeName.GROUND])
        workflow.add_node(NodeName.BFS_ROUTE, nodes[NodeName.BFS_ROUTE])
        workflow.add_node(NodeName.SCAN, nodes[NodeName.SCAN])
        workflow.add_node(NodeName.EXECUTE, nodes[NodeName.EXECUTE])
        workflow.add_node(NodeName.NAVIGATE, nodes[NodeName.NAVIGATE])
        workflow.add_node(NodeName.RECORD, nodes[NodeName.RECORD])

        # 2. Entry point
        workflow.set_entry_point(NodeName.GROUND)

        # 3. Conditional edges driven by the DFS phase machine
        workflow.add_conditional_edges(
            NodeName.GROUND,
            provider.after_ground,
            {NodeName.BFS_ROUTE: NodeName.BFS_ROUTE, NodeName.END: NodeName.END},
        )

        workflow.add_conditional_edges(
            NodeName.BFS_ROUTE,
            provider.after_bfs_route,
            {
                NodeName.SCAN: NodeName.SCAN,
                NodeName.NAVIGATE: NodeName.NAVIGATE,
                NodeName.END: NodeName.END,
            },
        )

        workflow.add_conditional_edges(
            NodeName.SCAN,
            provider.after_scan,
            {
                NodeName.EXECUTE: NodeName.EXECUTE,
                NodeName.BFS_ROUTE: NodeName.BFS_ROUTE,
                NodeName.END: NodeName.END,
            },
        )

        # 4. Both action paths converge on record, which loops back to ground
        workflow.add_edge(NodeName.EXECUTE, NodeName.RECORD)
        workflow.add_edge(NodeName.NAVIGATE, NodeName.RECORD)

        workflow.add_conditional_edges(
            NodeName.RECORD,
            provider.after_record,
            {NodeName.GROUND: NodeName.GROUND, NodeName.END: NodeName.END},
        )

        return workflow.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
