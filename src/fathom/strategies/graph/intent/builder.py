"""
Graph builder for intent execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional

from langgraph.graph import END, StateGraph

from fathom.constants.graph import GraphKey, NodeName
from fathom.interfaces.graph import GraphBuilder
from fathom.strategies.graph.intent.nodes import build_intent_nodes
from fathom.strategies.graph.state import IntentGraphState

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from fathom.strategies.graph.context import GraphContext


class IntentGraphBuilder(GraphBuilder):
    """
    Builder class for the Intent Execution Graph.
    Implements clean separation of node definition, routing, and compilation.
    """

    def __init__(self, context: GraphContext) -> None:
        """
        Initialize builder with shared graph context.
        """
        self.__context = context

    def build(
        self,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        interrupt_before: Optional[List[str]] = None,
    ) -> CompiledStateGraph:
        """
        Builds and compiles the Intent Execution Graph.
        """

        # 1. Build Nodes
        nodes = build_intent_nodes(context=self.__context)

        # 2. Define Workflow
        workflow = StateGraph(state_schema=IntentGraphState)

        workflow.add_node(NodeName.GROUND, nodes[NodeName.GROUND])
        workflow.add_node(NodeName.ANALYZE, nodes[NodeName.ANALYZE])
        workflow.add_node(NodeName.EXECUTE, nodes[NodeName.EXECUTE])
        workflow.add_node(NodeName.RECORD, nodes[NodeName.RECORD])

        # 3. Define Edges
        workflow.set_entry_point(key=NodeName.GROUND)

        workflow.add_edge(start_key=NodeName.GROUND, end_key=NodeName.ANALYZE)

        # Conditional Routing
        workflow.add_conditional_edges(
            source=NodeName.ANALYZE,
            path=self.__route_after_analyze,
            path_map={
                NodeName.EXECUTE: NodeName.EXECUTE,
                NodeName.GROUND: NodeName.GROUND,
                NodeName.END: END,
            },
        )

        workflow.add_edge(start_key=NodeName.EXECUTE, end_key=NodeName.RECORD)

        workflow.add_conditional_edges(
            source=NodeName.RECORD,
            path=self.__route_after_record,
            path_map={NodeName.GROUND: NodeName.GROUND, NodeName.END: END},
        )

        # 4. Compile with injected dependencies
        return workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=interrupt_before or [],
        )

    def __route_after_analyze(
        self, state: IntentGraphState
    ) -> Literal[NodeName.EXECUTE, NodeName.GROUND, NodeName.END]:
        """Routes execution after analysis node."""
        if state.get(GraphKey.IS_COMPLETE):
            return NodeName.END

        if state.get(GraphKey.SHOULD_RETRY):
            return NodeName.GROUND

        if not state.get(GraphKey.PLANNED_STEP):
            return NodeName.END

        return NodeName.EXECUTE

    def __route_after_record(
        self, state: IntentGraphState
    ) -> Literal[NodeName.GROUND, NodeName.END]:
        """Routes execution after recording node."""
        if state.get(GraphKey.IS_COMPLETE):
            return NodeName.END

        if self.__context.agent_state.step_count >= self.__context.max_steps:
            return NodeName.END

        if self.__context.is_cancelled:
            return NodeName.END

        return NodeName.GROUND
