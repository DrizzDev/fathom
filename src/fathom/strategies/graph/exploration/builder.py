"""
Graph builder for exploration execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from langgraph.graph import END, StateGraph

from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.exploration.nodes import build_exploration_nodes
from fathom.strategies.graph.exploration.state import ExplorationGraphState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_exploration_graph(
    context: GraphContext,
) -> CompiledStateGraph:
    """
    Builds the Exploration Execution Graph.
    """
    
    nodes = build_exploration_nodes(context=context)
    
    workflow = StateGraph(ExplorationGraphState)
    
    workflow.add_node("ground", nodes["ground"])
    workflow.add_node("bfs_route", nodes["bfs_route"])
    workflow.add_node("scan", nodes["scan"])
    workflow.add_node("execute", nodes["execute"])
    workflow.add_node("navigate", nodes["navigate"])
    workflow.add_node("record", nodes["record"])
    
    workflow.set_entry_point("ground")
    
    workflow.add_edge("ground", "bfs_route")
    
    def route_bfs(state: ExplorationGraphState) -> Literal["scan", "navigate", "end"]:
        phase = state.get("bfs_phase", "scan")
        if context.is_cancelled: return "end"
        if state.get("is_complete"): return "end"
        
        if phase == "scan":
            return "scan"
        return "navigate" # Fallback/TODO

    workflow.add_conditional_edges("bfs_route", route_bfs, ["scan", "navigate", "end"])
    
    def route_scan(state: ExplorationGraphState) -> Literal["execute", "bfs_route", "end"]:
        if context.is_cancelled: return "end"
        
        if state.get("content_exhausted"):
            # Switch phase? For now just loop or end.
            return "end" # Stop if screen exhausted (simple version)
            
        if not state.get("action"):
            return "bfs_route"
            
        return "execute"

    workflow.add_conditional_edges("scan", route_scan, ["execute", "bfs_route", "end"])
    
    workflow.add_edge("execute", "record")
    workflow.add_edge("navigate", "record")
    
    def route_record(state: ExplorationGraphState) -> Literal["ground", "end"]:
        if state.get("is_complete") or context.is_cancelled:
            return "end"
        return "ground"

    workflow.add_conditional_edges("record", route_record, ["ground", "end"])
    
    return workflow.compile()
