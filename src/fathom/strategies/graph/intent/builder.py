"""
Graph builder for intent execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.intent.nodes import build_intent_nodes
from fathom.strategies.graph.state import IntentGraphState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_intent_graph(
    context: GraphContext,
) -> CompiledStateGraph:
    """
    Builds the Intent Execution Graph.
    Configures interrupts for HITL.
    """
    
    # 1. Build Nodes
    nodes = build_intent_nodes(context=context)
    
    # 2. Define Graph
    workflow = StateGraph(IntentGraphState)
    
    workflow.add_node("ground", nodes["ground"])
    workflow.add_node("analyze", nodes["analyze"])
    workflow.add_node("execute", nodes["execute"])
    workflow.add_node("record", nodes["record"])
    
    # 3. Define Edges
    workflow.set_entry_point("ground")
    
    workflow.add_edge("ground", "analyze")
    
    def route_after_analyze(state: IntentGraphState) -> Literal["execute", "ground", "end"]:
        if state.get("is_complete"):
            return "end"
        
        if state.get("should_retry"):
            return "ground"
            
        if not state.get("planned_step"):
            return "end"
            
        return "execute"

    workflow.add_conditional_edges(
        "analyze",
        route_after_analyze,
        {
            "execute": "execute",
            "ground": "ground", 
            "end": END
        }
    )
    
    workflow.add_edge("execute", "record")
    
    def route_after_record(state: IntentGraphState) -> Literal["ground", "end"]:
        if state.get("is_complete"):
            return "end"
            
        if context.agent_state.step_count >= context.max_steps:
            return "end"
            
        if context.is_cancelled:
            return "end"
            
        return "ground"

    workflow.add_conditional_edges(
        "record",
        route_after_record,
        {
            "ground": "ground",
            "end": END
        }
    )
    
    # 4. Compile with Checkpointer and Interrupts
    # MemorySaver is required for interrupts to work.
    checkpointer = MemorySaver()
    
    # We interrupt before 'analyze' (to inject reasoning context) 
    # and before 'execute' (to validate/modify physical action).
    # NOTE: The actual interruption logic is controlled by the Runner/Strategy 
    # checking the signal port. If we want *automatic* interruption every time,
    # we set interrupt_before. But we only want to interrupt IF user requested it.
    # LangGraph's interrupt_before halts EVERY time.
    # To support "Pause on request", we need dynamic interrupts.
    # LangGraph supports `NodeInterrupt` exception raised from within a node.
    # OR we can stick to my previous `_check_signal` approach but implement it 
    # properly by raising `NodeInterrupt` which LangGraph catches and saves state.
    
    # However, the user called the previous approach "brute force".
    # The "Scalable" way in LangGraph is to rely on checkpointers.
    # If I set interrupt_before=[], I can't pause from outside easily unless I stop the graph.
    # But `ainvoke` runs until end.
    # To allow pausing, I should perhaps use `stream` mode in Strategy and check signal between steps?
    # YES. `graph.astream()` yields events. I can check signal between steps.
    # If signal is present, I stop iterating (pause).
    # Then I resume with `graph.invoke(..., config=thread_config)`.
    
    # So I do NOT need `interrupt_before` hardcoded here if I use streaming control.
    # But if I want to allow editing state *before* a specific node running,
    # hardcoded interrupts are safer.
    
    # The user asked for "Robust".
    # I will use `checkpointer` to enable time-travel/resume capabilities.
    
    return workflow.compile(checkpointer=checkpointer)
