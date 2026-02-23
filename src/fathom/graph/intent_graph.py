"""
LangGraph StateGraph for intent-based Fathom workflows.

Topology::

    ┌────────┐
    │ ground │──capture_failed?──→ END
    └───┬────┘
        │
    ┌───▼────────┐
    │  validate  │
    └───┬────────┘
        │
    ┌───▼──────┐
    │ hierarchy│
    └───┬──────┘
        │
    ┌───▼────────┐       ┌───────┐
    │  analyze   │─retry→│ ground│
    └───┬────────┘       └───────┘
        │
        ├─complete (no step)──→ END
        │
    ┌───▼────────┐
    │  resolve   │
    └───┬────────┘
        │
    ┌───▼────────┐
    │  execute   │
    └───┬────────┘
        │
    ┌───▼────────┐
    │  record    │──can_continue?──→ ground (loop)
    └───┬────────┘
        │
       END
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import asyncio

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph.state import CompiledStateGraph

    from fathom.graph.nodes import NodeContext

from logging import getLogger

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from fathom.agent.planner import StepPlanner
from fathom.graph.nodes import (
    NodeContext,
    build_nodes,
    make_route_after_analyze,
    make_route_after_ground,
    make_route_after_record,
)
from fathom.graph.state import FathomGraphState
from fathom.interfaces import IMemoryProvider
from fathom.tools.capture import CaptureTool
from fathom.tools.device import DeviceTool

logger = getLogger(__name__)


def build_intent_graph(
    intent: str,
    planner: StepPlanner,
    device: DeviceTool,
    capture: CaptureTool,
    memory: IMemoryProvider,
    *,
    max_steps: int = 100,
    use_xml: bool = False,
    step_timeout: float = 15.0,
    workflow_id: str = "default",
    checkpointer: Optional[MemorySaver] = None,
    cancel_event: Optional[asyncio.Event] = None,
    pause_event: Optional[asyncio.Event] = None,
    package_name: str = "",
    human_in_loop: bool = False,
) -> tuple["CompiledStateGraph[Any, Any, Any]", NodeContext]:
    """
    Build and compile a LangGraph :class:`StateGraph` for intent execution.

    The returned compiled graph accepts a :class:`FathomGraphState` initial
    state and drives the full *ground → hierarchy → analyze → resolve →
    execute → record* cycle until the intent completes or max steps are
    reached.

    Parameters
    ----------
    intent:
        The goal to achieve.
    planner:
        Pre-built ``StepPlanner`` (carries the VisionTool / LLM reference).
    device:
        ADB device tool for physical interactions.
    capture:
        Screen capture tool.
    memory:
        Persistent memory provider (SQLite, etc.).
    max_steps:
        Safety-limit on loop iterations.
    use_xml:
        Whether XML hierarchy grounding is enabled.
    step_timeout:
        Per-step timeout in seconds.
    workflow_id:
        Unique run identifier (used for checkpoints).
    checkpointer:
        Optional LangGraph checkpointer. Defaults to in-memory.
    cancel_event:
        Optional ``asyncio.Event`` signalled on workflow cancellation.
    package_name:
        Target app package name for memory scoping.

    Returns
    -------
    tuple[CompiledStateGraph, NodeContext]
        A compiled graph and its shared context.
    """

    # ── 1. Shared context (closed over by node functions) ──────────────
    ctx = NodeContext(
        intent=intent,
        planner=planner,
        device=device,
        capture=capture,
        memory=memory,
        max_steps=max_steps,
        use_xml=use_xml,
        step_timeout=step_timeout,
        workflow_id=workflow_id,
        cancel_event=cancel_event,
        pause_event=pause_event,
        package_name=package_name,
        human_in_loop=human_in_loop,
    )

    # ── 2. Build node functions ────────────────────────────────────────
    nodes = build_nodes(ctx)

    # ── 3. Assemble the graph ──────────────────────────────────────────
    graph = StateGraph(FathomGraphState)

    graph.add_node("ground", nodes["ground"])
    graph.add_node("validate", nodes["validate"])
    graph.add_node("hierarchy", nodes["hierarchy"])
    graph.add_node("analyze", nodes["analyze"])
    graph.add_node("resolve", nodes["resolve"])
    graph.add_node("execute", nodes["execute"])
    graph.add_node("record", nodes["record"])

    # ── 4. Wire edges ─────────────────────────────────────────────────
    graph.set_entry_point("ground")

    graph.add_conditional_edges(
        "ground",
        make_route_after_ground(ctx),
        {"validate": "validate", "done": END},
    )

    graph.add_edge("validate", "hierarchy")

    graph.add_edge("hierarchy", "analyze")

    graph.add_conditional_edges(
        "analyze",
        make_route_after_analyze(ctx),
        {"resolve": "resolve", "ground": "ground", "done": END},
    )

    graph.add_edge("resolve", "execute")
    graph.add_edge("execute", "record")

    graph.add_conditional_edges(
        "record",
        make_route_after_record(ctx),
        {"ground": "ground", "done": END},
    )

    # ── 5. Compile ────────────────────────────────────────────────────
    saver = checkpointer or MemorySaver()
    compiled = graph.compile(checkpointer=saver)

    logger.info(
        f"Intent graph compiled  (nodes={len(nodes)}, intent={intent!r}, max_steps={max_steps})"
    )

    return compiled, ctx
